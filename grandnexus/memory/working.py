"""
working.py - Hybrid Working Memory for GrandNexus

This module unifies:
  1) Solid LRU / LFU / priority eviction logic.
  2) Text + embedding-based retrieval approach (like a "Gemini + ChatGPT" synergy).
  3) Optional expiration times for short-lived items.
  4) Recency bias if desired.
  5) Potential auto-transfer to another store on eviction.

It is fully thread-safe via RLock, and can be integrated with other memory or
cognition modules in GrandNexus.
"""

import math  # réparation 6.5 : fourni avant par l'espace de noms partagé du notebook
import logging
import threading
import time
import uuid
from collections import deque, defaultdict
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Callable, Tuple, Union
import numpy as np


@dataclass
class WorkingMemoryItem:
    """
    Represents a single item stored in working memory.

    Attributes:
        item_id (str): Unique identifier.
        content (Any): Actual data (text, partial results, etc.).
        source (str): Which module or process inserted it.
        timestamp (float): Creation or last update time.
        importance (float): Salience or relevance (0..1).
        metadata (Dict[str, Any]): Additional info.
        expiration_time (Optional[float]): If set, we consider the item expired after that time.
        embedding (Optional[np.ndarray]): If config provides an embedding_function, we store the computed vector here.
    """
    item_id: str
    content: Any
    source: str
    timestamp: float = field(default_factory=time.time)
    importance: float = 0.5
    metadata: Dict[str, Any] = field(default_factory=dict)
    expiration_time: Optional[float] = None
    embedding: Optional[np.ndarray] = None

    def __repr__(self) -> str:
        return (f"<WorkingMemoryItem id={self.item_id[-8:]}, "
                f"source={self.source}, importance={self.importance:.2f}>")

class WorkingMemoryConfig:
    """
    Configuration for the WorkingMemory.

    Attributes:
        capacity (int): Max number of items allowed.
        eviction_policy (str): 'LRU', 'LFU', or 'priority'.
        default_importance (float): Default if importance not provided
        expiration_enabled (bool): If True, we consider item.expiration_time for removal
        default_expiration_time (Optional[float]): If set, used when adding items
        embedding_function (Optional[Callable[[Any], np.ndarray]]): For item embeddings
        recency_bias (float): Weighted factor for recency in retrieval scoring
        auto_transfer_callback (Optional[Callable[[WorkingMemoryItem], None]]):
            If set, we call it before eviction, allowing items to be "saved" elsewhere
    """
    def __init__(self,
                 capacity: int = 100,
                 eviction_policy: str = "LRU",
                 default_importance: float = 0.5,
                 expiration_enabled: bool = False,
                 default_expiration_time: Optional[float] = None,
                 embedding_function: Optional[Callable[[Any], np.ndarray]] = None,
                 recency_bias: float = 0.0,
                 auto_transfer_callback: Optional[Callable[[WorkingMemoryItem], None]] = None
                 ):
        self.capacity = capacity
        self.eviction_policy = eviction_policy
        self.default_importance = default_importance
        self.expiration_enabled = expiration_enabled
        self.default_expiration_time = default_expiration_time
        self.embedding_function = embedding_function
        self.recency_bias = recency_bias
        self.auto_transfer_callback = auto_transfer_callback


class WorkingMemory:
    """
    A robust working memory module for short-lived data, with concurrency,
    multiple eviction policies, item embeddings, text-based + embedding retrieval, etc.

    - store_item(): Add new item
    - get_item(): Retrieve by ID
    - retrieve(): High-level retrieval by text / embedding / importance
    - update_item(), remove_item()
    - Evictions done automatically when capacity is exceeded
    - Optionally call auto_transfer_callback on items about to be evicted
    - Clear or get_all_items for debugging
    """

    def __init__(self, config: WorkingMemoryConfig):
        self.logger = logging.getLogger("WorkingMemory")
        self.config = config
        self.lock = threading.RLock()

        # item_id -> WorkingMemoryItem
        self.memory: Dict[str, WorkingMemoryItem] = {}

        # LRU uses a deque for the order (most recently used at the right)
        self._access_order: deque[str] = deque()
        # LFU uses a map from item_id -> access_count
        self._access_counts: Dict[str, int] = defaultdict(int)

        # Recent retrievals/operations cache for optimizing frequently accessed items
        self._recent_operations_cache: Dict[str, Any] = {}
        self._cache_hits = 0
        self._cache_misses = 0

        self.logger.info(f"WorkingMemory initialized: capacity={config.capacity}, eviction={config.eviction_policy}")

        # Register with memory manager if available
        if hasattr(config, 'memory_manager') and config.memory_manager:
            self._register_with_memory_manager(config.memory_manager)

    def store_item(self,
                   content: Any,
                   source: str,
                   item_id: Optional[str] = None,
                   importance: Optional[float] = None,
                   metadata: Optional[Dict[str, Any]] = None,
                   expiration_time: Optional[float] = None,
                   embedding: Optional[np.ndarray] = None) -> str:
        """
        Add a new item to working memory. May trigger eviction if over capacity.

        Args:
            content: The item content
            source: Which module or subsystem created it
            item_id: If you want to specify a custom ID (else we create a new UUID)
            importance: 0..1, if omitted use config.default_importance
            metadata: Additional info
            expiration_time: If omitted but expiration_enabled, use default_expiration_time
            embedding: Pre-computed embedding vector for similarity searches

        Returns:
            The item_id of the newly stored item
        """
        with self.lock:
            item_id = item_id or str(uuid.uuid4())
            if item_id in self.memory:
                self.logger.warning(f"Item {item_id} already exists; overwriting.")
                self._remove_tracking(item_id)

            if importance is None:
                importance = self.config.default_importance
            if self.config.expiration_enabled and expiration_time is None and self.config.default_expiration_time is not None:
                expiration_time = time.time() + self.config.default_expiration_time

            metadata = metadata or {}
            metadata["source"] = source

            # Create the item
            now = time.time()
            item = WorkingMemoryItem(
                item_id=item_id,
                content=content,
                source=source,
                timestamp=now,
                importance=importance,
                metadata=metadata,
                expiration_time=expiration_time,
                embedding=embedding
            )

            # Generate embedding if needed and function provided
            if embedding is None and self.config.embedding_function:
                try:
                    item.embedding = self.config.embedding_function(content)
                except Exception as e:
                    self.logger.error(f"Failed to embed content for item {item_id}: {e}", exc_info=True)

            # Store the item
            self.memory[item_id] = item

            # Track access patterns based on policy
            if self.config.eviction_policy == "LRU":
                self._access_order.append(item_id)
            elif self.config.eviction_policy == "LFU":
                self._access_counts[item_id] = 1

            # Clear operation cache entry if it exists
            if item_id in self._recent_operations_cache:
                del self._recent_operations_cache[item_id]

            # Check capacity and evict if needed
            self._evict_if_needed()

            # Log with detailed context
            expiry_str = f", expires in {int(expiration_time - now)}s" if expiration_time else ""
            self.logger.debug(f"Stored item {item_id} from {source}, importance={importance:.2f}{expiry_str}")

            # Notify listeners if configured
            if hasattr(self.config, 'on_store_callbacks') and self.config.on_store_callbacks:
                for callback in self.config.on_store_callbacks:
                    try:
                        callback(item)
                    except Exception as e:
                        self.logger.error(f"Error in on_store callback: {e}", exc_info=True)

            return item_id

    def get_item(self, item_id: str) -> Optional[WorkingMemoryItem]:
        """
        Retrieve an item by ID, checking expiration, updating LRU or LFU data.
        Uses caching for frequently accessed items.
        """
        # Check cache first for frequent operations
        if item_id in self._recent_operations_cache:
            self._cache_hits += 1
            return self._recent_operations_cache[item_id]

        self._cache_misses += 1

        with self.lock:
            it = self.memory.get(item_id)
            if not it:
                return None

            # Check expiration
            if self.config.expiration_enabled and it.expiration_time is not None:
                if time.time() > it.expiration_time:
                    self.logger.debug(f"Item {item_id} expired; removing.")
                    self.remove_item(item_id)
                    return None

            # Access update
            self._record_access(item_id)

            # Cache for future operations if frequently accessed
            if it.access_count > 3:  # Only cache items accessed multiple times
                self._recent_operations_cache[item_id] = it
                # Keep cache size reasonable
                if len(self._recent_operations_cache) > min(100, self.config.capacity // 5):
                    # Remove oldest entries when cache gets too large
                    oldest = next(iter(self._recent_operations_cache))
                    del self._recent_operations_cache[oldest]

            return it

    def retrieve(self,
                 query_text: Optional[str] = None,
                 query_embedding: Optional[np.ndarray] = None,
                 similarity_metric: str = "cosine",
                 min_importance: float = 0.0,
                 max_results: int = 10,
                 metadata_filter: Optional[Dict[str, Any]] = None,
                 source_filter: Optional[Union[str, List[str]]] = None) -> List[WorkingMemoryItem]:
        """
        A combined retrieval approach that merges:
          - text-based matching (if content is str and query_text is given)
          - embedding-based similarity (if query_embedding is provided and item has an embedding)
          - item importance
          - optional recency bias
          - metadata and source filtering

        We compute a total score for each item, then sort descending and return the top N.
        """
        t0 = time.time()

        # Prepare source filter as a set for faster lookup
        if source_filter:
            if isinstance(source_filter, str):
                source_filter_set = {source_filter}
            else:
                source_filter_set = set(source_filter)
        else:
            source_filter_set = None

        with self.lock:
            # Pre-filter expired items
            now = time.time()
            valid_items = []
            for item in self.memory.values():
                # Skip expired items
                if self.config.expiration_enabled and item.expiration_time is not None and now > item.expiration_time:
                    continue

                # Skip items below minimum importance
                if item.importance < min_importance:
                    continue

                # Apply source filter if specified
                if source_filter_set and item.source not in source_filter_set:
                    continue

                # Apply metadata filter if specified
                if metadata_filter:
                    if not all(item.metadata.get(k) == v for k, v in metadata_filter.items()):
                        continue

                valid_items.append(item)

            # Exit early if no valid items
            if not valid_items:
                self.logger.debug("retrieve: no valid items matched filters")
                return []

            # Calculate scores for valid items
            scored_items = []
            for item in valid_items:
                score = self._calculate_retrieval_score(
                    item,
                    query_text,
                    query_embedding,
                    similarity_metric,
                    now
                )
                scored_items.append((score, item))

            # Sort by score descending and take top N
            scored_items.sort(reverse=True)
            results = [item for _, item in scored_items[:max_results]]

            # Update access records for returned items
            for item in results:
                self._record_access(item.item_id)

            dt = time.time() - t0
            self.logger.debug(f"retrieve: returned {len(results)} items in {dt:.4f}s from {len(valid_items)} candidates")
            return results

    def _calculate_retrieval_score(self,
                                 item: WorkingMemoryItem,
                                 query_text: Optional[str],
                                 query_embedding: Optional[np.ndarray],
                                 similarity_metric: str,
                                 now: float) -> float:
        """
        Calculate a combined score for an item during retrieval.
        Considers text match, embedding similarity, importance, and recency.
        """
        score = item.importance  # Start with base importance

        # Text match component
        if query_text and isinstance(item.content, str):
            text_score = self._text_similarity(item.content, query_text)
            score += text_score

        # Embedding similarity component
        if query_embedding is not None and item.embedding is not None:
            try:
                sim = self._vector_similarity(query_embedding, item.embedding, similarity_metric)
                score += sim
            except Exception as e:
                self.logger.error(f"Error computing embedding similarity: {e}", exc_info=True)

        # Recency bias component
        if self.config.recency_bias > 0.0:
            age = now - item.timestamp
            # Exponential decay based on recency
            recency_factor = np.exp(-age * self.config.recency_decay_rate)
            recency_bonus = self.config.recency_bias * recency_factor
            score += recency_bonus

        # Access frequency component (optional)
        if hasattr(self.config, 'access_frequency_weight') and self.config.access_frequency_weight > 0:
            access_bonus = math.log(1 + item.access_count) * self.config.access_frequency_weight
            score += access_bonus

        return score

    def _text_similarity(self, text1: str, text2: str) -> float:
        """
        Calculate text similarity between two strings.
        Uses a simple approach based on word overlap and substring matching.
        """
        # Method 1: Simple word overlap
        words1 = set(text1.lower().split())
        words2 = set(text2.lower().split())

        if not words1 or not words2:
            return 0.0

        overlap = len(words1.intersection(words2))
        word_sim = overlap / (len(words1) + len(words2) - overlap)  # Jaccard similarity

        # Method 2: Substring match
        substring_sim = 0.0
        if text2.lower() in text1.lower():
            substring_sim = len(text2) / len(text1)

        # Return max of the two methods
        return max(word_sim, substring_sim)

    def _vector_similarity(self, vec1: np.ndarray, vec2: np.ndarray, metric: str) -> float:
        """
        Compute vector similarity using the specified metric.
        """
        # Ensure inputs are numpy arrays
        if not isinstance(vec1, np.ndarray):
            vec1 = np.array(vec1)
        if not isinstance(vec2, np.ndarray):
            vec2 = np.array(vec2)

        # Ensure dimensions match
        if vec1.shape != vec2.shape:
            if len(vec1.shape) == 1 and len(vec2.shape) == 1:
                # Reshape to match dimensions if both are 1D but different lengths
                min_len = min(vec1.shape[0], vec2.shape[0])
                vec1 = vec1[:min_len]
                vec2 = vec2[:min_len]
            else:
                self.logger.warning(f"Incompatible embedding dimensions: {vec1.shape} vs {vec2.shape}")
                return 0.0

        if metric == "cosine":
            # Calculate cosine similarity: dot(a, b) / (||a|| * ||b||)
            norm1 = np.linalg.norm(vec1)
            norm2 = np.linalg.norm(vec2)

            if norm1 > 0 and norm2 > 0:
                return np.dot(vec1, vec2) / (norm1 * norm2)
            return 0.0

        elif metric == "euclidean":
            # Calculate negative Euclidean distance (negative since higher values = more similar)
            return -np.linalg.norm(vec1 - vec2)

        elif metric == "dot":
            # Simple dot product
            return float(np.dot(vec1, vec2))

        else:
            self.logger.warning(f"Unknown similarity metric: {metric}, defaulting to cosine")
            return self._vector_similarity(vec1, vec2, "cosine")

    def update_item(self,
                    item_id: str,
                    new_content: Optional[Any] = None,
                    new_importance: Optional[float] = None,
                    new_metadata: Optional[Dict[str, Any]] = None,
                    new_embedding: Optional[np.ndarray] = None,
                    reset_expiration: bool = False) -> bool:
        """
        Update an existing item (content, importance, metadata).
        Recompute embedding if content changes.

        Args:
            item_id: The ID of the item to update
            new_content: New content to set (None to keep current)
            new_importance: New importance to set (None to keep current)
            new_metadata: Metadata to update (None to keep current)
            new_embedding: New embedding vector (None to recompute if needed)
            reset_expiration: If True, reset the expiration timer

        Returns:
            bool: True if successful, False if item not found
        """
        with self.lock:
            it = self.memory.get(item_id)
            if not it:
                self.logger.warning(f"update_item: item {item_id} not found.")
                return False

            modified = False

            # Update content if provided
            if new_content is not None:
                it.content = new_content
                modified = True

                # Auto-recompute embedding if content changed and no new embedding provided
                if new_embedding is None and self.config.embedding_function:
                    try:
                        it.embedding = self.config.embedding_function(new_content)
                    except Exception as e:
                        self.logger.error(f"Embedding update failed for {item_id}: {e}", exc_info=True)

            # Update embedding if provided
            if new_embedding is not None:
                it.embedding = new_embedding
                modified = True

            # Update importance if provided
            if new_importance is not None:
                it.importance = max(0.0, min(1.0, new_importance))
                modified = True

            # Update metadata if provided
            if new_metadata is not None:
                it.metadata.update(new_metadata)
                modified = True

            # Reset expiration if requested
            if reset_expiration and it.expiration_time is not None:
                if self.config.default_expiration_time:
                    it.expiration_time = time.time() + self.config.default_expiration_time
                    modified = True

            # Update timestamp and access tracking
            if modified:
                it.timestamp = time.time()
                self._record_access(item_id)

                # Clear cache entry if it exists
                if item_id in self._recent_operations_cache:
                    del self._recent_operations_cache[item_id]

                self.logger.debug(f"update_item => item {item_id} updated.")

                # Notify listeners if configured
                if hasattr(self.config, 'on_update_callbacks') and self.config.on_update_callbacks:
                    for callback in self.config.on_update_callbacks:
                        try:
                            callback(it)
                        except Exception as e:
                            self.logger.error(f"Error in on_update callback: {e}", exc_info=True)

            return True

    def remove_item(self, item_id: str) -> bool:
        """
        Remove an item from working memory.

        Args:
            item_id: The item ID to remove

        Returns:
            bool: True if item was removed, False if not found
        """
        with self.lock:
            if item_id not in self.memory:
                self.logger.warning(f"Cannot remove item {item_id}: not found.")
                return False

            # Get the item before removing it for callbacks
            item = self.memory[item_id]

            # Remove from tracking structures
            self._remove_tracking(item_id)

            # Remove from memory
            del self.memory[item_id]

            # Remove from operation cache
            if item_id in self._recent_operations_cache:
                del self._recent_operations_cache[item_id]

            self.logger.debug(f"Removed item {item_id} from working memory.")

            # Notify listeners if configured
            if hasattr(self.config, 'on_remove_callbacks') and self.config.on_remove_callbacks:
                for callback in self.config.on_remove_callbacks:
                    try:
                        callback(item)
                    except Exception as e:
                        self.logger.error(f"Error in on_remove callback: {e}", exc_info=True)

            return True

    def _evict_if_needed(self) -> None:
        """
        Check if we're over capacity and evict items if needed.
        """
        if len(self.memory) <= self.config.capacity:
            return

        # Calculate how many items to evict
        to_evict = len(self.memory) - self.config.capacity
        self.logger.debug(f"Need to evict {to_evict} items to respect capacity")

        # Use the appropriate eviction policy
        if self.config.eviction_policy == "LRU":
            self._evict_lru(to_evict)
        elif self.config.eviction_policy == "LFU":
            self._evict_lfu(to_evict)
        elif self.config.eviction_policy == "priority":
            self._evict_priority(to_evict)
        else:
            self.logger.warning(f"Unknown eviction policy: {self.config.eviction_policy}, defaulting to LRU")
            self._evict_lru(to_evict)

    def _evict_lru(self, count: int) -> None:
        """
        Evict the least recently used items.
        """
        evicted = 0
        while evicted < count and self._access_order:
            # Remove from the left (least recently used)
            item_id = self._access_order.popleft()
            if item_id in self.memory:
                item = self.memory[item_id]
                self._auto_transfer_if_needed(item)
                del self.memory[item_id]

                # Remove from other tracking structures
                if item_id in self._access_counts:
                    del self._access_counts[item_id]
                if item_id in self._recent_operations_cache:
                    del self._recent_operations_cache[item_id]

                evicted += 1

        self.logger.debug(f"LRU eviction: removed {evicted} items")

    def _evict_lfu(self, count: int) -> None:
        """
        Evict the least frequently used items.
        """
        if not self._access_counts:
            return

        # Sort items by access count (ascending)
        sorted_items = sorted(self._access_counts.items(), key=lambda x: x[1])

        evicted = 0
        for item_id, _ in sorted_items:
            if item_id in self.memory and evicted < count:
                item = self.memory[item_id]
                self._auto_transfer_if_needed(item)
                del self.memory[item_id]

                # Remove from other tracking structures
                try:
                    self._access_order.remove(item_id)
                except ValueError:
                    pass
                del self._access_counts[item_id]
                if item_id in self._recent_operations_cache:
                    del self._recent_operations_cache[item_id]

                evicted += 1

            if evicted >= count:
                break

        self.logger.debug(f"LFU eviction: removed {evicted} items")

    def _evict_priority(self, count: int) -> None:
        """
        Evict items based on priority (importance, recency, etc.)
        """
        # Build scoring function
        now = time.time()

        def score_item(item: WorkingMemoryItem) -> float:
            """
            Calculate a score for each item - lower score = higher priority to keep
            (i.e., higher score = more likely to be evicted)
            """
            importance_factor = 1.0 - item.importance  # Invert importance (lower importance = higher score)
            recency_factor = (now - item.timestamp) / 3600.0  # Hours since last update
            access_factor = 1.0 / (1.0 + math.log(1.0 + item.access_count))  # Inverse of access count

            # Combined score (higher = more likely to evict)
            return (importance_factor * 0.6) + (recency_factor * 0.3) + (access_factor * 0.1)

        # Calculate scores for all items
        scored_items = [(score_item(item), item_id) for item_id, item in self.memory.items()]

        # Sort by score (highest first = most likely to evict)
        scored_items.sort(reverse=True)

        # Evict the highest-scored items
        evicted = 0
        for _, item_id in scored_items:
            if evicted >= count:
                break

            if item_id in self.memory:
                item = self.memory[item_id]
                self._auto_transfer_if_needed(item)
                del self.memory[item_id]

                # Remove from tracking structures
                self._remove_tracking(item_id)
                if item_id in self._recent_operations_cache:
                    del self._recent_operations_cache[item_id]

                evicted += 1

        self.logger.debug(f"Priority eviction: removed {evicted} items")

    def _auto_transfer_if_needed(self, item: WorkingMemoryItem) -> None:
        """
        If auto_transfer_callback is configured, call it with the item
        about to be evicted.
        """
        if not hasattr(self.config, 'auto_transfer_callback') or not self.config.auto_transfer_callback:
            return

        try:
            self.config.auto_transfer_callback(item)
            self.logger.debug(f"Auto-transferred item {item.item_id} to another memory store")
        except Exception as e:
            self.logger.error(f"Error in auto_transfer_callback for {item.item_id}: {e}", exc_info=True)

    def _record_access(self, item_id: str) -> None:
        """
        Update access tracking structures when an item is accessed.
        """
        if item_id not in self.memory:
            return

        # Update the item's access count
        self.memory[item_id].access_count += 1

        # Update LRU tracking if enabled
        if self.config.eviction_policy == "LRU":
            # Move to the end of the access order (most recently used)
            try:
                self._access_order.remove(item_id)
            except ValueError:
                pass  # Not found, which is fine
            self._access_order.append(item_id)

        # Update LFU tracking if enabled
        elif self.config.eviction_policy == "LFU":
            self._access_counts[item_id] += 1

    def _remove_tracking(self, item_id: str) -> None:
        """
        Remove an item from all tracking structures.
        """
        # LRU tracking
        try:
            self._access_order.remove(item_id)
        except ValueError:
            pass  # Not found, which is fine

        # LFU tracking
        if item_id in self._access_counts:
            del self._access_counts[item_id]

        # Operation cache
        if item_id in self._recent_operations_cache:
            del self._recent_operations_cache[item_id]

    def clear(self) -> None:
        """
        Remove all items from working memory.
        """
        with self.lock:
            # Get copy of items for callbacks
            items = list(self.memory.values())

            # Clear all data structures
            self.memory.clear()
            self._access_order.clear()
            self._access_counts.clear()
            self._recent_operations_cache.clear()
            self._cache_hits = 0
            self._cache_misses = 0

            self.logger.info(f"Working memory cleared ({len(items)} items removed)")

            # Notify listeners if configured
            if hasattr(self.config, 'on_clear_callbacks') and self.config.on_clear_callbacks:
                for callback in self.config.on_clear_callbacks:
                    try:
                        callback(items)
                    except Exception as e:
                        self.logger.error(f"Error in on_clear callback: {e}", exc_info=True)

    def get_all_items(self) -> List[WorkingMemoryItem]:
        """
        Returns a snapshot (copy) of all items currently in working memory.
        """
        with self.lock:
            return list(self.memory.values())

    def get_stats(self) -> Dict[str, Any]:
        """
        Return statistics about the working memory.
        """
        with self.lock:
            return {
                "capacity": self.config.capacity,
                "current_size": len(self.memory),
                "eviction_policy": self.config.eviction_policy,
                "access_order_size": len(self._access_order),
                "lfu_tracking_size": len(self._access_counts),
                "cache_hits": self._cache_hits,
                "cache_misses": self._cache_misses,
                "cache_hit_ratio": self._cache_hits / (self._cache_hits + self._cache_misses) if (self._cache_hits + self._cache_misses) > 0 else 0,
                "items_by_source": self._count_items_by_source(),
                "average_importance": self._calculate_average_importance()
            }

    def _count_items_by_source(self) -> Dict[str, int]:
        """
        Count items by source.
        """
        counts = defaultdict(int)
        for item in self.memory.values():
            counts[item.source] += 1
        return dict(counts)

    def _calculate_average_importance(self) -> float:
        """
        Calculate the average importance of all items.
        """
        if not self.memory:
            return 0.0

        total = sum(item.importance for item in self.memory.values())
        return total / len(self.memory)

    def _register_with_memory_manager(self, memory_manager: Any) -> None:
        """
        Register this working memory instance with a memory manager.
        This allows for integration with the broader memory system.
        """
        try:
            if hasattr(memory_manager, 'register_memory_module'):
                memory_manager.register_memory_module('working_memory', self)
                self.logger.info(f"Registered with memory manager: {memory_manager}")
            else:
                self.logger.warning(f"Memory manager does not support registration: {memory_manager}")
        except Exception as e:
            self.logger.error(f"Error registering with memory manager: {e}", exc_info=True)
