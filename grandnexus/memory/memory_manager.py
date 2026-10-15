"""
memory_manager.py - Hierarchical Memory System for GrandNexus

This module implements a multi-level memory architecture:
- Short-term, medium-term, and long-term memory stores
- Adaptive consolidation based on relevance, usage frequency, and recency
- Background maintenance thread for periodic decay, pruning, and auto-saving
- Flexible serialization (JSON or pickle) for storing memory to disk
- Rich metadata indexing and searching
- Thread-safe operations for concurrency

It combines ideas from the Claude/Gemini prototypes and extends them with:
- Weighted scoring for consolidation and eviction
- Configurable decay rates per tier
- Hooks for integrated logging and potential life-cycle linking
- Optional synergy with other modules, e.g., hooking into a 'registry' or 'core' if needed
"""

import logging
import threading
import time
import json
import os
import math
import pickle
import uuid
import heapq
import numpy as np
from collections import defaultdict
from typing import Dict, List, Any, Optional, Tuple, Union, Set


class MemoryItem:
        """
        Represents a single piece of information stored in the memory system.

        Attributes:
            memory_id (str): Unique identifier for this memory (UUID4 by default).
            content (Any): The actual data stored (can be text, JSON, custom class, etc.).
            metadata (Dict[str, Any]): Arbitrary metadata about this memory (source, type, tags, etc.).
            relevance (float): Importance score in [0,1].
            created_at (float): Timestamp when the memory was created.
            last_accessed (float): Timestamp of the most recent access.
            access_count (int): Number of times this memory has been accessed.
            embedding (Optional[np.ndarray]): Vector representation for similarity comparisons.
            expiration (Optional[float]): Time when this memory should be considered expired, if applicable.
        """

        def __init__(self,
                    content: Any,
                    metadata: Optional[Dict[str, Any]] = None,
                    relevance: float = 0.5,
                    memory_id: Optional[str] = None,
                    embedding: Optional[np.ndarray] = None,
                    expiration: Optional[float] = None):
            self.content = content
            self.metadata = metadata or {}
            self.relevance = max(0.0, min(1.0, relevance))  # clamp
            self.created_at = time.time()
            self.last_accessed = self.created_at
            self.access_count = 0
            self.memory_id = memory_id or str(uuid.uuid4())
            self.embedding = embedding
            self.expiration = expiration
            self.modified_count = 0  # Track modifications

        def access(self) -> None:
            """Mark this memory as accessed, updating last_accessed and incrementing the counter."""
            self.last_accessed = time.time()
            self.access_count += 1

        def update_relevance(self,
                            new_relevance: Optional[float] = None,
                            increment: Optional[float] = None,
                            context: Optional[Dict[str, Any]] = None) -> None:
            """
            Update the relevance score either by setting it absolutely (new_relevance)
            or by incrementing it (increment). Optional context can be provided for logging.

            Args:
                new_relevance: Optional absolute value to set (between 0-1)
                increment: Optional amount to add to current relevance
                context: Optional contextual information about why relevance is changing
            """
            old_relevance = self.relevance

            if new_relevance is not None:
                self.relevance = max(0.0, min(1.0, new_relevance))
            elif increment is not None:
                self.relevance = max(0.0, min(1.0, self.relevance + increment))

            # Record modification and context
            if old_relevance != self.relevance:
                self.modified_count += 1
                if context and isinstance(context, dict):
                    # Optionally store relevance change history in metadata
                    history = self.metadata.setdefault("relevance_history", [])
                    history.append({
                        "timestamp": time.time(),
                        "old": old_relevance,
                        "new": self.relevance,
                        "context": context
                    })

        def update_content(self, new_content: Any, preserve_metadata: bool = True) -> None:
            """
            Update the content of this memory item.

            Args:
                new_content: The new content to store
                preserve_metadata: If True, keep existing metadata, otherwise reset it
            """
            self.content = new_content
            if not preserve_metadata:
                self.metadata = {}
            self.modified_count += 1
            self.last_accessed = time.time()  # Consider this an access

        def update_embedding(self, new_embedding: Optional[np.ndarray] = None) -> None:
            """Update the embedding vector for this memory."""
            self.embedding = new_embedding
            self.modified_count += 1

        def set_expiration(self, expiration_time: Optional[float]) -> None:
            """Set or clear the expiration time for this memory."""
            self.expiration = expiration_time

        def is_expired(self) -> bool:
            """Check if this memory has expired."""
            if self.expiration is None:
                return False
            return time.time() >= self.expiration

        def age(self) -> float:
            """Returns how many seconds have passed since creation."""
            return time.time() - self.created_at

        def idle_time(self) -> float:
            """Returns how many seconds have passed since last access."""
            return time.time() - self.last_accessed

        def to_dict(self) -> Dict[str, Any]:
            """
            Convert MemoryItem to a dictionary for serialization.
            Note: embeddings must be handled separately if they need serialization.
            """
            result = {
                "memory_id": self.memory_id,
                "content": self.content,
                "metadata": self.metadata,
                "relevance": self.relevance,
                "created_at": self.created_at,
                "last_accessed": self.last_accessed,
                "access_count": self.access_count,
                "modified_count": self.modified_count
            }

            if self.expiration is not None:
                result["expiration"] = self.expiration

            # Embedding handling needs special care
            if self.embedding is not None:
                if isinstance(self.embedding, np.ndarray):
                    # Convert to list for JSON serialization
                    result["embedding_shape"] = self.embedding.shape
                    result["embedding_data"] = self.embedding.tolist()
                else:
                    result["embedding_data"] = self.embedding

            return result

        @classmethod
        def from_dict(cls, data: Dict[str, Any]) -> 'MemoryItem':
            """
            Create a MemoryItem from a dictionary, presumably read from disk.
            """
            # Handle embedding if present
            embedding = None
            if "embedding_data" in data:
                if "embedding_shape" in data:
                    # Reconstruct numpy array
                    embedding = np.array(data["embedding_data"])
                else:
                    embedding = data["embedding_data"]

            # Create the item with basic attributes
            item = cls(
                content=data["content"],
                metadata=data["metadata"],
                relevance=data["relevance"],
                memory_id=data["memory_id"],
                embedding=embedding,
                expiration=data.get("expiration")
            )

            # Restore additional attributes
            item.created_at = data["created_at"]
            item.last_accessed = data["last_accessed"]
            item.access_count = data["access_count"]
            if "modified_count" in data:
                item.modified_count = data["modified_count"]

            return item

        def __repr__(self) -> str:
            expiry_info = f", expires={self.expiration:.1f}" if self.expiration else ""
            return (f"<MemoryItem id={self.memory_id[-8:]}, rel={self.relevance:.2f}, "
                    f"age={self.age():.1f}s, acc={self.access_count}{expiry_info}>")


class NexusMemoryManager:
    """
    Hierarchical memory manager for GrandNexus, featuring:
      - short-term, medium-term, and long-term stores
      - dynamic consolidation and relevance decay
      - flexible persistence (pickle or JSON)
      - powerful metadata indexing
      - capacity-limited tiers with eviction

    Typical usage:
      1) Instantiate with desired capacities, intervals, thresholds, etc.
      2) Call start() to enable the background maintenance loop
      3) Store / retrieve items via store(), retrieve(), or query()
      4) stop() to gracefully shut down (auto-saves if configured)

    The memory manager is thread-safe, using an RLock for synchronization.
    """

    def __init__(self,
                 short_term_capacity: int = 100,
                 medium_term_capacity: int = 300,
                 long_term_capacity: int = 5000,
                 consolidation_interval: float = 60.0,
                 storage_file: Optional[str] = "nexus_memory.dat",
                 serialization_format: str = "pickle",  # can be "json" or "pickle"
                 auto_save: bool = True,
                 save_interval: float = 300.0,
                 relevance_threshold: float = 0.3,
                 short_decay_rate: float = 0.05,
                 medium_decay_rate: float = 0.02,
                 long_decay_rate: float = 0.005,
                 threshold_short_to_medium: float = 0.4,
                 threshold_medium_to_long: float = 0.7,
                 consolidation_access_for_short: int = 3,
                 consolidation_access_for_medium: int = 10
                 ):
        """
        Args:
            short_term_capacity (int): Maximum items in short-term store
            medium_term_capacity (int): Max items in medium-term
            long_term_capacity (int): Max items in long-term
            consolidation_interval (float): How often (sec) we run the consolidation/decay/prune loop
            storage_file (str, optional): Path to persist memories. If None, no persistence.
            serialization_format (str): "json" or "pickle" for on-disk format
            auto_save (bool): If True, we auto-save periodically
            save_interval (float): How often (sec) to auto-save
            relevance_threshold (float): Memories with relevance below this get pruned
            short_decay_rate (float): Decay factor for short-term store
            medium_decay_rate (float): Decay factor for medium-term
            long_decay_rate (float): Decay factor for long-term
            threshold_short_to_medium (float): Relevance threshold for short->medium promotion
            threshold_medium_to_long (float): Relevance threshold for medium->long promotion
            consolidation_access_for_short (int): Access count threshold for short->medium
            consolidation_access_for_medium (int): Access count threshold for medium->long
        """
        self.logger = logging.getLogger("NexusMemoryManager")

        # capacities
        self.short_term_capacity = short_term_capacity
        self.medium_term_capacity = medium_term_capacity
        self.long_term_capacity = long_term_capacity

        # data stores: memory_id -> MemoryItem
        self.short_term: Dict[str, MemoryItem] = {}
        self.medium_term: Dict[str, MemoryItem] = {}
        self.long_term: Dict[str, MemoryItem] = {}

        # intervals & config
        self.consolidation_interval = consolidation_interval
        self.storage_file = storage_file
        self.serialization_format = serialization_format
        self.auto_save = auto_save
        self.save_interval = save_interval
        self.relevance_threshold = relevance_threshold

        self.short_decay_rate = short_decay_rate
        self.medium_decay_rate = medium_decay_rate
        self.long_decay_rate = long_decay_rate

        # thresholds for internal promotions
        self.threshold_short_to_medium = threshold_short_to_medium
        self.threshold_medium_to_long = threshold_medium_to_long

        # optional additional constraints
        self.consolidation_access_for_short = consolidation_access_for_short
        self.consolidation_access_for_medium = consolidation_access_for_medium

        # concurrency
        self.lock = threading.RLock()
        self.running = False
        self.maintenance_thread: Optional[threading.Thread] = None

        # indexing (metadata_key -> {value -> set(memory_ids)})
        self.indexes: Dict[str, Dict[Any, Set[str]]] = defaultdict(lambda: defaultdict(set))

        # Attempt to load existing memories
        if self.storage_file and os.path.exists(self.storage_file):
            self._load_memories()

        self.logger.info(
            f"NexusMemoryManager initialized: short={short_term_capacity}, "
            f"medium={medium_term_capacity}, long={long_term_capacity}, file={self.storage_file}"
        )

    def start(self) -> None:
        """Starts the background maintenance thread if not already running."""
        with self.lock:
            if self.running:
                self.logger.warning("NexusMemoryManager is already running.")
                return
            self.running = True

        self.maintenance_thread = threading.Thread(
            target=self._maintenance_loop,
            name="NexusMemoryMaintenance",
            daemon=True
        )
        self.maintenance_thread.start()
        self.logger.info("NexusMemoryManager started with background maintenance loop.")

    def stop(self) -> None:
        """Stops the maintenance thread and performs a final save if configured."""
        with self.lock:
            if not self.running:
                self.logger.warning("NexusMemoryManager is not running.")
                return
            self.running = False

        if self.maintenance_thread:
            self.maintenance_thread.join(timeout=5.0)

        if self.storage_file and self.auto_save:
            self._save_memories()

        self.logger.info("NexusMemoryManager stopped.")

    def store(self,
              content: Any,
              metadata: Optional[Dict[str, Any]] = None,
              relevance: float = 0.5,
              memory_id: Optional[str] = None,
              embedding: Optional[np.ndarray] = None,
              expiration: Optional[float] = None) -> str:
        """
        Store a new memory item and decide which tier (short/medium/long) it goes into
        based on the initial relevance. Updates the index and ensures capacities.

        Args:
            content: The data to store
            metadata: Additional information about this memory
            relevance: Importance score from 0.0 to 1.0
            memory_id: Optional custom ID (UUID generated if None)
            embedding: Optional vector representation for similarity searches
            expiration: Optional timestamp when this memory should expire

        Returns:
            memory_id: The identifier of the stored memory
        """
        metadata = metadata or {}
        item = MemoryItem(content, metadata, relevance, memory_id, embedding, expiration)

        with self.lock:
            # Determine which store to use based on relevance thresholds
            if item.relevance >= self.threshold_medium_to_long:
                self.long_term[item.memory_id] = item
                tier = "long-term"
            elif item.relevance >= self.threshold_short_to_medium:
                self.medium_term[item.memory_id] = item
                tier = "medium-term"
            else:
                self.short_term[item.memory_id] = item
                tier = "short-term"

            self._index_memory(item)
            self._check_capacities()

            # Enhance logging with more context
            expiry_info = f", expires at {time.strftime('%H:%M:%S', time.localtime(expiration))}" if expiration else ""
            self.logger.info(f"Stored item {item.memory_id} in {tier} memory (relevance={item.relevance:.2f}{expiry_info})")

            # For semantic memories, optionally extract concepts and add additional indexing
            if metadata.get("memory_type") == "semantic" and self.config.synergy_with_memory:
                self._process_semantic_memory(item)

        return item.memory_id

    def retrieve(self, memory_id: str) -> Optional[MemoryItem]:
        """
        Retrieve a memory by ID from any store. Mark it accessed.
        """
        with self.lock:
            item = (self.short_term.get(memory_id)
                    or self.medium_term.get(memory_id)
                    or self.long_term.get(memory_id))
            if item:
                item.access()
                self.logger.debug(f"Retrieved memory {memory_id} (count={item.access_count})")
                return item
            else:
                self.logger.warning(f"Memory {memory_id} not found.")
                return None

    def retrieve_content(self, memory_id: str) -> Any:
        """Convenience method to retrieve only the content of a memory."""
        item = self.retrieve(memory_id)
        return item.content if item else None

    def delete(self, memory_id: str) -> bool:
        """Delete a memory from whichever store it is in."""
        with self.lock:
            item = None
            if memory_id in self.short_term:
                item = self.short_term.pop(memory_id)
            elif memory_id in self.medium_term:
                item = self.medium_term.pop(memory_id)
            elif memory_id in self.long_term:
                item = self.long_term.pop(memory_id)

            if item:
                self._remove_from_indexes(item)
                self.logger.info(f"Deleted memory {memory_id} from memory.")
                return True
            else:
                self.logger.warning(f"Delete failed: Memory {memory_id} not found.")
                return False

    def query(self,
              metadata_filter: Optional[Dict[str, Any]] = None,
              min_relevance: float = 0.0,
              max_results: int = 10,
              include_content: bool = True) -> List[Union[str, Tuple[str, Any]]]:
        """
        Query memories by (optional) metadata filters and minimal relevance.
        Returns up to max_results memory IDs or (memory_id, content) tuples.
        """
        with self.lock:
            candidate_ids = self._get_candidate_ids(metadata_filter)

            results: List[Union[str, Tuple[str, Any]]] = []
            for mid in candidate_ids:
                item = (self.short_term.get(mid)
                        or self.medium_term.get(mid)
                        or self.long_term.get(mid))
                if not item:
                    continue
                if item.relevance < min_relevance:
                    continue
                # Check metadata
                if metadata_filter and not all(item.metadata.get(k) == v for k, v in metadata_filter.items()):
                    continue
                # Mark it accessed
                item.access()
                # Decide what to return
                if include_content:
                    results.append((mid, item.content))
                else:
                    results.append(mid)
                if len(results) >= max_results:
                    break
            return results

    def update_relevance(self,
                         memory_id: str,
                         new_relevance: Optional[float] = None,
                         increment: Optional[float] = None) -> bool:
        """
        Update the relevance of a memory item. If this triggers a threshold crossing,
        the item may be moved between short/medium/long.
        """
        with self.lock:
            item = (self.short_term.get(memory_id)
                    or self.medium_term.get(memory_id)
                    or self.long_term.get(memory_id))
            if not item:
                self.logger.warning(f"Cannot update relevance: {memory_id} not found.")
                return False

            old_rel = item.relevance
            item.update_relevance(new_relevance, increment)
            self.logger.info(f"Updated relevance for {memory_id}: {old_rel:.2f} -> {item.relevance:.2f}")
            self._check_store_transition(item)
            return True

    def get_stats(self) -> Dict[str, Any]:
        """Obtain statistics about how many items each store holds, capacities, etc."""
        with self.lock:
            return {
                "short_count": len(self.short_term),
                "medium_count": len(self.medium_term),
                "long_count": len(self.long_term),
                "short_capacity": self.short_term_capacity,
                "medium_capacity": self.medium_term_capacity,
                "long_capacity": self.long_term_capacity,
                "total_memories": (len(self.short_term)
                                   + len(self.medium_term)
                                   + len(self.long_term))
            }

    def clear_memories(self, store_name: Optional[str] = None) -> int:
        """
        Clear the specified store or all stores if store_name is None.
        Returns the number of items cleared.
        """
        with self.lock:
            count = 0
            if store_name in (None, "short_term"):
                for m in list(self.short_term.values()):
                    self._remove_from_indexes(m)
                count += len(self.short_term)
                self.short_term.clear()
            if store_name in (None, "medium_term"):
                for m in list(self.medium_term.values()):
                    self._remove_from_indexes(m)
                count += len(self.medium_term)
                self.medium_term.clear()
            if store_name in (None, "long_term"):
                for m in list(self.long_term.values()):
                    self._remove_from_indexes(m)
                count += len(self.long_term)
                self.long_term.clear()

            if store_name is None:
                self._clear_all_indexes()

            self.logger.info(f"Cleared {count} memories from {store_name or 'all stores'}.")
            return count

    # -----------------------------------------------------
    #  Internal / Protected Methods
    # -----------------------------------------------------

    def _maintenance_loop(self) -> None:
        """Background thread that handles consolidation, decay, pruning, and auto-save."""
        self.logger.info("Memory maintenance loop started.")
        last_consolidation = time.time()
        last_save = time.time()

        while True:
            with self.lock:
                if not self.running:
                    break
            time.sleep(1.0)  # check every second

            now = time.time()
            if now - last_consolidation >= self.consolidation_interval:
                try:
                    with self.lock:
                        self._consolidate_memories()
                        self._apply_relevance_decay()
                        self._prune_low_relevance()
                    last_consolidation = now
                except Exception as e:
                    self.logger.exception(f"Error during memory consolidation: {e}")

            if self.auto_save and self.storage_file and (now - last_save >= self.save_interval):
                try:
                    with self.lock:
                        self._save_memories()
                    last_save = now
                except Exception as e:
                    self.logger.exception(f"Error during auto-save: {e}")

        self.logger.info("Memory maintenance loop ended.")

    def _consolidate_memories(self) -> None:
        """
        Periodically promote items from short-term -> medium-term or
        medium-term -> long-term based on relevance thresholds, access counts,
        and additional heuristics for cognitive-like memory management.
        """
        # For tracking consolidation statistics
        stats = {
            "short_to_medium": 0,
            "medium_to_long": 0,
            "short_expired": 0,
            "medium_expired": 0,
            "long_expired": 0
        }

        # Check for and handle expired memories first
        now = time.time()
        expired_short = self._prune_expired_memories(self.short_term)
        expired_medium = self._prune_expired_memories(self.medium_term)
        expired_long = self._prune_expired_memories(self.long_term)

        stats["short_expired"] = expired_short
        stats["medium_expired"] = expired_medium
        stats["long_expired"] = expired_long

        # short->medium promotion
        to_medium = []
        for mid, item in self.short_term.items():
            if self._should_promote_to_medium(item):
                to_medium.append(mid)

        for mid in to_medium:
            if mid in self.short_term:  # double-check
                mem = self.short_term.pop(mid)
                # Adjust relevance to ensure it meets medium-term threshold
                if mem.relevance < self.threshold_short_to_medium:
                    mem.relevance = self.threshold_short_to_medium
                self.medium_term[mid] = mem
                stats["short_to_medium"] += 1

        # medium->long promotion
        to_long = []
        for mid, item in self.medium_term.items():
            if self._should_promote_to_long(item):
                to_long.append(mid)

        for mid in to_long:
            if mid in self.medium_term:
                mem = self.medium_term.pop(mid)
                # Adjust relevance to ensure it meets long-term threshold
                if mem.relevance < self.threshold_medium_to_long:
                    mem.relevance = self.threshold_medium_to_long
                self.long_term[mid] = mem
                stats["medium_to_long"] += 1

        # After consolidation, re-check capacities
        self._check_capacities()

        # Log consolidation statistics if any changes were made
        total_changes = sum(stats.values())
        if total_changes > 0:
            self.logger.info(f"Memory consolidation: {stats['short_to_medium']} items promoted to medium-term, "
                            f"{stats['medium_to_long']} to long-term. "
                            f"Expired: {stats['short_expired']} short, {stats['medium_expired']} medium, "
                            f"{stats['long_expired']} long.")

    def _should_promote_to_medium(self, item: MemoryItem) -> bool:
        """
        Determine if a short-term memory should be promoted to medium-term.
        Uses multiple heuristics for a more cognitive-like memory system.
        """
        # Core criteria from configuration
        if item.relevance >= self.threshold_short_to_medium:
            return True

        if item.access_count >= self.consolidation_access_for_short:
            return True

        # Additional heuristics
        # 1. Recent frequent access relative to age
        if item.age() > 0 and item.access_count / item.age() > 0.1:  # Accessed every 10 seconds on average
            return True

        # 2. High relevance combined with recent access
        if item.relevance > 0.3 and (time.time() - item.last_accessed) < 60:  # Relevance > 0.3 and accessed in last minute
            return True

        # 3. Item has been modified multiple times, suggesting importance
        if item.modified_count > 2:
            return True

        # 4. Tagged as important in metadata
        if item.metadata.get("importance", 0) > 0.7:
            return True

        # 5. Associated with other long-term memories (requires implementation)
        if item.metadata.get("associated_with_long_term", False):
            return True

        return False

    def _should_promote_to_long(self, item: MemoryItem) -> bool:
        """
        Determine if a medium-term memory should be promoted to long-term.
        Uses multiple heuristics for a more cognitive-like memory system.
        """
        # Core criteria from configuration
        if item.relevance >= self.threshold_medium_to_long:
            return True

        if item.access_count >= self.consolidation_access_for_medium:
            return True

        # Additional heuristics
        # 1. Consistently accessed over long period
        if item.age() > 3600 and item.access_count > 5:  # Older than 1 hour and accessed multiple times
            return True

        # 2. High-quality embeddings (if applicable)
        if "embedding_quality" in item.metadata and item.metadata["embedding_quality"] > 0.8:
            return True

        # 3. Explicitly marked for long-term storage
        if item.metadata.get("store_long_term", False):
            return True

        # 4. Contains critical information types
        critical_types = {"core_concept", "fundamental_rule", "identity_info", "critical_fact"}
        if any(tag in critical_types for tag in item.metadata.get("tags", [])):
            return True

        return False

    def _prune_expired_memories(self, memory_store: Dict[str, MemoryItem]) -> int:
        """
        Remove expired memories from the given store.
        Returns the number of items removed.
        """
        expired_ids = []
        now = time.time()

        for mid, memory in memory_store.items():
            if memory.expiration and now >= memory.expiration:
                expired_ids.append(mid)

        for mid in expired_ids:
            memory = memory_store.pop(mid)
            self._remove_from_indexes(memory)

        return len(expired_ids)

    def _apply_relevance_decay(self) -> None:
        """
        Apply time-based decay to relevance. Items in short-term decay faster, etc.
        This is a simple approach where we reduce relevance proportionally to how long
        they've been idle since last_accessed.
        """
        now = time.time()

        def apply_decay(store: Dict[str, MemoryItem], decay_rate: float):
            for mem in store.values():
                idle_time = now - mem.last_accessed
                # scale the decay by a factor of (idle_time / consolidation_interval) for instance
                factor = (idle_time / self.consolidation_interval)
                decay_amount = decay_rate * factor
                if decay_amount > 0:
                    mem.update_relevance(increment=-decay_amount)

        with self.lock:
            apply_decay(self.short_term, self.short_decay_rate)
            apply_decay(self.medium_term, self.medium_decay_rate)
            apply_decay(self.long_term, self.long_decay_rate)

    def _prune_low_relevance(self) -> None:
        """
        Remove items whose relevance is below the global threshold from each store.
        """
        pruned_count = 0

        def prune_store(store: Dict[str, MemoryItem]) -> None:
            nonlocal pruned_count
            to_remove = [mid for mid, m in store.items() if m.relevance < self.relevance_threshold]
            for mid in to_remove:
                memory = store.pop(mid)
                self._remove_from_indexes(memory)
            pruned_count += len(to_remove)

        with self.lock:
            prune_store(self.short_term)
            prune_store(self.medium_term)
            prune_store(self.long_term)

        if pruned_count:
            self.logger.info(f"Pruned {pruned_count} low-relevance items.")

    def _check_capacities(self) -> None:
        """
        Ensure that none of the stores exceed capacity. If they do, we evict the
        lowest "score" items.
        """
        if len(self.short_term) > self.short_term_capacity:
            over = len(self.short_term) - self.short_term_capacity
            self._evict_from_store("short_term", over)

        if len(self.medium_term) > self.medium_term_capacity:
            over = len(self.medium_term) - self.medium_term_capacity
            self._evict_from_store("medium_term", over)

        if len(self.long_term) > self.long_term_capacity:
            over = len(self.long_term) - self.long_term_capacity
            self._evict_from_store("long_term", over)

    def _evict_from_store(self, store_name: str, count: int) -> None:
        """
        Evict `count` items from the specified store, choosing the ones
        with the 'lowest score'. For example, we can define a simple score
        that combines relevance, access_count, and recency.
        """
        if count <= 0:
            return

        store = getattr(self, store_name, None)
        if not store:
            return

        heap: List[Tuple[float, str]] = []
        now = time.time()

        def score(mem: MemoryItem) -> float:
            # a simple combined metric: relevance * log(access+1), then penalize old last_access
            # lower => more likely to be evicted
            access_factor = math.log(mem.access_count + 1)
            recency_factor = 1.0 / (1.0 + (now - mem.last_accessed))
            # we want a higher value => better => so we invert it for the min-heap
            return -(mem.relevance * access_factor * recency_factor)

        # build a min-heap by pushing negative scores
        for mid, mem in store.items():
            heap.append((score(mem), mid))
        heapq.heapify(heap)

        evicted = 0
        while evicted < count and heap:
            _, mem_id = heapq.heappop(heap)
            if mem_id in store:
                memory = store.pop(mem_id)
                self._remove_from_indexes(memory)
                evicted += 1

        self.logger.info(f"Evicted {evicted} items from {store_name} to respect capacity.")

    def _check_store_transition(self, memory: MemoryItem) -> None:
        """
        If the memory's new relevance crosses a threshold, move it to
        the appropriate store. Similar logic to store() method.
        """
        mid = memory.memory_id
        # figure out which store it belongs in
        if memory.relevance >= self.threshold_medium_to_long:
            target_store = "long_term"
        elif memory.relevance >= self.threshold_short_to_medium:
            target_store = "medium_term"
        else:
            target_store = "short_term"

        # see which store it's currently in
        in_store = None
        if mid in self.short_term:
            in_store = "short_term"
        elif mid in self.medium_term:
            in_store = "medium_term"
        elif mid in self.long_term:
            in_store = "long_term"

        if in_store != target_store:
            # remove from old store
            if in_store == "short_term":
                del self.short_term[mid]
            elif in_store == "medium_term":
                del self.medium_term[mid]
            elif in_store == "long_term":
                del self.long_term[mid]

            # put it in the correct store
            getattr(self, target_store)[mid] = memory
            self.logger.info(f"Moved memory {mid} from {in_store} to {target_store} after relevance change.")
            self._check_capacities()

    def _index_memory(self, memory: MemoryItem) -> None:
        """
        Add references to the 'indexes' dictionary for quick searching by metadata keys.
        We only index simple, hashable values (str, int, etc.).
        """
        mid = memory.memory_id
        for k, v in memory.metadata.items():
            if isinstance(v, (str, int, float, bool)):
                self.indexes[k][v].add(mid)

    def _remove_from_indexes(self, memory: MemoryItem) -> None:
        """Remove memory's references from the indexes."""
        mid = memory.memory_id
        for k, v in memory.metadata.items():
            if isinstance(v, (str, int, float, bool)) and v in self.indexes[k]:
                self.indexes[k][v].discard(mid)
                if not self.indexes[k][v]:
                    del self.indexes[k][v]
            if not self.indexes[k]:
                del self.indexes[k]

    def _clear_all_indexes(self) -> None:
        """Clear the entire indexing structure."""
        self.indexes = defaultdict(lambda: defaultdict(set))

    def _get_candidate_ids(self, metadata_filter: Optional[Dict[str, Any]]) -> List[str]:
        """Return a list of memory IDs that match the given metadata filter via indexes."""
        if not metadata_filter:
            # return all known IDs
            return (list(self.short_term.keys())
                    + list(self.medium_term.keys())
                    + list(self.long_term.keys()))

        candidate_sets: List[Set[str]] = []
        for k, v in metadata_filter.items():
            if k in self.indexes and v in self.indexes[k]:
                candidate_sets.append(self.indexes[k][v])
            else:
                # no match
                return []

        # intersect all sets
        if not candidate_sets:
            return []
        result = candidate_sets[0]
        for s in candidate_sets[1:]:
            result = result.intersection(s)

        return list(result)


    def _process_semantic_memory(self, item: MemoryItem) -> None:
        """
        Special processing for semantic memories, such as concept extraction
        and integration with other memory systems.
        """
        # Extract potential concepts from content
        concepts = []
        if isinstance(item.content, str):
            # Simple keyword extraction (placeholder for more sophisticated NLP)
            # In a real implementation, you might use a keyword extractor or NER model
            words = item.content.lower().split()
            # Filter out common stop words and short words
            stop_words = {'the', 'and', 'a', 'an', 'in', 'on', 'at', 'to', 'for', 'of', 'with'}
            potential_concepts = [w for w in words if w not in stop_words and len(w) > 3]
            # Remove duplicates
            concepts = list(set(potential_concepts))

        # Store extracted concepts in metadata
        if concepts:
            item.metadata["extracted_concepts"] = concepts
            self.logger.debug(f"Extracted concepts from memory {item.memory_id}: {concepts}")

        # Check for relationships with existing items
        if hasattr(self, "semantic_memory") and self.semantic_memory:
            # This would require access to the SemanticMemory module
            # For now, just mark that semantic processing was attempted
            item.metadata["semantic_processed"] = True
            self.logger.debug(f"Semantic memory processing attempted for {item.memory_id}")
        else:
            # Basic standalone processing
            item.metadata["semantic_processed"] = True
            self.logger.debug(f"Processed semantic memory {item.memory_id}")

    def _save_memories(self) -> None:
        """Persist all memories in short/medium/long to disk using the chosen format."""
        if not self.storage_file:
            return

        data_to_save = {
            "short_term": {mid: mem.to_dict() for mid, mem in self.short_term.items()},
            "medium_term": {mid: mem.to_dict() for mid, mem in self.medium_term.items()},
            "long_term": {mid: mem.to_dict() for mid, mem in self.long_term.items()}
        }

        try:
            if self.serialization_format == "pickle":
                with open(self.storage_file, 'wb') as f:
                    pickle.dump(data_to_save, f)
            elif self.serialization_format == "json":
                with open(self.storage_file, 'w') as f:
                    json.dump(data_to_save, f)
            else:
                self.logger.error(f"Unsupported serialization format: {self.serialization_format}")
                return

            self.logger.info(f"Saved memories to {self.storage_file} [{self.serialization_format}].")

        except Exception as e:
            self.logger.exception(f"Error saving memories: {e}")

    def _load_memories(self) -> None:
        """Load memories from disk into short/medium/long, re-building indexes."""
        if not self.storage_file or not os.path.exists(self.storage_file):
            return

        try:
            if self.serialization_format == "pickle":
                with open(self.storage_file, 'rb') as f:
                    loaded = pickle.load(f)
            elif self.serialization_format == "json":
                with open(self.storage_file, 'r') as f:
                    loaded = json.load(f)
            else:
                self.logger.error(f"Unsupported format: {self.serialization_format}")
                return

            self.short_term.clear()
            self.medium_term.clear()
            self.long_term.clear()
            self._clear_all_indexes()

            # Rebuild from loaded data
            for store_name in ("short_term", "medium_term", "long_term"):
                store_dict = loaded.get(store_name, {})
                for mid, mem_data in store_dict.items():
                    item = MemoryItem.from_dict(mem_data)
                    getattr(self, store_name)[mid] = item
                    self._index_memory(item)

            self.logger.info(f"Loaded memories from {self.storage_file} [{self.serialization_format}].")

        except Exception as e:
            self.logger.exception(f"Error loading memories: {e}")
