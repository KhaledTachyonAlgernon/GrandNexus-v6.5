"""
neural_reasoner.py - Neural reasoning module for GrandNexus

This module implements neural network-based reasoning capabilities for GrandNexus,
allowing for vector-based reasoning, pattern matching, and semantic similarity
computation within the cognitive architecture. It serves as a complement to
symbolic reasoning approaches, excelling at fuzzy pattern recognition and handling
uncertainty in knowledge.

Key features:
- Integration with embedding spaces for semantic reasoning
- Neural retrieval and similarity-based inference
- Hybrid neuro-symbolic operations supporting higher reasoning modules
- Adaptive weighting of neural vs symbolic evidence
"""

import logging
import threading
import time
import uuid
from typing import Dict, List, Any, Optional, Callable, Union, Tuple, Set
from dataclasses import dataclass, field
from enum import Enum, auto
import numpy as np
from collections import defaultdict

# Import from GrandNexus modules
from grandnexus.core.nexus_core import NexusCore
from grandnexus.memory.semantic import SemanticMemory
from grandnexus.memory.episodic import EpisodicMemory
from grandnexus.memory.working import WorkingMemory
from grandnexus.graph.graph_engine import create_graph_engine, AbstractGraphEngine

# -----------------------------------------------------------------------------
# Data Structures and Types
# -----------------------------------------------------------------------------

class EvidenceSourceType(Enum):
    """Types of sources that can provide reasoning evidence"""
    EMBEDDING = auto()       # Embeddings space similarity
    RETRIEVAL = auto()       # Retrieved information from memory
    GRAPH = auto()           # Graph-based inference
    PATTERN = auto()         # Pattern matching
    ANALOGY = auto()         # Analogical reasoning
    EXTERNAL = auto()        # External source (e.g., LLM)

class ReasoningTask(Enum):
    """Types of neural reasoning tasks"""
    SIMILARITY = auto()      # Compute similarity between concepts
    CLASSIFICATION = auto()  # Classify a concept
    CLUSTERING = auto()      # Group similar concepts
    COMPLETION = auto()      # Complete a partial pattern
    RETRIEVAL = auto()       # Retrieve similar concepts
    INFERENCE = auto()       # Infer new knowledge
    ANALOGY = auto()         # Find or complete analogies

@dataclass
class NeuralEvidence:
    """Evidence retrieved or computed by neural reasoning processes"""
    evidence_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    source_type: EvidenceSourceType = EvidenceSourceType.EMBEDDING
    content: Any = None
    confidence: float = 0.5
    relevance: float = 0.8
    embedding: Optional[np.ndarray] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class NeuralReasoningRequest:
    """Request for neural reasoning"""
    request_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    task_type: ReasoningTask = ReasoningTask.SIMILARITY
    query: Any = None
    query_embedding: Optional[np.ndarray] = None
    context: Optional[Dict[str, Any]] = None
    parameters: Dict[str, Any] = field(default_factory=dict)
    evidence_sources: List[EvidenceSourceType] = field(default_factory=list)
    timestamp: float = field(default_factory=time.time)

@dataclass
class NeuralReasoningResult:
    """Result of neural reasoning"""
    request_id: str
    conclusion: Any = None
    confidence: float = 0.0
    evidence: List[NeuralEvidence] = field(default_factory=list)
    embedding: Optional[np.ndarray] = None
    explanation: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    processing_time: float = 0.0

# -----------------------------------------------------------------------------
# Neural Reasoner Class
# -----------------------------------------------------------------------------

class NeuralReasoner:
    """
    Neural reasoning component for GrandNexus that leverages vector embeddings
    and neural networks to perform reasoning operations that complement
    traditional symbolic approaches.

    This module excels at similarity-based reasoning, analogy detection,
    and handling fuzzy or uncertain knowledge.
    """

    def __init__(self,
                 nexus_core: Optional[NexusCore] = None,
                 semantic_memory: Optional[SemanticMemory] = None,
                 episodic_memory: Optional[EpisodicMemory] = None,
                 working_memory: Optional[WorkingMemory] = None,
                 graph_engine: Optional[AbstractGraphEngine] = None,
                 embedding_dim: int = 768,
                 similarity_threshold: float = 0.7,
                 use_llm_embeddings: bool = True):
        """
        Initialize the neural reasoner.

        Args:
            nexus_core: Reference to the central nexus core
            semantic_memory: Reference to semantic memory for concept retrieval
            episodic_memory: Reference to episodic memory for experience retrieval
            working_memory: Reference to working memory for context
            graph_engine: Reference to graph engine for graph operations
            embedding_dim: Dimension of embedding vectors
            similarity_threshold: Default threshold for similarity matches
            use_llm_embeddings: Whether to use LLM-based embeddings
        """
        self.logger = logging.getLogger("GrandNexus.Reasoning.Neural")

        # Store module references
        self.nexus_core = nexus_core
        self.semantic_memory = semantic_memory
        self.episodic_memory = episodic_memory
        self.working_memory = working_memory
        self.graph_engine = graph_engine

        # Configuration
        self.embedding_dim = embedding_dim
        self.similarity_threshold = similarity_threshold
        self.use_llm_embeddings = use_llm_embeddings

        # Internal state
        self.instance_id = str(uuid.uuid4())[:8]
        self.running = False
        self.initialized = False
        self.lock = threading.RLock()

        # Cache frequently accessed embeddings
        self.embedding_cache = {}
        self.max_cache_size = 1000

        # Performance metrics and activation tracking
        self.metrics = defaultdict(list)
        self.task_history = []
        self.last_activation_time = time.time()

        # Adaptive weights for different evidence sources
        self.evidence_weights = {
            EvidenceSourceType.EMBEDDING: 1.0,
            EvidenceSourceType.RETRIEVAL: 0.8,
            EvidenceSourceType.GRAPH: 0.7,
            EvidenceSourceType.PATTERN: 0.6,
            EvidenceSourceType.ANALOGY: 0.5,
            EvidenceSourceType.EXTERNAL: 0.9
        }

        self.logger.info(f"NeuralReasoner initialized with ID={self.instance_id}")

    def initialize(self) -> bool:
        """Initialize the neural reasoner and establish connections"""
        if self.initialized:
            return True

        with self.lock:
            try:
                # Check for essential dependencies
                if self.semantic_memory is None:
                    self.logger.warning("Neural reasoner initialized without semantic memory")

                # Initialize any pre-trained models or weights if needed
                # [In a real implementation, this could load pre-trained models or embeddings]

                # Connect with NexusCore for messaging/coordination if available
                if self.nexus_core:
                    self._register_with_nexus()

                self.initialized = True
                self.logger.info("Neural reasoner initialization complete")
                return True
            except Exception as e:
                self.logger.error(f"Failed to initialize neural reasoner: {str(e)}")
                return False

    def start(self) -> bool:
        """Start the neural reasoner"""
        if not self.initialized:
            if not self.initialize():
                return False

        with self.lock:
            if self.running:
                return True

            try:
                # Start any background processes if needed
                # [In a real implementation, this might start background workers]

                self.running = True
                self.logger.info("Neural reasoner started successfully")
                return True
            except Exception as e:
                self.logger.error(f"Failed to start neural reasoner: {str(e)}")
                return False

    def stop(self) -> bool:
        """Stop the neural reasoner"""
        with self.lock:
            if not self.running:
                return True

            try:
                # Stop any background processes
                # [In a real implementation, this would clean up resources]

                self.running = False
                self.logger.info("Neural reasoner stopped successfully")
                return True
            except Exception as e:
                self.logger.error(f"Error stopping neural reasoner: {str(e)}")
                return False

    def process(self, request: NeuralReasoningRequest) -> NeuralReasoningResult:
        """
        Process a neural reasoning request.

        Args:
            request: The neural reasoning request to process

        Returns:
            The result of neural reasoning
        """
        if not self.running:
            self.logger.warning("Neural reasoner not running, attempting to start")
            if not self.start():
                raise RuntimeError("Neural reasoner could not be started")

        self.logger.info(f"Processing request {request.request_id}, task: {request.task_type}")
        self.last_activation_time = time.time()

        start_time = time.time()

        # Prepare the result object
        result = NeuralReasoningResult(request_id=request.request_id)

        try:
            # Process different types of reasoning tasks
            if request.task_type == ReasoningTask.SIMILARITY:
                self._process_similarity_task(request, result)
            elif request.task_type == ReasoningTask.CLASSIFICATION:
                self._process_classification_task(request, result)
            elif request.task_type == ReasoningTask.CLUSTERING:
                self._process_clustering_task(request, result)
            elif request.task_type == ReasoningTask.COMPLETION:
                self._process_completion_task(request, result)
            elif request.task_type == ReasoningTask.RETRIEVAL:
                self._process_retrieval_task(request, result)
            elif request.task_type == ReasoningTask.INFERENCE:
                self._process_inference_task(request, result)
            elif request.task_type == ReasoningTask.ANALOGY:
                self._process_analogy_task(request, result)
            else:
                raise ValueError(f"Unknown reasoning task type: {request.task_type}")

            # Record processing time
            result.processing_time = time.time() - start_time

            # Update metrics
            self.metrics["processing_time"].append(result.processing_time)
            self.metrics["confidence"].append(result.confidence)

            # Store in history
            self.task_history.append({
                "request_id": request.request_id,
                "task_type": request.task_type,
                "timestamp": time.time(),
                "confidence": result.confidence
            })

            self.logger.info(f"Completed request {request.request_id} in {result.processing_time:.3f}s")
            return result

        except Exception as e:
            # Log error and return a result indicating failure
            self.logger.error(f"Error processing request {request.request_id}: {str(e)}")

            result.confidence = 0.0
            result.explanation = f"Error: {str(e)}"
            result.processing_time = time.time() - start_time

            return result

    def compute_similarity(self, entity1: Any, entity2: Any,
                          method: str = "cosine") -> float:
        """
        Compute semantic similarity between two entities.

        Args:
            entity1: First entity (can be text, concept ID, or embedding)
            entity2: Second entity (can be text, concept ID, or embedding)
            method: Similarity method ('cosine', 'dot', 'euclidean')

        Returns:
            Similarity score between 0 and 1
        """
        # Get embeddings for both entities
        emb1 = self._get_embedding(entity1)
        emb2 = self._get_embedding(entity2)

        if emb1 is None or emb2 is None:
            self.logger.warning(f"Could not get embeddings for similarity computation")
            return 0.0

        # Compute similarity based on specified method
        if method == "cosine":
            return self._cosine_similarity(emb1, emb2)
        elif method == "dot":
            return self._dot_similarity(emb1, emb2)
        elif method == "euclidean":
            return self._euclidean_similarity(emb1, emb2)
        else:
            self.logger.warning(f"Unknown similarity method: {method}, using cosine")
            return self._cosine_similarity(emb1, emb2)

    def find_nearest(self, query: Any, candidates: List[Any],
                    top_k: int = 5, method: str = "cosine") -> List[Tuple[Any, float]]:
        """
        Find the nearest entities to a query from a list of candidates.

        Args:
            query: Query entity (text, concept ID, or embedding)
            candidates: List of candidate entities
            top_k: Number of top results to return
            method: Similarity method to use

        Returns:
            List of (entity, similarity) tuples for the top matches
        """
        # Get query embedding
        query_embedding = self._get_embedding(query)
        if query_embedding is None:
            self.logger.warning(f"Could not get embedding for query")
            return []

        # Compute similarity with all candidates
        similarities = []
        for candidate in candidates:
            candidate_embedding = self._get_embedding(candidate)
            if candidate_embedding is not None:
                if method == "cosine":
                    sim = self._cosine_similarity(query_embedding, candidate_embedding)
                elif method == "dot":
                    sim = self._dot_similarity(query_embedding, candidate_embedding)
                elif method == "euclidean":
                    sim = self._euclidean_similarity(query_embedding, candidate_embedding)
                else:
                    sim = self._cosine_similarity(query_embedding, candidate_embedding)

                similarities.append((candidate, sim))

        # Sort by similarity (descending) and return top_k
        similarities.sort(key=lambda x: x[1], reverse=True)
        return similarities[:top_k]

    def complete_pattern(self, pattern: List[Any], options: List[Any]) -> Tuple[Any, float]:
        """
        Complete a pattern by finding the best continuation.

        Args:
            pattern: List of entities forming a pattern
            options: List of candidate entities to complete the pattern

        Returns:
            Tuple of (best_completion, confidence)
        """
        if not pattern or len(pattern) < 2:
            self.logger.warning("Pattern too short for completion")
            return (None, 0.0)

        if not options:
            self.logger.warning("No options provided for pattern completion")
            return (None, 0.0)

        # Convert pattern to embeddings
        pattern_embeddings = [self._get_embedding(item) for item in pattern]
        pattern_embeddings = [emb for emb in pattern_embeddings if emb is not None]

        if len(pattern_embeddings) < 2:
            self.logger.warning("Insufficient valid embeddings in pattern")
            return (None, 0.0)

        # Simple pattern continuation: predict by vector arithmetic
        # Example: A is to B as C is to ?
        # We compute D = C + (B - A) in the embedding space
        if len(pattern_embeddings) >= 3:
            # Take the last 3 embeddings to form A, B, C for A:B::C:?
            a, b, c = pattern_embeddings[-3:]
            target = c + (b - a)

            # Normalize target
            target = target / np.linalg.norm(target)

            # Find closest option to target
            best_option = None
            best_similarity = -1.0

            for option in options:
                option_emb = self._get_embedding(option)
                if option_emb is not None:
                    sim = self._cosine_similarity(target, option_emb)
                    if sim > best_similarity:
                        best_similarity = sim
                        best_option = option

            return (best_option, best_similarity)

        # Fallback: average embedding similarity to pattern
        similarities = []
        for option in options:
            option_emb = self._get_embedding(option)
            if option_emb is not None:
                # Compute average similarity to pattern items
                avg_sim = sum(self._cosine_similarity(option_emb, pattern_emb)
                              for pattern_emb in pattern_embeddings) / len(pattern_embeddings)
                similarities.append((option, avg_sim))

        if not similarities:
            return (None, 0.0)

        # Return the option with highest similarity
        similarities.sort(key=lambda x: x[1], reverse=True)
        return similarities[0]

    def find_cluster_centers(self, entities: List[Any], n_clusters: int = 3) -> List[Any]:
        """
        Find cluster centers among entities in embedding space.

        Args:
            entities: List of entities to cluster
            n_clusters: Number of clusters to identify

        Returns:
            List of entities representing cluster centers
        """
        # Convert entities to embeddings
        embeddings = []
        valid_entities = []

        for entity in entities:
            emb = self._get_embedding(entity)
            if emb is not None:
                embeddings.append(emb)
                valid_entities.append(entity)

        if len(embeddings) < n_clusters:
            self.logger.warning(f"Not enough valid entities for {n_clusters} clusters")
            return valid_entities  # Return all valid entities if too few

        # Convert to numpy array
        embeddings_array = np.array(embeddings)

        # Simple k-means clustering
        # In a real implementation, we would use a full sklearn KMeans
        # Here we'll do a simplified version

        # Initialize random centers
        np.random.seed(42)  # For reproducibility
        center_indices = np.random.choice(len(embeddings), n_clusters, replace=False)
        centers = embeddings_array[center_indices]

        # Run a few iterations of k-means
        for _ in range(5):
            # Assign to clusters
            distances = np.array([[np.linalg.norm(emb - center) for center in centers]
                                 for emb in embeddings_array])
            cluster_assignments = np.argmin(distances, axis=1)

            # Update centers
            for i in range(n_clusters):
                cluster_points = embeddings_array[cluster_assignments == i]
                if len(cluster_points) > 0:
                    centers[i] = np.mean(cluster_points, axis=0)

        # Find entities closest to final centers
        center_entities = []
        for center in centers:
            distances = [np.linalg.norm(emb - center) for emb in embeddings_array]
            closest_idx = np.argmin(distances)
            center_entities.append(valid_entities[closest_idx])

        return center_entities

    def adapt_weights(self, feedback: Dict[EvidenceSourceType, float]) -> None:
        """
        Adapt weights of different evidence sources based on feedback.

        Args:
            feedback: Dictionary mapping evidence sources to feedback scores (-1 to 1)
        """
        with self.lock:
            for source_type, score in feedback.items():
                if source_type in self.evidence_weights:
                    # Apply bounded update with diminishing returns for extreme values
                    current = self.evidence_weights[source_type]
                    delta = score * 0.1 * (1.0 - current if score > 0 else current)
                    self.evidence_weights[source_type] = max(0.1, min(1.0, current + delta))

            self.logger.info(f"Updated evidence weights: {self.evidence_weights}")

    def get_activation(self) -> float:
        """
        Get the current activation level of the neural reasoner.

        Returns:
            Activation level between 0 and 1, indicating recent usage
        """
        time_since_active = time.time() - self.last_activation_time
        # Exponential decay of activation
        return np.exp(-time_since_active / 3600)  # Half-life of about 1 hour

    def clear_cache(self) -> None:
        """Clear the embedding cache"""
        with self.lock:
            self.embedding_cache.clear()
            self.logger.info("Embedding cache cleared")

    # ------------------------------------------------------------------------
    # Internal methods
    # ------------------------------------------------------------------------

    def _register_with_nexus(self) -> None:
        """Register with NexusCore for coordination"""
        try:
            # Register as a module
            dependencies = []
            if self.semantic_memory:
                dependencies.append("semantic_memory")
            if self.episodic_memory:
                dependencies.append("episodic_memory")
            if self.working_memory:
                dependencies.append("working_memory")
            if self.graph_engine:
                dependencies.append("graph_engine")

            self.nexus_core.register_module(
                name="neural_reasoner",
                module=self,
                dependencies=dependencies
            )

            self.logger.info("Successfully registered with NexusCore")
        except Exception as e:
            self.logger.error(f"Failed to register with NexusCore: {str(e)}")

    def _get_embedding(self, entity: Any) -> Optional[np.ndarray]:
        """
        Get embedding for an entity. The entity can be:
        - A text string
        - A concept ID from semantic memory
        - A pre-computed embedding

        If caching is enabled, will check cache first.

        Args:
            entity: The entity to get embedding for

        Returns:
            Embedding vector or None if unavailable
        """
        # Check if input is already an embedding
        if isinstance(entity, np.ndarray) and entity.ndim == 1:
            return entity / np.linalg.norm(entity)  # Normalize

        # Generate cache key
        cache_key = str(entity)

        # Check cache first
        if cache_key in self.embedding_cache:
            return self.embedding_cache[cache_key]

        # Try to get embedding based on entity type
        embedding = None

        if isinstance(entity, str):
            # Check if it's a concept ID in semantic memory
            if self.semantic_memory:
                try:
                    concept = self.semantic_memory.get_concept(entity)
                    if concept and concept.embedding is not None:
                        embedding = concept.embedding
                except Exception as e:
                    self.logger.debug(f"Error retrieving concept {entity}: {str(e)}")

            # If not found, treat as text and get embedding
            if embedding is None and self.use_llm_embeddings:
                embedding = self._get_text_embedding(entity)

        # Store in cache if valid
        if embedding is not None:
            # Normalize embedding
            embedding = embedding / np.linalg.norm(embedding)

            # Add to cache
            self.embedding_cache[cache_key] = embedding

            # Manage cache size
            if len(self.embedding_cache) > self.max_cache_size:
                # Simple strategy: remove a random key
                keys = list(self.embedding_cache.keys())
                key_to_remove = keys[np.random.randint(0, len(keys))]
                del self.embedding_cache[key_to_remove]

        return embedding

    def _get_text_embedding(self, text: str) -> Optional[np.ndarray]:
        """
        Get embedding for a text string using LLM or other embedding sources.

        Args:
            text: Text to get embedding for

        Returns:
            Embedding vector or None if unavailable
        """
        # In a real implementation, this would call an embedding model or service
        # For example, using OpenAI or a local model via HuggingFace

        # Simplified mock implementation for demonstration purposes
        try:
            # Check if we have access to an LLM interface through NexusCore
            if self.nexus_core:
                llm_interface = self.nexus_core.get_module("llm_interface")
                if llm_interface and hasattr(llm_interface, "create_embedding"):
                    self.logger.debug(f"Getting embedding for text via llm_interface")
                    return llm_interface.create_embedding(text)

            # Fallback to a mock embedding (random but deterministic based on input)
            self.logger.debug(f"Generating mock embedding for text")
            # Create a deterministic but unique hash from the text
            np.random.seed(hash(text) % 2**32)
            embedding = np.random.randn(self.embedding_dim)
            return embedding / np.linalg.norm(embedding)  # Normalize

        except Exception as e:
            self.logger.warning(f"Error generating embedding for text: {str(e)}")
            return None

    # Similarity metrics

    def _cosine_similarity(self, vec1: np.ndarray, vec2: np.ndarray) -> float:
        """
        Compute cosine similarity between two vectors.
        Assumes vectors are already normalized.

        Args:
            vec1, vec2: Vectors to compare

        Returns:
            Cosine similarity score (0-1)
        """
        # For normalized vectors, cosine similarity = dot product
        similarity = np.dot(vec1, vec2)

        # Clamp to -1..1 range (in case of numerical issues)
        similarity = max(-1.0, min(1.0, similarity))

        # Map from -1..1 to 0..1
        return (similarity + 1) / 2

    def _dot_similarity(self, vec1: np.ndarray, vec2: np.ndarray) -> float:
        """
        Compute dot product similarity, scaled to 0-1 range.

        Args:
            vec1, vec2: Vectors to compare

        Returns:
            Normalized similarity score (0-1)
        """
        # For normalized vectors, we can map from -1..1 to 0..1
        raw_dot = np.dot(vec1, vec2)
        return (raw_dot + 1) / 2

    def _euclidean_similarity(self, vec1: np.ndarray, vec2: np.ndarray) -> float:
        """
        Compute similarity based on Euclidean distance.

        Args:
            vec1, vec2: Vectors to compare

        Returns:
            Similarity score (0-1), where 1 = identical
        """
        distance = np.linalg.norm(vec1 - vec2)
        # Convert distance to similarity: 0 distance = 1 similarity, large distance = 0 similarity
        return 1 / (1 + distance)

    # Task processing methods

    def _process_similarity_task(self, request: NeuralReasoningRequest,
                                result: NeuralReasoningResult) -> None:
        """
        Process similarity computation between entities.

        Args:
            request: The reasoning request
            result: Result object to populate
        """
        if not isinstance(request.query, list) or len(request.query) != 2:
            raise ValueError("Similarity task requires query to be a list of two items")

        entity1, entity2 = request.query

        # Get embeddings if not provided
        emb1 = request.query_embedding if request.query_embedding is not None else self._get_embedding(entity1)
        emb2 = self._get_embedding(entity2)

        if emb1 is None or emb2 is None:
            raise ValueError("Could not get embeddings for similarity computation")

        # Get similarity method from parameters or use default
        method = request.parameters.get("similarity_method", "cosine")

        # Compute similarity
        if method == "cosine":
            similarity = self._cosine_similarity(emb1, emb2)
        elif method == "dot":
            similarity = self._dot_similarity(emb1, emb2)
        elif method == "euclidean":
            similarity = self._euclidean_similarity(emb1, emb2)
        else:
            self.logger.warning(f"Unknown similarity method: {method}, using cosine")
            similarity = self._cosine_similarity(emb1, emb2)

        # Create evidence
        evidence = NeuralEvidence(
            source_type=EvidenceSourceType.EMBEDDING,
            content=f"Similarity between '{entity1}' and '{entity2}': {similarity:.3f}",
            confidence=0.9,  # High confidence for direct computation
            relevance=1.0,
            embedding=None,
            metadata={
                "entity1": entity1,
                "entity2": entity2,
                "similarity": similarity,
                "method": method
            }
        )

        # Populate result
        result.conclusion = similarity
        result.confidence = 0.9  # High confidence for direct computation
        result.evidence = [evidence]
        result.explanation = f"Computed {method} similarity between '{entity1}' and '{entity2}'"

    def _process_classification_task(self, request: NeuralReasoningRequest,
                                    result: NeuralReasoningResult) -> None:
        """
        Process classification of entities into categories.

        Args:
            request: The reasoning request
            result: Result object to populate
        """
        # Extract query and ensure it's the right structure
        if not isinstance(request.query, dict) or "entity" not in request.query or "categories" not in request.query:
            raise ValueError("Classification task requires query with 'entity' and 'categories'")

        entity = request.query["entity"]
        categories = request.query["categories"]

        if not isinstance(categories, list) or len(categories) < 2:
            raise ValueError("Categories must be a list with at least 2 items")

        # Get embedding for entity
        entity_emb = request.query_embedding if request.query_embedding is not None else self._get_embedding(entity)

        if entity_emb is None:
            raise ValueError(f"Could not get embedding for entity: {entity}")

        # Compute similarity with each category
        similarities = []
        category_embeddings = {}

        for category in categories:
            category_emb = self._get_embedding(category)
            if category_emb is not None:
                sim = self._cosine_similarity(entity_emb, category_emb)
                similarities.append((category, sim))
                category_embeddings[category] = category_emb

        if not similarities:
            raise ValueError("Could not compute similarities with any category")

        # Sort by similarity
        similarities.sort(key=lambda x: x[1], reverse=True)

        # Apply any threshold from parameters
        threshold = request.parameters.get("classification_threshold", 0.5)

        # Filter by threshold
        valid_categories = [(cat, sim) for cat, sim in similarities if sim >= threshold]

        # For each valid category, add an evidence
        evidence_list = []
        for category, similarity in valid_categories:
            evidence = NeuralEvidence(
                source_type=EvidenceSourceType.EMBEDDING,
                content=f"Entity '{entity}' matches category '{category}' with score {similarity:.3f}",
                confidence=similarity,
                relevance=1.0,
                metadata={
                    "entity": entity,
                    "category": category,
                    "similarity": similarity,
                    "threshold": threshold
                }
            )
            evidence_list.append(evidence)

        # Compute overall confidence based on contrast between top categories
        if len(similarities) >= 2:
            top_sim = similarities[0][1]
            second_sim = similarities[1][1]
            contrast = (top_sim - second_sim) / top_sim if top_sim > 0 else 0
            confidence = top_sim * (0.5 + 0.5 * contrast)  # Boost confidence if clear contrast
        else:
            confidence = similarities[0][1] if similarities else 0.0

        # Populate result
        best_category = similarities[0][0] if similarities else None
        result.conclusion = best_category
        result.confidence = confidence
        result.evidence = evidence_list
        result.embedding = entity_emb

        # Build explanation
        if valid_categories:
            if len(valid_categories) == 1:
                result.explanation = f"Entity '{entity}' classified as '{best_category}' with confidence {confidence:.3f}"
            else:
                categories_str = ", ".join([f"'{cat}' ({sim:.3f})" for cat, sim in valid_categories[:3]])
                result.explanation = f"Entity '{entity}' matches multiple categories: {categories_str}"
        else:
            result.explanation = f"Entity '{entity}' did not match any category above threshold {threshold}"

    def _process_clustering_task(self, request: NeuralReasoningRequest,
                                result: NeuralReasoningResult) -> None:
        """
        Process clustering of entities.

        Args:
            request: The reasoning request
            result: Result object to populate
        """
        # Extract query and ensure it's the right structure
        if not isinstance(request.query, list):
            raise ValueError("Clustering task requires query to be a list of entities")

        entities = request.query

        if len(entities) < 2:
            raise ValueError("Need at least 2 entities for clustering")

        # Get number of clusters from parameters or derive from data size
        n_clusters = request.parameters.get("n_clusters")
        if n_clusters is None:
            # Heuristic: sqrt of number of entities, bounded
            n_clusters = min(max(2, int(np.sqrt(len(entities)))), 10)

        # Get embeddings for all entities
        entity_embeddings = []
        valid_entities = []

        for entity in entities:
            emb = self._get_embedding(entity)
            if emb is not None:
                entity_embeddings.append(emb)
                valid_entities.append(entity)

        if len(valid_entities) < n_clusters:
            n_clusters = len(valid_entities)
            self.logger.warning(f"Reduced number of clusters to {n_clusters} due to insufficient valid entities")

        if len(valid_entities) < 2:
            raise ValueError("Need at least 2 valid entities with embeddings for clustering")

        # Convert to numpy array for clustering
        embeddings_array = np.array(entity_embeddings)

        # Simple k-means clustering
        # In a real implementation, we would use a full sklearn KMeans
        # Here we'll do a simplified version

        # Initialize random centers
        np.random.seed(42)  # For reproducibility
        center_indices = np.random.choice(len(embeddings_array), n_clusters, replace=False)
        centers = embeddings_array[center_indices].copy()

        # Run a few iterations of k-means
        for _ in range(5):
            # Assign to clusters
            distances = np.array([[np.linalg.norm(emb - center) for center in centers]
                                for emb in embeddings_array])
            cluster_assignments = np.argmin(distances, axis=1)

            # Update centers
            for i in range(n_clusters):
                cluster_points = embeddings_array[cluster_assignments == i]
                if len(cluster_points) > 0:
                    centers[i] = np.mean(cluster_points, axis=0)

        # Collect results
        clusters = [[] for _ in range(n_clusters)]
        for idx, cluster_idx in enumerate(cluster_assignments):
            clusters[cluster_idx].append(valid_entities[idx])

        # Find entities closest to final centers (cluster representatives)
        representatives = []
        for i, center in enumerate(centers):
            if len(clusters[i]) > 0:  # Ensure cluster is not empty
                distances = [np.linalg.norm(embeddings_array[j] - center)
                             for j, entity in enumerate(valid_entities)
                             if cluster_assignments[j] == i]
                closest_idx = np.argmin(distances)
                # Find the entity in this cluster with this index
                cluster_entities = [valid_entities[j] for j, ca in enumerate(cluster_assignments) if ca == i]
                representatives.append(cluster_entities[closest_idx])
            else:
                representatives.append(None)

        # Create evidence for each cluster
        evidence_list = []
        for i, (cluster, representative) in enumerate(zip(clusters, representatives)):
            if representative is not None:
                evidence = NeuralEvidence(
                    source_type=EvidenceSourceType.PATTERN,
                    content=f"Cluster {i+1} with {len(cluster)} entities, representative: '{representative}'",
                    confidence=0.8,  # Fixed confidence for clustering
                    relevance=1.0,
                    metadata={
                        "cluster_index": i,
                        "cluster_size": len(cluster),
                        "cluster_entities": cluster,
                        "representative": representative
                    }
                )
                evidence_list.append(evidence)

        # Compute overall quality metric (simplified)
        # In a real implementation, we would use proper clustering metrics like silhouette score
        # Here we'll use a simple heuristic: average distance to cluster center
        intra_cluster_distances = []
        for i in range(n_clusters):
            cluster_points = embeddings_array[cluster_assignments == i]
            if len(cluster_points) > 0:
                distances = [np.linalg.norm(point - centers[i]) for point in cluster_points]
                intra_cluster_distances.extend(distances)

        avg_distance = np.mean(intra_cluster_distances) if intra_cluster_distances else 1.0
        quality_score = 1.0 / (1.0 + avg_distance)  # Convert to 0-1 score, higher is better

        # Populate result
        result.conclusion = {
            "clusters": clusters,
            "representatives": representatives,
            "num_clusters": n_clusters,
            "quality_score": quality_score
        }
        result.confidence = quality_score
        result.evidence = evidence_list

        # Build explanation
        result.explanation = f"Clustered {len(valid_entities)} entities into {n_clusters} groups with quality score {quality_score:.3f}"

    def _process_completion_task(self, request: NeuralReasoningRequest,
                                result: NeuralReasoningResult) -> None:
        """
        Process pattern completion task.

        Args:
            request: The reasoning request
            result: Result object to populate
        """
        # Extract query and ensure it's the right structure
        if not isinstance(request.query, dict) or "pattern" not in request.query or "options" not in request.query:
            raise ValueError("Completion task requires query with 'pattern' and 'options'")

        pattern = request.query["pattern"]
        options = request.query["options"]

        if not isinstance(pattern, list) or len(pattern) < 2:
            raise ValueError("Pattern must be a list with at least 2 items")

        if not isinstance(options, list) or len(options) < 1:
            raise ValueError("Options must be a list with at least 1 item")

        # Convert pattern to embeddings
        pattern_embeddings = []
        valid_pattern_items = []

        for item in pattern:
            emb = self._get_embedding(item)
            if emb is not None:
                pattern_embeddings.append(emb)
                valid_pattern_items.append(item)

        if len(pattern_embeddings) < 2:
            raise ValueError("Need at least 2 valid items with embeddings in the pattern")

        # Determine completion type from parameters
        completion_type = request.parameters.get("completion_type", "analogy")

        if completion_type == "analogy" and len(pattern_embeddings) >= 3:
            # A:B::C:? style analogy
            a, b, c = pattern_embeddings[-3:]
            target = c + (b - a)
            # Normalize
            target = target / np.linalg.norm(target)

            # Find closest option
            best_option = None
            best_similarity = -1.0

            for option in options:
                option_emb = self._get_embedding(option)
                if option_emb is not None:
                    sim = self._cosine_similarity(target, option_emb)
                    if sim > best_similarity:
                        best_similarity = sim
                        best_option = option

            if best_option is None:
                raise ValueError("Could not find a valid completion option")

            # Create evidence
            a_item, b_item, c_item = valid_pattern_items[-3:]
            evidence = NeuralEvidence(
                source_type=EvidenceSourceType.ANALOGY,
                content=f"Analogy completion: '{a_item}' is to '{b_item}' as '{c_item}' is to '{best_option}'",
                confidence=best_similarity,
                relevance=1.0,
                metadata={
                    "a": a_item,
                    "b": b_item,
                    "c": c_item,
                    "d": best_option,
                    "similarity": best_similarity,
                    "completion_type": "analogy"
                }
            )

            # Populate result
            result.conclusion = best_option
            result.confidence = best_similarity
            result.evidence = [evidence]
            result.explanation = f"Completed analogy pattern using vector arithmetic: '{a_item}' is to '{b_item}' as '{c_item}' is to '{best_option}' (confidence: {best_similarity:.3f})"

        else:
            # Simpler pattern continuation using average similarity
            similarities = []
            for option in options:
                option_emb = self._get_embedding(option)
                if option_emb is not None:
                    # Compute average similarity to pattern items
                    avg_sim = sum(self._cosine_similarity(option_emb, pattern_emb)
                                for pattern_emb in pattern_embeddings) / len(pattern_embeddings)
                    similarities.append((option, avg_sim))

            if not similarities:
                raise ValueError("Could not find a valid completion option")

            # Get best match
            similarities.sort(key=lambda x: x[1], reverse=True)
            best_option, best_similarity = similarities[0]

            # Create evidence
            evidence = NeuralEvidence(
                source_type=EvidenceSourceType.PATTERN,
                content=f"Pattern completion: '{best_option}' continues pattern with similarity {best_similarity:.3f}",
                confidence=best_similarity,
                relevance=1.0,
                metadata={
                    "pattern_items": valid_pattern_items,
                    "completion": best_option,
                    "similarity": best_similarity,
                    "completion_type": "continuation"
                }
            )

            # Populate result
            result.conclusion = best_option
            result.confidence = best_similarity
            result.evidence = [evidence]
            result.explanation = f"Completed pattern using average similarity, adding '{best_option}' (confidence: {best_similarity:.3f})"

    def _process_retrieval_task(self, request: NeuralReasoningRequest,
                               result: NeuralReasoningResult) -> None:
        """
        Process retrieval of similar entities.

        Args:
            request: The reasoning request
            result: Result object to populate
        """
        # Get query embedding
        if request.query_embedding is not None:
            query_emb = request.query_embedding
        elif request.query is not None:
            query_emb = self._get_embedding(request.query)
        else:
            raise ValueError("Retrieval task requires query or query_embedding")

        if query_emb is None:
            raise ValueError("Could not get embedding for query")

        # Get parameters
        top_k = request.parameters.get("top_k", 5)
        similarity_threshold = request.parameters.get("similarity_threshold", self.similarity_threshold)

        # Use different evidence sources based on request
        evidence_sources = request.evidence_sources or [EvidenceSourceType.RETRIEVAL]

        all_results = []

        # Use semantic memory if available and requested
        if self.semantic_memory is not None and EvidenceSourceType.RETRIEVAL in evidence_sources:
            try:
                # Try to query semantic memory with embedding
                query = {
                    "semantic_query": request.query if isinstance(request.query, str) else None,
                    "embedding": query_emb,
                    "max_results": top_k
                }

                semantic_results = self.semantic_memory.query_concepts(query)

                for concept in semantic_results:
                    # Calculate similarity if not provided
                    if hasattr(concept, 'embedding') and concept.embedding is not None:
                        similarity = self._cosine_similarity(query_emb, concept.embedding)
                    else:
                        similarity = 0.5  # Default if no embedding

                    if similarity >= similarity_threshold:
                        # Create evidence
                        evidence = NeuralEvidence(
                            source_type=EvidenceSourceType.RETRIEVAL,
                            content=concept,
                            confidence=similarity,
                            relevance=similarity,
                            embedding=concept.embedding if hasattr(concept, 'embedding') else None,
                            metadata={
                                "concept_id": concept.concept_id if hasattr(concept, 'concept_id') else None,
                                "name": concept.name if hasattr(concept, 'name') else str(concept),
                                "similarity": similarity,
                                "source": "semantic_memory"
                            }
                        )
                        all_results.append((evidence, similarity))
            except Exception as e:
                self.logger.warning(f"Error retrieving from semantic memory: {str(e)}")

        # Use graph engine if available and requested
        if self.graph_engine is not None and EvidenceSourceType.GRAPH in evidence_sources:
            try:
                # Try to find similar nodes in the graph
                node_criteria = {}
                if isinstance(request.query, str):
                    node_criteria = {"label": request.query}  # Simple text match

                # For a real implementation, this would use embedding-based graph search
                # Here we use a simplified approach
                node_ids = self.graph_engine.query_nodes(node_criteria)

                for node_id in node_ids[:top_k]:
                    try:
                        node = self.graph_engine.get_node(node_id)

                        # Calculate similarity if node has embedding
                        if hasattr(node, 'embedding') and node.embedding is not None:
                            similarity = self._cosine_similarity(query_emb, node.embedding)
                        else:
                            similarity = 0.4  # Lower default for graph match without embedding

                        if similarity >= similarity_threshold:
                            # Create evidence
                            evidence = NeuralEvidence(
                                source_type=EvidenceSourceType.GRAPH,
                                content=node,
                                confidence=similarity,
                                relevance=similarity,
                                embedding=node.embedding if hasattr(node, 'embedding') else None,
                                metadata={
                                    "node_id": node_id,
                                    "label": node.label if hasattr(node, 'label') else str(node),
                                    "similarity": similarity,
                                    "source": "graph_engine"
                                }
                            )
                            all_results.append((evidence, similarity))
                    except Exception as e:
                        self.logger.debug(f"Error retrieving node {node_id}: {str(e)}")
            except Exception as e:
                self.logger.warning(f"Error retrieving from graph engine: {str(e)}")

        # Use episodic memory if available and requested
        if self.episodic_memory is not None and EvidenceSourceType.RETRIEVAL in evidence_sources:
            try:
                # Here we would query episodic memory with the embedding
                # This is a simplified placeholder - real implementation would depend on the episodic memory interface

                # Add episodic memory results to all_results
                pass
            except Exception as e:
                self.logger.warning(f"Error retrieving from episodic memory: {str(e)}")

        # Sort all results by weighted similarity
        all_results.sort(key=lambda x: x[1], reverse=True)

        # Take top results
        top_results = all_results[:top_k]

        if not top_results:
            # No results found
            result.conclusion = []
            result.confidence = 0.0
            result.evidence = []
            result.explanation = "No items retrieved matching the query"
            return

        # Extract evidence and items
        evidence_list = [item[0] for item in top_results]
        retrieved_items = [e.content for e in evidence_list]

        # Weighted average confidence based on top matches
        confidence = sum(item[1] for item in top_results) / len(top_results)

        # Populate result
        result.conclusion = retrieved_items
        result.confidence = confidence
        result.evidence = evidence_list
        result.embedding = query_emb

        # Build explanation
        if isinstance(request.query, str):
            query_str = f"'{request.query}'"
        else:
            query_str = "the provided query"

        top_matches_str = ", ".join([
            f"'{e.metadata.get('name', str(e.content))}' ({e.confidence:.2f})"
            for e in evidence_list[:3]
        ])

        result.explanation = f"Retrieved {len(retrieved_items)} items similar to {query_str}. Top matches: {top_matches_str}"

    def _process_inference_task(self, request: NeuralReasoningRequest,
                               result: NeuralReasoningResult) -> None:
        """
        Process inference from neural evidence.

        Args:
            request: The reasoning request
            result: Result object to populate
        """
        # Extract inference type from parameters
        inference_type = request.parameters.get("inference_type", "property")

        # Handle different types of inferences
        if inference_type == "property":
            self._process_property_inference(request, result)
        elif inference_type == "relation":
            self._process_relation_inference(request, result)
        elif inference_type == "category":
            self._process_category_inference(request, result)
        else:
            raise ValueError(f"Unknown inference type: {inference_type}")

    def _process_property_inference(self, request: NeuralReasoningRequest,
                                   result: NeuralReasoningResult) -> None:
        """
        Infer properties of an entity based on similar entities.

        Args:
            request: The reasoning request
            result: Result object to populate
        """
        # Extract query entity
        if not isinstance(request.query, dict) or "entity" not in request.query:
            raise ValueError("Property inference requires query with 'entity'")

        entity = request.query["entity"]
        properties_to_infer = request.query.get("properties", [])

        # Get entity embedding
        entity_emb = request.query_embedding if request.query_embedding is not None else self._get_embedding(entity)

        if entity_emb is None:
            raise ValueError(f"Could not get embedding for entity: {entity}")

        # Find similar entities in semantic memory
        similar_entities = []
        properties_counts = defaultdict(int)
        properties_confidence = defaultdict(float)
        evidence_list = []

        # Use semantic memory if available
        if self.semantic_memory is not None:
            try:
                query = {
                    "semantic_query": entity if isinstance(entity, str) else None,
                    "embedding": entity_emb,
                    "max_results": 10
                }

                similar_concepts = self.semantic_memory.query_concepts(query)

                for concept in similar_concepts:
                    # Skip if it's the same entity
                    if hasattr(concept, 'concept_id') and str(concept.concept_id) == str(entity):
                        continue

                    # Calculate similarity
                    if hasattr(concept, 'embedding') and concept.embedding is not None:
                        similarity = self._cosine_similarity(entity_emb, concept.embedding)
                    else:
                        similarity = 0.5  # Default

                    # Only consider sufficiently similar entities
                    if similarity >= self.similarity_threshold:
                        similar_entities.append((concept, similarity))

                        # Extract properties from concept
                        if hasattr(concept, 'properties'):
                            for prop_name, prop_value in concept.properties.items():
                                if not properties_to_infer or prop_name in properties_to_infer:
                                    properties_counts[prop_name] += 1
                                    properties_confidence[prop_name] += similarity

                        # Create evidence
                        evidence = NeuralEvidence(
                            source_type=EvidenceSourceType.RETRIEVAL,
                            content=f"Similar entity: {concept.name if hasattr(concept, 'name') else str(concept)}",
                            confidence=similarity,
                            relevance=similarity,
                            metadata={
                                "entity": concept.concept_id if hasattr(concept, 'concept_id') else str(concept),
                                "similarity": similarity,
                                "properties": concept.properties if hasattr(concept, 'properties') else {}
                            }
                        )
                        evidence_list.append(evidence)
            except Exception as e:
                self.logger.warning(f"Error retrieving similar entities: {str(e)}")

        # Now infer properties based on what we found
        inferred_properties = {}

        for prop_name, count in properties_counts.items():
            if count >= 2:  # Require at least 2 similar entities to have this property
                confidence = properties_confidence[prop_name] / count
                inferred_properties[prop_name] = {
                    'confidence': confidence,
                    'support': count
                }

        if not inferred_properties:
            # No properties inferred
            result.conclusion = {}
            result.confidence = 0.0
            result.evidence = evidence_list
            result.explanation = f"Could not infer any properties for '{entity}' from similar entities"
            return

        # Add inferred properties evidence
        for prop_name, info in inferred_properties.items():
            evidence = NeuralEvidence(
                source_type=EvidenceSourceType.INFERENCE,
                content=f"Inferred property: {prop_name} for '{entity}'",
                confidence=info['confidence'],
                relevance=1.0,
                metadata={
                    "property": prop_name,
                    "confidence": info['confidence'],
                    "support": info['support'],
                    "inference_type": "property"
                }
            )
            evidence_list.append(evidence)

        # Compute overall confidence (average of property confidences)
        confidence = sum(info['confidence'] for info in inferred_properties.values()) / len(inferred_properties)

        # Populate result
        result.conclusion = inferred_properties
        result.confidence = confidence
        result.evidence = evidence_list
        result.embedding = entity_emb

        # Build explanation
        props_str = ", ".join([f"'{prop}' ({info['confidence']:.2f})"
                              for prop, info in list(inferred_properties.items())[:3]])

        result.explanation = f"Inferred {len(inferred_properties)} properties for '{entity}'. Top properties: {props_str}"

    def _process_relation_inference(self, request: NeuralReasoningRequest,
                                   result: NeuralReasoningResult) -> None:
        """
        Infer relations between entities.

        Args:
            request: The reasoning request
            result: Result object to populate
        """
        # Extract source and target entities
        if not isinstance(request.query, dict) or "source" not in request.query or "target" not in request.query:
            raise ValueError("Relation inference requires query with 'source' and 'target'")

        source = request.query["source"]
        target = request.query["target"]

        # Get embeddings
        source_emb = self._get_embedding(source)
        target_emb = self._get_embedding(target)

        if source_emb is None or target_emb is None:
            raise ValueError(f"Could not get embeddings for entities")

        # Check for direct relation in graph if available
        direct_relations = []
        if self.graph_engine is not None:
            try:
                # Convert to node IDs if needed
                source_id = source
                target_id = target

                # Check for edges between source and target
                edges = self.graph_engine.get_node_neighbors(source_id)
                for edge_id, edge_type, neighbor_id in edges:
                    if neighbor_id == target_id:
                        direct_relations.append((edge_type, edge_id))
            except Exception as e:
                self.logger.debug(f"Error checking direct relations: {str(e)}")

        # Find similar entity pairs in the graph to infer likely relations
        similar_relations = defaultdict(float)
        relation_evidence = defaultdict(list)

        # Helper to find similar pairs
        def find_similar_pairs():
            # Find entities similar to source
            similar_sources = self._find_similar_entities(source, source_emb, 5)

            # Find entities similar to target
            similar_targets = self._find_similar_entities(target, target_emb, 5)

            # Check relations between similar entity pairs
            for sim_source, source_sim in similar_sources:
                for sim_target, target_sim in similar_targets:
                    # Skip if it's the original pair
                    if str(sim_source) == str(source) and str(sim_target) == str(target):
                        continue

                    # Check relations between this pair
                    if self.graph_engine is not None:
                        try:
                            # Get source ID
                            source_id = sim_source

                            # Check edges
                            edges = self.graph_engine.get_node_neighbors(source_id)
                            for edge_id, edge_type, neighbor_id in edges:
                                if neighbor_id == str(sim_target):
                                    # Weight by product of similarities
                                    weight = source_sim * target_sim
                                    similar_relations[edge_type] += weight

                                    # Add as evidence
                                    relation_evidence[edge_type].append({
                                        "source": sim_source,
                                        "target": sim_target,
                                        "source_sim": source_sim,
                                        "target_sim": target_sim,
                                        "weight": weight,
                                        "edge_id": edge_id
                                    })
                        except Exception as e:
                            self.logger.debug(f"Error checking relations between similar entities: {str(e)}")

        # Find similar pairs to infer relations
        find_similar_pairs()

        # Prepare evidence list
        evidence_list = []

        # Add direct relation evidence if any
        for edge_type, edge_id in direct_relations:
            evidence = NeuralEvidence(
                source_type=EvidenceSourceType.GRAPH,
                content=f"Direct relation '{edge_type}' exists between '{source}' and '{target}'",
                confidence=1.0,  # High confidence for direct relation
                relevance=1.0,
                metadata={
                    "source": source,
                    "target": target,
                    "relation": edge_type,
                    "edge_id": edge_id,
                    "direct": True
                }
            )
            evidence_list.append(evidence)

        # Add similar relation evidence
        for relation_type, weight in similar_relations.items():
            # Normalize weight to 0-1 based on number of supporting pairs
            support_count = len(relation_evidence[relation_type])
            confidence = min(0.9, weight / support_count) if support_count > 0 else 0.0

            # Only include if confidence above threshold
            if confidence >= 0.3:  # Minimum threshold for relations
                # Get supporting examples (up to 3)
                examples = relation_evidence[relation_type][:3]
                examples_str = ", ".join([
                    f"'{ex['source']}' -> '{ex['target']}'" for ex in examples
                ])

                evidence = NeuralEvidence(
                    source_type=EvidenceSourceType.INFERENCE,
                    content=f"Inferred relation '{relation_type}' between '{source}' and '{target}' based on similar pairs",
                    confidence=confidence,
                    relevance=1.0,
                    metadata={
                        "source": source,
                        "target": target,
                        "relation": relation_type,
                        "support_count": support_count,
                        "weight": weight,
                        "examples": examples
                    }
                )
                evidence_list.append(evidence)

        # Combine direct and inferred relations
        inferred_relations = {}

        # First add direct relations with highest confidence
        for edge_type, _ in direct_relations:
            inferred_relations[edge_type] = {
                'confidence': 1.0,
                'direct': True
            }

        # Then add inferred relations if not already present
        for relation_type, weight in similar_relations.items():
            if relation_type not in inferred_relations:
                support_count = len(relation_evidence[relation_type])
                confidence = min(0.9, weight / support_count) if support_count > 0 else 0.0

                if confidence >= 0.3:  # Minimum threshold
                    inferred_relations[relation_type] = {
                        'confidence': confidence,
                        'direct': False,
                        'support_count': support_count
                    }

        if not inferred_relations:
            # No relations inferred
            result.conclusion = {}
            result.confidence = 0.0
            result.evidence = evidence_list
            result.explanation = f"Could not infer any relations between '{source}' and '{target}'"
            return

        # Compute overall confidence
        confidence = max(info['confidence'] for info in inferred_relations.values())

        # Populate result
        result.conclusion = inferred_relations
        result.confidence = confidence
        result.evidence = evidence_list

        # Build explanation
        if direct_relations:
            rel_types = [edge_type for edge_type, _ in direct_relations]
            rel_str = ", ".join([f"'{rel}'" for rel in rel_types])
            result.explanation = f"Found direct relation(s) {rel_str} between '{source}' and '{target}'"
        else:
            inferred_rels = [(rel, info['confidence']) for rel, info in inferred_relations.items()
                            if not info.get('direct', False)]
            inferred_rels.sort(key=lambda x: x[1], reverse=True)

            if inferred_rels:
                rel_str = ", ".join([f"'{rel}' ({conf:.2f})" for rel, conf in inferred_rels[:3]])
                result.explanation = f"Inferred relation(s) {rel_str} between '{source}' and '{target}'"
            else:
                result.explanation = f"No relations inferred between '{source}' and '{target}'"

    def _process_category_inference(self, request: NeuralReasoningRequest,
                                   result: NeuralReasoningResult) -> None:
        """
        Infer categories for an entity.

        Args:
            request: The reasoning request
            result: Result object to populate
        """
        # Extract entity
        if not isinstance(request.query, dict) or "entity" not in request.query:
            raise ValueError("Category inference requires query with 'entity'")

        entity = request.query["entity"]

        # Get entity embedding
        entity_emb = request.query_embedding if request.query_embedding is not None else self._get_embedding(entity)

        if entity_emb is None:
            raise ValueError(f"Could not get embedding for entity: {entity}")

        # Find similar entities in semantic memory
        similar_entities = []
        categories_counts = defaultdict(int)
        categories_confidence = defaultdict(float)
        evidence_list = []

        # Use semantic memory if available
        if self.semantic_memory is not None:
            try:
                query = {
                    "semantic_query": entity if isinstance(entity, str) else None,
                    "embedding": entity_emb,
                    "max_results": 10
                }

                similar_concepts = self.semantic_memory.query_concepts(query)

                for concept in similar_concepts:
                    # Skip if it's the same entity
                    if hasattr(concept, 'concept_id') and str(concept.concept_id) == str(entity):
                        continue

                    # Calculate similarity
                    if hasattr(concept, 'embedding') and concept.embedding is not None:
                        similarity = self._cosine_similarity(entity_emb, concept.embedding)
                    else:
                        similarity = 0.5  # Default

                    # Only consider sufficiently similar entities
                    if similarity >= self.similarity_threshold:
                        similar_entities.append((concept, similarity))

                        # Extract categories from concept
                        if hasattr(concept, 'categories'):
                            for category in concept.categories:
                                categories_counts[category] += 1
                                categories_confidence[category] += similarity

                        # Also check for "is_a" relations in graph if available
                        if self.graph_engine is not None and hasattr(concept, 'concept_id'):
                            try:
                                edges = self.graph_engine.get_node_neighbors(concept.concept_id)
                                for _, edge_type, neighbor_id in edges:
                                    if edge_type == "is_a" or edge_type == "instance_of" or edge_type == "type":
                                        # Get the target node (category)
                                        category_node = self.graph_engine.get_node(neighbor_id)
                                        if category_node:
                                            category = category_node.label if hasattr(category_node, 'label') else str(neighbor_id)
                                            categories_counts[category] += 1
                                            categories_confidence[category] += similarity
                            except Exception as e:
                                self.logger.debug(f"Error checking graph relations: {str(e)}")

                        # Create evidence
                        evidence = NeuralEvidence(
                            source_type=EvidenceSourceType.RETRIEVAL,
                            content=f"Similar entity: {concept.name if hasattr(concept, 'name') else str(concept)}",
                            confidence=similarity,
                            relevance=similarity,
                            metadata={
                                "entity": concept.concept_id if hasattr(concept, 'concept_id') else str(concept),
                                "similarity": similarity,
                                "categories": concept.categories if hasattr(concept, 'categories') else []
                            }
                        )
                        evidence_list.append(evidence)
            except Exception as e:
                self.logger.warning(f"Error retrieving similar entities: {str(e)}")

        # Now infer categories based on what we found
        inferred_categories = {}

        for category, count in categories_counts.items():
            if count >= 2:  # Require at least 2 similar entities to have this category
                confidence = categories_confidence[category] / count
                inferred_categories[category] = {
                    'confidence': confidence,
                    'support': count
                }

        if not inferred_categories:
            # No categories inferred
            result.conclusion = {}
            result.confidence = 0.0
            result.evidence = evidence_list
            result.explanation = f"Could not infer any categories for '{entity}' from similar entities"
            return

        # Add inferred categories evidence
        for category, info in inferred_categories.items():
            evidence = NeuralEvidence(
                source_type=EvidenceSourceType.INFERENCE,
                content=f"Inferred category: {category} for '{entity}'",
                confidence=info['confidence'],
                relevance=1.0,
                metadata={
                    "category": category,
                    "confidence": info['confidence'],
                    "support": info['support'],
                    "inference_type": "category"
                }
            )
            evidence_list.append(evidence)

        # Compute overall confidence (average of category confidences)
        confidence = sum(info['confidence'] for info in inferred_categories.values()) / len(inferred_categories)

        # Populate result
        result.conclusion = inferred_categories
        result.confidence = confidence
        result.evidence = evidence_list
        result.embedding = entity_emb

        # Build explanation
        categories_str = ", ".join([f"'{cat}' ({info['confidence']:.2f})"
                                   for cat, info in list(inferred_categories.items())[:3]])

        result.explanation = f"Inferred {len(inferred_categories)} categories for '{entity}'. Top categories: {categories_str}"

    def _process_analogy_task(self, request: NeuralReasoningRequest,
                             result: NeuralReasoningResult) -> None:
        """
        Process analogy-based reasoning (A is to B as C is to ?).

        Args:
            request: The reasoning request
            result: Result object to populate
        """
        # Extract query and ensure it's the right structure
        if not isinstance(request.query, dict) or "a" not in request.query or "b" not in request.query or "c" not in request.query:
            raise ValueError("Analogy task requires query with 'a', 'b', and 'c'")

        # Extract analogy components
        a = request.query["a"]
        b = request.query["b"]
        c = request.query["c"]
        options = request.query.get("options", [])

        # Get embeddings
        a_emb = self._get_embedding(a)
        b_emb = self._get_embedding(b)
        c_emb = self._get_embedding(c)

        if a_emb is None or b_emb is None or c_emb is None:
            raise ValueError("Could not get embeddings for all analogy components")

        # Calculate target embedding using vector arithmetic
        # D = C + (B - A)
        target_emb = c_emb + (b_emb - a_emb)

        # Normalize
        target_emb = target_emb / np.linalg.norm(target_emb)

        # If options are provided, find the best match
        if options:
            similarities = []
            for option in options:
                option_emb = self._get_embedding(option)
                if option_emb is not None:
                    sim = self._cosine_similarity(target_emb, option_emb)
                    similarities.append((option, sim))

            if not similarities:
                raise ValueError("Could not compute embeddings for any options")

            # Sort by similarity
            similarities.sort(key=lambda x: x[1], reverse=True)
            best_match, similarity = similarities[0]

            # Add evidence for top matches
            evidence_list = []
            for option, sim in similarities[:3]:  # Top 3 matches
                evidence = NeuralEvidence(
                    source_type=EvidenceSourceType.ANALOGY,
                    content=f"Analogy match: '{option}' with similarity {sim:.3f}",
                    confidence=sim,
                    relevance=1.0,
                    metadata={
                        "a": a,
                        "b": b,
                        "c": c,
                        "d": option,
                        "similarity": sim
                    }
                )
                evidence_list.append(evidence)

            # Populate result
            result.conclusion = best_match
            result.confidence = similarity
            result.evidence = evidence_list
            result.embedding = target_emb

            # Build explanation
            result.explanation = f"Analogy '{a}' is to '{b}' as '{c}' is to '{best_match}' with confidence {similarity:.3f}"

        else:
            # If no options provided, find similar entities to target embedding
            similar_entities = self._find_similar_entities_to_embedding(target_emb, 5)

            if not similar_entities:
                result.conclusion = None
                result.confidence = 0.0
                result.evidence = []
                result.embedding = target_emb
                result.explanation = f"Could not find any entities similar to the target of analogy '{a}' is to '{b}' as '{c}' is to ?"
                return

            # Add evidence for similar entities
            evidence_list = []
            for entity, sim in similar_entities:
                evidence = NeuralEvidence(
                    source_type=EvidenceSourceType.ANALOGY,
                    content=f"Analogy completion: '{entity}' with similarity {sim:.3f}",
                    confidence=sim,
                    relevance=1.0,
                    metadata={
                        "a": a,
                        "b": b,
                        "c": c,
                        "d": entity,
                        "similarity": sim
                    }
                )
                evidence_list.append(evidence)

            # Best match is the highest similarity
            best_match, similarity = similar_entities[0]

            # Populate result
            result.conclusion = best_match
            result.confidence = similarity
            result.evidence = evidence_list
            result.embedding = target_emb

            # Build explanation
            result.explanation = f"Analogy '{a}' is to '{b}' as '{c}' is to '{best_match}' with confidence {similarity:.3f}"

    def _find_similar_entities(self, entity: Any, entity_emb: np.ndarray,
                              top_k: int = 5) -> List[Tuple[Any, float]]:
        """
        Find entities similar to the given entity.

        Args:
            entity: Entity to find similar entities for
            entity_emb: Embedding of the entity
            top_k: Number of top similar entities to return

        Returns:
            List of (entity, similarity) tuples
        """
        similar_entities = []

        # Use semantic memory if available
        if self.semantic_memory is not None:
            try:
                query = {
                    "semantic_query": entity if isinstance(entity, str) else None,
                    "embedding": entity_emb,
                    "max_results": top_k * 2  # Get more and then filter
                }

                similar_concepts = self.semantic_memory.query_concepts(query)

                for concept in similar_concepts:
                    # Skip if it's the same entity
                    if hasattr(concept, 'concept_id') and str(concept.concept_id) == str(entity):
                        continue

                    # Calculate similarity
                    if hasattr(concept, 'embedding') and concept.embedding is not None:
                        similarity = self._cosine_similarity(entity_emb, concept.embedding)
                    else:
                        similarity = 0.5  # Default

                    # Only consider sufficiently similar entities
                    if similarity >= self.similarity_threshold:
                        entity_id = concept.concept_id if hasattr(concept, 'concept_id') else str(concept)
                        similar_entities.append((entity_id, similarity))
            except Exception as e:
                self.logger.debug(f"Error finding similar entities in semantic memory: {str(e)}")

        # Use graph engine if available and needed
        if self.graph_engine is not None and len(similar_entities) < top_k:
            try:
                # Find similar nodes in the graph
                # This is a simplified approach - in a real implementation,
                # would use proper embedding-based similarity search

                # Assume entity is an ID or can be converted to string
                entity_str = str(entity)

                # Get properties of the entity if it's a node
                try:
                    entity_node = self.graph_engine.get_node(entity_str)
                    node_criteria = {}

                    if hasattr(entity_node, 'label'):
                        node_criteria["label"] = entity_node.label

                    # Find nodes with matching criteria
                    node_ids = self.graph_engine.query_nodes(node_criteria)

                    for node_id in node_ids:
                        # Skip if it's the same entity
                        if node_id == entity_str:
                            continue

                        try:
                            node = self.graph_engine.get_node(node_id)

                            # Calculate similarity if node has embedding
                            if hasattr(node, 'embedding') and node.embedding is not None:
                                similarity = self._cosine_similarity(entity_emb, node.embedding)

                                # Only consider sufficiently similar entities
                                if similarity >= self.similarity_threshold:
                                    similar_entities.append((node_id, similarity))
                        except Exception as e:
                            self.logger.debug(f"Error processing node {node_id}: {str(e)}")

                except Exception as e:
                    self.logger.debug(f"Entity not found in graph or error: {str(e)}")
            except Exception as e:
                self.logger.debug(f"Error finding similar entities in graph: {str(e)}")

        # Sort by similarity and return top_k
        similar_entities.sort(key=lambda x: x[1], reverse=True)
        return similar_entities[:top_k]

    def _find_similar_entities_to_embedding(self, embedding: np.ndarray,
                                          top_k: int = 5) -> List[Tuple[Any, float]]:
        """
        Find entities similar to the given embedding.

        Args:
            embedding: Embedding vector to find similar entities for
            top_k: Number of top similar entities to return

        Returns:
            List of (entity, similarity) tuples
        """
        similar_entities = []

        # Use semantic memory if available
        if self.semantic_memory is not None:
            try:
                query = {
                    "embedding": embedding,
                    "max_results": top_k
                }

                similar_concepts = self.semantic_memory.query_concepts(query)

                for concept in similar_concepts:
                    # Calculate similarity
                    if hasattr(concept, 'embedding') and concept.embedding is not None:
                        similarity = self._cosine_similarity(embedding, concept.embedding)
                    else:
                        similarity = 0.5  # Default

                    # Only consider sufficiently similar entities
                    if similarity >= self.similarity_threshold:
                        entity_id = concept.name if hasattr(concept, 'name') else str(concept)
                        similar_entities.append((entity_id, similarity))
            except Exception as e:
                self.logger.debug(f"Error finding similar entities in semantic memory: {str(e)}")

        # Sort by similarity and return top_k
        similar_entities.sort(key=lambda x: x[1], reverse=True)
        return similar_entities[:top_k]

# End of NeuralReasoner class