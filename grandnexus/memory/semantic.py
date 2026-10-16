"""
semantic.py - Enhanced Semantic Memory for GrandNexus

This module combines the best features of the original implementation with
approaches inspired by BioNeMo for biological ontologies, providing:

1. Multi-level abstraction for concepts
2. Dynamic weights for different relation types
3. Improved ontological reasoning with transitive inference
4. Activation spreading for related concepts
5. Context-aware retrieval with relevance scoring
"""

import logging
import threading
import uuid
import math
import numpy as np
from dataclasses import dataclass, field
from typing import (
    Any, Dict, List, Optional, Set, Tuple, Callable, Union
)
from collections import defaultdict, deque
from enum import Enum, auto
# Add: check if it's coherent
from grandnexus.graph.graph_engine import AbstractGraphEngine, create_graph_engine, GraphQuery, GraphNode


class AbstractionLevel(Enum):
    """Represents different levels of abstraction for concepts."""
    CONCRETE = 1      # Very specific instances or entities
    SPECIFIC = 2      # Specific types/categories
    GENERAL = 3       # General categories
    ABSTRACT = 4      # Abstract concepts
    UNIVERSAL = 5     # Universal principles or concepts


class RelationType(str, Enum):
    """Standard relation types with predefined semantics."""
    IS_A = "is_a"                 # Inheritance/subsumption (e.g., "dog is_a animal")
    PART_OF = "part_of"           # Composition (e.g., "wheel part_of car")
    HAS_PROPERTY = "has_property" # Attributes (e.g., "apple has_property red")
    RELATED_TO = "related_to"     # Generic association
    CAUSES = "causes"             # Causal relationship
    PRECEDED_BY = "preceded_by"   # Temporal precedence
    FOLLOWED_BY = "followed_by"   # Temporal succession
    SIMILAR_TO = "similar_to"     # Similarity or analogy
    OPPOSITE_OF = "opposite_of"   # Opposition or antonymy
    INSTANCE_OF = "instance_of"   # Instance relationship (e.g., "Fido instance_of dog")
    DEFINED_AS = "defined_as"     # Definition relationship


@dataclass
class Concept:
    """
    Represents a concept in the semantic memory with enhanced BioNeMo-inspired features.

    Attributes:
        concept_id (str): Unique identifier.
        name (str): Human-readable name (e.g., "Dog").
        description (str): Optional textual description.
        properties (Dict[str, Any]): Arbitrary attributes or key-value pairs.
        embedding (Optional[np.ndarray]): Vector representation for similarity calculations.
        synonyms (Set[str]): Alternate names or labels for the concept.
        importance (float): A measure of how relevant or "important" this concept is (0..1).
        abstraction_level (AbstractionLevel): The level of abstraction of this concept.
        activation (float): Current activation level (for spreading activation).
        last_accessed (float): Timestamp of last access (for recency effects).
        access_count (int): Number of times this concept has been accessed.
        confidence (float): Confidence in the concept's validity (0..1).
        source (str): Source of the concept (e.g., "user", "llm", "reasoning").
    """
    concept_id: str
    name: str
    description: str = ""
    properties: Dict[str, Any] = field(default_factory=dict)
    embedding: Optional[np.ndarray] = None
    synonyms: Set[str] = field(default_factory=set)
    importance: float = 0.5
    abstraction_level: AbstractionLevel = AbstractionLevel.SPECIFIC
    activation: float = 0.0
    last_accessed: float = field(default_factory=lambda: 0.0)
    access_count: int = 0
    confidence: float = 1.0
    source: str = "user"

    def __repr__(self):
        return (f"<Concept id={self.concept_id}, "
                f"name={self.name}, level={self.abstraction_level.name}, "
                f"importance={self.importance:.2f}, activation={self.activation:.2f}>")


@dataclass
class Relation:
    """
    Represents a relationship between two concepts with enhanced features.

    Attributes:
        relation_id (str): Unique identifier for this relation.
        source_concept_id (str): ID of the source concept.
        relation_type (str): Type/category of the relationship.
        target_concept_id (str): ID of the target concept.
        weight (float): Strength/confidence of the relationship (0..1).
        properties (Dict[str, Any]): Additional info about the relation.
        bidirectional (bool): Whether the relation works in both directions.
        transitive (bool): Whether the relation is transitive.
        temporal (bool): Whether the relation has temporal significance.
        created_at (float): Timestamp when relation was created.
        confidence (float): Confidence in this relation's validity (0..1).
        source (str): Source of this relation (e.g., "user", "inference").
    """
    relation_id: str
    source_concept_id: str
    relation_type: Union[RelationType, str]
    target_concept_id: str
    weight: float = 1.0
    properties: Dict[str, Any] = field(default_factory=dict)
    bidirectional: bool = False
    transitive: bool = False
    temporal: bool = False
    created_at: float = field(default_factory=lambda: 0.0)
    confidence: float = 1.0
    source: str = "user"

    def __post_init__(self):
        """Automatically set typical properties for standard relation types."""
        if isinstance(self.relation_type, str) and not isinstance(self.relation_type, RelationType):
            # Try to convert string to RelationType if possible
            try:
                self.relation_type = RelationType(self.relation_type)
            except ValueError:
                # Keep as string if not a standard type
                pass

        # Set defaults for standard relation types
        if isinstance(self.relation_type, RelationType):
            if self.relation_type == RelationType.IS_A:
                self.transitive = True
            elif self.relation_type == RelationType.PART_OF:
                self.transitive = True
            elif self.relation_type == RelationType.SIMILAR_TO:
                self.bidirectional = True
            elif self.relation_type == RelationType.OPPOSITE_OF:
                self.bidirectional = True
            elif self.relation_type == RelationType.PRECEDED_BY or self.relation_type == RelationType.FOLLOWED_BY:
                self.temporal = True

    def __repr__(self):
        return (f"<Relation id={self.relation_id}, "
                f"{self.source_concept_id} -{self.relation_type}-> "
                f"{self.target_concept_id}, weight={self.weight:.2f}, confidence={self.confidence:.2f}>")


@dataclass
class SemanticQuery:
    """
    Enhanced query container for specifying parameters when searching for concepts.

    Attributes:
        text (Optional[str]): A text string to match against a concept's name or synonyms.
        embedding (Optional[np.ndarray]): If set, we do an embedding-based similarity search.
        similarity_metric (str): 'cosine' or 'euclidean'.
        importance_threshold (float): Minimum concept importance required.
        max_results (int): Limit on returned concepts.
        relationship_filters (Optional[List[Tuple[str, str]]]):
            A list of (relation_type, concept_id). The concept must have a relationship of
            that type to the specified concept_id.
        synonyms_match (bool): If True, we also match `text` against synonyms.
        abstraction_levels (Optional[List[AbstractionLevel]]): Filter by abstraction levels.
        min_confidence (float): Minimum confidence level for retrieved concepts.
        context_boost (Dict[str, float]): Boost certain concept types or properties.
        recency_bias (float): Weight to give to recently accessed concepts (0-1).
        activation_threshold (float): Only return concepts with activation above this.
        include_inferred (bool): Whether to include inferred relationships.
        sort_by (str): How to sort results ('relevance', 'importance', 'confidence', etc.)
    """
    text: Optional[str] = None
    embedding: Optional[np.ndarray] = None
    similarity_metric: str = "cosine"
    importance_threshold: float = 0.0
    max_results: int = 20
    relationship_filters: Optional[List[Tuple[Union[RelationType, str], str]]] = None
    synonyms_match: bool = True
    abstraction_levels: Optional[List[AbstractionLevel]] = None
    min_confidence: float = 0.0
    context_boost: Dict[str, float] = field(default_factory=dict)
    recency_bias: float = 0.2
    activation_threshold: float = 0.0
    include_inferred: bool = True
    sort_by: str = "relevance"


class SemanticMemoryConfig:
    """
    Enhanced configuration for SemanticMemory.

    Attributes:
        embedding_function (Optional[Callable[[str], np.ndarray]]):
            A function that takes a string (name + synonyms) and returns an np.ndarray embedding.
        auto_link_graph (bool): If True, we call _sync_to_graph_subsystem whenever a concept changes.
        recency_decay (float): For advanced scoring if we track recency.
        max_concepts (int): If > 0, we limit the number of concepts and prune if exceeded.
        prune_policy (str): E.g. 'lowest_importance', 'oldest', or 'lowest_score'.
        activation_decay (float): Rate at which activation decays over time.
        activation_spread_factor (float): How much activation spreads to neighbors.
        relation_confidence_threshold (float): Minimum confidence for relations to be used in inference.
        inference_depth_limit (int): Maximum depth for transitive inference.
        enable_spreading_activation (bool): Whether to use spreading activation for search.
        max_activation_spreading_steps (int): Maximum steps for spreading activation.
        ontology_validation (bool): Validate ontological consistency when adding relations.
        default_concept_confidence (float): Default confidence for new concepts if not specified.
        context_window_size (int): Number of recent concepts to maintain in context window.
        similarity_threshold (float): Threshold for considering two embeddings similar.
        use_memory_consolidation (bool): Whether to periodically consolidate memory.
        consolidation_interval (int): How many operations before consolidation.
    """
    def __init__(self,
                 embedding_function: Optional[Callable[[str], np.ndarray]] = None,
                 auto_link_graph: bool = False,
                 recency_decay: float = 0.01,
                 max_concepts: int = 0,
                 prune_policy: str = "lowest_importance",
                 activation_decay: float = 0.05,
                 activation_spread_factor: float = 0.5,
                 relation_confidence_threshold: float = 0.5,
                 inference_depth_limit: int = 5,
                 enable_spreading_activation: bool = True,
                 max_activation_spreading_steps: int = 3,
                 ontology_validation: bool = True,
                 default_concept_confidence: float = 0.8,
                 context_window_size: int = 50,
                 similarity_threshold: float = 0.75,
                 use_memory_consolidation: bool = True,
                 consolidation_interval: int = 100):
        self.embedding_function = embedding_function
        self.auto_link_graph = auto_link_graph
        self.recency_decay = recency_decay
        self.max_concepts = max_concepts
        self.prune_policy = prune_policy
        self.activation_decay = activation_decay
        self.activation_spread_factor = activation_spread_factor
        self.relation_confidence_threshold = relation_confidence_threshold
        self.inference_depth_limit = inference_depth_limit
        self.enable_spreading_activation = enable_spreading_activation
        self.max_activation_spreading_steps = max_activation_spreading_steps
        self.ontology_validation = ontology_validation
        self.default_concept_confidence = default_concept_confidence
        self.context_window_size = context_window_size
        self.similarity_threshold = similarity_threshold
        self.use_memory_consolidation = use_memory_consolidation
        self.consolidation_interval = consolidation_interval


class SemanticMemory:
    """
    Enhanced semantic memory for GrandNexus.
    Stores concepts, each with synonyms, properties, abstraction levels, and importance.
    Also stores relations with weights, confidence, and specialized attributes.

    Main capabilities:
     - Add / get / update / remove concepts with abstraction levels
     - Add / get / update / remove relations with dynamic weights
     - Query concepts using text, embedding, or relationship filters
     - Advanced inference with transitive relationships
     - Spreading activation for context-aware retrieval
     - Ontological validation and consistency checking
    """

    def __init__(self, config: SemanticMemoryConfig, graph_engine: Optional[AbstractGraphEngine] = None):
        """Initialize the semantic memory with the given configuration."""
        self.logger = logging.getLogger("SemanticMemory")
        self.config = config
        self.lock = threading.RLock()

        # Graph engine integration - NEW
        self.graph_engine = graph_engine
        if self.graph_engine is None and config.auto_link_graph:
            self.logger.info("Creating default in-memory graph engine for semantic memory")
            self.graph_engine = create_graph_engine("inmemory")

        # Primary storage
        self.concepts: Dict[str, Concept] = {}
        self.relations: Dict[str, Relation] = {}

        # Enhanced indexing structures beyond the original implementation
        self.relations_by_source: Dict[str, List[Relation]] = defaultdict(list)
        self.relations_by_target: Dict[str, List[Relation]] = defaultdict(list)
        self.relations_by_type: Dict[str, List[Relation]] = defaultdict(list)

        # BioNeMo-inspired additions
        self.concepts_by_abstraction: Dict[AbstractionLevel, List[str]] = {
            level: [] for level in AbstractionLevel
        }
        self.context_window: deque[str] = deque(maxlen=config.context_window_size)
        self.concept_activations: Dict[str, float] = defaultdict(float)
        self.operation_counter: int = 0

        self.logger.info(f"SemanticMemory initialized (auto_link_graph={config.auto_link_graph}, "
                        f"ontology_validation={config.ontology_validation}, "
                        f"graph_engine={'enabled' if self.graph_engine else 'disabled'})")

    def _sync_concept_to_graph(self, concept: Concept) -> None:
        """Synchronize a concept to the graph engine if enabled."""
        if not self.graph_engine or not self.config.auto_link_graph:
            return

        try:
            # Add or update node in graph
            properties = {
                "type": "concept",
                "name": concept.name,
                "description": concept.description,
                "abstraction_level": concept.abstraction_level.value,
                "importance": concept.importance,
                "confidence": concept.confidence,
                "source": concept.source,
                **concept.properties
            }

            # Add synonyms as a property
            if concept.synonyms:
                properties["synonyms"] = list(concept.synonyms)

            # Check if node exists
            existing_nodes = self.graph_engine.query_nodes({"type": "concept", "name": concept.name})

            if existing_nodes:
                # Update existing node
                self.graph_engine.update_node(concept.concept_id, properties)
            else:
                # Add new node
                self.graph_engine.add_node(
                    node_id=concept.concept_id,
                    label="Concept",
                    properties=properties
                )

            # Store embedding if available
            if concept.embedding is not None and hasattr(self.graph_engine.get_node(concept.concept_id), 'embedding'):
                node = self.graph_engine.get_node(concept.concept_id)
                node.embedding = concept.embedding

        except Exception as e:
            self.logger.warning(f"Failed to sync concept {concept.concept_id} to graph: {e}")

    def _sync_relation_to_graph(self, relation: Relation) -> None:
        """Synchronize a relation to the graph engine if enabled."""
        if not self.graph_engine or not self.config.auto_link_graph:
            return

        try:
            # Add edge in graph
            properties = {
                "confidence": relation.confidence,
                "source": relation.source,
                "temporal": relation.temporal,
                **relation.properties
            }

            self.graph_engine.add_edge(
                source_id=relation.source_concept_id,
                target_id=relation.target_concept_id,
                edge_type=str(relation.relation_type),
                edge_id=relation.relation_id,
                properties=properties,
                weight=relation.weight,
                bidirectional=relation.bidirectional
            )

        except Exception as e:
            self.logger.warning(f"Failed to sync relation {relation.relation_id} to graph: {e}")

    def _navigate_knowledge_graph(self, start_concept_id: str, query: str, max_depth: int = 3) -> List[str]:
        """
        Navigate the knowledge graph to find related concepts based on a query.

        Args:
            start_concept_id: Starting concept ID
            query: Query string to guide navigation
            max_depth: Maximum depth to traverse

        Returns:
            List of relevant concept IDs found
        """
        if not self.graph_engine:
            return []

        try:
            # Use graph engine to execute query
            graph_query = GraphQuery(
                start_nodes=[start_concept_id],
                node_criteria={"type": "concept"},
                max_depth=max_depth,
                min_activation=0.1
            )

            results = self.graph_engine.execute_query(graph_query)

            # Extract concept IDs from results
            concept_ids = list(results["nodes"].keys())

            # If we have an embedding function, we can rank by similarity to query
            if self.config.embedding_function and query:
                query_embedding = self.config.embedding_function(query)

                # Rank concepts by embedding similarity
                ranked_concepts = []
                for concept_id in concept_ids:
                    if concept_id in self.concepts and self.concepts[concept_id].embedding is not None:
                        similarity = self._similarity(
                            query_embedding,
                            self.concepts[concept_id].embedding,
                            self.config.similarity_metric
                        )
                        ranked_concepts.append((concept_id, similarity))

                # Sort by similarity and return IDs
                ranked_concepts.sort(key=lambda x: x[1], reverse=True)
                return [cid for cid, _ in ranked_concepts]

            return concept_ids

        except Exception as e:
            self.logger.warning(f"Failed to navigate knowledge graph: {e}")
            return []

    # --------------------------------------------------------------------------
    # CONCEPT METHODS
    # --------------------------------------------------------------------------

    def add_concept(self,
                   concept: Union[Concept, Dict[str, Any]]) -> str:
        """
        Insert a new concept. If config.embedding_function is set, compute an embedding from
        concept.name + synonyms. Supports both Concept objects and dictionaries.

        Args:
            concept: Either a Concept object or a dictionary with concept attributes

        Returns:
            str: The concept_id of the created/updated concept
        """
        with self.lock:
            # Convert dictionary to Concept if needed
            if isinstance(concept, dict):
                concept_id = concept.get("concept_id") or str(uuid.uuid4())

                # Handle abstraction_level if provided as string
                abstraction_level = concept.get("abstraction_level", AbstractionLevel.SPECIFIC)
                if isinstance(abstraction_level, str):
                    try:
                        abstraction_level = AbstractionLevel[abstraction_level.upper()]
                    except KeyError:
                        abstraction_level = AbstractionLevel.SPECIFIC

                concept_obj = Concept(
                    concept_id=concept_id,
                    name=concept.get("name", ""),
                    description=concept.get("description", ""),
                    properties=concept.get("properties", {}),
                    embedding=concept.get("embedding"),
                    synonyms=set(concept.get("synonyms", [])),
                    importance=concept.get("importance", 0.5),
                    abstraction_level=abstraction_level,
                    confidence=concept.get("confidence", self.config.default_concept_confidence),
                    source=concept.get("source", "user")
                )
                concept = concept_obj
            else:
                # Ensure concept_id exists
                if not concept.concept_id:
                    concept.concept_id = str(uuid.uuid4())

            # Check if concept already exists
            if concept.concept_id in self.concepts:
                self.logger.warning(f"Concept {concept.concept_id} already exists. Use update_concept to modify.")
                return concept.concept_id

            # Generate embedding if needed
            if self.config.embedding_function and not concept.embedding:
                text_for_embedding = concept.name
                if concept.synonyms:
                    text_for_embedding += " " + " ".join(concept.synonyms)
                if concept.description:
                    text_for_embedding += " " + concept.description

                try:
                    concept.embedding = self.config.embedding_function(text_for_embedding)
                except Exception as e:
                    self.logger.error(f"Failed embedding for concept {concept.concept_id}: {e}", exc_info=True)

            # Set timestamps and initialize activation
            import time
            current_time = time.time()
            concept.last_accessed = current_time
            concept.activation = 1.0  # New concepts start highly activated

            # Store the concept
            self.concepts[concept.concept_id] = concept

            # Update indices
            self.concepts_by_abstraction[concept.abstraction_level].append(concept.concept_id)
            self.context_window.append(concept.concept_id)
            self.concept_activations[concept.concept_id] = 1.0

            # Track operation count for potential consolidation
            self.operation_counter += 1
            if self.config.use_memory_consolidation and self.operation_counter >= self.config.consolidation_interval:
                self._consolidate_memory()
                self.operation_counter = 0

            # Sync to graph if enabled
            if self.config.auto_link_graph:
                self._sync_to_graph_subsystem(concept)

            self.logger.debug(f"Added concept: {concept}")
            # Sync to graph engine - NEW
            self._sync_concept_to_graph(concept)


        return concept.concept_id

    def get_concept(self, concept_id: str, update_activation: bool = True) -> Optional[Concept]:
        """
        Retrieve a concept by ID, optionally updating its activation and access metadata.

        Args:
            concept_id: ID of the concept to retrieve
            update_activation: Whether to update activation and access timestamp/count

        Returns:
            The Concept object or None if not found
        """
        with self.lock:
            concept = self.concepts.get(concept_id)
            if not concept:
                return None

            if update_activation:
                import time
                current_time = time.time()
                concept.last_accessed = current_time
                concept.access_count += 1
                concept.activation = 1.0  # Full activation when directly accessed
                self.concept_activations[concept_id] = 1.0

                # Update context window
                if concept_id in self.context_window:
                    # Move to most recent position if already in window
                    self.context_window.remove(concept_id)
                self.context_window.append(concept_id)

            return concept

    def update_concept(self,
                      concept: Union[Concept, Dict[str, Any]],
                      partial_update: bool = False) -> bool:
        """
        Update an existing concept. If synonyms changed, we might recalc embedding.

        Args:
            concept: The concept object or dictionary with updated values
            partial_update: If True, only update the provided fields

        Returns:
            bool: True if the update was successful
        """
        with self.lock:
            # Convert dictionary to concept if needed
            concept_id = None
            if isinstance(concept, dict):
                concept_id = concept.get("concept_id")
                if not concept_id:
                    self.logger.error("Cannot update concept: concept_id not provided in dictionary")
                    return False
            else:
                concept_id = concept.concept_id

            # Check if concept exists
            if concept_id not in self.concepts:
                self.logger.warning(f"Cannot update concept {concept_id}, it doesn't exist. Use add_concept.")
                return False

            existing_concept = self.concepts[concept_id]

            # Update the concept
            if isinstance(concept, dict):
                # Partial update with dictionary
                if "name" in concept:
                    existing_concept.name = concept["name"]
                if "description" in concept:
                    existing_concept.description = concept["description"]
                if "properties" in concept:
                    if partial_update:
                        existing_concept.properties.update(concept["properties"])
                    else:
                        existing_concept.properties = concept["properties"]
                if "synonyms" in concept:
                    existing_concept.synonyms = set(concept["synonyms"])
                if "importance" in concept:
                    existing_concept.importance = concept["importance"]
                if "abstraction_level" in concept:
                    abstraction_level = concept["abstraction_level"]
                    if isinstance(abstraction_level, str):
                        try:
                            abstraction_level = AbstractionLevel[abstraction_level.upper()]
                        except KeyError:
                            abstraction_level = existing_concept.abstraction_level

                    # Update abstraction index if changed
                    if abstraction_level != existing_concept.abstraction_level:
                        self.concepts_by_abstraction[existing_concept.abstraction_level].remove(concept_id)
                        existing_concept.abstraction_level = abstraction_level
                        self.concepts_by_abstraction[abstraction_level].append(concept_id)
                if "confidence" in concept:
                    existing_concept.confidence = concept["confidence"]
                if "source" in concept:
                    existing_concept.source = concept["source"]
                if "embedding" in concept:
                    existing_concept.embedding = concept["embedding"]
            else:
                # Replace with new concept but keep activation data
                old_activation = existing_concept.activation
                old_access_count = existing_concept.access_count
                old_last_accessed = existing_concept.last_accessed

                # Update abstraction index if changed
                if concept.abstraction_level != existing_concept.abstraction_level:
                    self.concepts_by_abstraction[existing_concept.abstraction_level].remove(concept_id)
                    self.concepts_by_abstraction[concept.abstraction_level].append(concept_id)

                # Replace the concept
                self.concepts[concept_id] = concept

                # Preserve activation data
                concept.activation = old_activation
                concept.access_count = old_access_count
                concept.last_accessed = old_last_accessed

            # Recalculate embedding if necessary
            need_embedding_update = False
            if isinstance(concept, dict):
                need_embedding_update = "name" in concept or "synonyms" in concept or "description" in concept
            else:
                need_embedding_update = True

            if need_embedding_update and self.config.embedding_function:
                try:
                    updated_concept = self.concepts[concept_id]
                    text_for_embedding = updated_concept.name
                    if updated_concept.synonyms:
                        text_for_embedding += " " + " ".join(updated_concept.synonyms)
                    if updated_concept.description:
                        text_for_embedding += " " + updated_concept.description

                    updated_concept.embedding = self.config.embedding_function(text_for_embedding)
                except Exception as e:
                    self.logger.error(f"Failed to update embedding for concept {concept_id}: {e}", exc_info=True)

            # Update context window
            if concept_id in self.context_window:
                self.context_window.remove(concept_id)
            self.context_window.append(concept_id)

            # Track operation
            self.operation_counter += 1

            # Sync to graph if enabled
            if self.config.auto_link_graph:
                self._sync_to_graph_subsystem(self.concepts[concept_id])

            self.logger.debug(f"Updated concept: {self.concepts[concept_id]}")
            return True

    def remove_concept(self, concept_id: str) -> bool:
        """
        Remove a concept and all its relations from memory.

        Args:
            concept_id: ID of the concept to remove

        Returns:
            bool: True if removed, False if not found
        """
        with self.lock:
            concept = self.concepts.pop(concept_id, None)
            if not concept:
                self.logger.warning(f"Cannot remove concept {concept_id}: doesn't exist.")
                return False

            # Remove from abstraction index
            if concept_id in self.concepts_by_abstraction[concept.abstraction_level]:
                self.concepts_by_abstraction[concept.abstraction_level].remove(concept_id)

            # Remove from context window and activations
            if concept_id in self.context_window:
                self.context_window.remove(concept_id)
            if concept_id in self.concept_activations:
                del self.concept_activations[concept_id]

            # Remove all relations involving this concept
            relations_to_remove = []
            for rel_id, relation in self.relations.items():
                if relation.source_concept_id == concept_id or relation.target_concept_id == concept_id:
                    relations_to_remove.append(rel_id)

            for rel_id in relations_to_remove:
                self.remove_relation(rel_id)

            self.logger.debug(f"Removed concept {concept_id} and {len(relations_to_remove)} related relations")
            return True

    # --------------------------------------------------------------------------
    # RELATION METHODS
    # --------------------------------------------------------------------------

    def add_relation(self,
                    relation: Union[Relation, Dict[str, Any]]) -> str:
        """
        Create a new relationship between two existing concepts.

        Args:
            relation: A Relation object or dictionary with relation data

        Returns:
            str: The relation_id of the created relation
        """
        with self.lock:
            # Convert dictionary to Relation if needed
            if isinstance(relation, dict):
                relation_id = relation.get("relation_id") or str(uuid.uuid4())
                source_id = relation.get("source_concept_id")
                target_id = relation.get("target_concept_id")
                relation_type = relation.get("relation_type")

                if not source_id or not target_id or not relation_type:
                    self.logger.error("Cannot add relation: missing required fields (source_id, target_id, relation_type)")
                    return ""

                relation_obj = Relation(
                    relation_id=relation_id,
                    source_concept_id=source_id,
                    relation_type=relation_type,
                    target_concept_id=target_id,
                    weight=relation.get("weight", 1.0),
                    properties=relation.get("properties", {}),
                    bidirectional=relation.get("bidirectional", False),
                    transitive=relation.get("transitive", False),
                    temporal=relation.get("temporal", False),
                    confidence=relation.get("confidence", self.config.default_concept_confidence),
                    source=relation.get("source", "user")
                )
                relation = relation_obj
            else:
                # Ensure relation_id exists
                if not relation.relation_id:
                    relation.relation_id = str(uuid.uuid4())

            # Check if relation already exists
            if relation.relation_id in self.relations:
                self.logger.warning(f"Relation {relation.relation_id} already exists. Use update_relation to modify.")
                return relation.relation_id

            # Verify concepts exist
            if relation.source_concept_id not in self.concepts:
                self.logger.warning(f"Source concept {relation.source_concept_id} doesn't exist.")
                return ""
            if relation.target_concept_id not in self.concepts:
                self.logger.warning(f"Target concept {relation.target_concept_id} doesn't exist.")
                return ""

            # Set created timestamp
            import time
            relation.created_at = time.time()

            # Validate ontology if enabled
            if self.config.ontology_validation:
                validation_result = self._validate_relation_ontology(relation)
                if not validation_result["valid"]:
                    self.logger.warning(f"Ontology validation failed for relation: {validation_result['reason']}")
                    # We might still add it but with reduced confidence
                    relation.confidence *= 0.5

            # Store the relation
            self.relations[relation.relation_id] = relation

            # Update indices
            self.relations_by_source[relation.source_concept_id].append(relation)
            self.relations_by_target[relation.target_concept_id].append(relation)
            self.relations_by_type[str(relation.relation_type)].append(relation)

            # Handle bidirectional relations by creating the inverse
            if relation.bidirectional and relation.source_concept_id != relation.target_concept_id:
                inverse_id = f"inverse_{relation.relation_id}"
                inverse_relation = Relation(
                    relation_id=inverse_id,
                    source_concept_id=relation.target_concept_id,
                    relation_type=relation.relation_type,
                    target_concept_id=relation.source_concept_id,
                    weight=relation.weight,
                    properties=dict(relation.properties),
                    bidirectional=True,
                    transitive=relation.transitive,
                    temporal=relation.temporal,
                    created_at=relation.created_at,
                    confidence=relation.confidence,
                    source=f"inverse_of_{relation.source}"
                )

                self.relations[inverse_id] = inverse_relation
                self.relations_by_source[inverse_relation.source_concept_id].append(inverse_relation)
                self.relations_by_target[inverse_relation.target_concept_id].append(inverse_relation)
                self.relations_by_type[str(inverse_relation.relation_type)].append(inverse_relation)

            # Update context for both concepts
            self.context_window.append(relation.source_concept_id)
            self.context_window.append(relation.target_concept_id)

            # Spread activation between related concepts
            source_concept = self.concepts[relation.source_concept_id]
            target_concept = self.concepts[relation.target_concept_id]

            # Source activates target
            activation_spread = source_concept.activation * relation.weight * self.config.activation_spread_factor
            target_concept.activation = min(1.0, target_concept.activation + activation_spread)
            self.concept_activations[relation.target_concept_id] = target_concept.activation

            # Target activates source (weaker)
            reverse_spread = target_concept.activation * relation.weight * self.config.activation_spread_factor * 0.5
            source_concept.activation = min(1.0, source_concept.activation + reverse_spread)
            self.concept_activations[relation.source_concept_id] = source_concept.activation

            # Track operation
            self.operation_counter += 1

            self.logger.debug(f"Added relation: {relation}")

            self._sync_relation_to_graph(relation)
            return relation.relation_id

    def get_relation(self, relation_id: str) -> Optional[Relation]:
        """
        Retrieve a relation by its ID.

        Args:
            relation_id: ID of the relation to retrieve

        Returns:
            The Relation object or None if not found
        """
        with self.lock:
            return self.relations.get(relation_id)

    def update_relation(self,
                       relation: Union[Relation, Dict[str, Any]],
                       partial_update: bool = False) -> bool:
        """
        Update an existing relation.

        Args:
            relation: The relation object or dictionary with updated values
            partial_update: If True, only update the provided fields

        Returns:
            bool: True if the update was successful
        """
        with self.lock:
            # Convert dictionary to relation if needed
            relation_id = None
            if isinstance(relation, dict):
                relation_id = relation.get("relation_id")
                if not relation_id:
                    self.logger.error("Cannot update relation: relation_id not provided in dictionary")
                    return False
            else:
                relation_id = relation.relation_id

            # Check if relation exists
            if relation_id not in self.relations:
                self.logger.warning(f"Cannot update relation {relation_id}, it doesn't exist. Use add_relation.")
                return False

            existing_relation = self.relations[relation_id]

            # Remember old indices for removal
            old_source_id = existing_relation.source_concept_id
            old_target_id = existing_relation.target_concept_id
            old_type = str(existing_relation.relation_type)

            # Update the relation
            if isinstance(relation, dict):
                # Partial update with dictionary
                changed_source = False
                changed_target = False
                changed_type = False

                if "source_concept_id" in relation and relation["source_concept_id"] != existing_relation.source_concept_id:
                    if relation["source_concept_id"] in self.concepts:
                        existing_relation.source_concept_id = relation["source_concept_id"]
                        changed_source = True
                    else:
                        self.logger.warning(f"Source concept {relation['source_concept_id']} doesn't exist.")
                        return False

                if "target_concept_id" in relation and relation["target_concept_id"] != existing_relation.target_concept_id:
                    if relation["target_concept_id"] in self.concepts:
                        existing_relation.target_concept_id = relation["target_concept_id"]
                        changed_target = True
                    else:
                        self.logger.warning(f"Target concept {relation['target_concept_id']} doesn't exist.")
                        return False

                if "relation_type" in relation and relation["relation_type"] != existing_relation.relation_type:
                    existing_relation.relation_type = relation["relation_type"]
                    changed_type = True

                if "weight" in relation:
                    existing_relation.weight = relation["weight"]
                if "properties" in relation:
                    if partial_update:
                        existing_relation.properties.update(relation["properties"])
                    else:
                        existing_relation.properties = relation["properties"]
                if "bidirectional" in relation:
                    existing_relation.bidirectional = relation["bidirectional"]
                if "transitive" in relation:
                    existing_relation.transitive = relation["transitive"]
                if "temporal" in relation:
                    existing_relation.temporal = relation["temporal"]
                if "confidence" in relation:
                    existing_relation.confidence = relation["confidence"]
                if "source" in relation:
                    existing_relation.source = relation["source"]

                # Update indices if necessary
                if changed_source or changed_target or changed_type:
                    # Remove from old indices
                    if old_source_id in self.relations_by_source and existing_relation in self.relations_by_source[old_source_id]:
                        self.relations_by_source[old_source_id].remove(existing_relation)
                    if old_target_id in self.relations_by_target and existing_relation in self.relations_by_target[old_target_id]:
                        self.relations_by_target[old_target_id].remove(existing_relation)
                    if old_type in self.relations_by_type and existing_relation in self.relations_by_type[old_type]:
                        self.relations_by_type[old_type].remove(existing_relation)

                    # Add to new indices
                    self.relations_by_source[existing_relation.source_concept_id].append(existing_relation)
                    self.relations_by_target[existing_relation.target_concept_id].append(existing_relation)
                    self.relations_by_type[str(existing_relation.relation_type)].append(existing_relation)
            else:
                # Full replacement
                old_relation = self.relations[relation_id]

                # Remove from old indices
                if old_source_id in self.relations_by_source and old_relation in self.relations_by_source[old_source_id]:
                    self.relations_by_source[old_source_id].remove(old_relation)
                if old_target_id in self.relations_by_target and old_relation in self.relations_by_target[old_target_id]:
                    self.relations_by_target[old_target_id].remove(old_relation)
                if old_type in self.relations_by_type and old_relation in self.relations_by_type[old_type]:
                    self.relations_by_type[old_type].remove(old_relation)

                # Verify concepts exist
                if relation.source_concept_id not in self.concepts:
                    self.logger.warning(f"Source concept {relation.source_concept_id} doesn't exist.")
                    return False
                if relation.target_concept_id not in self.concepts:
                    self.logger.warning(f"Target concept {relation.target_concept_id} doesn't exist.")
                    return False

                # Replace relation and update indices
                self.relations[relation_id] = relation
                self.relations_by_source[relation.source_concept_id].append(relation)
                self.relations_by_target[relation.target_concept_id].append(relation)
                self.relations_by_type[str(relation.relation_type)].append(relation)

            # Validate ontology if enabled
            if self.config.ontology_validation:
                validation_result = self._validate_relation_ontology(self.relations[relation_id])
                if not validation_result["valid"]:
                    self.logger.warning(f"Ontology validation failed for updated relation: {validation_result['reason']}")
                    # Reduce confidence
                    self.relations[relation_id].confidence *= 0.5

            # Update context window
            self.context_window.append(self.relations[relation_id].source_concept_id)
            self.context_window.append(self.relations[relation_id].target_concept_id)

            # Track operation
            self.operation_counter += 1

            self.logger.debug(f"Updated relation: {self.relations[relation_id]}")
            return True

    def remove_relation(self, relation_id: str) -> bool:
        """
        Remove a relation from the semantic memory.

        Args:
            relation_id: ID of the relation to remove

        Returns:
            bool: True if removed, False if not found
        """
        with self.lock:
            relation = self.relations.pop(relation_id, None)
            if not relation:
                self.logger.warning(f"Cannot remove relation {relation_id}: doesn't exist.")
                return False

            # Remove from indices
            if relation in self.relations_by_source[relation.source_concept_id]:
                self.relations_by_source[relation.source_concept_id].remove(relation)
            if relation in self.relations_by_target[relation.target_concept_id]:
                self.relations_by_target[relation.target_concept_id].remove(relation)

            rel_type = str(relation.relation_type)
            if relation in self.relations_by_type[rel_type]:
                self.relations_by_type[rel_type].remove(relation)

            # If it was bidirectional, try to remove the inverse relation too
            if relation.bidirectional:
                inverse_id = f"inverse_{relation_id}"
                if inverse_id in self.relations:
                    self.remove_relation(inverse_id)

            # For completeness, also check if this relation is an inverse of another
            if relation_id.startswith("inverse_"):
                original_id = relation_id[8:]  # Remove "inverse_" prefix
                if original_id in self.relations:
                    original = self.relations[original_id]
                    if original.bidirectional:
                        original.bidirectional = False  # No longer bidirectional

            self.logger.debug(f"Removed relation {relation_id}")
            return True

    def get_relations_by_source(self, concept_id: str) -> List[Relation]:
        """
        Get all relations where the specified concept is the source.

        Args:
            concept_id: ID of the source concept

        Returns:
            List of relations
        """
        with self.lock:
            return list(self.relations_by_source.get(concept_id, []))

    def get_relations_by_target(self, concept_id: str) -> List[Relation]:
        """
        Get all relations where the specified concept is the target.

        Args:
            concept_id: ID of the target concept

        Returns:
            List of relations
        """
        with self.lock:
            return list(self.relations_by_target.get(concept_id, []))

    def get_relations_by_type(self, relation_type: Union[RelationType, str]) -> List[Relation]:
        """
        Get all relations of the specified type.

        Args:
            relation_type: Type of relations to retrieve

        Returns:
            List of relations
        """
        with self.lock:
            return list(self.relations_by_type.get(str(relation_type), []))

    def query_relations(self,
                       source_concept_id: Optional[str] = None,
                       relation_type: Optional[Union[RelationType, str]] = None,
                       target_concept_id: Optional[str] = None,
                       min_weight: float = 0.0,
                       min_confidence: float = 0.0) -> List[Relation]:
        """
        Flexible query for relations based on source, type, target, and other criteria.

        Args:
            source_concept_id: Optional source concept ID filter
            relation_type: Optional relation type filter
            target_concept_id: Optional target concept ID filter
            min_weight: Minimum relation weight
            min_confidence: Minimum confidence level

        Returns:
            List of matching relations
        """
        with self.lock:
            results = []

            # Determine the most efficient way to search
            if source_concept_id and source_concept_id in self.relations_by_source:
                # Start with source-based index
                candidates = self.relations_by_source[source_concept_id]
            elif target_concept_id and target_concept_id in self.relations_by_target:
                # Start with target-based index
                candidates = self.relations_by_target[target_concept_id]
            elif relation_type and str(relation_type) in self.relations_by_type:
                # Start with type-based index
                candidates = self.relations_by_type[str(relation_type)]
            else:
                # No specific index, search all relations
                candidates = list(self.relations.values())

            # Apply all filters
            for relation in candidates:
                if (source_concept_id is None or relation.source_concept_id == source_concept_id) and \
                   (target_concept_id is None or relation.target_concept_id == target_concept_id) and \
                   (relation_type is None or str(relation.relation_type) == str(relation_type)) and \
                   relation.weight >= min_weight and \
                   relation.confidence >= min_confidence:
                    results.append(relation)

            return results

    # --------------------------------------------------------------------------
    # QUERY CONCEPTS
    # --------------------------------------------------------------------------

    def query_concepts(self, query: SemanticQuery) -> List[Concept]:
        """
        Enhanced concept retrieval using advanced criteria including activation,
        abstraction levels, and spreading activation.

        Args:
            query: SemanticQuery object with search parameters

        Returns:
            List of matching concepts
        """
        with self.lock:
            # Optional spreading activation
            if self.config.enable_spreading_activation and query.text:
                self._apply_spreading_activation(query.text)

            # Build candidate set
            candidates = set()

            # Apply text-based filtering if specified
            if query.text is not None:
                # Find concepts matching the text query
                for concept_id, concept in self.concepts.items():
                    if query.text.lower() in concept.name.lower():
                        candidates.add(concept_id)
                    elif query.synonyms_match and concept.synonyms:
                        if any(query.text.lower() in syn.lower() for syn in concept.synonyms):
                            candidates.add(concept_id)

                # If no direct matches found and embedding_function is available, try embedding similarity
                if not candidates and self.config.embedding_function and query.embedding is None:
                    try:
                        query.embedding = self.config.embedding_function(query.text)
                    except Exception as e:
                        self.logger.error(f"Error generating embedding for query text: {e}", exc_info=True)

            # Apply relationship filters if specified
            if query.relationship_filters:
                rel_candidates = set()
                first_filter = True

                for rel_type, related_concept_id in query.relationship_filters:
                    filter_matches = set()

                    # Find concepts with the specified relationship
                    if related_concept_id in self.concepts:
                        # Find concepts that have this relationship
                        relations = self.query_relations(
                            relation_type=rel_type,
                            target_concept_id=related_concept_id,
                            min_confidence=query.min_confidence
                        )

                        # Add sources to candidates
                        for rel in relations:
                            filter_matches.add(rel.source_concept_id)

                        # Include inferred relationships if requested
                        if query.include_inferred and isinstance(rel_type, RelationType) and rel_type == RelationType.IS_A:
                            # Get transitive is_a relationships
                            transitive_matches = self._find_transitive_sources(related_concept_id, rel_type)
                            filter_matches.update(transitive_matches)

                    # Intersect with previous filters' results
                    if first_filter:
                        rel_candidates = filter_matches
                        first_filter = False
                    else:
                        rel_candidates &= filter_matches

                # If we have text matches, intersect with relationship matches
                if query.text is not None:
                    if candidates:
                        candidates &= rel_candidates
                    else:
                        candidates = rel_candidates
                else:
                    candidates = rel_candidates

            # If neither text nor relationship filters specified, use all concepts
            if query.text is None and not query.relationship_filters:
                candidates = set(self.concepts.keys())

            # Apply abstraction level filtering if specified
            if query.abstraction_levels:
                valid_levels = set(query.abstraction_levels)
                candidates = {cid for cid in candidates
                             if self.concepts[cid].abstraction_level in valid_levels}

            # Apply activation threshold if specified
            if query.activation_threshold > 0:
                candidates = {cid for cid in candidates
                             if self.concepts[cid].activation >= query.activation_threshold}

            # Apply importance threshold filtering
            if query.importance_threshold > 0:
                candidates = {cid for cid in candidates
                             if self.concepts[cid].importance >= query.importance_threshold}

            # Apply confidence threshold filtering
            if query.min_confidence > 0:
                candidates = {cid for cid in candidates
                             if self.concepts[cid].confidence >= query.min_confidence}

            # Compute scores and rank results
            scored_results = []
            for concept_id in candidates:
                concept = self.concepts[concept_id]
                # Calculate combined score
                score = self._compute_concept_score(concept, query)
                scored_results.append((score, concept))

            # Sort by score (descending)
            scored_results.sort(reverse=True, key=lambda x: x[0])

            # Get top results
            top_concepts = [concept for _, concept in scored_results[:query.max_results]]

            # Update context window and activations for retrieved concepts
            for concept in top_concepts:
                if concept.concept_id in self.context_window:
                    self.context_window.remove(concept.concept_id)
                self.context_window.append(concept.concept_id)

                # Increase activation for retrieved concepts
                concept.activation = min(1.0, concept.activation + 0.2)
                self.concept_activations[concept.concept_id] = concept.activation

                # Update access metadata
                import time
                concept.last_accessed = time.time()
                concept.access_count += 1

            return top_concepts

    def _compute_concept_score(self, concept: Concept, query: SemanticQuery) -> float:
        """
        Calculate a combined score for a concept based on the query parameters.

        Args:
            concept: Concept to score
            query: Query parameters

        Returns:
            Float score (higher is better)
        """
        score = 0.0

        # Importance component
        score += concept.importance * 0.2

        # Activation component
        score += concept.activation * 0.1

        # Recency component (if recency_bias > 0)
        if query.recency_bias > 0 and concept.last_accessed > 0:
            import time
            time_factor = math.exp(-self.config.recency_decay * (time.time() - concept.last_accessed))
            score += time_factor * query.recency_bias

        # Text match component
        if query.text is not None:
            if query.text.lower() in concept.name.lower():
                score += 0.5
            elif query.synonyms_match and any(query.text.lower() in syn.lower() for syn in concept.synonyms):
                score += 0.3

        # Embedding similarity component
        if query.embedding is not None and concept.embedding is not None:
            similarity = self._similarity(query.embedding, concept.embedding, query.similarity_metric)
            score += similarity * 0.3

        # Context boost component
        for boost_key, boost_value in query.context_boost.items():
            if boost_key in concept.properties:
                score += boost_value
            elif boost_key == concept.abstraction_level.name:
                score += boost_value

        # Apply confidence factor
        score *= concept.confidence

        return score

    def _similarity(self, vec1: np.ndarray, vec2: np.ndarray, metric: str) -> float:
        """
        Compute vector similarity using the specified metric.

        Args:
            vec1: First vector
            vec2: Second vector
            metric: Similarity metric name

        Returns:
            Similarity score (higher means more similar)
        """
        if metric == "cosine":
            dot_product = np.dot(vec1, vec2)
            norm1 = np.linalg.norm(vec1)
            norm2 = np.linalg.norm(vec2)
            if norm1 > 0 and norm2 > 0:
                return dot_product / (norm1 * norm2)
            return 0.0
        elif metric == "euclidean":
            # Convert distance to similarity (lower distance = higher similarity)
            dist = np.linalg.norm(vec1 - vec2)
            return 1.0 / (1.0 + dist)
        else:
            self.logger.warning(f"Unknown similarity metric: {metric}, defaulting to cosine")
            return self._similarity(vec1, vec2, "cosine")

    def _apply_spreading_activation(self, seed_text: str) -> None:
        """
        Apply spreading activation from concepts matching the seed text.

        Args:
            seed_text: Text to search for initial concepts to activate
        """
        # Find initial concepts to activate
        activated_concepts = set()

        # Find concepts matching the seed text
        for concept_id, concept in self.concepts.items():
            if seed_text.lower() in concept.name.lower():
                activated_concepts.add(concept_id)
                concept.activation = 1.0  # Full activation
                self.concept_activations[concept_id] = 1.0
            elif any(seed_text.lower() in syn.lower() for syn in concept.synonyms):
                activated_concepts.add(concept_id)
                concept.activation = 0.8  # High but not full activation
                self.concept_activations[concept_id] = 0.8

        # Process spreading in steps
        for step in range(self.config.max_activation_spreading_steps):
            newly_activated = set()

            # Process each activated concept
            for concept_id in activated_concepts:
                # Spread to related concepts (outgoing relations)
                for relation in self.relations_by_source.get(concept_id, []):
                    if relation.target_concept_id not in activated_concepts:
                        target_concept = self.concepts[relation.target_concept_id]
                        # Calculate spread amount
                        spread = self.concepts[concept_id].activation * relation.weight * self.config.activation_spread_factor
                        # Apply spread with decay based on step
                        decay_factor = 1.0 / (step + 1)
                        spread *= decay_factor
                        # Update activation
                        target_concept.activation = min(1.0, target_concept.activation + spread)
                        self.concept_activations[relation.target_concept_id] = target_concept.activation
                        # Mark as newly activated if significant
                        if spread > 0.05:
                            newly_activated.add(relation.target_concept_id)

                # Also consider incoming relations (weaker)
                for relation in self.relations_by_target.get(concept_id, []):
                    if relation.source_concept_id not in activated_concepts:
                        source_concept = self.concepts[relation.source_concept_id]
                        # Calculate reverse spread (weaker)
                        spread = self.concepts[concept_id].activation * relation.weight * self.config.activation_spread_factor * 0.5
                        # Apply spread with decay based on step
                        decay_factor = 1.0 / (step + 1)
                        spread *= decay_factor
                        # Update activation
                        source_concept.activation = min(1.0, source_concept.activation + spread)
                        self.concept_activations[relation.source_concept_id] = source_concept.activation
                        # Mark as newly activated if significant
                        if spread > 0.05:
                            newly_activated.add(relation.source_concept_id)

            # Add newly activated concepts
            activated_concepts.update(newly_activated)

            # If no new concepts were activated, stop spreading
            if not newly_activated:
                break

    # --------------------------------------------------------------------------
    # INFERENCE METHODS
    # --------------------------------------------------------------------------

    def infer_relation(self,
                      source_concept_id: str,
                      relation_type: Union[RelationType, str],
                      target_concept_id: str,
                      min_confidence: float = 0.5) -> bool:
        """
        Check if a relation between source and target exists directly or can be inferred.

        Args:
            source_concept_id: Source concept ID
            relation_type: Relation type to check
            target_concept_id: Target concept ID
            min_confidence: Minimum confidence threshold

        Returns:
            bool: True if the relation exists or can be inferred
        """
        with self.lock:
            # Check if concepts exist
            if source_concept_id not in self.concepts or target_concept_id not in self.concepts:
                return False

            # First, check for direct relations
            direct_relations = self.query_relations(
                source_concept_id=source_concept_id,
                relation_type=relation_type,
                target_concept_id=target_concept_id,
                min_confidence=min_confidence
            )

            if direct_relations:
                return True

            # For transitive relations, try inference
            if isinstance(relation_type, RelationType) and \
               (relation_type == RelationType.IS_A or relation_type == RelationType.PART_OF) or \
               (isinstance(relation_type, str) and relation_type in ("is_a", "part_of")):
                return self._check_transitive_relation(
                    source_concept_id,
                    relation_type,
                    target_concept_id,
                    min_confidence,
                    depth=0,
                    visited=set()
                )

            return False

    def get_related_concepts(self,
                            concept_id: str,
                            relation_type: Optional[Union[RelationType, str]] = None,
                            max_depth: int = 1,
                            min_confidence: float = 0.0) -> Dict[str, float]:
        """
        Get concepts related to the given concept with relationship strength scores.

        Args:
            concept_id: ID of the concept to find relations for
            relation_type: Optional specific relation type to filter by
            max_depth: Maximum traversal depth for relations (1 = direct relations only)
            min_confidence: Minimum confidence threshold for relations

        Returns:
            Dictionary mapping related concept IDs to relevance scores
        """
        with self.lock:
            if concept_id not in self.concepts:
                return {}

            result = {}  # concept_id -> relevance score
            visited = set([concept_id])  # Avoid cycles

            def traverse_relations(current_id: str, depth: int, path_weight: float):
                if depth > max_depth:
                    return

                # Get outgoing relations
                relations = []
                if relation_type is not None:
                    relations = self.query_relations(
                        source_concept_id=current_id,
                        relation_type=relation_type,
                        min_confidence=min_confidence
                    )
                else:
                    relations = self.get_relations_by_source(current_id)
                    relations = [r for r in relations if r.confidence >= min_confidence]

                # Process each relation
                for rel in relations:
                    if rel.target_concept_id not in visited:
                        # Calculate new weight (diminishes with depth)
                        new_weight = path_weight * rel.weight * (1.0 / (depth + 1))

                        # Add or update the target in results
                        result[rel.target_concept_id] = max(
                            result.get(rel.target_concept_id, 0.0),
                            new_weight
                        )

                        # Mark as visited
                        visited.add(rel.target_concept_id)

                        # Recurse deeper
                        traverse_relations(rel.target_concept_id, depth + 1, new_weight)

            # Start traversal
            traverse_relations(concept_id, 1, 1.0)

            return result

    def _check_transitive_relation(self,
                                 source_id: str,
                                 relation_type: Union[RelationType, str],
                                 target_id: str,
                                 min_confidence: float,
                                 depth: int,
                                 visited: Set[str]) -> bool:
        """
        Recursive helper to check for transitive relations.

        Args:
            source_id: Current source concept
            relation_type: Relation type to check
            target_id: Ultimate target concept
            min_confidence: Minimum confidence threshold
            depth: Current recursion depth
            visited: Set of already visited concepts to prevent cycles

        Returns:
            bool: True if a valid path exists
        """
        # Prevent excessive recursion or cycles
        if depth >= self.config.inference_depth_limit or source_id in visited:
            return False

        # Mark as visited
        visited.add(source_id)

        # Get direct relations of the specified type
        relations = self.query_relations(
            source_concept_id=source_id,
            relation_type=relation_type,
            min_confidence=min_confidence
        )

        # Check each relation
        for relation in relations:
            # Check if this relation leads to target
            if relation.target_concept_id == target_id:
                return True

            # Recursively check from this target
            if self._check_transitive_relation(
                relation.target_concept_id,
                relation_type,
                target_id,
                min_confidence,
                depth + 1,
                visited
            ):
                return True

        return False

    def _find_transitive_sources(self,
                               concept_id: str,
                               relation_type: Union[RelationType, str]) -> Set[str]:
        """
        Find all concepts that have a transitive relationship to the given concept.
        This is useful for IS_A relationships (find all subtypes).

        Args:
            concept_id: Target concept ID
            relation_type: Relation type to traverse

        Returns:
            Set of concept IDs with transitive relationships
        """
        results = set()
        rel_type_str = str(relation_type)

        # Get all direct relations of this type targeting the concept
        direct_relations = self.query_relations(
            relation_type=relation_type,
            target_concept_id=concept_id
        )

        # Add direct sources
        for rel in direct_relations:
            results.add(rel.source_concept_id)

        # For each direct source, recursively find its sources
        for rel in direct_relations:
            # Only recurse for transitive relation types
            if rel.transitive:
                indirect_sources = self._find_transitive_sources(rel.source_concept_id, relation_type)
                results.update(indirect_sources)

        return results

    def _validate_relation_ontology(self, relation: Relation) -> Dict[str, Any]:
        """
        Validate if a relation is consistent with ontological constraints.

        Args:
            relation: The relation to validate

        Returns:
            Dict with 'valid' (bool) and optional 'reason' for failure
        """
        # Default valid result
        result = {"valid": True}

        # Skip validation if not enabled
        if not self.config.ontology_validation:
            return result

        source_concept = self.concepts.get(relation.source_concept_id)
        target_concept = self.concepts.get(relation.target_concept_id)

        if not source_concept or not target_concept:
            result["valid"] = False
            result["reason"] = "Source or target concept not found"
            return result

        # Check abstraction level consistency for IS_A relationships
        if isinstance(relation.relation_type, RelationType) and relation.relation_type == RelationType.IS_A:
            # For IS_A, source should generally be at the same or lower abstraction level than target
            if source_concept.abstraction_level.value > target_concept.abstraction_level.value:
                result["valid"] = False
                result["reason"] = f"Invalid IS_A abstraction levels: {source_concept.abstraction_level.name} -> {target_concept.abstraction_level.name}"
                return result

        # Check for circularity in transitive relationships
        if relation.transitive and relation.source_concept_id != relation.target_concept_id:
            # Check if the target already has a path to the source (which would create a cycle)
            if self._check_transitive_relation(
                relation.target_concept_id,
                relation.relation_type,
                relation.source_concept_id,
                0.0,  # No confidence threshold for this check
                0,
                set()
            ):
                result["valid"] = False
                result["reason"] = "Relation would create a circular dependency"
                return result

        return result

    # --------------------------------------------------------------------------
    # MEMORY MANAGEMENT METHODS
    # --------------------------------------------------------------------------

    def _decay_activations(self) -> None:
        """
        Apply decay to all concept activations based on config.activation_decay.
        """
        with self.lock:
            for concept_id, concept in self.concepts.items():
                if concept.activation > 0:
                    # Apply exponential decay
                    concept.activation *= (1.0 - self.config.activation_decay)
                    if concept.activation < 0.01:  # Minimum threshold
                        concept.activation = 0.0
                    self.concept_activations[concept_id] = concept.activation

    def _consolidate_memory(self) -> None:
        """
        Perform memory consolidation operations:
        1. Decay activations
        2. Prune concepts if over capacity
        3. Update concept importances based on access patterns
        """
        with self.lock:
            # Decay activations
            self._decay_activations()

            # Adjust concept importances based on access patterns
            self._update_concept_importances()

            # Prune if necessary
            if self.config.max_concepts > 0 and len(self.concepts) > self.config.max_concepts:
                self._prune_concepts(len(self.concepts) - self.config.max_concepts)

    def _update_concept_importances(self) -> None:
        """
        Update concept importances based on access patterns and relationship structure.
        """
        import time
        current_time = time.time()

        for concept_id, concept in self.concepts.items():
            # Initialize factors
            recency_factor = 0.0
            centrality_factor = 0.0
            access_factor = 0.0

            # Recency factor - how recently the concept was accessed
            if concept.last_accessed > 0:
                time_since_access = current_time - concept.last_accessed
                recency_factor = math.exp(-self.config.recency_decay * time_since_access)

            # Centrality factor - how many relations involve this concept
            outgoing_relations = len(self.relations_by_source.get(concept_id, []))
            incoming_relations = len(self.relations_by_target.get(concept_id, []))
            relation_count = outgoing_relations + incoming_relations
            if relation_count > 0:
                centrality_factor = min(1.0, math.log(1 + relation_count) / 5.0)

            # Access factor - how frequently the concept is accessed
            if concept.access_count > 0:
                access_factor = min(1.0, math.log(1 + concept.access_count) / 5.0)

            # Combine factors into new importance
            # Higher weights for recency and centrality, lower for access count
            new_importance = (
                recency_factor * 0.4 +
                centrality_factor * 0.4 +
                access_factor * 0.2
            )

            # Smooth change by blending with current importance
            concept.importance = 0.7 * concept.importance + 0.3 * new_importance

    def _prune_concepts(self, count: int) -> None:
        """
        Remove the least important concepts based on prune_policy.

        Args:
            count: Number of concepts to remove
        """
        if count <= 0:
            return

        candidates = []

        if self.config.prune_policy == "lowest_importance":
            # Sort by importance (ascending)
            candidates = sorted(self.concepts.items(), key=lambda x: x[1].importance)
        elif self.config.prune_policy == "lowest_activation":
            # Sort by activation (ascending)
            candidates = sorted(self.concepts.items(), key=lambda x: x[1].activation)
        elif self.config.prune_policy == "lowest_confidence":
            # Sort by confidence (ascending)
            candidates = sorted(self.concepts.items(), key=lambda x: x[1].confidence)
        else:
            # Default to lowest importance
            candidates = sorted(self.concepts.items(), key=lambda x: x[1].importance)

        # Remove the least important concepts
        for i in range(min(count, len(candidates))):
            concept_id, _ = candidates[i]
            self.remove_concept(concept_id)

        self.logger.info(f"Pruned {count} concepts using policy '{self.config.prune_policy}'")

    def clear_all(self) -> None:
        """
        Clear all concepts and relations from memory.
        """
        with self.lock:
            self.concepts.clear()
            self.relations.clear()
            self.relations_by_source.clear()
            self.relations_by_target.clear()
            self.relations_by_type.clear()
            self.concepts_by_abstraction = {level: [] for level in AbstractionLevel}
            self.context_window.clear()
            self.concept_activations.clear()
            self.operation_counter = 0

        self.logger.info("Semantic memory cleared.")

    # --------------------------------------------------------------------------
    # GRAPH INTEGRATION
    # --------------------------------------------------------------------------

    def _sync_to_graph_subsystem(self, concept: Concept) -> None:
        """
        Sync concept to an external graph engine (if configured).
        This is a stub for integration with graph_engine.py.

        Args:
            concept: Concept to sync to graph
        """
        if not self.config.auto_link_graph:
            return

        self.logger.debug(f"Would sync concept {concept.concept_id} to graph subsystem (stub)")
        # In a real implementation, call graph_engine methods

    # --------------------------------------------------------------------------
    # IMPORT/EXPORT METHODS
    # --------------------------------------------------------------------------

    def export_to_dict(self) -> Dict[str, Any]:
        """
        Export the semantic memory state to a dictionary for serialization.

        Returns:
            Dictionary representation of the semantic memory
        """
        with self.lock:
            # Convert concepts to serializable dicts
            concepts_dict = {}
            for cid, concept in self.concepts.items():
                concept_dict = {
                    "concept_id": concept.concept_id,
                    "name": concept.name,
                    "description": concept.description,
                    "properties": concept.properties,
                    "synonyms": list(concept.synonyms),
                    "importance": concept.importance,
                    "abstraction_level": concept.abstraction_level.name,
                    "activation": concept.activation,
                    "last_accessed": concept.last_accessed,
                    "access_count": concept.access_count,
                    "confidence": concept.confidence,
                    "source": concept.source
                }

                # Handle embeddings specially
                if concept.embedding is not None:
                    concept_dict["embedding"] = concept.embedding.tolist()

                concepts_dict[cid] = concept_dict

            # Convert relations to serializable dicts
            relations_dict = {}
            for rid, relation in self.relations.items():
                relation_dict = {
                    "relation_id": relation.relation_id,
                    "source_concept_id": relation.source_concept_id,
                    "relation_type": str(relation.relation_type),
                    "target_concept_id": relation.target_concept_id,
                    "weight": relation.weight,
                    "properties": relation.properties,
                    "bidirectional": relation.bidirectional,
                    "transitive": relation.transitive,
                    "temporal": relation.temporal,
                    "created_at": relation.created_at,
                    "confidence": relation.confidence,
                    "source": relation.source
                }

                relations_dict[rid] = relation_dict

            # Build the export dict
            export_data = {
                "concepts": concepts_dict,
                "relations": relations_dict,
                "context_window": list(self.context_window),
                "config": {
                    "max_concepts": self.config.max_concepts,
                    "prune_policy": self.config.prune_policy,
                    "activation_decay": self.config.activation_decay,
                    "activation_spread_factor": self.config.activation_spread_factor,
                    "relation_confidence_threshold": self.config.relation_confidence_threshold,
                    "inference_depth_limit": self.config.inference_depth_limit,
                    "enable_spreading_activation": self.config.enable_spreading_activation,
                    "max_activation_spreading_steps": self.config.max_activation_spreading_steps,
                    "ontology_validation": self.config.ontology_validation,
                    "default_concept_confidence": self.config.default_concept_confidence,
                    "context_window_size": self.config.context_window_size,
                    "similarity_threshold": self.config.similarity_threshold,
                    "use_memory_consolidation": self.config.use_memory_consolidation,
                    "consolidation_interval": self.config.consolidation_interval
                }
            }

            return export_data

    def import_from_dict(self, data: Dict[str, Any]) -> bool:
        """
        Import semantic memory state from a dictionary.

        Args:
            data: Dictionary representation of semantic memory state

        Returns:
            bool: True if import was successful
        """
        with self.lock:
            try:
                # Clear existing data
                self.clear_all()

                # Update config values if present
                if "config" in data:
                    for key, value in data["config"].items():
                        if hasattr(self.config, key):
                            setattr(self.config, key, value)

                # Import concepts
                for cid, concept_dict in data.get("concepts", {}).items():
                    # Handle embeddings
                    embedding = None
                    if "embedding" in concept_dict:
                        embedding = np.array(concept_dict["embedding"])

                    # Parse abstraction level
                    abstraction_level = AbstractionLevel.SPECIFIC
                    if "abstraction_level" in concept_dict:
                        level_name = concept_dict["abstraction_level"]
                        try:
                            abstraction_level = AbstractionLevel[level_name]
                        except KeyError:
                            pass

                    concept = Concept(
                        concept_id=concept_dict["concept_id"],
                        name=concept_dict["name"],
                        description=concept_dict.get("description", ""),
                        properties=concept_dict.get("properties", {}),
                        embedding=embedding,
                        synonyms=set(concept_dict.get("synonyms", [])),
                        importance=concept_dict.get("importance", 0.5),
                        abstraction_level=abstraction_level,
                        activation=concept_dict.get("activation", 0.0),
                        last_accessed=concept_dict.get("last_accessed", 0.0),
                        access_count=concept_dict.get("access_count", 0),
                        confidence=concept_dict.get("confidence", 1.0),
                        source=concept_dict.get("source", "user")
                    )

                    # Store without using add_concept to preserve exact values
                    self.concepts[concept.concept_id] = concept
                    self.concept_activations[concept.concept_id] = concept.activation
                    self.concepts_by_abstraction[concept.abstraction_level].append(concept.concept_id)

                # Import relations
                for rid, relation_dict in data.get("relations", {}).items():
                    relation = Relation(
                        relation_id=relation_dict["relation_id"],
                        source_concept_id=relation_dict["source_concept_id"],
                        relation_type=relation_dict["relation_type"],
                        target_concept_id=relation_dict["target_concept_id"],
                        weight=relation_dict.get("weight", 1.0),
                        properties=relation_dict.get("properties", {}),
                        bidirectional=relation_dict.get("bidirectional", False),
                        transitive=relation_dict.get("transitive", False),
                        temporal=relation_dict.get("temporal", False),
                        created_at=relation_dict.get("created_at", 0.0),
                        confidence=relation_dict.get("confidence", 1.0),
                        source=relation_dict.get("source", "user")
                    )

                    # Store without using add_relation to preserve exact values
                    self.relations[relation.relation_id] = relation
                    self.relations_by_source[relation.source_concept_id].append(relation)
                    self.relations_by_target[relation.target_concept_id].append(relation)
                    self.relations_by_type[str(relation.relation_type)].append(relation)

                # Import context window
                self.context_window = deque(
                    data.get("context_window", []),
                    maxlen=self.config.context_window_size
                )

                self.logger.info(f"Imported {len(self.concepts)} concepts and {len(self.relations)} relations")
                return True

            except Exception as e:
                self.logger.error(f"Error importing semantic memory: {e}", exc_info=True)
                # Try to cleanup after failed import
                self.clear_all()
                return False

    # --------------------------------------------------------------------------
    # STATS & DIAGNOSTICS
    # --------------------------------------------------------------------------

    def get_stats(self) -> Dict[str, Any]:
        """
        Get statistics about the semantic memory state.

        Returns:
            Dictionary with statistics
        """
        with self.lock:
            stats = {
                "concept_count": len(self.concepts),
                "relation_count": len(self.relations),
                "abstraction_level_counts": {
                    level.name: len(concepts)
                    for level, concepts in self.concepts_by_abstraction.items()
                },
                "relation_type_counts": {
                    rtype: len(relations)
                    for rtype, relations in self.relations_by_type.items()
                },
                "avg_importance": sum(c.importance for c in self.concepts.values()) / max(1, len(self.concepts)),
                "avg_confidence": sum(c.confidence for c in self.concepts.values()) / max(1, len(self.concepts)),
                "avg_relations_per_concept": len(self.relations) / max(1, len(self.concepts)),
                "activated_concepts": len([c for c in self.concepts.values() if c.activation > 0]),
                "context_window_size": len(self.context_window),
                "operation_counter": self.operation_counter
            }

            return stats

    def analyze_concept_network(self) -> Dict[str, Any]:
        """
        Analyze the concept network structure to identify key concepts,
        clusters, and other network properties.

        Returns:
            Dictionary with network analysis results
        """
        with self.lock:
            if not self.concepts:
                return {"error": "No concepts in memory"}

            try:
                # This would require networkx, which might not be available
                # We'll use a more basic approach that doesn't require external dependencies

                # Find central concepts (most connections)
                connection_counts = {}
                for cid in self.concepts:
                    outgoing = len(self.relations_by_source.get(cid, []))
                    incoming = len(self.relations_by_target.get(cid, []))
                    connection_counts[cid] = outgoing + incoming

                # Sort by connection count
                central_concepts = sorted(
                    connection_counts.items(),
                    key=lambda x: x[1],
                    reverse=True
                )[:10]  # Top 10

                # Find isolated concepts (no connections)
                isolated_concepts = [
                    cid for cid in self.concepts
                    if len(self.relations_by_source.get(cid, [])) == 0 and
                       len(self.relations_by_target.get(cid, [])) == 0
                ]

                # Find concept clusters (very basic approach)
                # Group concepts that share relations
                basic_clusters = []
                remaining = set(self.concepts.keys())

                while remaining:
                    # Start a new cluster
                    if not remaining:
                        break

                    seed = next(iter(remaining))
                    cluster = {seed}
                    to_process = {seed}

                    # Grow the cluster
                    while to_process:
                        current = to_process.pop()
                        # Find connected concepts
                        for rel in self.relations_by_source.get(current, []):
                            if rel.target_concept_id in remaining:
                                cluster.add(rel.target_concept_id)
                                to_process.add(rel.target_concept_id)

                        for rel in self.relations_by_target.get(current, []):
                            if rel.source_concept_id in remaining:
                                cluster.add(rel.source_concept_id)
                                to_process.add(rel.source_concept_id)

                    # Only add clusters with multiple concepts
                    if len(cluster) > 1:
                        basic_clusters.append(list(cluster))

                    # Remove cluster concepts from remaining
                    remaining -= cluster

                return {
                    "central_concepts": [
                        {"concept_id": cid, "name": self.concepts[cid].name, "connections": count}
                        for cid, count in central_concepts
                    ],
                    "isolated_concepts": [
                        {"concept_id": cid, "name": self.concepts[cid].name}
                        for cid in isolated_concepts
                    ],
                    "basic_clusters": [
                        {
                            "size": len(cluster),
                            "concepts": [
                                {"concept_id": cid, "name": self.concepts[cid].name}
                                for cid in cluster
                            ]
                        }
                        for cluster in basic_clusters
                    ],
                    "total_concepts": len(self.concepts),
                    "total_relations": len(self.relations),
                    "average_connectivity": sum(connection_counts.values()) / max(1, len(self.concepts))
                }

            except Exception as e:
                self.logger.error(f"Error analyzing concept network: {e}", exc_info=True)
                return {"error": f"Analysis failed: {str(e)}"}