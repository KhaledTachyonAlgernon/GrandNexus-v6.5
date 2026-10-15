"""
graph_pattern_interface.py - Interface between GraphEngine and PatternMining

This module defines the integration points between the graph engine and
pattern mining modules, allowing them to work as a cohesive unit while maintaining
separation of concerns. It serves as the bridge between low-level graph operations
and high-level cognitive pattern analysis.
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional, Union, Tuple, Set
from dataclasses import dataclass, field

from grandnexus.graph.graph_engine import (
    create_graph_engine, AbstractGraphEngine, GraphNode, GraphEdge, GraphPattern,
    GraphQuery, NodeNotFoundError, EdgeNotFoundError
)

@dataclass
class PatternMatchResult:
    """
    Represents the result of a pattern matching operation.
    """
    subgraph_nodes: List[str] = field(default_factory=list)
    subgraph_edges: List[str] = field(default_factory=list)
    similarity_score: float = 0.0
    pattern_type: Optional[str] = None
    extra_info: Dict[str, Any] = field(default_factory=dict)

    def __repr__(self):
        pattern_info = f", type={self.pattern_type}" if self.pattern_type else ""
        return (f"<PatternMatchResult nodes={len(self.subgraph_nodes)}, "
                f"edges={len(self.subgraph_edges)}, sim={self.similarity_score:.2f}{pattern_info}>")


@dataclass
class FrequentPattern:
    """
    Represents a discovered frequent pattern in the graph.
    """
    pattern_id: str
    nodes: List[str]
    edges: List[str]
    support: int
    pattern_embedding: Optional[Any] = None
    signature: str = ""

    def __repr__(self):
        return f"<FrequentPattern id={self.pattern_id}, support={self.support}, nodes={len(self.nodes)}>"


class PatternMiningInterface(ABC):
    """
    Abstract interface that defines how PatternMining should interact with GraphEngine.
    This allows for multiple implementations while maintaining consistent interfaces.
    """

    @abstractmethod
    def find_patterns(self,
                     pattern_spec: Dict[str, Any],
                     similarity_threshold: Optional[float] = None,
                     max_results: int = 10) -> List[PatternMatchResult]:
        """
        Searches the graph for subgraphs matching the given pattern specification.
        """
        pass

    @abstractmethod
    def extract_frequent_subgraphs(self,
                                  min_support: int = 2,
                                  max_size: int = 5) -> List[FrequentPattern]:
        """
        Discovers frequently occurring patterns in the graph.
        """
        pass

    @abstractmethod
    def detect_anomalies(self,
                        property_outliers: bool = True,
                        structural_anomalies: bool = True,
                        threshold: float = 2.0) -> List[Dict[str, Any]]:
        """
        Detects nodes or relationships that appear anomalous.
        """
        pass

    @abstractmethod
    def find_similar_patterns(self,
                             pattern: Union[PatternMatchResult, Dict[str, Any]],
                             top_n: int = 5) -> List[Tuple[PatternMatchResult, float]]:
        """
        Finds patterns similar to the provided pattern using embedding similarity.
        """
        pass

    @abstractmethod
    def get_pattern_statistics(self) -> Dict[str, Any]:
        """
        Returns statistics about the discovered patterns.
        """
        pass


class GraphPatternBridge:
    """
    Bridge class that connects GraphEngine with PatternMining.
    This class maintains the state needed for both modules to collaborate effectively
    while respecting GrandNexus's fractal architecture.
    """

    def __init__(self, graph_engine: AbstractGraphEngine, pattern_mining: PatternMiningInterface):
        """
        Initialize the bridge with both required components.

        Args:
            graph_engine: Instance of a GraphEngine implementation
            pattern_mining: Instance of a PatternMining implementation
        """
        self.graph_engine = graph_engine
        self.pattern_mining = pattern_mining
        self.logger = logging.getLogger("GraphPatternBridge")

    def convert_pattern_to_graphpattern(self, pattern: FrequentPattern) -> GraphPattern:
        """
        Converts a FrequentPattern from PatternMining to a GraphPattern for GraphEngine.

        Args:
            pattern: A FrequentPattern object

        Returns:
            A GraphPattern object that can be used with GraphEngine
        """
        return GraphPattern(
            pattern_id=pattern.pattern_id,
            nodes=pattern.nodes,
            edges=pattern.edges,
            frequency=pattern.support,
            significance=0.5,  # Default value, could be calculated
            embedding=pattern.pattern_embedding
        )

    def convert_graphpattern_to_pattern(self, graph_pattern: GraphPattern) -> FrequentPattern:
        """
        Converts a GraphPattern from GraphEngine to a FrequentPattern for PatternMining.

        Args:
            graph_pattern: A GraphPattern object

        Returns:
            A FrequentPattern object that can be used with PatternMining
        """
        return FrequentPattern(
            pattern_id=graph_pattern.pattern_id,
            nodes=graph_pattern.nodes,
            edges=graph_pattern.edges,
            support=graph_pattern.frequency,
            pattern_embedding=graph_pattern.embedding,
            signature=""  # Would need to be computed
        )

    def store_discovered_patterns(self, patterns: List[FrequentPattern]) -> None:
        """
        Stores discovered patterns from PatternMining into the GraphEngine.
        This makes the patterns part of the graph itself, enabling meta-reasoning.

        Args:
            patterns: List of discovered FrequentPattern objects
        """
        for pattern in patterns:
            # Convert to GraphPattern
            graph_pattern = self.convert_pattern_to_graphpattern(pattern)

            # Create a special node for the pattern
            pattern_node_id = f"pattern_{pattern.pattern_id}"
            properties = {
                "type": "pattern",
                "support": pattern.support,
                "signature": pattern.signature,
            }

            try:
                # Add the pattern node
                self.graph_engine.add_node(node_id=pattern_node_id,
                                          label="Pattern",
                                          properties=properties)

                # Link the pattern node to all nodes in the pattern
                for node_id in pattern.nodes:
                    self.graph_engine.add_edge(
                        source_id=pattern_node_id,
                        target_id=node_id,
                        edge_type="PATTERN_CONTAINS",
                        properties={"pattern_id": pattern.pattern_id}
                    )
            except Exception as e:
                self.logger.error(f"Failed to store pattern {pattern.pattern_id}: {e}")

    def apply_pattern_activation(self, pattern: FrequentPattern, activation: float = 0.8) -> Dict[str, float]:
        """
        Activates all nodes in a pattern, allowing for pattern-based spreading activation.

        Args:
            pattern: The pattern whose nodes should be activated
            activation: Initial activation value for the pattern nodes

        Returns:
            Dictionary mapping node IDs to their activation values
        """
        # First, activate all nodes in the pattern
        for node_id in pattern.nodes:
            try:
                node = self.graph_engine.get_node(node_id)
                node.activation = activation
            except NodeNotFoundError:
                self.logger.warning(f"Node {node_id} from pattern {pattern.pattern_id} not found")

        # Then use spreading activation to propagate
        return self.graph_engine.spread_activation(
            source_nodes=pattern.nodes,
            decay_factor=0.5,
            max_depth=2
        )

    def get_pattern_subgraph(self, pattern_id: str) -> Tuple[List[GraphNode], List[GraphEdge]]:
        """
        Retrieves all nodes and edges that are part of a specific pattern.

        Args:
            pattern_id: ID of the pattern to retrieve

        Returns:
            Tuple of (nodes, edges) lists representing the pattern subgraph
        """
        pattern_node_id = f"pattern_{pattern_id}"

        try:
            # Find all nodes connected to the pattern node
            neighbors = self.graph_engine.get_node_neighbors(
                node_id=pattern_node_id,
                edge_types=["PATTERN_CONTAINS"]
            )

            node_ids = [neighbor_id for neighbor_id, _, _ in neighbors]
            nodes = [self.graph_engine.get_node(node_id) for node_id in node_ids]

            # Find all edges between these nodes
            edges = []
            for i, src_id in enumerate(node_ids):
                for dst_id in node_ids[i+1:]:
                    # Check for edges in both directions
                    out_neighbors = self.graph_engine.get_node_neighbors(
                        node_id=src_id,
                        direction="outgoing"
                    )
                    for neighbor_id, edge_id, _ in out_neighbors:
                        if neighbor_id == dst_id:
                            edges.append(self.graph_engine.get_edge(edge_id))

                    in_neighbors = self.graph_engine.get_node_neighbors(
                        node_id=src_id,
                        direction="incoming"
                    )
                    for neighbor_id, edge_id, _ in in_neighbors:
                        if neighbor_id == dst_id:
                            edges.append(self.graph_engine.get_edge(edge_id))

            return nodes, edges

        except NodeNotFoundError:
            self.logger.warning(f"Pattern node {pattern_node_id} not found")
            return [], []

    def register_with_nexus_core(self, nexus_core):
        """
        Registers this bridge with NexusCore to enable communication with other modules.

        Args:
            nexus_core: The NexusCore instance
        """
        # Register callbacks with NexusCore for handling pattern-related events
        nexus_core.register_handler("graph.pattern_discovered", self._handle_pattern_discovered)
        nexus_core.register_handler("memory.concept_stored", self._handle_concept_stored)
        nexus_core.register_handler("reasoning.relation_inferred", self._handle_relation_inferred)

    def _handle_pattern_discovered(self, event_data: Dict[str, Any]) -> None:
        """Handler for pattern discovery events"""
        # Implementation would dispatch to appropriate methods
        pass

    def _handle_concept_stored(self, event_data: Dict[str, Any]) -> None:
        """Handler for memory concept storage events"""
        # Check if new concept forms patterns with existing concepts
        pass

    def _handle_relation_inferred(self, event_data: Dict[str, Any]) -> None:
        """Handler for reasoning relation inference events"""
        # Check if new relation completes or modifies known patterns
        pass


# Import here to avoid circular imports
import logging