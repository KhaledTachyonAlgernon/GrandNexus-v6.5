"""
graph_engine.py - Advanced Graph Engine Module for GrandNexus

This module provides a comprehensive graph-based representation system for GrandNexus,
integrating seamlessly with memory and cognitive modules.

Features:
- Abstract base class with consistent interface
- Multiple graph engine implementations (InMemory, Neo4j)
- Cognitive graph operations (activation spreading, pattern mining)
- Advanced query capabilities
- Integration with semantic and episodic memory
- Support for both symbolic and subsymbolic representations
"""

import logging
import time
import uuid
import threading
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Union, Tuple, Set, Callable
from collections import defaultdict, deque
import math
import json

# Optional dependencies for vector operations and mathematical functions
try:
    import numpy as np
    HAS_NUMPY = True
except ImportError:
    HAS_NUMPY = False

# Optional dependency for in-memory graph analytics
try:
    import networkx as nx
    HAS_NETWORKX = True
except ImportError:
    HAS_NETWORKX = False

# Optional dependency for Neo4j
try:
    from neo4j import GraphDatabase, Driver
    HAS_NEO4J = True
except ImportError:
    HAS_NEO4J = False


# -----------------------------------------------------------------------------
# Core Data Structures
# -----------------------------------------------------------------------------

@dataclass
class GraphNode:
    """
    Represents a node in the graph with cognitive properties.

    Attributes:
        node_id: Unique identifier for the node
        label: Primary category or type of the node
        properties: Dictionary of node attributes
        embedding: Optional vector representation for similarity operations
        activation: Current activation level (0.0 to 1.0, higher = more activated)
        last_accessed: Timestamp of last access (for decay calculations)
        importance: Long-term importance score (0.0 to 1.0)
        creation_time: When the node was created
    """
    node_id: str
    label: str
    properties: Dict[str, Any] = field(default_factory=dict)
    embedding: Optional[Any] = None
    activation: float = 0.0
    last_accessed: float = field(default_factory=time.time)
    importance: float = 0.5
    creation_time: float = field(default_factory=time.time)

    def access(self) -> None:
        """Update activation and last_accessed time when node is accessed."""
        self.activation = 1.0
        self.last_accessed = time.time()

    def decay_activation(self, decay_rate: float, current_time: Optional[float] = None) -> None:
        """
        Apply decay to node activation based on time since last access.

        Args:
            decay_rate: Rate of activation decay (higher = faster decay)
            current_time: Current time or None to use current system time
        """
        if current_time is None:
            current_time = time.time()

        time_since_access = max(0.001, current_time - self.last_accessed)
        decay_factor = math.exp(-decay_rate * time_since_access)
        self.activation *= decay_factor

        # Ensure activation stays in valid range
        self.activation = max(0.0, min(1.0, self.activation))


@dataclass
class GraphEdge:
    """
    Represents a directed edge in the graph with cognitive properties.

    Attributes:
        edge_id: Unique identifier for the edge
        source_id: ID of the source node
        target_id: ID of the target node
        type: Relationship type (e.g., "IS_A", "PART_OF")
        properties: Dictionary of edge attributes
        weight: Strength of the connection (0.0 to 1.0, higher = stronger)
        bidirectional: Whether the relationship applies in both directions
        creation_time: When the edge was created
    """
    edge_id: str
    source_id: str
    target_id: str
    type: str
    properties: Dict[str, Any] = field(default_factory=dict)
    weight: float = 1.0
    bidirectional: bool = False
    creation_time: float = field(default_factory=time.time)


@dataclass
class GraphPattern:
    """
    Represents a discovered pattern in the graph.

    Attributes:
        pattern_id: Unique identifier for the pattern
        nodes: List of node IDs in this pattern
        edges: List of edge IDs in this pattern
        frequency: Number of occurrences in the graph
        significance: Measure of pattern importance
        embedding: Vector representation of the pattern
    """
    pattern_id: str
    nodes: List[str]
    edges: List[str]
    frequency: int = 1
    significance: float = 0.5
    embedding: Optional[Any] = None


@dataclass
class GraphQuery:
    """
    A flexible query specification for graph operations.

    Attributes:
        node_criteria: Criteria for matching nodes
        edge_criteria: Criteria for matching edges
        start_nodes: Optional set of starting node IDs
        max_depth: Maximum traversal depth
        max_results: Maximum number of results to return
        include_inactive: Whether to include nodes below activation threshold
        min_activation: Minimum activation level to consider
        sort_by: How to sort results ("activation", "importance", "recency")
    """
    node_criteria: Dict[str, Any] = field(default_factory=dict)
    edge_criteria: Dict[str, Any] = field(default_factory=dict)
    start_nodes: Optional[List[str]] = None
    max_depth: int = 3
    max_results: int = 100
    include_inactive: bool = False
    min_activation: float = 0.0
    sort_by: str = "activation"


# -----------------------------------------------------------------------------
# Exception Classes
# -----------------------------------------------------------------------------

class GraphEngineError(Exception):
    """Base class for graph-related errors."""
    pass

class NodeNotFoundError(GraphEngineError):
    """Raised when a node is not found."""
    pass

class EdgeNotFoundError(GraphEngineError):
    """Raised when an edge is not found."""
    pass

class InvalidQueryError(GraphEngineError):
    """Raised when a query is malformed."""
    pass


# -----------------------------------------------------------------------------
# Abstract Base Class
# -----------------------------------------------------------------------------

class AbstractGraphEngine(ABC):
    """
    Abstract base class defining the interface for graph engines in GrandNexus.

    This class defines the core operations that any graph engine implementation
    must support, ensuring a consistent interface regardless of the underlying
    storage or processing mechanism.
    """

    @abstractmethod
    def add_node(self, node_id: Optional[str] = None, label: str = "Node",
                properties: Optional[Dict[str, Any]] = None) -> str:
        """
        Add a node to the graph.

        Args:
            node_id: Optional node ID (generated if None)
            label: Primary type or category of the node
            properties: Node attributes

        Returns:
            The ID of the created node
        """
        pass

    @abstractmethod
    def get_node(self, node_id: str) -> GraphNode:
        """
        Retrieve a node by ID.

        Args:
            node_id: ID of the node to retrieve

        Returns:
            The GraphNode object

        Raises:
            NodeNotFoundError: If node with given ID doesn't exist
        """
        pass

    @abstractmethod
    def update_node(self, node_id: str, properties: Dict[str, Any]) -> None:
        """
        Update a node's properties.

        Args:
            node_id: ID of the node to update
            properties: Properties to update (merged with existing)

        Raises:
            NodeNotFoundError: If node with given ID doesn't exist
        """
        pass

    @abstractmethod
    def delete_node(self, node_id: str) -> None:
        """
        Delete a node and its connected edges.

        Args:
            node_id: ID of the node to delete

        Raises:
            NodeNotFoundError: If node with given ID doesn't exist
        """
        pass

    @abstractmethod
    def add_edge(self, source_id: str, target_id: str, edge_type: str,
                edge_id: Optional[str] = None, properties: Optional[Dict[str, Any]] = None,
                weight: float = 1.0, bidirectional: bool = False) -> str:
        """
        Add an edge between two nodes.

        Args:
            source_id: ID of the source node
            target_id: ID of the target node
            edge_type: Type of relationship
            edge_id: Optional edge ID (generated if None)
            properties: Edge attributes
            weight: Strength of connection (0.0 to 1.0)
            bidirectional: Whether the relationship is bi-directional

        Returns:
            The ID of the created edge

        Raises:
            NodeNotFoundError: If source or target node doesn't exist
        """
        pass

    @abstractmethod
    def get_edge(self, edge_id: str) -> GraphEdge:
        """
        Retrieve an edge by ID.

        Args:
            edge_id: ID of the edge to retrieve

        Returns:
            The GraphEdge object

        Raises:
            EdgeNotFoundError: If edge with given ID doesn't exist
        """
        pass

    @abstractmethod
    def update_edge(self, edge_id: str, properties: Dict[str, Any]) -> None:
        """
        Update an edge's properties.

        Args:
            edge_id: ID of the edge to update
            properties: Properties to update (merged with existing)

        Raises:
            EdgeNotFoundError: If edge with given ID doesn't exist
        """
        pass

    @abstractmethod
    def delete_edge(self, edge_id: str) -> None:
        """
        Delete an edge.

        Args:
            edge_id: ID of the edge to delete

        Raises:
            EdgeNotFoundError: If edge with given ID doesn't exist
        """
        pass

    @abstractmethod
    def get_node_neighbors(self, node_id: str, edge_types: Optional[List[str]] = None,
                          direction: str = "outgoing") -> List[Tuple[str, str, str]]:
        """
        Get neighboring nodes connected to a node.

        Args:
            node_id: ID of the node
            edge_types: Optional filter for specific edge types
            direction: "outgoing", "incoming", or "both"

        Returns:
            List of (neighbor_id, edge_id, edge_type) tuples

        Raises:
            NodeNotFoundError: If node with given ID doesn't exist
        """
        pass

    @abstractmethod
    def query_nodes(self, criteria: Dict[str, Any]) -> List[str]:
        """
        Find nodes matching the specified criteria.

        Args:
            criteria: Dictionary of property constraints

        Returns:
            List of matching node IDs
        """
        pass

    @abstractmethod
    def query_edges(self, criteria: Dict[str, Any]) -> List[str]:
        """
        Find edges matching the specified criteria.

        Args:
            criteria: Dictionary of property constraints

        Returns:
            List of matching edge IDs
        """
        pass

    @abstractmethod
    def find_paths(self, start_id: str, end_id: str, max_depth: int = 3,
                  edge_types: Optional[List[str]] = None) -> List[List[str]]:
        """
        Find paths between two nodes.

        Args:
            start_id: ID of the starting node
            end_id: ID of the ending node
            max_depth: Maximum path length
            edge_types: Optional filter for specific edge types

        Returns:
            List of paths, where each path is a list of node IDs
        """
        pass

    @abstractmethod
    def execute_query(self, query: GraphQuery) -> Dict[str, Any]:
        """
        Execute a complex graph query.

        Args:
            query: GraphQuery object specifying the query parameters

        Returns:
            Dictionary with query results
        """
        pass

    @abstractmethod
    def spread_activation(self, source_nodes: List[str], decay_factor: float = 0.5,
                         max_depth: int = 3) -> Dict[str, float]:
        """
        Spread activation from source nodes through the graph.

        Args:
            source_nodes: Starting nodes for activation spread
            decay_factor: How quickly activation diminishes (0-1)
            max_depth: Maximum steps to spread

        Returns:
            Dictionary mapping node IDs to activation values
        """
        pass

    @abstractmethod
    def get_activated_subgraph(self, threshold: float = 0.2) -> Tuple[List[str], List[str]]:
        """
        Get the currently activated subgraph.

        Args:
            threshold: Minimum activation level

        Returns:
            Tuple of (node_ids, edge_ids) for the activated subgraph
        """
        pass

    @abstractmethod
    def detect_patterns(self, min_frequency: int = 2, max_size: int = 5) -> List[GraphPattern]:
        """
        Detect frequent patterns in the graph.

        Args:
            min_frequency: Minimum number of occurrences
            max_size: Maximum pattern size (nodes)

        Returns:
            List of GraphPattern objects
        """
        pass

    @abstractmethod
    def clear(self) -> None:
        """
        Clear all nodes and edges from the graph.
        """
        pass

    @abstractmethod
    def serialize(self) -> Dict[str, Any]:
        """
        Serialize the graph to a dictionary.

        Returns:
            Dictionary representation of the graph
        """
        pass

    @abstractmethod
    def deserialize(self, data: Dict[str, Any]) -> None:
        """
        Load the graph from a serialized representation.

        Args:
            data: Dictionary containing serialized graph
        """
        pass

    @abstractmethod
    def close(self) -> None:
        """
        Close the graph engine and release any resources.
        """
        pass


# -----------------------------------------------------------------------------
# In-Memory Graph Engine Implementation
# -----------------------------------------------------------------------------

class InMemoryGraphEngine(AbstractGraphEngine):
    """
    In-memory implementation of the graph engine with cognitive capabilities.

    This implementation keeps all nodes and edges in memory, optimized for
    cognitive operations like activation spreading and pattern detection.
    Optionally uses NetworkX for advanced graph algorithms.
    """

    def __init__(self, use_networkx: bool = True, decay_rate: float = 0.1):
        """
        Initialize the in-memory graph engine.

        Args:
            use_networkx: Whether to use NetworkX for graph algorithms (if available)
            decay_rate: Default activation decay rate
        """
        self.logger = logging.getLogger("InMemoryGraphEngine")
        self.use_networkx = use_networkx and HAS_NETWORKX
        self.decay_rate = decay_rate

        # Primary storage
        self.nodes: Dict[str, GraphNode] = {}
        self.edges: Dict[str, GraphEdge] = {}

        # Indices for efficient access
        self.outgoing_edges: Dict[str, List[str]] = defaultdict(list)  # node_id -> [edge_ids]
        self.incoming_edges: Dict[str, List[str]] = defaultdict(list)  # node_id -> [edge_ids]
        self.edges_by_type: Dict[str, List[str]] = defaultdict(list)   # type -> [edge_ids]
        self.nodes_by_label: Dict[str, List[str]] = defaultdict(list)  # label -> [node_ids]

        # NetworkX graph (if enabled)
        self._nx_graph = nx.MultiDiGraph() if self.use_networkx else None

        # Concurrency control
        self.lock = threading.RLock()

        self.logger.info(f"InMemoryGraphEngine initialized (use_networkx={self.use_networkx})")

    def add_node(self, node_id: Optional[str] = None, label: str = "Node",
                properties: Optional[Dict[str, Any]] = None) -> str:
        """Add a node to the graph."""
        with self.lock:
            # Generate ID if not provided
            node_id = node_id or str(uuid.uuid4())
            properties = properties or {}

            # Create node
            node = GraphNode(
                node_id=node_id,
                label=label,
                properties=properties.copy()
            )

            # Store node
            self.nodes[node_id] = node
            self.nodes_by_label[label].append(node_id)

            # Add to NetworkX graph if enabled
            if self.use_networkx and self._nx_graph is not None:
                attrs = dict(properties)
                attrs.update({
                    'label': label,
                    'activation': node.activation,
                    'importance': node.importance
                })
                self._nx_graph.add_node(node_id, **attrs)

            self.logger.debug(f"Added node {node_id} with label '{label}'")
            return node_id

    def get_node(self, node_id: str) -> GraphNode:
        """Retrieve a node by ID."""
        with self.lock:
            if node_id not in self.nodes:
                raise NodeNotFoundError(f"Node {node_id} not found")

            node = self.nodes[node_id]
            node.access()  # Update activation and last_accessed
            return node

    def update_node(self, node_id: str, properties: Dict[str, Any]) -> None:
        """Update a node's properties."""
        with self.lock:
            if node_id not in self.nodes:
                raise NodeNotFoundError(f"Node {node_id} not found")

            node = self.nodes[node_id]

            # Update properties
            node.properties.update(properties)
            node.last_accessed = time.time()

            # Update NetworkX if enabled
            if self.use_networkx and self._nx_graph is not None:
                for key, value in properties.items():
                    self._nx_graph.nodes[node_id][key] = value

                # Update common attributes
                self._nx_graph.nodes[node_id]['activation'] = node.activation
                self._nx_graph.nodes[node_id]['importance'] = node.importance

            self.logger.debug(f"Updated node {node_id}")

    def delete_node(self, node_id: str) -> None:
        """Delete a node and its connected edges."""
        with self.lock:
            if node_id not in self.nodes:
                raise NodeNotFoundError(f"Node {node_id} not found")

            # Delete connected edges first
            edges_to_delete = []
            edges_to_delete.extend(self.outgoing_edges.get(node_id, []))
            edges_to_delete.extend(self.incoming_edges.get(node_id, []))

            for edge_id in edges_to_delete:
                if edge_id in self.edges:  # Check to avoid duplicate deletions
                    self._delete_edge_internal(edge_id)

            # Remove from label index
            label = self.nodes[node_id].label
            if node_id in self.nodes_by_label.get(label, []):
                self.nodes_by_label[label].remove(node_id)

            # Remove from main storage
            node = self.nodes.pop(node_id)

            # Remove from NetworkX if enabled
            if self.use_networkx and self._nx_graph is not None:
                if self._nx_graph.has_node(node_id):
                    self._nx_graph.remove_node(node_id)

            # Clean up edge indices
            if node_id in self.outgoing_edges:
                del self.outgoing_edges[node_id]
            if node_id in self.incoming_edges:
                del self.incoming_edges[node_id]

            self.logger.debug(f"Deleted node {node_id} and its connected edges")

    def add_edge(self, source_id: str, target_id: str, edge_type: str,
                edge_id: Optional[str] = None, properties: Optional[Dict[str, Any]] = None,
                weight: float = 1.0, bidirectional: bool = False) -> str:
        """Add an edge between two nodes."""
        with self.lock:
            # Verify nodes exist
            if source_id not in self.nodes:
                raise NodeNotFoundError(f"Source node {source_id} not found")
            if target_id not in self.nodes:
                raise NodeNotFoundError(f"Target node {target_id} not found")

            # Generate ID if not provided
            edge_id = edge_id or str(uuid.uuid4())
            properties = properties or {}

            # Create edge
            edge = GraphEdge(
                edge_id=edge_id,
                source_id=source_id,
                target_id=target_id,
                type=edge_type,
                properties=properties.copy(),
                weight=weight,
                bidirectional=bidirectional
            )

            # Store edge
            self.edges[edge_id] = edge
            self.outgoing_edges[source_id].append(edge_id)
            self.incoming_edges[target_id].append(edge_id)
            self.edges_by_type[edge_type].append(edge_id)

            # Add to NetworkX graph if enabled
            if self.use_networkx and self._nx_graph is not None:
                attrs = dict(properties)
                attrs.update({
                    'edge_id': edge_id,
                    'type': edge_type,
                    'weight': weight,
                    'bidirectional': bidirectional
                })
                self._nx_graph.add_edge(source_id, target_id, **attrs)

            # If bidirectional, add inverse edge automatically
            if bidirectional and source_id != target_id:
                inverse_id = f"{edge_id}_inverse"
                inverse_props = properties.copy()
                inverse_props['inverse_of'] = edge_id

                inverse_edge = GraphEdge(
                    edge_id=inverse_id,
                    source_id=target_id,
                    target_id=source_id,
                    type=edge_type,
                    properties=inverse_props,
                    weight=weight,
                    bidirectional=True
                )

                self.edges[inverse_id] = inverse_edge
                self.outgoing_edges[target_id].append(inverse_id)
                self.incoming_edges[source_id].append(inverse_id)
                self.edges_by_type[edge_type].append(inverse_id)

                if self.use_networkx and self._nx_graph is not None:
                    inverse_attrs = dict(inverse_props)
                    inverse_attrs.update({
                        'edge_id': inverse_id,
                        'type': edge_type,
                        'weight': weight,
                        'bidirectional': True
                    })
                    self._nx_graph.add_edge(target_id, source_id, **inverse_attrs)

            self.logger.debug(f"Added edge {edge_id} from {source_id} to {target_id} with type '{edge_type}'")
            return edge_id

    def get_edge(self, edge_id: str) -> GraphEdge:
        """Retrieve an edge by ID."""
        with self.lock:
            if edge_id not in self.edges:
                raise EdgeNotFoundError(f"Edge {edge_id} not found")

            return self.edges[edge_id]

    def update_edge(self, edge_id: str, properties: Dict[str, Any]) -> None:
        """Update an edge's properties."""
        with self.lock:
            if edge_id not in self.edges:
                raise EdgeNotFoundError(f"Edge {edge_id} not found")

            edge = self.edges[edge_id]

            # Update properties
            edge.properties.update(properties)

            # Update weight if provided
            if 'weight' in properties:
                edge.weight = float(properties['weight'])

            # Update NetworkX if enabled
            if self.use_networkx and self._nx_graph is not None:
                # Get the edge in NetworkX
                source_id, target_id = edge.source_id, edge.target_id
                if self._nx_graph.has_edge(source_id, target_id):
                    # Find the edge with matching edge_id
                    for _, _, edge_data in self._nx_graph.edges(data=True):
                        if edge_data.get('edge_id') == edge_id:
                            # Update properties
                            for key, value in properties.items():
                                edge_data[key] = value
                            break

            # If this is part of a bidirectional pair, update the inverse too
            if edge.bidirectional:
                inverse_id = f"{edge_id}_inverse"
                if inverse_id in self.edges:
                    inverse_edge = self.edges[inverse_id]
                    for key, value in properties.items():
                        if key != 'inverse_of':  # Skip this to prevent recursion
                            inverse_edge.properties[key] = value

                    if 'weight' in properties:
                        inverse_edge.weight = float(properties['weight'])

            self.logger.debug(f"Updated edge {edge_id}")

    def delete_edge(self, edge_id: str) -> None:
        """Delete an edge."""
        with self.lock:
            if edge_id not in self.edges:
                raise EdgeNotFoundError(f"Edge {edge_id} not found")

            self._delete_edge_internal(edge_id)

            # If this is part of a bidirectional pair, delete the inverse too
            inverse_id = f"{edge_id}_inverse"
            if inverse_id in self.edges:
                self._delete_edge_internal(inverse_id)

            self.logger.debug(f"Deleted edge {edge_id}")

    def _delete_edge_internal(self, edge_id: str) -> None:
        """Internal helper to delete an edge without checking existence."""
        edge = self.edges[edge_id]

        # Remove from indices
        if edge_id in self.outgoing_edges.get(edge.source_id, []):
            self.outgoing_edges[edge.source_id].remove(edge_id)
        if edge_id in self.incoming_edges.get(edge.target_id, []):
            self.incoming_edges[edge.target_id].remove(edge_id)
        if edge_id in self.edges_by_type.get(edge.type, []):
            self.edges_by_type[edge.type].remove(edge_id)

        # Remove from NetworkX if enabled
        if self.use_networkx and self._nx_graph is not None:
            source_id, target_id = edge.source_id, edge.target_id
            # Remove the specific edge with this edge_id
            edges_to_remove = []
            for u, v, k, data in self._nx_graph.edges(source_id, target_id, keys=True, data=True):
                if data.get('edge_id') == edge_id:
                    edges_to_remove.append((u, v, k))

            for u, v, k in edges_to_remove:
                self._nx_graph.remove_edge(u, v, k)

        # Remove from main storage
        del self.edges[edge_id]

    def get_node_neighbors(self, node_id: str, edge_types: Optional[List[str]] = None,
                          direction: str = "outgoing") -> List[Tuple[str, str, str]]:
        """Get neighboring nodes connected to a node."""
        with self.lock:
            if node_id not in self.nodes:
                raise NodeNotFoundError(f"Node {node_id} not found")

            result = []

            if direction in ["outgoing", "both"]:
                for edge_id in self.outgoing_edges.get(node_id, []):
                    if edge_id in self.edges:  # Ensure edge still exists
                        edge = self.edges[edge_id]
                        if edge_types is None or edge.type in edge_types:
                            result.append((edge.target_id, edge_id, edge.type))

            if direction in ["incoming", "both"]:
                for edge_id in self.incoming_edges.get(node_id, []):
                    if edge_id in self.edges:  # Ensure edge still exists
                        edge = self.edges[edge_id]
                        if edge_types is None or edge.type in edge_types:
                            result.append((edge.source_id, edge_id, edge.type))

            # Update activation for the node
            self.nodes[node_id].access()

            return result

    def query_nodes(self, criteria: Dict[str, Any]) -> List[str]:
        """Find nodes matching the specified criteria."""
        with self.lock:
            result = []

            # Optimize for label queries
            if 'label' in criteria and len(criteria) == 1:
                label = criteria['label']
                return list(self.nodes_by_label.get(label, []))

            # General matching
            for node_id, node in self.nodes.items():
                if self._match_criteria(node.properties, criteria):
                    result.append(node_id)

            return result

    def query_edges(self, criteria: Dict[str, Any]) -> List[str]:
        """Find edges matching the specified criteria."""
        with self.lock:
            result = []

            # Optimize for type queries
            if 'type' in criteria and len(criteria) == 1:
                edge_type = criteria['type']
                return list(self.edges_by_type.get(edge_type, []))

            # General matching
            for edge_id, edge in self.edges.items():
                if self._match_criteria(edge.properties, criteria):
                    result.append(edge_id)

            return result

    def _match_criteria(self, properties: Dict[str, Any], criteria: Dict[str, Any]) -> bool:
        """Check if properties match the given criteria."""
        for key, value in criteria.items():
            if key not in properties:
                return False

            prop_value = properties[key]

            # Handle comparison operators
            if isinstance(value, dict):
                for op, op_value in value.items():
                    if op == '$gt':
                        if not (prop_value > op_value):
                            return False
                    elif op == '$gte':
                        if not (prop_value >= op_value):
                            return False
                    elif op == '$lt':
                        if not (prop_value < op_value):
                            return False
                    elif op == '$lte':
                        if not (prop_value <= op_value):
                            return False
                    elif op == '$ne':
                        if not (prop_value != op_value):
                            return False
                    elif op == '$in':
                        if not (prop_value in op_value):
                            return False
                    else:
                        # Unknown operator
                        return False
            else:
                # Simple equality
                if prop_value != value:
                    return False

        return True

    def find_paths(self, start_id: str, end_id: str, max_depth: int = 3,
                  edge_types: Optional[List[str]] = None) -> List[List[str]]:
        """Find paths between two nodes."""
        with self.lock:
            # Verify nodes exist
            if start_id not in self.nodes:
                raise NodeNotFoundError(f"Start node {start_id} not found")
            if end_id not in self.nodes:
                raise NodeNotFoundError(f"End node {end_id} not found")

            # Use NetworkX if available for path finding
            if self.use_networkx and self._nx_graph is not None:
                return self._find_paths_networkx(start_id, end_id, max_depth, edge_types)
            else:
                return self._find_paths_bfs(start_id, end_id, max_depth, edge_types)

    def _find_paths_networkx(self, start_id: str, end_id: str, max_depth: int,
                           edge_types: Optional[List[str]]) -> List[List[str]]:
        """Find paths using NetworkX algorithms."""
        try:
            # If edge_types is specified, create a view of the graph with only those types
            if edge_types:
                view = nx.subgraph_view(
                    self._nx_graph,
                    filter_edge=lambda u, v, k: self._nx_graph[u][v][k].get('type') in edge_types
                )
            else:
                view = self._nx_graph

            # Find simple paths with cutoff
            all_paths = list(nx.all_simple_paths(view, start_id, end_id, cutoff=max_depth))
            return all_paths

        except (nx.NetworkXError, nx.NetworkXNoPath):
            return []

    def _find_paths_bfs(self, start_id: str, end_id: str, max_depth: int,
                      edge_types: Optional[List[str]]) -> List[List[str]]:
        """Find paths using breadth-first search (fallback if NetworkX is unavailable)."""
        if start_id == end_id:
            return [[start_id]]

        paths = []
        queue = deque([(start_id, [start_id], 0)])
        visited = set([start_id])

        while queue:
            node_id, path, depth = queue.popleft()

            if depth >= max_depth:
                continue

            neighbors = self.get_node_neighbors(node_id, edge_types, "outgoing")

            for neighbor_id, _, _ in neighbors:
                if neighbor_id == end_id:
                    paths.append(path + [neighbor_id])
                elif neighbor_id not in visited:
                    visited.add(neighbor_id)
                    queue.append((neighbor_id, path + [neighbor_id], depth + 1))

        return paths

    def execute_query(self, query: GraphQuery) -> Dict[str, Any]:
        """Execute a complex graph query."""
        with self.lock:
            result = {
                'nodes': [],
                'edges': [],
                'metadata': {
                    'query_time': time.time(),
                    'node_count': 0,
                    'edge_count': 0
                }
            }

            # Find starting nodes
            start_nodes = []
            if query.start_nodes:
                # Use specified start nodes
                start_nodes = [node_id for node_id in query.start_nodes if node_id in self.nodes]
            else:
                # Find nodes matching criteria
                start_nodes = self.query_nodes(query.node_criteria)

            if not start_nodes:
                return result

            # Apply activation threshold
            if not query.include_inactive and query.min_activation > 0:
                start_nodes = [node_id for node_id in start_nodes
                              if self.nodes[node_id].activation >= query.min_activation]

            # Build result nodes and edges
            processed_nodes = set()
            processed_edges = set()

            # Process start nodes
            for node_id in start_nodes:
                if node_id not in processed_nodes:
                    processed_nodes.add(node_id)
                    result['nodes'].append(self._node_to_dict(self.nodes[node_id]))

            # Expand graph if needed
            if query.max_depth > 0:
                for start_id in start_nodes:
                    self._expand_query_from_node(
                        start_id, query, processed_nodes, processed_edges, result, 0
                    )

            # Sort results if requested
            if query.sort_by == "activation":
                result['nodes'].sort(key=lambda n: n.get('activation', 0), reverse=True)
            elif query.sort_by == "importance":
                result['nodes'].sort(key=lambda n: n.get('importance', 0), reverse=True)
            elif query.sort_by == "recency":
                result['nodes'].sort(key=lambda n: n.get('last_accessed', 0), reverse=True)

            # Limit results if needed
            if query.max_results > 0:
                result['nodes'] = result['nodes'][:query.max_results]

                # Rebuild edges to only include those connected to returned nodes
                node_ids = {n['id'] for n in result['nodes']}
                result['edges'] = [e for e in result['edges']
                                  if e['source'] in node_ids and e['target'] in node_ids]

            # Update metadata
            result['metadata']['node_count'] = len(result['nodes'])
            result['metadata']['edge_count'] = len(result['edges'])

            return result

    def _expand_query_from_node(self, node_id: str, query: GraphQuery,
                              processed_nodes: Set[str], processed_edges: Set[str],
                              result: Dict[str, Any], depth: int) -> None:
        """Helper method to expand a query from a node recursively."""
        if depth >= query.max_depth:
            return

        # Get neighbors
        neighbors = self.get_node_neighbors(node_id, None, "both")

        for neighbor_id, edge_id, edge_type in neighbors:
            # Check edge criteria
            edge = self.edges[edge_id]
            if query.edge_criteria and not self._match_criteria(edge.properties, query.edge_criteria):
                continue

            # Process edge if not already processed
            if edge_id not in processed_edges:
                processed_edges.add(edge_id)
                result['edges'].append(self._edge_to_dict(edge))

            # Process neighbor node if not already processed
            if neighbor_id not in processed_nodes:
                neighbor = self.nodes[neighbor_id]

                # Check node criteria and activation
                if (query.node_criteria and not self._match_criteria(neighbor.properties, query.node_criteria)):
                    continue

                if not query.include_inactive and neighbor.activation < query.min_activation:
                    continue

                processed_nodes.add(neighbor_id)
                result['nodes'].append(self._node_to_dict(neighbor))

                # Recurse
                self._expand_query_from_node(
                    neighbor_id, query, processed_nodes, processed_edges, result, depth + 1
                )

    def _node_to_dict(self, node: GraphNode) -> Dict[str, Any]:
        """Convert a node to a dictionary representation."""
        return {
            'id': node.node_id,
            'label': node.label,
            'properties': node.properties.copy(),
            'activation': node.activation,
            'importance': node.importance,
            'last_accessed': node.last_accessed,
            'creation_time': node.creation_time
        }

    def _edge_to_dict(self, edge: GraphEdge) -> Dict[str, Any]:
        """Convert an edge to a dictionary representation."""
        return {
            'id': edge.edge_id,
            'source': edge.source_id,
            'target': edge.target_id,
            'type': edge.type,
            'properties': edge.properties.copy(),
            'weight': edge.weight,
            'bidirectional': edge.bidirectional,
            'creation_time': edge.creation_time
        }

    def spread_activation(self, source_nodes: List[str], decay_factor: float = 0.5,
                         max_depth: int = 3) -> Dict[str, float]:
        """Spread activation from source nodes through the graph."""
        with self.lock:
            # Verify nodes exist
            for node_id in source_nodes:
                if node_id not in self.nodes:
                    raise NodeNotFoundError(f"Source node {node_id} not found")

            # Initialize activation
            activation = {}
            for node_id in self.nodes:
                if node_id in source_nodes:
                    activation[node_id] = 1.0  # Full activation for source nodes
                    self.nodes[node_id].activation = 1.0
                else:
                    activation[node_id] = self.nodes[node_id].activation

            # Spread activation with BFS
            queue = deque([(node_id, 0) for node_id in source_nodes])
            visited = set(source_nodes)

            while queue:
                node_id, depth = queue.popleft()

                if depth >= max_depth:
                    continue

                current_activation = activation[node_id]

                # Get outgoing neighbors
                neighbors = self.get_node_neighbors(node_id, None, "outgoing")

                # Calculate activation to spread
                spread_activation = current_activation * decay_factor

                for neighbor_id, edge_id, _ in neighbors:
                    edge = self.edges[edge_id]

                    # Weight by edge weight
                    neighbor_activation = spread_activation * edge.weight

                    # Add to existing activation
                    activation[neighbor_id] = min(1.0, activation[neighbor_id] + neighbor_activation)

                    # Update node activation
                    self.nodes[neighbor_id].activation = activation[neighbor_id]

                    # Continue spreading if not visited
                    if neighbor_id not in visited:
                        visited.add(neighbor_id)
                        queue.append((neighbor_id, depth + 1))

            # Return only non-zero activations
            return {node_id: act for node_id, act in activation.items() if act > 0}

    def get_activated_subgraph(self, threshold: float = 0.2) -> Tuple[List[str], List[str]]:
        """Get the currently activated subgraph."""
        with self.lock:
            # Find activated nodes
            activated_nodes = [node_id for node_id, node in self.nodes.items()
                              if node.activation >= threshold]

            # Find edges between activated nodes
            activated_edges = []
            for node_id in activated_nodes:
                for edge_id in self.outgoing_edges.get(node_id, []):
                    edge = self.edges.get(edge_id)
                    if edge and edge.target_id in activated_nodes:
                        activated_edges.append(edge_id)

            return activated_nodes, activated_edges

    def detect_patterns(self, min_frequency: int = 2, max_size: int = 5) -> List[GraphPattern]:
        """Detect frequent patterns in the graph."""
        with self.lock:
            if self.use_networkx and self._nx_graph is not None and HAS_NUMPY:
                return self._detect_patterns_networkx(min_frequency, max_size)
            else:
                return self._detect_patterns_basic(min_frequency, max_size)

    def _detect_patterns_networkx(self, min_frequency: int, max_size: int) -> List[GraphPattern]:
        """Detect patterns using NetworkX algorithms."""
        try:
            # This is a simplified approach using motifs/subgraphs
            # A production implementation would use a more sophisticated algorithm
            patterns = []

            # Sample subgraphs of different sizes
            for size in range(2, max_size + 1):
                # Find connected components
                for c in nx.connected_components(self._nx_graph.to_undirected()):
                    if len(c) == size:
                        subgraph = self._nx_graph.subgraph(c)
                        pattern_id = f"pattern_{len(patterns)}"

                        # Create pattern
                        pattern = GraphPattern(
                            pattern_id=pattern_id,
                            nodes=list(subgraph.nodes()),
                            edges=[edge[2]['edge_id'] for edge in subgraph.edges(data=True)
                                   if 'edge_id' in edge[2]],
                            frequency=self._count_pattern_occurrences(subgraph),
                            significance=0.5,  # Placeholder
                            embedding=None  # Placeholder
                        )

                        if pattern.frequency >= min_frequency:
                            patterns.append(pattern)

            return patterns

        except Exception as e:
            self.logger.error(f"Error detecting patterns: {e}")
            return []

    def _count_pattern_occurrences(self, pattern_graph: 'nx.Graph') -> int:
        """Count occurrences of a pattern in the graph."""
        # This is a simplified implementation
        # In a production system, this would use subgraph isomorphism algorithms
        return 2  # Placeholder

    def _detect_patterns_basic(self, min_frequency: int, max_size: int) -> List[GraphPattern]:
        """Basic pattern detection fallback."""
        # This is a very simplified approach that just looks for star patterns
        # A production implementation would use more sophisticated algorithms
        patterns = []

        # Look for hub nodes (nodes with many connections)
        for node_id, node in self.nodes.items():
            outgoing = self.outgoing_edges.get(node_id, [])
            incoming = self.incoming_edges.get(node_id, [])

            if len(outgoing) + len(incoming) >= 3:  # Hub threshold
                # Create a star pattern
                pattern_id = f"pattern_{len(patterns)}"

                # Collect connected nodes and edges
                connected_nodes = [node_id]
                pattern_edges = []

                for edge_id in outgoing:
                    edge = self.edges[edge_id]
                    connected_nodes.append(edge.target_id)
                    pattern_edges.append(edge_id)

                # Truncate to max_size
                if len(connected_nodes) > max_size:
                    connected_nodes = connected_nodes[:max_size]
                    pattern_edges = pattern_edges[:max_size-1]

                pattern = GraphPattern(
                    pattern_id=pattern_id,
                    nodes=connected_nodes,
                    edges=pattern_edges,
                    frequency=min_frequency,  # Placeholder
                    significance=0.5,  # Placeholder
                    embedding=None  # Placeholder
                )

                patterns.append(pattern)

        return patterns

    def clear(self) -> None:
        """Clear all nodes and edges from the graph."""
        with self.lock:
            self.nodes.clear()
            self.edges.clear()
            self.outgoing_edges.clear()
            self.incoming_edges.clear()
            self.edges_by_type.clear()
            self.nodes_by_label.clear()

            if self.use_networkx and self._nx_graph is not None:
                self._nx_graph.clear()

            self.logger.info("Graph cleared")

    def serialize(self) -> Dict[str, Any]:
        """Serialize the graph to a dictionary."""
        with self.lock:
            data = {
                'nodes': [],
                'edges': [],
                'metadata': {
                    'timestamp': time.time(),
                    'node_count': len(self.nodes),
                    'edge_count': len(self.edges)
                }
            }

            # Serialize nodes
            for node_id, node in self.nodes.items():
                node_data = {
                    'id': node.node_id,
                    'label': node.label,
                    'properties': node.properties,
                    'activation': node.activation,
                    'importance': node.importance,
                    'creation_time': node.creation_time,
                    'last_accessed': node.last_accessed
                }

                # Handle embedding if present
                if node.embedding is not None and HAS_NUMPY:
                    if isinstance(node.embedding, np.ndarray):
                        node_data['embedding'] = node.embedding.tolist()
                    else:
                        node_data['embedding'] = node.embedding

                data['nodes'].append(node_data)

            # Serialize edges
            for edge_id, edge in self.edges.items():
                edge_data = {
                    'id': edge.edge_id,
                    'source': edge.source_id,
                    'target': edge.target_id,
                    'type': edge.type,
                    'properties': edge.properties,
                    'weight': edge.weight,
                    'bidirectional': edge.bidirectional,
                    'creation_time': edge.creation_time
                }

                data['edges'].append(edge_data)

            return data

    def deserialize(self, data: Dict[str, Any]) -> None:
        """Load the graph from a serialized representation."""
        with self.lock:
            # Clear existing data
            self.clear()

            # Load nodes first
            for node_data in data.get('nodes', []):
                node_id = node_data['id']

                # Create embedding if present
                embedding = None
                if 'embedding' in node_data and HAS_NUMPY:
                    embedding_data = node_data['embedding']
                    if embedding_data:
                        embedding = np.array(embedding_data)

                # Create node
                node = GraphNode(
                    node_id=node_id,
                    label=node_data['label'],
                    properties=node_data.get('properties', {}),
                    embedding=embedding,
                    activation=node_data.get('activation', 0.0),
                    last_accessed=node_data.get('last_accessed', time.time()),
                    importance=node_data.get('importance', 0.5),
                    creation_time=node_data.get('creation_time', time.time())
                )

                # Add to storage
                self.nodes[node_id] = node
                self.nodes_by_label[node.label].append(node_id)

                # Add to NetworkX if enabled
                if self.use_networkx and self._nx_graph is not None:
                    attrs = dict(node.properties)
                    attrs.update({
                        'label': node.label,
                        'activation': node.activation,
                        'importance': node.importance
                    })
                    self._nx_graph.add_node(node_id, **attrs)

            # Load edges
            for edge_data in data.get('edges', []):
                edge_id = edge_data['id']
                source_id = edge_data['source']
                target_id = edge_data['target']

                # Skip if source or target doesn't exist
                if source_id not in self.nodes or target_id not in self.nodes:
                    self.logger.warning(f"Skipping edge {edge_id}: source or target not found")
                    continue

                # Create edge
                edge = GraphEdge(
                    edge_id=edge_id,
                    source_id=source_id,
                    target_id=target_id,
                    type=edge_data['type'],
                    properties=edge_data.get('properties', {}),
                    weight=edge_data.get('weight', 1.0),
                    bidirectional=edge_data.get('bidirectional', False),
                    creation_time=edge_data.get('creation_time', time.time())
                )

                # Add to storage (avoid using add_edge to prevent duplicate bidirectional edges)
                self.edges[edge_id] = edge
                self.outgoing_edges[source_id].append(edge_id)
                self.incoming_edges[target_id].append(edge_id)
                self.edges_by_type[edge.type].append(edge_id)

                # Add to NetworkX if enabled
                if self.use_networkx and self._nx_graph is not None:
                    attrs = dict(edge.properties)
                    attrs.update({
                        'edge_id': edge_id,
                        'type': edge.type,
                        'weight': edge.weight,
                        'bidirectional': edge.bidirectional
                    })
                    self._nx_graph.add_edge(source_id, target_id, **attrs)

            self.logger.info(f"Deserialized graph with {len(self.nodes)} nodes and {len(self.edges)} edges")

    def close(self) -> None:
        """Close the graph engine and release any resources."""
        # For in-memory graph, there's not much to close
        # This method is primarily for interface consistency with other implementations
        self.logger.info("InMemoryGraphEngine closed")


# -----------------------------------------------------------------------------
# Neo4j Graph Engine Implementation
# -----------------------------------------------------------------------------

class Neo4jGraphEngine(AbstractGraphEngine):
    """
    Neo4j-based implementation of the graph engine.

    This implementation uses Neo4j as a backend store, providing scalability
    for large graphs while maintaining the cognitive capabilities.
    """

    def __init__(self, uri: str, user: str, password: str, database: str = "neo4j",
                decay_rate: float = 0.1, use_apoc: bool = False):
        """
        Initialize the Neo4j graph engine.

        Args:
            uri: Neo4j connection URI (e.g., "bolt://localhost:7687")
            user: Neo4j username
            password: Neo4j password
            database: Neo4j database name
            decay_rate: Default activation decay rate
            use_apoc: Whether to use APOC procedures (must be installed in Neo4j)
        """
        if not HAS_NEO4J:
            raise ImportError("Neo4j driver not available. Please install it with 'pip install neo4j'.")

        self.logger = logging.getLogger("Neo4jGraphEngine")
        self.uri = uri
        self.user = user
        self.password = password
        self.database = database
        self.decay_rate = decay_rate
        self.use_apoc = use_apoc

        # Initialize driver
        self.driver = GraphDatabase.driver(uri, auth=(user, password))

        # Test connection
        self._test_connection()

        # Ensure constraints and indices
        self._setup_schema()

        # Local caches to avoid repeated database lookups
        self._node_cache: Dict[str, GraphNode] = {}
        self._edge_cache: Dict[str, GraphEdge] = {}

        # Concurrency control
        self.lock = threading.RLock()

        self.logger.info(f"Neo4jGraphEngine initialized (connected to {uri}, database={database})")

    def _test_connection(self) -> None:
        """Test the Neo4j connection."""
        try:
            with self.driver.session(database=self.database) as session:
                result = session.run("RETURN 1 AS test")
                test_value = result.single()["test"]
                if test_value != 1:
                    raise ConnectionError("Neo4j connection test failed")
        except Exception as e:
            self.logger.error(f"Failed to connect to Neo4j: {e}")
            raise ConnectionError(f"Could not connect to Neo4j: {e}")

    def _setup_schema(self) -> None:
        """Set up constraints and indices for optimized performance."""
        try:
            with self.driver.session(database=self.database) as session:
                # Create constraints
                session.run("CREATE CONSTRAINT node_id IF NOT EXISTS FOR (n:Node) REQUIRE n.node_id IS UNIQUE")
                session.run("CREATE CONSTRAINT edge_id IF NOT EXISTS FOR ()-[r:RELATES]-() REQUIRE r.edge_id IS UNIQUE")

                # Create indices
                session.run("CREATE INDEX node_label IF NOT EXISTS FOR (n:Node) ON (n.label)")
                session.run("CREATE INDEX node_activation IF NOT EXISTS FOR (n:Node) ON (n.activation)")
                session.run("CREATE INDEX edge_type IF NOT EXISTS FOR ()-[r:RELATES]-() ON (r.type)")
        except Exception as e:
            self.logger.warning(f"Failed to set up schema: {e}. Continuing without constraints.")

    def add_node(self, node_id: Optional[str] = None, label: str = "Node",
                properties: Optional[Dict[str, Any]] = None) -> str:
        """Add a node to the graph."""
        with self.lock:
            # Generate ID if not provided
            node_id = node_id or str(uuid.uuid4())
            properties = properties or {}

            # Create node record
            cypher = """
            CREATE (n:Node {node_id: $node_id, label: $label,
                        activation: $activation,
                        importance: $importance,
                        creation_time: $creation_time,
                        last_accessed: $last_accessed})
            SET n += $properties
            RETURN n.node_id AS node_id
            """

            params = {
                "node_id": node_id,
                "label": label,
                "activation": 0.0,
                "importance": 0.5,
                "creation_time": time.time(),
                "last_accessed": time.time(),
                "properties": properties
            }

            try:
                with self.driver.session(database=self.database) as session:
                    result = session.run(cypher, params)
                    created_id = result.single()["node_id"]

                    # Create and cache the node
                    node = GraphNode(
                        node_id=created_id,
                        label=label,
                        properties=properties.copy(),
                        activation=0.0,
                        importance=0.5
                    )
                    self._node_cache[created_id] = node

                    self.logger.debug(f"Added node {created_id} with label '{label}'")
                    return created_id
            except Exception as e:
                self.logger.error(f"Error adding node: {e}")
                raise GraphEngineError(f"Failed to add node: {e}")

    def get_node(self, node_id: str) -> GraphNode:
        """Retrieve a node by ID."""
        with self.lock:
            # Check cache first
            if node_id in self._node_cache:
                node = self._node_cache[node_id]
                node.access()  # Update activation and last_accessed
                self._update_node_activation(node_id, node.activation, node.last_accessed)
                return node

            # Query the database
            cypher = """
            MATCH (n:Node {node_id: $node_id})
            SET n.activation = 1.0, n.last_accessed = $current_time
            RETURN n.node_id AS id, n.label AS label,
                   n.activation AS activation, n.importance AS importance,
                   n.creation_time AS creation_time, n.last_accessed AS last_accessed,
                   properties(n) AS properties
            """

            params = {
                "node_id": node_id,
                "current_time": time.time()
            }

            try:
                with self.driver.session(database=self.database) as session:
                    result = session.run(cypher, params)
                    record = result.single()

                    if not record:
                        raise NodeNotFoundError(f"Node {node_id} not found")

                    # Extract properties, excluding known attributes
                    props = dict(record["properties"])
                    for key in ["node_id", "label", "activation", "importance", "creation_time", "last_accessed"]:
                        if key in props:
                            del props[key]

                    # Create and cache node
                    node = GraphNode(
                        node_id=record["id"],
                        label=record["label"],
                        properties=props,
                        activation=record["activation"],
                        last_accessed=record["last_accessed"],
                        importance=record["importance"],
                        creation_time=record["creation_time"]
                    )

                    self._node_cache[node_id] = node
                    return node
            except Exception as e:
                if isinstance(e, NodeNotFoundError):
                    raise
                self.logger.error(f"Error retrieving node: {e}")
                raise GraphEngineError(f"Failed to retrieve node: {e}")

    def _update_node_activation(self, node_id: str, activation: float, last_accessed: float) -> None:
        """Update a node's activation and last_accessed time in the database."""
        cypher = """
        MATCH (n:Node {node_id: $node_id})
        SET n.activation = $activation, n.last_accessed = $last_accessed
        """

        params = {
            "node_id": node_id,
            "activation": activation,
            "last_accessed": last_accessed
        }

        try:
            with self.driver.session(database=self.database) as session:
                session.run(cypher, params)
        except Exception as e:
            self.logger.warning(f"Failed to update node activation: {e}")

    def update_node(self, node_id: str, properties: Dict[str, Any]) -> None:
        """Update a node's properties."""
        with self.lock:
            # Update cache if present
            if node_id in self._node_cache:
                node = self._node_cache[node_id]
                node.properties.update(properties)
                node.last_accessed = time.time()

            # Update in database
            cypher = """
            MATCH (n:Node {node_id: $node_id})
            SET n += $properties, n.last_accessed = $current_time
            """

            params = {
                "node_id": node_id,
                "properties": properties,
                "current_time": time.time()
            }

            try:
                with self.driver.session(database=self.database) as session:
                    result = session.run(cypher, params)
                    if result.consume().counters.nodes_created == 0 and \
                       result.consume().counters.properties_set == 0:
                        raise NodeNotFoundError(f"Node {node_id} not found")

                    self.logger.debug(f"Updated node {node_id}")
            except Exception as e:
                if isinstance(e, NodeNotFoundError):
                    raise
                self.logger.error(f"Error updating node: {e}")
                raise GraphEngineError(f"Failed to update node: {e}")

    def delete_node(self, node_id: str) -> None:
        """Delete a node and its connected edges."""
        with self.lock:
            # Remove from cache
            if node_id in self._node_cache:
                del self._node_cache[node_id]

            # Delete from database
            cypher = """
            MATCH (n:Node {node_id: $node_id})
            OPTIONAL MATCH (n)-[r]-()
            DELETE r, n
            """

            params = {"node_id": node_id}

            try:
                with self.driver.session(database=self.database) as session:
                    result = session.run(cypher, params)
                    if result.consume().counters.nodes_deleted == 0:
                        raise NodeNotFoundError(f"Node {node_id} not found")

                    # Clean up edge cache - remove edges connected to this node
                    edge_ids_to_remove = []
                    for edge_id, edge in self._edge_cache.items():
                        if edge.source_id == node_id or edge.target_id == node_id:
                            edge_ids_to_remove.append(edge_id)

                    for edge_id in edge_ids_to_remove:
                        if edge_id in self._edge_cache:
                            del self._edge_cache[edge_id]

                    self.logger.debug(f"Deleted node {node_id} and its connected edges")
            except Exception as e:
                if isinstance(e, NodeNotFoundError):
                    raise
                self.logger.error(f"Error deleting node: {e}")
                raise GraphEngineError(f"Failed to delete node: {e}")

    def add_edge(self, source_id: str, target_id: str, edge_type: str,
                edge_id: Optional[str] = None, properties: Optional[Dict[str, Any]] = None,
                weight: float = 1.0, bidirectional: bool = False) -> str:
        """Add an edge between two nodes."""
        with self.lock:
            # Generate ID if not provided
            edge_id = edge_id or str(uuid.uuid4())
            properties = properties or {}

            # Create edge in database
            cypher = """
            MATCH (s:Node {node_id: $source_id})
            MATCH (t:Node {node_id: $target_id})
            CREATE (s)-[r:RELATES]->(t)
            SET r.edge_id = $edge_id, r.type = $edge_type, r.weight = $weight,
                r.bidirectional = $bidirectional, r.creation_time = $creation_time
            SET r += $properties
            RETURN r.edge_id AS edge_id
            """

            params = {
                "source_id": source_id,
                "target_id": target_id,
                "edge_id": edge_id,
                "edge_type": edge_type,
                "weight": weight,
                "bidirectional": bidirectional,
                "creation_time": time.time(),
                "properties": properties
            }

            try:
                with self.driver.session(database=self.database) as session:
                    result = session.run(cypher, params)
                    created_id = result.single()["edge_id"]

                    # Create and cache the edge
                    edge = GraphEdge(
                        edge_id=created_id,
                        source_id=source_id,
                        target_id=target_id,
                        type=edge_type,
                        properties=properties.copy(),
                        weight=weight,
                        bidirectional=bidirectional
                    )
                    self._edge_cache[created_id] = edge

                    # If bidirectional, add inverse edge
                    if bidirectional and source_id != target_id:
                        inverse_id = f"{edge_id}_inverse"
                        inverse_cypher = """
                        MATCH (s:Node {node_id: $target_id})
                        MATCH (t:Node {node_id: $source_id})
                        CREATE (s)-[r:RELATES]->(t)
                        SET r.edge_id = $edge_id, r.type = $edge_type, r.weight = $weight,
                            r.bidirectional = $bidirectional, r.creation_time = $creation_time,
                            r.inverse_of = $original_edge_id
                        SET r += $properties
                        RETURN r.edge_id AS edge_id
                        """

                        inverse_params = {
                            "source_id": source_id,
                            "target_id": target_id,
                            "edge_id": inverse_id,
                            "edge_type": edge_type,
                            "weight": weight,
                            "bidirectional": bidirectional,
                            "creation_time": time.time(),
                            "properties": properties,
                            "original_edge_id": edge_id
                        }

                        result = session.run(inverse_cypher, inverse_params)
                        inverse_edge = GraphEdge(
                            edge_id=inverse_id,
                            source_id=target_id,
                            target_id=source_id,
                            type=edge_type,
                            properties={**properties, "inverse_of": edge_id},
                            weight=weight,
                            bidirectional=True
                        )
                        self._edge_cache[inverse_id] = inverse_edge

                    self.logger.debug(f"Added edge {created_id} from {source_id} to {target_id} with type '{edge_type}'")
                    return created_id
            except Exception as e:
                self.logger.error(f"Error adding edge: {e}")
                raise GraphEngineError(f"Failed to add edge: {e}")

    def get_edge(self, edge_id: str) -> GraphEdge:
        """Retrieve an edge by ID."""
        with self.lock:
            # Check cache first
            if edge_id in self._edge_cache:
                return self._edge_cache[edge_id]

            # Query the database
            cypher = """
            MATCH (s)-[r:RELATES]->(t)
            WHERE r.edge_id = $edge_id
            RETURN r.edge_id AS id, s.node_id AS source_id, t.node_id AS target_id,
                   r.type AS type, r.weight AS weight, r.bidirectional AS bidirectional,
                   r.creation_time AS creation_time, properties(r) AS properties
            """

            params = {"edge_id": edge_id}

            try:
                with self.driver.session(database=self.database) as session:
                    result = session.run(cypher, params)
                    record = result.single()

                    if not record:
                        raise EdgeNotFoundError(f"Edge {edge_id} not found")

                    # Extract properties, excluding known attributes
                    props = dict(record["properties"])
                    for key in ["edge_id", "type", "weight", "bidirectional", "creation_time"]:
                        if key in props:
                            del props[key]

                    # Create and cache edge
                    edge = GraphEdge(
                        edge_id=record["id"],
                        source_id=record["source_id"],
                        target_id=record["target_id"],
                        type=record["type"],
                        properties=props,
                        weight=record["weight"],
                        bidirectional=record["bidirectional"],
                        creation_time=record["creation_time"]
                    )

                    self._edge_cache[edge_id] = edge
                    return edge
            except Exception as e:
                if isinstance(e, EdgeNotFoundError):
                    raise
                self.logger.error(f"Error retrieving edge: {e}")
                raise GraphEngineError(f"Failed to retrieve edge: {e}")

    def update_edge(self, edge_id: str, properties: Dict[str, Any]) -> None:
        """Update an edge's properties."""
        with self.lock:
            # Update cache if present
            if edge_id in self._edge_cache:
                edge = self._edge_cache[edge_id]
                edge.properties.update(properties)

            # Update in database
            cypher = """
            MATCH ()-[r:RELATES]->()
            WHERE r.edge_id = $edge_id
            SET r += $properties
            """

            params = {
                "edge_id": edge_id,
                "properties": properties
            }

            try:
                with self.driver.session(database=self.database) as session:
                    result = session.run(cypher, params)
                    if result.consume().counters.properties_set == 0:
                        raise EdgeNotFoundError(f"Edge {edge_id} not found")

                    # Update weight in cache if it changed
                    if "weight" in properties and edge_id in self._edge_cache:
                        self._edge_cache[edge_id].weight = properties["weight"]

                    self.logger.debug(f"Updated edge {edge_id}")
            except Exception as e:
                if isinstance(e, EdgeNotFoundError):
                    raise
                self.logger.error(f"Error updating edge: {e}")
                raise GraphEngineError(f"Failed to update edge: {e}")

    def delete_edge(self, edge_id: str) -> None:
        """Delete an edge."""
        with self.lock:
            # Remove from cache
            if edge_id in self._edge_cache:
                edge = self._edge_cache[edge_id]
                bidirectional = edge.bidirectional
                del self._edge_cache[edge_id]

                # Handle bidirectional edge
                if bidirectional:
                    inverse_id = f"{edge_id}_inverse"
                    if inverse_id in self._edge_cache:
                        del self._edge_cache[inverse_id]

            # Delete from database
            cypher = """
            MATCH ()-[r:RELATES]->()
            WHERE r.edge_id = $edge_id
            DELETE r
            """

            params = {"edge_id": edge_id}

            try:
                with self.driver.session(database=self.database) as session:
                    result = session.run(cypher, params)
                    if result.consume().counters.relationships_deleted == 0:
                        raise EdgeNotFoundError(f"Edge {edge_id} not found")

                    # Delete inverse edge if it exists
                    inverse_id = f"{edge_id}_inverse"
                    session.run(cypher, {"edge_id": inverse_id})

                    self.logger.debug(f"Deleted edge {edge_id}")
            except Exception as e:
                if isinstance(e, EdgeNotFoundError):
                    raise
                self.logger.error(f"Error deleting edge: {e}")
                raise GraphEngineError(f"Failed to delete edge: {e}")

    def get_node_neighbors(self, node_id: str, edge_types: Optional[List[str]] = None,
                          direction: str = "outgoing") -> List[Tuple[str, str, str]]:
        """Get neighboring nodes connected to a node."""
        with self.lock:
            # Prepare Cypher query based on direction
            if direction == "outgoing":
                cypher = """
                MATCH (n:Node {node_id: $node_id})-[r:RELATES]->(neighbor:Node)
                WHERE $edge_types IS NULL OR r.type IN $edge_types
                RETURN neighbor.node_id AS neighbor_id, r.edge_id AS edge_id, r.type AS edge_type
                """
            elif direction == "incoming":
                cypher = """
                MATCH (n:Node {node_id: $node_id})<-[r:RELATES]-(neighbor:Node)
                WHERE $edge_types IS NULL OR r.type IN $edge_types
                RETURN neighbor.node_id AS neighbor_id, r.edge_id AS edge_id, r.type AS edge_type
                """
            elif direction == "both":
                cypher = """
                MATCH (n:Node {node_id: $node_id})-[r:RELATES]-(neighbor:Node)
                WHERE $edge_types IS NULL OR r.type IN $edge_types
                RETURN neighbor.node_id AS neighbor_id, r.edge_id AS edge_id, r.type AS edge_type
                """
            else:
                raise ValueError(f"Invalid direction: {direction}. Must be 'outgoing', 'incoming', or 'both'.")

            params = {
                "node_id": node_id,
                "edge_types": edge_types
            }

            try:
                with self.driver.session(database=self.database) as session:
                    result = session.run(cypher, params)
                    neighbors = []

                    for record in result:
                        neighbors.append((
                            record["neighbor_id"],
                            record["edge_id"],
                            record["edge_type"]
                        ))

                    # Activate the node
                    self._update_node_activation(node_id, 1.0, time.time())

                    return neighbors
            except Exception as e:
                self.logger.error(f"Error getting node neighbors: {e}")
                raise GraphEngineError(f"Failed to get node neighbors: {e}")

    def query_nodes(self, criteria: Dict[str, Any]) -> List[str]:
        """Find nodes matching the specified criteria."""
        with self.lock:
            # Prepare WHERE clause from criteria
            where_clauses = []
            params = {}

            for i, (key, value) in enumerate(criteria.items()):
                param_name = f"param{i}"

                if isinstance(value, dict):
                    # Handle comparison operators
                    for op, op_value in value.items():
                        op_param = f"{param_name}_{op}"

                        if op == '$gt':
                            where_clauses.append(f"n.{key} > ${op_param}")
                        elif op == '$gte':
                            where_clauses.append(f"n.{key} >= ${op_param}")
                        elif op == '$lt':
                            where_clauses.append(f"n.{key} < ${op_param}")
                        elif op == '$lte':
                            where_clauses.append(f"n.{key} <= ${op_param}")
                        elif op == '$ne':
                            where_clauses.append(f"n.{key} <> ${op_param}")
                        elif op == '$in':
                            where_clauses.append(f"n.{key} IN ${op_param}")
                        else:
                            # Unknown operator, skip
                            continue

                        params[op_param] = op_value
                else:
                    # Simple equality
                    where_clauses.append(f"n.{key} = ${param_name}")
                    params[param_name] = value

            # Build query
            where_clause = " AND ".join(where_clauses) if where_clauses else "TRUE"

            cypher = f"""
            MATCH (n:Node)
            WHERE {where_clause}
            RETURN n.node_id AS node_id
            """

            try:
                with self.driver.session(database=self.database) as session:
                    result = session.run(cypher, params)
                    node_ids = [record["node_id"] for record in result]
                    return node_ids
            except Exception as e:
                self.logger.error(f"Error querying nodes: {e}")
                raise GraphEngineError(f"Failed to query nodes: {e}")

    def query_edges(self, criteria: Dict[str, Any]) -> List[str]:
        """Find edges matching the specified criteria."""
        with self.lock:
            # Prepare WHERE clause from criteria
            where_clauses = []
            params = {}

            for i, (key, value) in enumerate(criteria.items()):
                param_name = f"param{i}"

                if isinstance(value, dict):
                    # Handle comparison operators
                    for op, op_value in value.items():
                        op_param = f"{param_name}_{op}"

                        if op == '$gt':
                            where_clauses.append(f"r.{key} > ${op_param}")
                        elif op == '$gte':
                            where_clauses.append(f"r.{key} >= ${op_param}")
                        elif op == '$lt':
                            where_clauses.append(f"r.{key} < ${op_param}")
                        elif op == '$lte':
                            where_clauses.append(f"r.{key} <= ${op_param}")
                        elif op == '$ne':
                            where_clauses.append(f"r.{key} <> ${op_param}")
                        elif op == '$in':
                            where_clauses.append(f"r.{key} IN ${op_param}")
                        else:
                            # Unknown operator, skip
                            continue

                        params[op_param] = op_value
                else:
                    # Simple equality
                    where_clauses.append(f"r.{key} = ${param_name}")
                    params[param_name] = value

            # Build query
            where_clause = " AND ".join(where_clauses) if where_clauses else "TRUE"

            cypher = f"""
            MATCH ()-[r:RELATES]->()
            WHERE {where_clause}
            RETURN r.edge_id AS edge_id
            """

            try:
                with self.driver.session(database=self.database) as session:
                    result = session.run(cypher, params)
                    edge_ids = [record["edge_id"] for record in result]
                    return edge_ids
            except Exception as e:
                self.logger.error(f"Error querying edges: {e}")
                raise GraphEngineError(f"Failed to query edges: {e}")

    def find_paths(self, start_id: str, end_id: str, max_depth: int = 3,
                  edge_types: Optional[List[str]] = None) -> List[List[str]]:
        """Find paths between two nodes."""
        with self.lock:
            # Verify nodes exist
            if not self._node_exists(start_id):
                raise NodeNotFoundError(f"Start node {start_id} not found")
            if not self._node_exists(end_id):
                raise NodeNotFoundError(f"End node {end_id} not found")

            # Prepare Cypher query
            type_filter = ""
            if edge_types:
                type_filter = "AND r.type IN $edge_types"

            # Use shortest path algorithm if available, otherwise variable length path
            if self.use_apoc:
                cypher = f"""
                MATCH (start:Node {{node_id: $start_id}}), (end:Node {{node_id: $end_id}})
                CALL apoc.algo.allSimplePaths(start, end, 'RELATES>', $max_depth) YIELD path
                WITH path, [node IN nodes(path) | node.node_id] AS node_ids
                WHERE ALL(r IN relationships(path) WHERE true {type_filter})
                RETURN node_ids
                """
            else:
                cypher = f"""
                MATCH p = (start:Node {{node_id: $start_id}})-[r:RELATES*..{max_depth}]->(end:Node {{node_id: $end_id}})
                WHERE ALL(rel IN relationships(p) WHERE true {type_filter})
                RETURN [node IN nodes(p) | node.node_id] AS node_ids
                """

            params = {
                "start_id": start_id,
                "end_id": end_id,
                "max_depth": max_depth,
                "edge_types": edge_types
            }

            try:
                with self.driver.session(database=self.database) as session:
                    result = session.run(cypher, params)
                    paths = [record["node_ids"] for record in result]
                    return paths
            except Exception as e:
                self.logger.error(f"Error finding paths: {e}")
                raise GraphEngineError(f"Failed to find paths: {e}")

    def _node_exists(self, node_id: str) -> bool:
        """Check if a node exists in the database."""
        # Check cache first
        if node_id in self._node_cache:
            return True

        # Query database
        cypher = "MATCH (n:Node {node_id: $node_id}) RETURN count(n) AS count"
        params = {"node_id": node_id}

        try:
            with self.driver.session(database=self.database) as session:
                result = session.run(cypher, params)
                return result.single()["count"] > 0
        except Exception as e:
            self.logger.error(f"Error checking if node exists: {e}")
            return False

    def execute_query(self, query: GraphQuery) -> Dict[str, Any]:
        """Execute a complex graph query."""
        with self.lock:
            result = {
                'nodes': [],
                'edges': [],
                'metadata': {
                    'query_time': time.time(),
                    'node_count': 0,
                    'edge_count': 0
                }
            }

            # Prepare node criteria
            node_where_clauses = []
            node_params = {}
            for i, (key, value) in enumerate(query.node_criteria.items()):
                param_name = f"n_param{i}"
                node_where_clauses.append(f"n.{key} = ${param_name}")
                node_params[param_name] = value

            # Prepare edge criteria
            edge_where_clauses = []
            edge_params = {}
            for i, (key, value) in enumerate(query.edge_criteria.items()):
                param_name = f"e_param{i}"
                edge_where_clauses.append(f"r.{key} = ${param_name}")
                edge_params[param_name] = value

            # Combine all parameters
            all_params = {**node_params, **edge_params}
            all_params.update({
                "start_nodes": query.start_nodes,
                "max_depth": query.max_depth,
                "min_activation": query.min_activation
            })

            # Build query
            node_where = " AND ".join(node_where_clauses) if node_where_clauses else "TRUE"
            edge_where = " AND ".join(edge_where_clauses) if edge_where_clauses else "TRUE"

            activation_filter = ""
            if not query.include_inactive and query.min_activation > 0:
                activation_filter = f"AND n.activation >= {query.min_activation}"

            # Execute query based on complexity
            try:
                with self.driver.session(database=self.database) as session:
                    # Strategy depends on whether start nodes are specified
                    if query.start_nodes and len(query.start_nodes) > 0:
                        # Exploration from start nodes
                        cypher = f"""
                        MATCH (start:Node)
                        WHERE start.node_id IN $start_nodes
                        OPTIONAL MATCH p = (start)-[r:RELATES*..{query.max_depth}]-(n:Node)
                        WHERE {node_where} AND ALL(rel IN relationships(p) WHERE {edge_where})
                        {activation_filter}
                        WITH DISTINCT n, r
                        RETURN collect(DISTINCT {{
                            id: n.node_id,
                            label: n.label,
                            properties: properties(n),
                            activation: n.activation,
                            importance: n.importance
                        }}) AS nodes,
                        collect(DISTINCT {{
                            id: r.edge_id,
                            source: startNode(r).node_id,
                            target: endNode(r).node_id,
                            type: r.type,
                            properties: properties(r)
                        }}) AS edges
                        """
                    else:
                        # Query based on criteria
                        cypher = f"""
                        MATCH (n:Node)
                        WHERE {node_where} {activation_filter}
                        WITH n
                        LIMIT {query.max_results}
                        OPTIONAL MATCH (n)-[r:RELATES]-(m:Node)
                        WHERE {edge_where}
                        RETURN collect(DISTINCT {{
                            id: n.node_id,
                            label: n.label,
                            properties: properties(n),
                            activation: n.activation,
                            importance: n.importance
                        }}) AS nodes,
                        collect(DISTINCT {{
                            id: r.edge_id,
                            source: startNode(r).node_id,
                            target: endNode(r).node_id,
                            type: r.type,
                            properties: properties(r)
                        }}) AS edges
                        """

                    result_record = session.run(cypher, all_params).single()

                    if result_record:
                        result['nodes'] = result_record["nodes"]
                        result['edges'] = result_record["edges"]

                    # Update metadata
                    result['metadata']['node_count'] = len(result['nodes'])
                    result['metadata']['edge_count'] = len(result['edges'])

                    return result
            except Exception as e:
                self.logger.error(f"Error executing query: {e}")
                raise GraphEngineError(f"Failed to execute query: {e}")

    def spread_activation(self, source_nodes: List[str], decay_factor: float = 0.5,
                         max_depth: int = 3) -> Dict[str, float]:
        """Spread activation from source nodes through the graph."""
        with self.lock:
            # Verify nodes exist
            for node_id in source_nodes:
                if not self._node_exists(node_id):
                    raise NodeNotFoundError(f"Source node {node_id} not found")

            # Clear cache to ensure fresh activation values
            self._node_cache.clear()

            # First, set full activation for source nodes
            activate_cypher = """
            MATCH (n:Node)
            WHERE n.node_id IN $source_nodes
            SET n.activation = 1.0, n.last_accessed = $current_time
            """

            activation_results = {}

            try:
                with self.driver.session(database=self.database) as session:
                    # Set initial activation
                    session.run(activate_cypher, {
                        "source_nodes": source_nodes,
                        "current_time": time.time()
                    })

                    # Use different strategies depending on Neo4j version and extensions
                    if self.use_apoc:
                        # Use APOC for more efficient spreading
                        return self._spread_activation_apoc(source_nodes, decay_factor, max_depth)
                    else:
                        # Fallback to manual Cypher approach
                        return self._spread_activation_cypher(source_nodes, decay_factor, max_depth)
            except Exception as e:
                self.logger.error(f"Error spreading activation: {e}")
                raise GraphEngineError(f"Failed to spread activation: {e}")

    def _spread_activation_apoc(self, source_nodes: List[str], decay_factor: float,
                              max_depth: int) -> Dict[str, float]:
        """Spread activation using APOC procedures."""
        # This would use APOC's graph algorithms for more efficient propagation
        # Simplified version here
        cypher = """
        MATCH (start:Node)
        WHERE start.node_id IN $source_nodes
        CALL apoc.path.expandConfig(start, {
            relationshipFilter: 'RELATES>',
            minLevel: 1,
            maxLevel: $max_depth
        }) YIELD path
        WITH nodes(path) AS nodes, relationships(path) AS rels
        UNWIND nodes AS n
        WITH DISTINCT n,
             1.0 * (($decay_factor) ^ length(
                 shortestPath((start:Node)-[*]->(n))
             )) AS activation
        WHERE start.node_id IN $source_nodes
        SET n.activation = CASE
            WHEN n.activation < activation THEN activation
            ELSE n.activation
        END
        RETURN n.node_id AS node_id, n.activation AS activation
        """

        params = {
            "source_nodes": source_nodes,
            "max_depth": max_depth,
            "decay_factor": decay_factor
        }

        with self.driver.session(database=self.database) as session:
            result = session.run(cypher, params)
            activation_dict = {record["node_id"]: record["activation"] for record in result}
            return activation_dict

    def _spread_activation_cypher(self, source_nodes: List[str], decay_factor: float,
                               max_depth: int) -> Dict[str, float]:
        """Spread activation using pure Cypher."""
        # Iterative approach for Neo4j without APOC
        activation_dict = {node_id: 1.0 for node_id in source_nodes}

        with self.driver.session(database=self.database) as session:
            for depth in range(1, max_depth + 1):
                # Calculate decay for this level
                level_decay = decay_factor ** depth

                # Spread to neighbors at this depth
                cypher = """
                MATCH (start:Node)-[r:RELATES*EXACTLY {depth}]->(n:Node)
                WHERE start.node_id IN $source_nodes
                WITH DISTINCT n, $level_decay * r[-1].weight AS new_activation
                WHERE new_activation > n.activation
                SET n.activation = new_activation, n.last_accessed = $current_time
                RETURN n.node_id AS node_id, n.activation AS activation
                """

                params = {
                    "source_nodes": source_nodes,
                    "depth": depth,
                    "level_decay": level_decay,
                    "current_time": time.time()
                }

                result = session.run(cypher, params)
                for record in result:
                    activation_dict[record["node_id"]] = record["activation"]

            # Retrieve all activated nodes
            final_cypher = """
            MATCH (n:Node)
            WHERE n.activation > 0
            RETURN n.node_id AS node_id, n.activation AS activation
            """

            result = session.run(final_cypher)
            return {record["node_id"]: record["activation"] for record in result}

    def get_activated_subgraph(self, threshold: float = 0.2) -> Tuple[List[str], List[str]]:
        """Get the currently activated subgraph."""
        with self.lock:
            # Query for activated nodes
            nodes_cypher = """
            MATCH (n:Node)
            WHERE n.activation >= $threshold
            RETURN n.node_id AS node_id
            """

            # Query for edges between activated nodes
            edges_cypher = """
            MATCH (n:Node)-[r:RELATES]->(m:Node)
            WHERE n.activation >= $threshold AND m.activation >= $threshold
            RETURN r.edge_id AS edge_id
            """

            params = {"threshold": threshold}

            try:
                with self.driver.session(database=self.database) as session:
                    # Get activated nodes
                    nodes_result = session.run(nodes_cypher, params)
                    activated_nodes = [record["node_id"] for record in nodes_result]

                    # Get edges between activated nodes
                    edges_result = session.run(edges_cypher, params)
                    activated_edges = [record["edge_id"] for record in edges_result]

                    return activated_nodes, activated_edges
            except Exception as e:
                self.logger.error(f"Error getting activated subgraph: {e}")
                raise GraphEngineError(f"Failed to get activated subgraph: {e}")

    def detect_patterns(self, min_frequency: int = 2, max_size: int = 5) -> List[GraphPattern]:
        """Detect frequent patterns in the graph."""
        with self.lock:
            # Pattern detection is complex - this is a simplified implementation
            # In a production system, more sophisticated algorithms would be used

            # Use APOC if available
            if self.use_apoc:
                return self._detect_patterns_apoc(min_frequency, max_size)
            else:
                return self._detect_patterns_basic(min_frequency, max_size)

    def _detect_patterns_apoc(self, min_frequency: int, max_size: int) -> List[GraphPattern]:
        """Detect patterns using APOC."""
        # This would use APOC's graph algorithms
        # Simplified placeholder implementation
        patterns = []

        cypher = """
        CALL apoc.periodic.iterate(
            'MATCH (n:Node) RETURN n',
            'MATCH (n)-[r:RELATES*..{max_size}]-(m)
             WITH nodes(r) AS nodes, relationships(r) AS rels
             WHERE size(nodes) <= {max_size}
             RETURN nodes, rels, count(*) AS freq
             ORDER BY freq DESC
             LIMIT 10',
            {batchSize:100, parallel:true}
        )
        YIELD batches, total
        RETURN batches, total
        """

        # Simplified response
        for i in range(3):  # Just return a few placeholder patterns
            pattern = GraphPattern(
                pattern_id=f"pattern_{i}",
                nodes=[f"node_{j}" for j in range(3)],
                edges=[f"edge_{j}" for j in range(2)],
                frequency=min_frequency + i,
                significance=0.5 + (i * 0.1)
            )
            patterns.append(pattern)

        return patterns

    def _detect_patterns_basic(self, min_frequency: int, max_size: int) -> List[GraphPattern]:
        """Basic pattern detection without APOC."""
        # Very simplified approach - in production, use a more sophisticated algorithm
        patterns = []

        cypher = """
        MATCH (n:Node)
        WITH n.label AS label, count(*) AS count
        WHERE count >= $min_frequency
        MATCH (n:Node {label: label})-[r:RELATES*..{max_size}]-(m:Node {label: label})
        WITH label, collect(n.node_id) AS nodes, collect(r.edge_id) AS edges
        RETURN label, nodes, edges, count
        ORDER BY count DESC
        LIMIT 5
        """

        params = {
            "min_frequency": min_frequency,
            "max_size": max_size
        }

        try:
            with self.driver.session(database=self.database) as session:
                result = session.run(cypher, params)

                for i, record in enumerate(result):
                    pattern = GraphPattern(
                        pattern_id=f"pattern_{i}_{record['label']}",
                        nodes=record["nodes"],
                        edges=record["edges"],
                        frequency=record["count"],
                        significance=0.5  # Placeholder
                    )
                    patterns.append(pattern)

                return patterns
        except Exception as e:
            self.logger.error(f"Error detecting patterns: {e}")
            return []  # Return empty list on error

    def clear(self) -> None:
        """Clear all nodes and edges from the graph."""
        with self.lock:
            # Reset caches
            self._node_cache.clear()
            self._edge_cache.clear()

            # Delete all data
            cypher = "MATCH (n) DETACH DELETE n"

            try:
                with self.driver.session(database=self.database) as session:
                    session.run(cypher)
                    self.logger.info("Graph cleared")
            except Exception as e:
                self.logger.error(f"Error clearing graph: {e}")
                raise GraphEngineError(f"Failed to clear graph: {e}")

    def serialize(self) -> Dict[str, Any]:
        """Serialize the graph to a dictionary."""
        with self.lock:
            data = {
                'nodes': [],
                'edges': [],
                'metadata': {
                    'timestamp': time.time(),
                    'uri': self.uri,
                    'database': self.database
                }
            }

            # Query all nodes
            nodes_cypher = """
            MATCH (n:Node)
            RETURN n.node_id AS id, n.label AS label, n.activation AS activation,
                   n.importance AS importance, n.creation_time AS creation_time,
                   n.last_accessed AS last_accessed, properties(n) AS properties
            """

            # Query all edges
            edges_cypher = """
            MATCH (s:Node)-[r:RELATES]->(t:Node)
            RETURN r.edge_id AS id, s.node_id AS source, t.node_id AS target,
                   r.type AS type, r.weight AS weight, r.bidirectional AS bidirectional,
                   r.creation_time AS creation_time, properties(r) AS properties
            """

            try:
                with self.driver.session(database=self.database) as session:
                    # Get nodes
                    nodes_result = session.run(nodes_cypher)
                    for record in nodes_result:
                        props = dict(record["properties"])
                        # Remove standard properties
                        for key in ["node_id", "label", "activation", "importance",
                                   "creation_time", "last_accessed"]:
                            if key in props:
                                del props[key]

                        node_data = {
                            'id': record["id"],
                            'label': record["label"],
                            'properties': props,
                            'activation': record["activation"],
                            'importance': record["importance"],
                            'creation_time': record["creation_time"],
                            'last_accessed': record["last_accessed"]
                        }
                        data['nodes'].append(node_data)

                    # Get edges
                    edges_result = session.run(edges_cypher)
                    for record in edges_result:
                        props = dict(record["properties"])
                        # Remove standard properties
                        for key in ["edge_id", "type", "weight", "bidirectional", "creation_time"]:
                            if key in props:
                                del props[key]

                        edge_data = {
                            'id': record["id"],
                            'source': record["source"],
                            'target': record["target"],
                            'type': record["type"],
                            'properties': props,
                            'weight': record["weight"],
                            'bidirectional': record["bidirectional"],
                            'creation_time': record["creation_time"]
                        }
                        data['edges'].append(edge_data)

                    # Update metadata
                    data['metadata']['node_count'] = len(data['nodes'])
                    data['metadata']['edge_count'] = len(data['edges'])

                    return data
            except Exception as e:
                self.logger.error(f"Error serializing graph: {e}")
                raise GraphEngineError(f"Failed to serialize graph: {e}")

    def deserialize(self, data: Dict[str, Any]) -> None:
        """Load the graph from a serialized representation."""
        with self.lock:
            # Clear existing data first
            self.clear()

            try:
                with self.driver.session(database=self.database) as session:
                    # Create nodes first
                    for node_data in data.get('nodes', []):
                        node_id = node_data['id']
                        label = node_data['label']

                        # Prepare properties
                        props = {
                            'node_id': node_id,
                            'label': label,
                            'activation': node_data.get('activation', 0.0),
                            'importance': node_data.get('importance', 0.5),
                            'creation_time': node_data.get('creation_time', time.time()),
                            'last_accessed': node_data.get('last_accessed', time.time())
                        }

                        # Add custom properties
                        props.update(node_data.get('properties', {}))

                        # Create node
                        create_node_cypher = """
                        CREATE (n:Node)
                        SET n = $props
                        """

                        session.run(create_node_cypher, {'props': props})

                    # Create edges
                    for edge_data in data.get('edges', []):
                        edge_id = edge_data['id']
                        source_id = edge_data['source']
                        target_id = edge_data['target']
                        edge_type = edge_data['type']

                        # Prepare properties
                        props = {
                            'edge_id': edge_id,
                            'type': edge_type,
                            'weight': edge_data.get('weight', 1.0),
                            'bidirectional': edge_data.get('bidirectional', False),
                            'creation_time': edge_data.get('creation_time', time.time())
                        }

                        # Add custom properties
                        props.update(edge_data.get('properties', {}))

                        # Create edge
                        create_edge_cypher = """
                        MATCH (s:Node {node_id: $source_id})
                        MATCH (t:Node {node_id: $target_id})
                        CREATE (s)-[r:RELATES]->(t)
                        SET r = $props
                        """

                        edge_params = {
                            'source_id': source_id,
                            'target_id': target_id,
                            'props': props
                        }

                        session.run(create_edge_cypher, edge_params)

                    self.logger.info(f"Deserialized graph with {len(data.get('nodes', []))} nodes and {len(data.get('edges', []))} edges")
            except Exception as e:
                self.logger.error(f"Error deserializing graph: {e}")
                raise GraphEngineError(f"Failed to deserialize graph: {e}")

    def close(self) -> None:
        """Close the graph engine and release any resources."""
        with self.lock:
            try:
                if self.driver:
                    self.driver.close()
                    self.logger.info("Neo4jGraphEngine closed")
            except Exception as e:
                self.logger.error(f"Error closing Neo4j driver: {e}")


# -----------------------------------------------------------------------------
# Factory function to create graph engines
# -----------------------------------------------------------------------------

def create_graph_engine(engine_type: str, **kwargs) -> AbstractGraphEngine:
    """
    Factory function to create the appropriate graph engine.

    Args:
        engine_type: Type of graph engine to create ("inmemory" or "neo4j")
        **kwargs: Additional arguments to pass to the graph engine constructor

    Returns:
        An instance of AbstractGraphEngine

    Raises:
        ValueError: If the engine type is not supported
    """
    if engine_type.lower() == "inmemory":
        return InMemoryGraphEngine(**kwargs)
    elif engine_type.lower() == "neo4j":
        return Neo4jGraphEngine(**kwargs)
    else:
        raise ValueError(f"Unsupported graph engine type: {engine_type}")