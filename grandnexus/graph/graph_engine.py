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
import os
import random

# Optional dependencies
try:
    import numpy as np
    HAS_NUMPY = True
except ImportError:
    HAS_NUMPY = False

try:
    import networkx as nx
    HAS_NETWORKX = True
except ImportError:
    HAS_NETWORKX = False

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
    """A node in the graph with cognitive properties"""
    node_id: str
    label: str
    properties: Dict[str, Any] = field(default_factory=dict)
    embedding: Optional[Any] = None
    activation: float = 0.0
    last_accessed: float = field(default_factory=time.time)
    importance: float = 0.5
    creation_time: float = field(default_factory=time.time)

    def access(self) -> None:
        """Update activation and last_accessed time when accessed"""
        current_time = time.time()
        self.last_accessed = current_time
        # Increase activation, with saturation
        self.activation = min(1.0, self.activation + 0.2)

    def decay_activation(self, decay_rate: float, current_time: Optional[float] = None) -> None:
        """Apply decay to node activation"""
        if current_time is None:
            current_time = time.time()

        time_since_access = current_time - self.last_accessed
        decay_factor = math.exp(-decay_rate * time_since_access)
        self.activation *= decay_factor


@dataclass
class GraphEdge:
    """A directed edge in the graph with cognitive properties"""
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
    """A discovered pattern in the graph"""
    pattern_id: str
    nodes: List[str]
    edges: List[str]
    frequency: int = 1
    significance: float = 0.5
    embedding: Optional[Any] = None


@dataclass
class GraphQuery:
    """A flexible query specification for graph operations"""
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
    """Base class for graph-related errors"""
    pass

class NodeNotFoundError(GraphEngineError):
    """Raised when a node is not found"""
    pass

class EdgeNotFoundError(GraphEngineError):
    """Raised when an edge is not found"""
    pass

class InvalidQueryError(GraphEngineError):
    """Raised when a query is malformed"""
    pass


# -----------------------------------------------------------------------------
# Abstract Base Class
# -----------------------------------------------------------------------------

class AbstractGraphEngine(ABC):
    """Abstract base class defining the interface for graph engines"""

    @abstractmethod
    def add_node(self, node_id: Optional[str] = None, label: str = "Node",
                properties: Optional[Dict[str, Any]] = None) -> str:
        """Add a node to the graph"""
        pass

    @abstractmethod
    def get_node(self, node_id: str) -> GraphNode:
        """Retrieve a node by ID"""
        pass

    @abstractmethod
    def update_node(self, node_id: str, properties: Dict[str, Any]) -> None:
        """Update a node's properties"""
        pass

    @abstractmethod
    def delete_node(self, node_id: str) -> None:
        """Delete a node and its connected edges"""
        pass

    @abstractmethod
    def add_edge(self, source_id: str, target_id: str, edge_type: str,
                edge_id: Optional[str] = None, properties: Optional[Dict[str, Any]] = None,
                weight: float = 1.0, bidirectional: bool = False) -> str:
        """Add an edge between two nodes"""
        pass

    @abstractmethod
    def get_edge(self, edge_id: str) -> GraphEdge:
        """Retrieve an edge by ID"""
        pass

    @abstractmethod
    def update_edge(self, edge_id: str, properties: Dict[str, Any]) -> None:
        """Update an edge's properties"""
        pass

    @abstractmethod
    def delete_edge(self, edge_id: str) -> None:
        """Delete an edge"""
        pass

    @abstractmethod
    def get_node_neighbors(self, node_id: str, edge_types: Optional[List[str]] = None,
                          direction: str = "outgoing") -> List[Tuple[str, str, str]]:
        """Get neighboring nodes connected to a node"""
        pass

    @abstractmethod
    def query_nodes(self, criteria: Dict[str, Any]) -> List[str]:
        """Find nodes matching criteria"""
        pass

    @abstractmethod
    def query_edges(self, criteria: Dict[str, Any]) -> List[str]:
        """Find edges matching criteria"""
        pass

    @abstractmethod
    def find_paths(self, start_id: str, end_id: str, max_depth: int = 3,
                  edge_types: Optional[List[str]] = None) -> List[List[str]]:
        """Find paths between two nodes"""
        pass

    @abstractmethod
    def execute_query(self, query: GraphQuery) -> Dict[str, Any]:
        """Execute a complex graph query"""
        pass

    @abstractmethod
    def spread_activation(self, source_nodes: List[str], decay_factor: float = 0.5,
                         max_depth: int = 3) -> Dict[str, float]:
        """Spread activation from source nodes through the graph"""
        pass

    @abstractmethod
    def get_activated_subgraph(self, threshold: float = 0.2) -> Tuple[List[str], List[str]]:
        """Get the currently activated subgraph"""
        pass

    @abstractmethod
    def detect_patterns(self, min_frequency: int = 2, max_size: int = 5) -> List[GraphPattern]:
        """Detect frequent patterns in the graph"""
        pass

    @abstractmethod
    def clear(self) -> None:
        """Clear all nodes and edges from the graph"""
        pass

    @abstractmethod
    def serialize(self) -> Dict[str, Any]:
        """Serialize the graph to a dictionary"""
        pass

    @abstractmethod
    def deserialize(self, data: Dict[str, Any]) -> None:
        """Load the graph from a serialized representation"""
        pass

    @abstractmethod
    def close(self) -> None:
        """Close the graph engine and release resources"""
        pass


# -----------------------------------------------------------------------------
# In-Memory Graph Engine Implementation
# -----------------------------------------------------------------------------

class InMemoryGraphEngine(AbstractGraphEngine):
    """In-memory implementation of the graph engine"""

    def __init__(self, use_networkx: bool = True, decay_rate: float = 0.1):
        """Initialize the in-memory graph engine"""
        self.logger = logging.getLogger("GrandNexus.Graph.InMemoryEngine")
        self.instance_id = str(uuid.uuid4())[:8]
        self.decay_rate = decay_rate

        # Data storage
        self.nodes = {}  # node_id -> GraphNode
        self.edges = {}  # edge_id -> GraphEdge
        self.outgoing_edges = defaultdict(list)  # source_id -> [edge_id]
        self.incoming_edges = defaultdict(list)  # target_id -> [edge_id]

        # Activation tracking
        self.activation_updated = {}  # node_id -> last_update_time

        # NetworkX graph (optional)
        self.use_networkx = use_networkx and HAS_NETWORKX
        self.nx_graph = None
        self.nx_graph_dirty = True  # Needs to be rebuilt

        if self.use_networkx:
            if not HAS_NETWORKX:
                self.logger.warning("NetworkX not available; falling back to basic implementation")
                self.use_networkx = False
            else:
                self.nx_graph = nx.DiGraph()

        self.logger.info(f"InMemoryGraphEngine initialized with ID={self.instance_id}")

    def add_node(self, node_id: Optional[str] = None, label: str = "Node",
                properties: Optional[Dict[str, Any]] = None) -> str:
        """Add a node to the graph"""
        if node_id is None:
            node_id = str(uuid.uuid4())

        if node_id in self.nodes:
            self.logger.warning(f"Node {node_id} already exists, updating")
            self.update_node(node_id, properties or {})
            return node_id

        node = GraphNode(
            node_id=node_id,
            label=label,
            properties=properties or {},
            activation=0.1,  # Initial small activation
        )

        self.nodes[node_id] = node

        # Update NetworkX graph if used
        if self.use_networkx:
            self.nx_graph.add_node(
                node_id,
                label=label,
                properties=properties or {},
                activation=node.activation
            )
        else:
            self.nx_graph_dirty = True

        self.logger.debug(f"Added node {node_id} with label {label}")
        return node_id

    def get_node(self, node_id: str) -> GraphNode:
        """Retrieve a node by ID"""
        if node_id not in self.nodes:
            raise NodeNotFoundError(f"Node {node_id} not found")

        node = self.nodes[node_id]
        node.access()  # Update access time and activation
        return node

    def update_node(self, node_id: str, properties: Dict[str, Any]) -> None:
        """Update a node's properties"""
        if node_id not in self.nodes:
            raise NodeNotFoundError(f"Node {node_id} not found")

        node = self.nodes[node_id]

        # Update existing properties
        node.properties.update(properties)

        # Update NetworkX graph if used
        if self.use_networkx:
            for key, value in properties.items():
                self.nx_graph.nodes[node_id][key] = value
        else:
            self.nx_graph_dirty = True

        self.logger.debug(f"Updated node {node_id}")

    def delete_node(self, node_id: str) -> None:
        """Delete a node and its connected edges"""
        if node_id not in self.nodes:
            raise NodeNotFoundError(f"Node {node_id} not found")

        # Remove all edges connected to this node
        incoming = self.incoming_edges[node_id].copy()
        outgoing = self.outgoing_edges[node_id].copy()

        for edge_id in incoming + outgoing:
            self._delete_edge_internal(edge_id)

        # Remove node
        del self.nodes[node_id]
        if node_id in self.incoming_edges:
            del self.incoming_edges[node_id]
        if node_id in self.outgoing_edges:
            del self.outgoing_edges[node_id]
        if node_id in self.activation_updated:
            del self.activation_updated[node_id]

        # Update NetworkX graph if used
        if self.use_networkx:
            if node_id in self.nx_graph:
                self.nx_graph.remove_node(node_id)
        else:
            self.nx_graph_dirty = True

        self.logger.debug(f"Deleted node {node_id}")

    def add_edge(self, source_id: str, target_id: str, edge_type: str,
                edge_id: Optional[str] = None, properties: Optional[Dict[str, Any]] = None,
                weight: float = 1.0, bidirectional: bool = False) -> str:
        """Add an edge between two nodes"""
        # Verify nodes exist
        if source_id not in self.nodes:
            raise NodeNotFoundError(f"Source node {source_id} not found")
        if target_id not in self.nodes:
            raise NodeNotFoundError(f"Target node {target_id} not found")

        if edge_id is None:
            edge_id = str(uuid.uuid4())

        if edge_id in self.edges:
            self.logger.warning(f"Edge {edge_id} already exists, updating")
            self.update_edge(edge_id, properties or {})
            return edge_id

        edge = GraphEdge(
            edge_id=edge_id,
            source_id=source_id,
            target_id=target_id,
            type=edge_type,
            properties=properties or {},
            weight=weight,
            bidirectional=bidirectional
        )

        self.edges[edge_id] = edge
        self.outgoing_edges[source_id].append(edge_id)
        self.incoming_edges[target_id].append(edge_id)

        # Update NetworkX graph if used
        if self.use_networkx:
            self.nx_graph.add_edge(
                source_id,
                target_id,
                key=edge_id,
                type=edge_type,
                properties=properties or {},
                weight=weight,
                bidirectional=bidirectional
            )

            if bidirectional:
                self.nx_graph.add_edge(
                    target_id,
                    source_id,
                    key=f"{edge_id}_rev",
                    type=edge_type,
                    properties=properties or {},
                    weight=weight,
                    bidirectional=bidirectional
                )
        else:
            self.nx_graph_dirty = True

        self.logger.debug(f"Added edge {edge_id} from {source_id} to {target_id} of type {edge_type}")

        # If bidirectional, add reverse edge
        if bidirectional:
            # No need to add another edge in memory since the bidirectional flag
            # is already set, but make sure the edge is in both node's lists
            if edge_id not in self.outgoing_edges[target_id]:
                self.outgoing_edges[target_id].append(edge_id)
            if edge_id not in self.incoming_edges[source_id]:
                self.incoming_edges[source_id].append(edge_id)

        return edge_id

    def get_edge(self, edge_id: str) -> GraphEdge:
        """Retrieve an edge by ID"""
        if edge_id not in self.edges:
            raise EdgeNotFoundError(f"Edge {edge_id} not found")

        return self.edges[edge_id]

    def update_edge(self, edge_id: str, properties: Dict[str, Any]) -> None:
        """Update an edge's properties"""
        if edge_id not in self.edges:
            raise EdgeNotFoundError(f"Edge {edge_id} not found")

        edge = self.edges[edge_id]

        # Update existing properties
        edge.properties.update(properties)

        # Update NetworkX graph if used
        if self.use_networkx:
            # Find the edge in the graph
            if edge.source_id in self.nx_graph and edge.target_id in self.nx_graph[edge.source_id]:
                for key, value in properties.items():
                    for k in self.nx_graph[edge.source_id][edge.target_id]:
                        if k == edge_id:
                            self.nx_graph[edge.source_id][edge.target_id][k][key] = value
        else:
            self.nx_graph_dirty = True

        self.logger.debug(f"Updated edge {edge_id}")

    def delete_edge(self, edge_id: str) -> None:
        """Delete an edge"""
        if edge_id not in self.edges:
            raise EdgeNotFoundError(f"Edge {edge_id} not found")

        self._delete_edge_internal(edge_id)
        self.logger.debug(f"Deleted edge {edge_id}")

    def _delete_edge_internal(self, edge_id: str) -> None:
        """Internal helper to delete an edge"""
        edge = self.edges[edge_id]

        # Remove from indices
        if edge_id in self.outgoing_edges[edge.source_id]:
            self.outgoing_edges[edge.source_id].remove(edge_id)
        if edge_id in self.incoming_edges[edge.target_id]:
            self.incoming_edges[edge.target_id].remove(edge_id)

        # Remove reverse connections if bidirectional
        if edge.bidirectional:
            if edge_id in self.outgoing_edges[edge.target_id]:
                self.outgoing_edges[edge.target_id].remove(edge_id)
            if edge_id in self.incoming_edges[edge.source_id]:
                self.incoming_edges[edge.source_id].remove(edge_id)

        # Remove from edges dict
        del self.edges[edge_id]

        # Update NetworkX graph if used
        if self.use_networkx:
            if edge.source_id in self.nx_graph and edge.target_id in self.nx_graph[edge.source_id]:
                # Find the edge with the matching key
                for k in list(self.nx_graph[edge.source_id][edge.target_id].keys()):
                    if k == edge_id:
                        self.nx_graph.remove_edge(edge.source_id, edge.target_id, key=k)

                # If bidirectional, remove reverse edge
                if edge.bidirectional:
                    for k in list(self.nx_graph[edge.target_id][edge.source_id].keys()):
                        if k == f"{edge_id}_rev":
                            self.nx_graph.remove_edge(edge.target_id, edge.source_id, key=k)
        else:
            self.nx_graph_dirty = True

    def get_node_neighbors(self, node_id: str, edge_types: Optional[List[str]] = None,
                          direction: str = "outgoing") -> List[Tuple[str, str, str]]:
        """
        Get neighboring nodes connected to a node.

        Args:
            node_id: ID of the node
            edge_types: Optional filter for edge types
            direction: "outgoing", "incoming", or "both"

        Returns:
            List of (neighbor_id, edge_id, edge_type) tuples
        """
        if node_id not in self.nodes:
            raise NodeNotFoundError(f"Node {node_id} not found")

        neighbors = []

        # Get outgoing edges
        if direction in ["outgoing", "both"]:
            for edge_id in self.outgoing_edges[node_id]:
                edge = self.edges[edge_id]
                if edge_types is None or edge.type in edge_types:
                    neighbors.append((edge.target_id, edge_id, edge.type))

        # Get incoming edges
        if direction in ["incoming", "both"]:
            for edge_id in self.incoming_edges[node_id]:
                edge = self.edges[edge_id]
                if edge_types is None or edge.type in edge_types:
                    neighbors.append((edge.source_id, edge_id, edge.type))

        return neighbors

    def query_nodes(self, criteria: Dict[str, Any]) -> List[str]:
        """Find nodes matching criteria"""
        matching_nodes = []

        for node_id, node in self.nodes.items():
            if self._match_criteria(node.properties, criteria):
                matching_nodes.append(node_id)

        return matching_nodes

    def query_edges(self, criteria: Dict[str, Any]) -> List[str]:
        """Find edges matching criteria"""
        matching_edges = []

        for edge_id, edge in self.edges.items():
            if self._match_criteria(edge.properties, criteria):
                matching_edges.append(edge_id)

        return matching_edges

    def _match_criteria(self, properties: Dict[str, Any], criteria: Dict[str, Any]) -> bool:
        """Check if properties match given criteria"""
        for key, value in criteria.items():
            # Handle special cases with operators
            if isinstance(value, dict) and len(value) == 1:
                op = list(value.keys())[0]
                op_value = value[op]

                if op == "$gt":
                    if key not in properties or properties[key] <= op_value:
                        return False
                elif op == "$lt":
                    if key not in properties or properties[key] >= op_value:
                        return False
                elif op == "$gte":
                    if key not in properties or properties[key] < op_value:
                        return False
                elif op == "$lte":
                    if key not in properties or properties[key] > op_value:
                        return False
                elif op == "$ne":
                    if key in properties and properties[key] == op_value:
                        return False
                elif op == "$in":
                    if key not in properties or properties[key] not in op_value:
                        return False
                elif op == "$contains":
                    if (key not in properties or
                        not isinstance(properties[key], (str, list, set, dict)) or
                        op_value not in properties[key]):
                        return False
                else:
                    # Unknown operator
                    return False
            else:
                # Direct equality match
                if key not in properties or properties[key] != value:
                    return False

        return True

    def find_paths(self, start_id: str, end_id: str, max_depth: int = 3,
                  edge_types: Optional[List[str]] = None) -> List[List[str]]:
        """Find paths between two nodes"""
        if start_id not in self.nodes:
            raise NodeNotFoundError(f"Start node {start_id} not found")
        if end_id not in self.nodes:
            raise NodeNotFoundError(f"End node {end_id} not found")

        # Use NetworkX if available
        if self.use_networkx:
            return self._find_paths_networkx(start_id, end_id, max_depth, edge_types)

        # Fall back to BFS
        return self._find_paths_bfs(start_id, end_id, max_depth, edge_types)

    def _find_paths_networkx(self, start_id: str, end_id: str, max_depth: int,
                           edge_types: Optional[List[str]]) -> List[List[str]]:
        """Find paths using NetworkX algorithms"""
        # Make sure NetworkX graph is up to date
        if self.nx_graph_dirty:
            self._rebuild_nx_graph()

        # Create a subgraph with only the specified edge types if needed
        if edge_types:
            subgraph = nx.DiGraph()
            for u, v, data in self.nx_graph.edges(data=True):
                if data.get('type') in edge_types:
                    subgraph.add_edge(u, v)
        else:
            subgraph = self.nx_graph

        # Find all simple paths up to max_depth
        paths = []
        try:
            for path in nx.all_simple_paths(subgraph, start_id, end_id, cutoff=max_depth):
                paths.append(list(path))
        except (nx.NetworkXNoPath, nx.NodeNotFound):
            # No path exists or node not found
            pass

        return paths

    def _find_paths_bfs(self, start_id: str, end_id: str, max_depth: int,
                      edge_types: Optional[List[str]]) -> List[List[str]]:
        """Find paths using BFS (fallback if NetworkX unavailable)"""
        paths = []
        queue = deque([(start_id, [start_id], 0)])  # (node, path_so_far, depth)
        visited = set()  # For cycle detection

        while queue:
            node_id, path, depth = queue.popleft()

            if depth > max_depth:
                continue

            if node_id == end_id:
                paths.append(path)
                continue

            # Get neighbors
            neighbors = self.get_node_neighbors(node_id, edge_types, "outgoing")

            for neighbor_id, _, _ in neighbors:
                if neighbor_id in path:  # Avoid cycles
                    continue

                new_path = path + [neighbor_id]
                new_depth = depth + 1

                # Store path state
                queue.append((neighbor_id, new_path, new_depth))

        return paths

    def execute_query(self, query: GraphQuery) -> Dict[str, Any]:
        """Execute a complex graph query"""
        results = {
            "nodes": {},
            "edges": {},
            "metadata": {
                "query_time": time.time(),
                "node_count": 0,
                "edge_count": 0,
                "start_nodes": query.start_nodes
            }
        }

        # Start with node criteria if no start nodes specified
        if not query.start_nodes:
            matching_nodes = self.query_nodes(query.node_criteria)
            start_nodes = matching_nodes
        else:
            start_nodes = query.start_nodes

        # Track processed nodes to avoid cycles
        processed_nodes = set()
        processed_edges = set()

        # Expand from each start node
        for node_id in start_nodes:
            if node_id not in self.nodes:
                continue

            self._expand_query_from_node(
                node_id, query, processed_nodes, processed_edges, results, 0
            )

        # Set metadata
        results["metadata"]["node_count"] = len(results["nodes"])
        results["metadata"]["edge_count"] = len(results["edges"])

        return results

    def _expand_query_from_node(self, node_id: str, query: GraphQuery,
                              processed_nodes: Set[str], processed_edges: Set[str],
                              result: Dict[str, Any], depth: int) -> None:
        """Helper to expand a query from a node recursively"""
        if depth > query.max_depth or node_id in processed_nodes:
            return

        # Check activation threshold
        node = self.nodes[node_id]
        if not query.include_inactive and node.activation < query.min_activation:
            return

        # Add node to result if it passes criteria
        if not query.node_criteria or self._match_criteria(node.properties, query.node_criteria):
            result["nodes"][node_id] = self._node_to_dict(node)
            processed_nodes.add(node_id)

            # Find neighbors according to edge criteria
            neighbors = self.get_node_neighbors(node_id)

            for neighbor_id, edge_id, edge_type in neighbors:
                edge = self.edges[edge_id]

                # Add edge to result if it passes criteria
                if edge_id not in processed_edges and (
                   not query.edge_criteria or
                   self._match_criteria(edge.properties, query.edge_criteria)):
                    result["edges"][edge_id] = self._edge_to_dict(edge)
                    processed_edges.add(edge_id)

                    # Recursively expand neighbor
                    self._expand_query_from_node(
                        neighbor_id, query, processed_nodes, processed_edges, result, depth + 1
                    )

    def _node_to_dict(self, node: GraphNode) -> Dict[str, Any]:
        """Convert node to dictionary representation"""
        return {
            "id": node.node_id,
            "label": node.label,
            "properties": node.properties,
            "activation": node.activation,
            "last_accessed": node.last_accessed,
            "importance": node.importance,
            "creation_time": node.creation_time
        }

    def _edge_to_dict(self, edge: GraphEdge) -> Dict[str, Any]:
        """Convert edge to dictionary representation"""
        return {
            "id": edge.edge_id,
            "source": edge.source_id,
            "target": edge.target_id,
            "type": edge.type,
            "properties": edge.properties,
            "weight": edge.weight,
            "bidirectional": edge.bidirectional,
            "creation_time": edge.creation_time
        }

    def spread_activation(self, source_nodes: List[str], decay_factor: float = 0.5,
                         max_depth: int = 3) -> Dict[str, float]:
        """
        Spread activation from source nodes through the graph.

        Args:
            source_nodes: Starting nodes for activation
            decay_factor: How much activation decays per hop (0-1)
            max_depth: Maximum number of hops

        Returns:
            Dictionary mapping node IDs to activation values
        """
        # Initialize activation
        activation = defaultdict(float)

        # Set initial activation for source nodes
        for node_id in source_nodes:
            if node_id in self.nodes:
                self.nodes[node_id].access()  # Mark as accessed
                activation[node_id] = 1.0  # Full activation

        # Track visited nodes to avoid cycles
        visited = set(source_nodes)

        # For each depth level
        frontier = source_nodes
        for depth in range(1, max_depth + 1):
            new_frontier = []

            for node_id in frontier:
                if node_id not in self.nodes:
                    continue

                # Get outgoing neighbors
                neighbors = self.get_node_neighbors(node_id, direction="outgoing")

                # Calculate propagated activation (decayed by distance and edge weight)
                current_activation = activation[node_id]
                propagated_base = current_activation * (1 - decay_factor)

                for neighbor_id, edge_id, _ in neighbors:
                    if neighbor_id in visited:
                        continue

                    edge = self.edges[edge_id]

                    # Propagate activation, weighted by edge weight
                    propagated = propagated_base * edge.weight

                    # Add to neighbor's activation
                    activation[neighbor_id] += propagated

                    # Add to frontier for next iteration
                    new_frontier.append(neighbor_id)
                    visited.add(neighbor_id)

            frontier = new_frontier
            if not frontier:
                break  # No more nodes to process

        # Update node activation values in the graph
        current_time = time.time()
        for node_id, act_value in activation.items():
            if node_id in self.nodes:
                node = self.nodes[node_id]
                # Blend new activation with existing (favor higher values)
                node.activation = max(node.activation, act_value)
                # Update timestamp to reset decay
                if act_value > 0.1:  # Only if significant activation
                    node.last_accessed = current_time

        return dict(activation)

    def get_activated_subgraph(self, threshold: float = 0.2) -> Tuple[List[str], List[str]]:
        """
        Get the currently activated subgraph (nodes with activation >= threshold).

        Args:
            threshold: Minimum activation threshold

        Returns:
            Tuple of (node_ids, edge_ids) lists
        """
        # Decay all nodes' activation
        current_time = time.time()
        self._decay_all_activations(current_time)

        # Find activated nodes
        activated_nodes = []
        for node_id, node in self.nodes.items():
            if node.activation >= threshold:
                activated_nodes.append(node_id)

        # Find edges connecting activated nodes
        activated_edges = []
        for edge_id, edge in self.edges.items():
            if edge.source_id in activated_nodes and edge.target_id in activated_nodes:
                activated_edges.append(edge_id)

        return activated_nodes, activated_edges

    def detect_patterns(self, min_frequency: int = 2, max_size: int = 5) -> List[GraphPattern]:
        """
        Detect frequent patterns in the graph.

        Args:
            min_frequency: Minimum number of occurrences for a pattern
            max_size: Maximum number of nodes in a pattern

        Returns:
            List of detected patterns
        """
        if self.use_networkx:
            return self._detect_patterns_networkx(min_frequency, max_size)
        else:
            return self._detect_patterns_basic(min_frequency, max_size)

    def _detect_patterns_networkx(self, min_frequency: int, max_size: int) -> List[GraphPattern]:
        """Detect patterns using NetworkX algorithms"""
        # Make sure NetworkX graph is up to date
        if self.nx_graph_dirty:
            self._rebuild_nx_graph()

        patterns = []

        # This is a simplified implementation
        # Real pattern mining would use more sophisticated algorithms

        # For small max_size, we can use a brute force approach to find common subgraphs
        for size in range(2, min(max_size + 1, 4)):  # Limit to size 3 for efficiency
            # Find all connected subgraphs of size 'size'
            for node in self.nx_graph.nodes():
                # Extract a subgraph around this node
                subgraph_nodes = [node]
                frontier = [node]
                visited = {node}

                # BFS to build a subgraph of up to 'size' nodes
                while len(subgraph_nodes) < size and frontier:
                    new_frontier = []
                    for n in frontier:
                        for neighbor in self.nx_graph.neighbors(n):
                            if neighbor not in visited and len(subgraph_nodes) < size:
                                visited.add(neighbor)
                                subgraph_nodes.append(neighbor)
                                new_frontier.append(neighbor)
                    frontier = new_frontier

                if len(subgraph_nodes) == size:
                    # Create the subgraph
                    subgraph = self.nx_graph.subgraph(subgraph_nodes)

                    # Check if this subgraph appears elsewhere
                    frequency = self._count_pattern_occurrences(subgraph)

                    if frequency >= min_frequency:
                        # Convert to EdgeGraph format
                        pattern_id = str(uuid.uuid4())

                        # Get edge IDs
                        edges = []
                        for u, v, data in subgraph.edges(data=True):
                            if 'key' in data:
                                edges.append(data['key'])

                        patterns.append(GraphPattern(
                            pattern_id=pattern_id,
                            nodes=list(subgraph_nodes),
                            edges=edges,
                            frequency=frequency,
                            significance=0.5 + (frequency - min_frequency) * 0.1  # Simple heuristic
                        ))

        return patterns

    def _count_pattern_occurrences(self, pattern_graph: 'nx.Graph') -> int:
        """Count occurrences of a pattern in the graph"""
        # This is a simplified approach - full subgraph isomorphism is NP-hard
        # We just check for subgraphs with the same node/edge counts and similar connections

        pattern_node_count = pattern_graph.number_of_nodes()
        pattern_edge_count = pattern_graph.number_of_edges()

        # Quick signature based on degree sequence (sorted)
        pattern_degree_seq = sorted([d for n, d in pattern_graph.degree()])

        # Count matches
        count = 0

        # Check each node as a potential starting point
        for start_node in self.nx_graph.nodes():
            # Get subgraph around this node
            subgraph_nodes = [start_node]
            frontier = [start_node]
            visited = {start_node}

            # BFS to build a subgraph of matching size
            while len(subgraph_nodes) < pattern_node_count and frontier:
                new_frontier = []
                for n in frontier:
                    for neighbor in self.nx_graph.neighbors(n):
                        if neighbor not in visited and len(subgraph_nodes) < pattern_node_count:
                            visited.add(neighbor)
                            subgraph_nodes.append(neighbor)
                            new_frontier.append(neighbor)
                frontier = new_frontier

            if len(subgraph_nodes) == pattern_node_count:
                subgraph = self.nx_graph.subgraph(subgraph_nodes)

                # Compare simple metrics
                if (subgraph.number_of_edges() == pattern_edge_count and
                    sorted([d for n, d in subgraph.degree()]) == pattern_degree_seq):
                    count += 1

        return count

    def _detect_patterns_basic(self, min_frequency: int, max_size: int) -> List[GraphPattern]:
        """Basic pattern detection fallback"""
        patterns = []

        # This is a very simplified approach
        # Real pattern mining would use more sophisticated algorithms

        # Look for common edge types connecting the same node types
        edge_type_counts = defaultdict(int)

        for edge in self.edges.values():
            source_node = self.nodes[edge.source_id]
            target_node = self.nodes[edge.target_id]

            # Create a pattern signature
            pattern_sig = f"{source_node.label}--{edge.type}-->{target_node.label}"
            edge_type_counts[pattern_sig] += 1

        # Check which patterns are frequent
        for pattern_sig, count in edge_type_counts.items():
            if count >= min_frequency:
                # Find instances of this pattern
                source_label, edge_type, target_label = pattern_sig.split("--")
                edge_type = edge_type.strip(">")

                # Get some example instances for this pattern
                instances = []
                for edge in self.edges.values():
                    source_node = self.nodes[edge.source_id]
                    target_node = self.nodes[edge.target_id]

                    if (source_node.label == source_label and
                        edge.type == edge_type and
                        target_node.label == target_label):
                        instances.append((edge.source_id, edge.target_id, edge.edge_id))

                        if len(instances) >= 5:  # Limit to 5 examples
                            break

                if instances:
                    pattern_id = str(uuid.uuid4())
                    pattern_nodes = []
                    pattern_edges = []

                    # Use the first instance as the pattern example
                    pattern_nodes.append(instances[0][0])
                    pattern_nodes.append(instances[0][1])
                    pattern_edges.append(instances[0][2])

                    patterns.append(GraphPattern(
                        pattern_id=pattern_id,
                        nodes=pattern_nodes,
                        edges=pattern_edges,
                        frequency=count,
                        significance=0.5 + (count - min_frequency) * 0.1  # Simple heuristic
                    ))

        return patterns

    def clear(self) -> None:
        """Clear all nodes and edges from the graph"""
        self.nodes.clear()
        self.edges.clear()
        self.outgoing_edges.clear()
        self.incoming_edges.clear()
        self.activation_updated.clear()

        if self.use_networkx:
            self.nx_graph.clear()
        else:
            self.nx_graph_dirty = True

        self.logger.info("Graph cleared")

    def serialize(self) -> Dict[str, Any]:
        """Serialize the graph to a dictionary"""
        # Decay activations before serialization
        current_time = time.time()
        self._decay_all_activations(current_time)

        nodes_data = {}
        for node_id, node in self.nodes.items():
            # Convert potentially non-serializable data
            properties = {}
            for k, v in node.properties.items():
                try:
                    json.dumps({k: v})
                    properties[k] = v
                except (TypeError, OverflowError):
                    # If not serializable, convert to string
                    properties[k] = str(v)

            nodes_data[node_id] = {
                "label": node.label,
                "properties": properties,
                "activation": node.activation,
                "last_accessed": node.last_accessed,
                "importance": node.importance,
                "creation_time": node.creation_time
            }

            # Handle embeddings specially (if present and numpy array)
            if node.embedding is not None:
                if HAS_NUMPY and isinstance(node.embedding, np.ndarray):
                    nodes_data[node_id]["embedding"] = node.embedding.tolist()
                else:
                    # Try to convert to list if possible
                    try:
                        nodes_data[node_id]["embedding"] = list(node.embedding)
                    except:
                        pass  # Skip if not convertible

        edges_data = {}
        for edge_id, edge in self.edges.items():
            # Convert potentially non-serializable data
            properties = {}
            for k, v in edge.properties.items():
                try:
                    json.dumps({k: v})
                    properties[k] = v
                except (TypeError, OverflowError):
                    # If not serializable, convert to string
                    properties[k] = str(v)

            edges_data[edge_id] = {
                "source_id": edge.source_id,
                "target_id": edge.target_id,
                "type": edge.type,
                "properties": properties,
                "weight": edge.weight,
                "bidirectional": edge.bidirectional,
                "creation_time": edge.creation_time
            }

        return {
            "nodes": nodes_data,
            "edges": edges_data,
            "metadata": {
                "timestamp": current_time,
                "instance_id": self.instance_id,
                "node_count": len(nodes_data),
                "edge_count": len(edges_data)
            }
        }

    def deserialize(self, data: Dict[str, Any]) -> None:
        """Load the graph from a serialized representation"""
        # Clear existing graph
        self.clear()

        # Load nodes
        for node_id, node_data in data.get("nodes", {}).items():
            # Handle embeddings if present
            embedding = node_data.get("embedding")
            if embedding is not None and HAS_NUMPY:
                embedding = np.array(embedding)

            node = GraphNode(
                node_id=node_id,
                label=node_data.get("label", "Node"),
                properties=node_data.get("properties", {}),
                embedding=embedding,
                activation=node_data.get("activation", 0.0),
                last_accessed=node_data.get("last_accessed", time.time()),
                importance=node_data.get("importance", 0.5),
                creation_time=node_data.get("creation_time", time.time())
            )

            self.nodes[node_id] = node

        # Load edges
        for edge_id, edge_data in data.get("edges", {}).items():
            edge = GraphEdge(
                edge_id=edge_id,
                source_id=edge_data.get("source_id"),
                target_id=edge_data.get("target_id"),
                type=edge_data.get("type", "RELATED_TO"),
                properties=edge_data.get("properties", {}),
                weight=edge_data.get("weight", 1.0),
                bidirectional=edge_data.get("bidirectional", False),
                creation_time=edge_data.get("creation_time", time.time())
            )

            self.edges[edge_id] = edge

            # Update indices
            self.outgoing_edges[edge.source_id].append(edge_id)
            self.incoming_edges[edge.target_id].append(edge_id)

            if edge.bidirectional:
                if edge_id not in self.outgoing_edges[edge.target_id]:
                    self.outgoing_edges[edge.target_id].append(edge_id)
                if edge_id not in self.incoming_edges[edge.source_id]:
                    self.incoming_edges[edge.source_id].append(edge_id)

        # Rebuild NetworkX graph if used
        if self.use_networkx:
            self._rebuild_nx_graph()
        else:
            self.nx_graph_dirty = True

        self.logger.info(f"Loaded graph with {len(self.nodes)} nodes and {len(self.edges)} edges")

    def close(self) -> None:
        """Close the graph engine and release resources"""
        # Nothing to do for in-memory engine
        pass

    def _decay_all_activations(self, current_time: float) -> None:
        """Decay all nodes' activation based on time since last access"""
        for node_id, node in self.nodes.items():
            node.decay_activation(self.decay_rate, current_time)

    def _rebuild_nx_graph(self) -> None:
        """Rebuild the NetworkX graph from scratch"""
        if not self.use_networkx:
            return

        self.nx_graph.clear()

        # Add nodes
        for node_id, node in self.nodes.items():
            self.nx_graph.add_node(
                node_id,
                label=node.label,
                properties=node.properties,
                activation=node.activation
            )

        # Add edges
        for edge_id, edge in self.edges.items():
            self.nx_graph.add_edge(
                edge.source_id,
                edge.target_id,
                key=edge_id,
                type=edge.type,
                properties=edge.properties,
                weight=edge.weight,
                bidirectional=edge.bidirectional
            )

            if edge.bidirectional:
                self.nx_graph.add_edge(
                    edge.target_id,
                    edge.source_id,
                    key=f"{edge_id}_rev",
                    type=edge.type,
                    properties=edge.properties,
                    weight=edge.weight,
                    bidirectional=edge.bidirectional
                )

        self.nx_graph_dirty = False


# -----------------------------------------------------------------------------
# Neo4j Graph Engine Implementation
# -----------------------------------------------------------------------------

class Neo4jGraphEngine(AbstractGraphEngine):
    """Neo4j-based implementation of the graph engine"""

    def __init__(self, uri: str, user: str, password: str, database: str = "neo4j",
                decay_rate: float = 0.1, use_apoc: bool = False):
        """Initialize the Neo4j graph engine"""
        self.logger = logging.getLogger("GrandNexus.Graph.Neo4jEngine")
        self.instance_id = str(uuid.uuid4())[:8]
        self.decay_rate = decay_rate

        # Neo4j connection
        self.uri = uri
        self.user = user
        self.password = password
        self.database = database
        self.use_apoc = use_apoc

        # Local cache for node/edge data
        self.node_cache = {}  # node_id -> GraphNode
        self.edge_cache = {}  # edge_id -> GraphEdge

        # Initialize driver
        if not HAS_NEO4J:
            raise ImportError("Neo4j Python driver is not installed")

        self.driver = GraphDatabase.driver(uri, auth=(user, password))

        # Test connection
        self._test_connection()

        # Set up schema
        self._setup_schema()

        self.logger.info(f"Neo4jGraphEngine initialized with ID={self.instance_id}")

    def _test_connection(self) -> None:
        """Test the Neo4j connection"""
        try:
            with self.driver.session(database=self.database) as session:
                result = session.run("RETURN 1 as test")
                record = result.single()
                if record["test"] != 1:
                    raise ConnectionError("Unexpected response from Neo4j")

            self.logger.info("Successfully connected to Neo4j")
        except Exception as e:
            self.logger.error(f"Failed to connect to Neo4j: {str(e)}")
            raise

    def _setup_schema(self) -> None:
        """Set up constraints and indices for optimized performance"""
        try:
            with self.driver.session(database=self.database) as session:
                # Create constraint on node IDs
                session.run("""
                    CREATE CONSTRAINT node_id_unique IF NOT EXISTS
                    FOR (n:Node)
                    REQUIRE n.id IS UNIQUE
                """)

                # Create constraint on edge IDs
                session.run("""
                    CREATE CONSTRAINT edge_id_unique IF NOT EXISTS
                    FOR ()-[r:RELATED_TO]->()
                    REQUIRE r.id IS UNIQUE
                """)

                # Create index on node labels
                session.run("""
                    CREATE INDEX node_label_index IF NOT EXISTS
                    FOR (n:Node)
                    ON (n.label)
                """)

                # Create index on edge types
                session.run("""
                    CREATE INDEX edge_type_index IF NOT EXISTS
                    FOR ()-[r:RELATED_TO]->()
                    ON (r.type)
                """)

                # Create index on activation
                session.run("""
                    CREATE INDEX node_activation_index IF NOT EXISTS
                    FOR (n:Node)
                    ON (n.activation)
                """)

            self.logger.info("Neo4j schema setup successful")
        except Exception as e:
            self.logger.warning(f"Failed to set up Neo4j schema: {str(e)}")

    def add_node(self, node_id: Optional[str] = None, label: str = "Node",
                properties: Optional[Dict[str, Any]] = None) -> str:
        """Add a node to the graph"""
        if node_id is None:
            node_id = str(uuid.uuid4())

        if properties is None:
            properties = {}

        current_time = time.time()

        # Sanitize properties (ensure they're JSON-serializable)
        clean_props = {}
        for k, v in properties.items():
            try:
                json.dumps({k: v})
                clean_props[k] = v
            except (TypeError, OverflowError):
                # If not serializable, convert to string
                clean_props[k] = str(v)

        try:
            with self.driver.session(database=self.database) as session:
                # Check if node exists
                if self._node_exists(node_id):
                    # Update existing node
                    self.update_node(node_id, clean_props)
                    return node_id

                # Create node
                cypher = """
                CREATE (n:Node {
                    id: $id,
                    label: $label,
                    activation: $activation,
                    last_accessed: $last_accessed,
                    importance: $importance,
                    creation_time: $creation_time
                })
                SET n += $properties
                RETURN n
                """

                result = session.run(
                    cypher,
                    id=node_id,
                    label=label,
                    activation=0.1,  # Initial activation
                    last_accessed=current_time,
                    importance=0.5,  # Default importance
                    creation_time=current_time,
                    properties=clean_props
                )

                # Update cache
                self.node_cache[node_id] = GraphNode(
                    node_id=node_id,
                    label=label,
                    properties=clean_props,
                    activation=0.1,
                    last_accessed=current_time,
                    importance=0.5,
                    creation_time=current_time
                )

                self.logger.debug(f"Added node {node_id} with label {label}")
                return node_id

        except Exception as e:
            self.logger.error(f"Error adding node: {str(e)}")
            raise GraphEngineError(f"Failed to add node: {str(e)}")

    def get_node(self, node_id: str) -> GraphNode:
        """Retrieve a node by ID"""
        # Check cache first
        if node_id in self.node_cache:
            node = self.node_cache[node_id]
            node.access()  # Update access time and activation

            # Update activation in database (async)
            threading.Thread(
                target=self._update_node_activation,
                args=(node_id, node.activation, node.last_accessed)
            ).start()

            return node

        try:
            with self.driver.session(database=self.database) as session:
                cypher = """
                MATCH (n:Node {id: $id})
                SET n.activation = n.activation + 0.2,
                    n.last_accessed = $current_time
                RETURN n
                """

                current_time = time.time()

                result = session.run(cypher, id=node_id, current_time=current_time)
                record = result.single()

                if not record:
                    raise NodeNotFoundError(f"Node {node_id} not found")

                node_data = record["n"]

                # Extract node data
                node = GraphNode(
                    node_id=node_id,
                    label=node_data.get("label", "Node"),
                    properties={k: v for k, v in dict(node_data).items()
                               if k not in ["id", "label", "activation", "last_accessed",
                                           "importance", "creation_time"]},
                    activation=min(1.0, node_data.get("activation", 0.0) + 0.2),
                    last_accessed=current_time,
                    importance=node_data.get("importance", 0.5),
                    creation_time=node_data.get("creation_time", current_time)
                )

                # Update cache
                self.node_cache[node_id] = node

                return node

        except NodeNotFoundError:
            raise
        except Exception as e:
            self.logger.error(f"Error retrieving node: {str(e)}")
            raise GraphEngineError(f"Failed to retrieve node: {str(e)}")

    def _update_node_activation(self, node_id: str, activation: float, last_accessed: float) -> None:
        """Update a node's activation and last_accessed time"""
        try:
            with self.driver.session(database=self.database) as session:
                cypher = """
                MATCH (n:Node {id: $id})
                SET n.activation = $activation,
                    n.last_accessed = $last_accessed
                """

                session.run(
                    cypher,
                    id=node_id,
                    activation=min(1.0, activation),  # Cap at 1.0
                    last_accessed=last_accessed
                )

        except Exception as e:
            self.logger.warning(f"Failed to update node activation: {str(e)}")

    def update_node(self, node_id: str, properties: Dict[str, Any]) -> None:
        """Update a node's properties"""
        # Sanitize properties
        clean_props = {}
        for k, v in properties.items():
            try:
                json.dumps({k: v})
                clean_props[k] = v
            except (TypeError, OverflowError):
                clean_props[k] = str(v)

        try:
            with self.driver.session(database=self.database) as session:
                cypher = """
                MATCH (n:Node {id: $id})
                SET n += $properties
                RETURN n
                """

                result = session.run(cypher, id=node_id, properties=clean_props)
                record = result.single()

                if not record:
                    raise NodeNotFoundError(f"Node {node_id} not found")

                # Update cache if present
                if node_id in self.node_cache:
                    self.node_cache[node_id].properties.update(clean_props)

                self.logger.debug(f"Updated node {node_id}")

        except Exception as e:
            self.logger.error(f"Error updating node: {str(e)}")
            raise GraphEngineError(f"Failed to update node: {str(e)}")

    def delete_node(self, node_id: str) -> None:
        """Delete a node and its connected edges"""
        try:
            with self.driver.session(database=self.database) as session:
                # Delete node and its relationships
                cypher = """
                MATCH (n:Node {id: $id})
                OPTIONAL MATCH (n)-[r]-()
                DELETE r, n
                """

                result = session.run(cypher, id=node_id)

                # Update cache
                if node_id in self.node_cache:
                    del self.node_cache[node_id]

                # Also remove from edge cache any edges connected to this node
                edge_ids_to_remove = []
                for edge_id, edge in self.edge_cache.items():
                    if edge.source_id == node_id or edge.target_id == node_id:
                        edge_ids_to_remove.append(edge_id)

                for edge_id in edge_ids_to_remove:
                    if edge_id in self.edge_cache:
                        del self.edge_cache[edge_id]

                self.logger.debug(f"Deleted node {node_id}")

        except Exception as e:
            self.logger.error(f"Error deleting node: {str(e)}")
            raise GraphEngineError(f"Failed to delete node: {str(e)}")

    def add_edge(self, source_id: str, target_id: str, edge_type: str,
                edge_id: Optional[str] = None, properties: Optional[Dict[str, Any]] = None,
                weight: float = 1.0, bidirectional: bool = False) -> str:
        """Add an edge between two nodes"""
        if edge_id is None:
            edge_id = str(uuid.uuid4())

        if properties is None:
            properties = {}

        # Sanitize properties
        clean_props = {}
        for k, v in properties.items():
            try:
                json.dumps({k: v})
                clean_props[k] = v
            except (TypeError, OverflowError):
                clean_props[k] = str(v)

        current_time = time.time()

        try:
            with self.driver.session(database=self.database) as session:
                # Check if nodes exist
                if not self._node_exists(source_id):
                    raise NodeNotFoundError(f"Source node {source_id} not found")
                if not self._node_exists(target_id):
                    raise NodeNotFoundError(f"Target node {target_id} not found")

                # Check if edge exists
                result = session.run("""
                    MATCH (s:Node {id: $source_id})-[r:RELATED_TO {id: $edge_id}]->(t:Node {id: $target_id})
                    RETURN r
                """, source_id=source_id, target_id=target_id, edge_id=edge_id)

                if result.single():
                    # Edge exists, update properties
                    self.update_edge(edge_id, clean_props)
                    return edge_id

                # Create edge
                direction = "<->" if bidirectional else "->"

                cypher = f"""
                MATCH (s:Node {{id: $source_id}}), (t:Node {{id: $target_id}})
                CREATE (s)-[r:RELATED_TO {{
                    id: $edge_id,
                    type: $edge_type,
                    weight: $weight,
                    bidirectional: $bidirectional,
                    creation_time: $creation_time
                }}]{direction}(t)
                SET r += $properties
                RETURN r
                """

                result = session.run(
                    cypher,
                    source_id=source_id,
                    target_id=target_id,
                    edge_id=edge_id,
                    edge_type=edge_type,
                    weight=weight,
                    bidirectional=bidirectional,
                    creation_time=current_time,
                    properties=clean_props
                )

                # Update cache
                edge = GraphEdge(
                    edge_id=edge_id,
                    source_id=source_id,
                    target_id=target_id,
                    type=edge_type,
                    properties=clean_props,
                    weight=weight,
                    bidirectional=bidirectional,
                    creation_time=current_time
                )

                self.edge_cache[edge_id] = edge

                self.logger.debug(f"Added edge {edge_id} from {source_id} to {target_id} of type {edge_type}")
                return edge_id

        except (NodeNotFoundError, EdgeNotFoundError):
            raise
        except Exception as e:
            self.logger.error(f"Error adding edge: {str(e)}")
            raise GraphEngineError(f"Failed to add edge: {str(e)}")

    def get_edge(self, edge_id: str) -> GraphEdge:
        """Retrieve an edge by ID"""
        # Check cache first
        if edge_id in self.edge_cache:
            return self.edge_cache[edge_id]

        try:
            with self.driver.session(database=self.database) as session:
                cypher = """
                MATCH ()-[r:RELATED_TO {id: $id}]->()
                RETURN r, startNode(r) as source, endNode(r) as target
                """

                result = session.run(cypher, id=edge_id)
                record = result.single()

                if not record:
                    raise EdgeNotFoundError(f"Edge {edge_id} not found")

                edge_data = record["r"]
                source_node = record["source"]
                target_node = record["target"]

                # Extract edge data
                edge = GraphEdge(
                    edge_id=edge_id,
                    source_id=source_node["id"],
                    target_id=target_node["id"],
                    type=edge_data.get("type", "RELATED_TO"),
                    properties={k: v for k, v in dict(edge_data).items()
                               if k not in ["id", "type", "weight", "bidirectional", "creation_time"]},
                    weight=edge_data.get("weight", 1.0),
                    bidirectional=edge_data.get("bidirectional", False),
                    creation_time=edge_data.get("creation_time", time.time())
                )

                # Update cache
                self.edge_cache[edge_id] = edge

                return edge

        except EdgeNotFoundError:
            raise
        except Exception as e:
            self.logger.error(f"Error retrieving edge: {str(e)}")
            raise GraphEngineError(f"Failed to retrieve edge: {str(e)}")

    def update_edge(self, edge_id: str, properties: Dict[str, Any]) -> None:
        """Update an edge's properties"""
        # Sanitize properties
        clean_props = {}
        for k, v in properties.items():
            try:
                json.dumps({k: v})
                clean_props[k] = v
            except (TypeError, OverflowError):
                clean_props[k] = str(v)

        try:
            with self.driver.session(database=self.database) as session:
                cypher = """
                MATCH ()-[r:RELATED_TO {id: $id}]->()
                SET r += $properties
                RETURN r
                """

                result = session.run(cypher, id=edge_id, properties=clean_props)
                record = result.single()

                if not record:
                    raise EdgeNotFoundError(f"Edge {edge_id} not found")

                # Update cache if present
                if edge_id in self.edge_cache:
                    self.edge_cache[edge_id].properties.update(clean_props)

                self.logger.debug(f"Updated edge {edge_id}")

        except Exception as e:
            self.logger.error(f"Error updating edge: {str(e)}")
            raise GraphEngineError(f"Failed to update edge: {str(e)}")

    def delete_edge(self, edge_id: str) -> None:
        """Delete an edge"""
        try:
            with self.driver.session(database=self.database) as session:
                # Delete edge
                cypher = """
                MATCH ()-[r:RELATED_TO {id: $id}]->()
                DELETE r
                """

                result = session.run(cypher, id=edge_id)

                # Update cache
                if edge_id in self.edge_cache:
                    del self.edge_cache[edge_id]

                self.logger.debug(f"Deleted edge {edge_id}")

        except Exception as e:
            self.logger.error(f"Error deleting edge: {str(e)}")
            raise GraphEngineError(f"Failed to delete edge: {str(e)}")

    def _node_exists(self, node_id: str) -> bool:
        """Check if a node exists in the database"""
        try:
            with self.driver.session(database=self.database) as session:
                result = session.run(
                    "MATCH (n:Node {id: $id}) RETURN count(n) as count",
                    id=node_id
                )
                record = result.single()
                return record and record["count"] > 0
        except Exception as e:
            self.logger.error(f"Error checking node existence: {str(e)}")
            return False

    def get_node_neighbors(self, node_id: str, edge_types: Optional[List[str]] = None,
                          direction: str = "outgoing") -> List[Tuple[str, str, str]]:
        """
        Get neighboring nodes connected to a node.

        Args:
            node_id: ID of the node
            edge_types: Optional filter for edge types
            direction: "outgoing", "incoming", or "both"

        Returns:
            List of (neighbor_id, edge_id, edge_type) tuples
        """
        if not self._node_exists(node_id):
            raise NodeNotFoundError(f"Node {node_id} not found")

        neighbors = []

        try:
            with self.driver.session(database=self.database) as session:
                # Construct Cypher based on direction and edge types
                edge_filter = ""
                if edge_types:
                    type_list = ", ".join([f"'{t}'" for t in edge_types])
                    edge_filter = f"WHERE r.type IN [{type_list}]"

                if direction == "outgoing":
                    cypher = f"""
                    MATCH (n:Node {{id: $id}})-[r:RELATED_TO]->(neighbor:Node)
                    {edge_filter}
                    RETURN neighbor.id as neighbor_id, r.id as edge_id, r.type as edge_type
                    """
                elif direction == "incoming":
                    cypher = f"""
                    MATCH (n:Node {{id: $id}})<-[r:RELATED_TO]-(neighbor:Node)
                    {edge_filter}
                    RETURN neighbor.id as neighbor_id, r.id as edge_id, r.type as edge_type
                    """
                else:  # both
                    cypher = f"""
                    MATCH (n:Node {{id: $id}})-[r:RELATED_TO]-(neighbor:Node)
                    {edge_filter}
                    RETURN neighbor.id as neighbor_id, r.id as edge_id, r.type as edge_type
                    """

                result = session.run(cypher, id=node_id)

                for record in result:
                    neighbors.append((
                        record["neighbor_id"],
                        record["edge_id"],
                        record["edge_type"]
                    ))

                return neighbors

        except NodeNotFoundError:
            raise
        except Exception as e:
            self.logger.error(f"Error retrieving neighbors: {str(e)}")
            raise GraphEngineError(f"Failed to retrieve neighbors: {str(e)}")

    def query_nodes(self, criteria: Dict[str, Any]) -> List[str]:
        """Find nodes matching criteria"""
        if not criteria:
            # If no criteria, return all nodes (up to a limit)
            try:
                with self.driver.session(database=self.database) as session:
                    result = session.run(
                        "MATCH (n:Node) RETURN n.id as id LIMIT 1000"
                    )
                    return [record["id"] for record in result]
            except Exception as e:
                self.logger.error(f"Error querying nodes: {str(e)}")
                return []

        # Convert criteria to Cypher WHERE clause
        where_clauses = []
        params = {}

        for key, value in criteria.items():
            # Handle special operators
            if isinstance(value, dict) and len(value) == 1:
                op = list(value.keys())[0]
                op_value = value[op]

                if op == "$gt":
                    where_clauses.append(f"n.{key} > ${key}")
                    params[key] = op_value
                elif op == "$lt":
                    where_clauses.append(f"n.{key} < ${key}")
                    params[key] = op_value
                elif op == "$gte":
                    where_clauses.append(f"n.{key} >= ${key}")
                    params[key] = op_value
                elif op == "$lte":
                    where_clauses.append(f"n.{key} <= ${key}")
                    params[key] = op_value
                elif op == "$ne":
                    where_clauses.append(f"n.{key} <> ${key}")
                    params[key] = op_value
                elif op == "$in":
                    where_clauses.append(f"n.{key} IN ${key}")
                    params[key] = op_value
                elif op == "$contains":
                    where_clauses.append(f"n.{key} CONTAINS ${key}")
                    params[key] = op_value
                else:
                    # Unknown operator, use equality
                    where_clauses.append(f"n.{key} = ${key}")
                    params[key] = value
            else:
                # Direct equality
                where_clauses.append(f"n.{key} = ${key}")
                params[key] = value

        where_clause = " AND ".join(where_clauses)

        try:
            with self.driver.session(database=self.database) as session:
                cypher = f"""
                MATCH (n:Node)
                WHERE {where_clause}
                RETURN n.id as id
                """

                result = session.run(cypher, **params)
                return [record["id"] for record in result]

        except Exception as e:
            self.logger.error(f"Error querying nodes: {str(e)}")
            return []

    def query_edges(self, criteria: Dict[str, Any]) -> List[str]:
        """Find edges matching criteria"""
        if not criteria:
            # If no criteria, return all edges (up to a limit)
            try:
                with self.driver.session(database=self.database) as session:
                    result = session.run(
                        "MATCH ()-[r:RELATED_TO]->() RETURN r.id as id LIMIT 1000"
                    )
                    return [record["id"] for record in result]
            except Exception as e:
                self.logger.error(f"Error querying edges: {str(e)}")
                return []

        # Convert criteria to Cypher WHERE clause
        where_clauses = []
        params = {}

        for key, value in criteria.items():
            # Handle special operators
            if isinstance(value, dict) and len(value) == 1:
                op = list(value.keys())[0]
                op_value = value[op]

                if op == "$gt":
                    where_clauses.append(f"r.{key} > ${key}")
                    params[key] = op_value
                elif op == "$lt":
                    where_clauses.append(f"r.{key} < ${key}")
                    params[key] = op_value
                elif op == "$gte":
                    where_clauses.append(f"r.{key} >= ${key}")
                    params[key] = op_value
                elif op == "$lte":
                    where_clauses.append(f"r.{key} <= ${key}")
                    params[key] = op_value
                elif op == "$ne":
                    where_clauses.append(f"r.{key} <> ${key}")
                    params[key] = op_value
                elif op == "$in":
                    where_clauses.append(f"r.{key} IN ${key}")
                    params[key] = op_value
                elif op == "$contains":
                    where_clauses.append(f"r.{key} CONTAINS ${key}")
                    params[key] = op_value
                else:
                    # Unknown operator, use equality
                    where_clauses.append(f"r.{key} = ${key}")
                    params[key] = value
            else:
                # Direct equality
                where_clauses.append(f"r.{key} = ${key}")
                params[key] = value

        where_clause = " AND ".join(where_clauses)

        try:
            with self.driver.session(database=self.database) as session:
                cypher = f"""
                MATCH ()-[r:RELATED_TO]->()
                WHERE {where_clause}
                RETURN r.id as id
                """

                result = session.run(cypher, **params)
                return [record["id"] for record in result]

        except Exception as e:
            self.logger.error(f"Error querying edges: {str(e)}")
            return []

    def find_paths(self, start_id: str, end_id: str, max_depth: int = 3,
                  edge_types: Optional[List[str]] = None) -> List[List[str]]:
        """Find paths between two nodes"""
        if not self._node_exists(start_id):
            raise NodeNotFoundError(f"Start node {start_id} not found")
        if not self._node_exists(end_id):
            raise NodeNotFoundError(f"End node {end_id} not found")

        # Construct edge type filter if provided
        edge_filter = ""
        if edge_types:
            type_list = ", ".join([f"'{t}'" for t in edge_types])
            edge_filter = f"WHERE ALL(r IN relationships(p) WHERE r.type IN [{type_list}])"

        try:
            with self.driver.session(database=self.database) as session:
                cypher = f"""
                MATCH (start:Node {{id: $start_id}}), (end:Node {{id: $end_id}}),
                      p = shortestPath((start)-[:RELATED_TO*..{max_depth}]->(end))
                {edge_filter}
                RETURN [node in nodes(p) | node.id] as path
                """

                result = session.run(cypher, start_id=start_id, end_id=end_id)
                paths = [record["path"] for record in result]

                # If no paths found with shortestPath, try to find all paths (up to a limit)
                if not paths:
                    cypher = f"""
                    MATCH (start:Node {{id: $start_id}}), (end:Node {{id: $end_id}}),
                          p = (start)-[:RELATED_TO*..{max_depth}]->(end)
                    {edge_filter}
                    RETURN [node in nodes(p) | node.id] as path
                    LIMIT 10
                    """

                    result = session.run(cypher, start_id=start_id, end_id=end_id)
                    paths = [record["path"] for record in result]

                return paths

        except (NodeNotFoundError, EdgeNotFoundError):
            raise
        except Exception as e:
            self.logger.error(f"Error finding paths: {str(e)}")
            raise GraphEngineError(f"Failed to find paths: {str(e)}")

    def execute_query(self, query: GraphQuery) -> Dict[str, Any]:
        """Execute a complex graph query"""
        results = {
            "nodes": {},
            "edges": {},
            "metadata": {
                "query_time": time.time(),
                "node_count": 0,
                "edge_count": 0,
                "start_nodes": query.start_nodes
            }
        }

        # Construct Cypher query based on GraphQuery object

        # Start with node criteria if no start nodes specified
        if not query.start_nodes:
            # Convert node criteria to Cypher WHERE clause
            node_where_clauses = []
            node_params = {}

            for key, value in query.node_criteria.items():
                # Handle special operators
                if isinstance(value, dict) and len(value) == 1:
                    op = list(value.keys())[0]
                    op_value = value[op]

                    if op == "$gt":
                        node_where_clauses.append(f"n.{key} > ${key}")
                        node_params[key] = op_value
                    elif op == "$lt":
                        node_where_clauses.append(f"n.{key} < ${key}")
                        node_params[key] = op_value
                    elif op == "$gte":
                        node_where_clauses.append(f"n.{key} >= ${key}")
                        node_params[key] = op_value
                    elif op == "$lte":
                        node_where_clauses.append(f"n.{key} <= ${key}")
                        node_params[key] = op_value
                    elif op == "$ne":
                        node_where_clauses.append(f"n.{key} <> ${key}")
                        node_params[key] = op_value
                    elif op == "$in":
                        node_where_clauses.append(f"n.{key} IN ${key}")
                        node_params[key] = op_value
                    elif op == "$contains":
                        node_where_clauses.append(f"n.{key} CONTAINS ${key}")
                        node_params[key] = op_value
                    else:
                        # Unknown operator, use equality
                        node_where_clauses.append(f"n.{key} = ${key}")
                        node_params[key] = value
                else:
                    # Direct equality
                    node_where_clauses.append(f"n.{key} = ${key}")
                    node_params[key] = value

            node_where_clause = " AND ".join(node_where_clauses) if node_where_clauses else "1=1"

            # Execute query to find matching start nodes
            try:
                with self.driver.session(database=self.database) as session:
                    cypher = f"""
                    MATCH (n:Node)
                    WHERE {node_where_clause}
                    RETURN n.id as id
                    """

                    result = session.run(cypher, **node_params)
                    start_nodes = [record["id"] for record in result]
            except Exception as e:
                self.logger.error(f"Error finding start nodes: {str(e)}")
                return results
        else:
            start_nodes = query.start_nodes

        # No start nodes found
        if not start_nodes:
            return results

        # Construct edge criteria WHERE clause
        edge_where_clauses = []
        edge_params = {}

        for key, value in query.edge_criteria.items():
            # Handle special operators (similar to node criteria)
            if isinstance(value, dict) and len(value) == 1:
                op = list(value.keys())[0]
                op_value = value[op]

                # Similar to node criteria processing
                param_key = f"edge_{key}"  # Avoid parameter name conflicts

                if op == "$gt":
                    edge_where_clauses.append(f"r.{key} > ${param_key}")
                    edge_params[param_key] = op_value
                elif op == "$lt":
                    edge_where_clauses.append(f"r.{key} < ${param_key}")
                    edge_params[param_key] = op_value
                # ... other operators similarly
                else:
                    # Default to equality
                    edge_where_clauses.append(f"r.{key} = ${param_key}")
                    edge_params[param_key] = value
            else:
                # Direct equality
                param_key = f"edge_{key}"
                edge_where_clauses.append(f"r.{key} = ${param_key}")
                edge_params[param_key] = value

        edge_where_clause = " AND ".join(edge_where_clauses) if edge_where_clauses else "1=1"

        # Add activation filter if needed
        activation_filter = ""
        if not query.include_inactive and query.min_activation > 0:
            activation_filter = f"AND n2.activation >= {query.min_activation}"

        # Execute the main query for each start node
        for start_node_id in start_nodes:
            try:
                with self.driver.session(database=self.database) as session:
                    # Construct a variable-length path query with filters
                    cypher = f"""
                    MATCH (n:Node {{id: $start_id}})-[r:RELATED_TO*1..{query.max_depth}]->(n2:Node)
                    WHERE {edge_where_clause} {activation_filter}
                    WITH n, r, n2
                    RETURN n.id as start_id, n2.id as node_id,
                           [rel in r | rel.id] as edge_ids,
                           n2 as node_data
                    LIMIT {query.max_results}
                    """

                    params = {"start_id": start_node_id, **edge_params}

                    result = session.run(cypher, **params)

                    for record in result:
                        node_id = record["node_id"]
                        node_data = record["node_data"]
                        edge_ids = record["edge_ids"]

                        # Add node to results if not already present
                        if node_id not in results["nodes"]:
                            # Convert node data to dictionary
                            node_dict = dict(node_data)

                            # Format for consistency with in-memory version
                            node_obj = {
                                "id": node_id,
                                "label": node_dict.get("label", "Node"),
                                "properties": {k: v for k, v in node_dict.items()
                                            if k not in ["id", "label", "activation",
                                                        "last_accessed", "importance",
                                                        "creation_time"]},
                                "activation": node_dict.get("activation", 0.0),
                                "last_accessed": node_dict.get("last_accessed", time.time()),
                                "importance": node_dict.get("importance", 0.5),
                                "creation_time": node_dict.get("creation_time", time.time())
                            }

                            results["nodes"][node_id] = node_obj

                        # Add edges to results
                        for edge_id in edge_ids:
                            if edge_id not in results["edges"]:
                                # Fetch edge details
                                edge = self.get_edge(edge_id)

                                # Format for consistency
                                edge_obj = {
                                    "id": edge.edge_id,
                                    "source": edge.source_id,
                                    "target": edge.target_id,
                                    "type": edge.type,
                                    "properties": edge.properties,
                                    "weight": edge.weight,
                                    "bidirectional": edge.bidirectional,
                                    "creation_time": edge.creation_time
                                }

                                results["edges"][edge_id] = edge_obj

            except Exception as e:
                self.logger.error(f"Error executing graph query for start node {start_node_id}: {str(e)}")
                # Continue with other start nodes

        # Update metadata
        results["metadata"]["node_count"] = len(results["nodes"])
        results["metadata"]["edge_count"] = len(results["edges"])

        return results

    def spread_activation(self, source_nodes: List[str], decay_factor: float = 0.5,
                         max_depth: int = 3) -> Dict[str, float]:
        """
        Spread activation from source nodes through the graph.

        Args:
            source_nodes: Starting nodes for activation
            decay_factor: How much activation decays per hop (0-1)
            max_depth: Maximum number of hops

        Returns:
            Dictionary mapping node IDs to activation values
        """
        # Choose appropriate implementation based on configuration
        if self.use_apoc:
            return self._spread_activation_apoc(source_nodes, decay_factor, max_depth)
        else:
            return self._spread_activation_cypher(source_nodes, decay_factor, max_depth)

    def _spread_activation_apoc(self, source_nodes: List[str], decay_factor: float,
                              max_depth: int) -> Dict[str, float]:
        """Spread activation using APOC procedures"""
        # Requires APOC installed on Neo4j

        # Initialize activation map
        activation = {}
        for node_id in source_nodes:
            activation[node_id] = 1.0  # Full activation

        try:
            with self.driver.session(database=self.database) as session:
                # Update activation for source nodes
                for node_id in source_nodes:
                    session.run("""
                        MATCH (n:Node {id: $id})
                        SET n.activation = 1.0,
                            n.last_accessed = $current_time
                    """, id=node_id, current_time=time.time())

                # Use APOC's path expander to spread activation
                cypher = """
                UNWIND $source_nodes AS source_id
                MATCH (source:Node {id: source_id})
                CALL apoc.path.expandConfig(source, {
                    relationshipFilter: 'RELATED_TO>',
                    minLevel: 1,
                    maxLevel: $max_depth
                }) YIELD path
                WITH path, source,
                     [n IN nodes(path) | n.id] AS node_ids,
                     [r IN relationships(path) | r.weight] AS edge_weights,
                     length(path) AS depth
                UNWIND range(0, size(node_ids) - 1) AS idx
                WITH node_ids[idx] AS node_id,
                     CASE WHEN idx = 0 THEN 1.0
                          ELSE REDUCE(weight = 1.0, w IN edge_weights[0..idx] |
                               weight * w * (1 - $decay_factor)) END AS act
                MERGE (target:Node {id: node_id})
                ON MATCH SET target.activation =
                    CASE WHEN target.activation < act
                    THEN act
                    ELSE target.activation END,
                    target.last_accessed = CASE WHEN act > 0.1
                                          THEN $current_time
                                          ELSE target.last_accessed END
                RETURN node_id, act as activation
                """

                result = session.run(
                    cypher,
                    source_nodes=source_nodes,
                    decay_factor=decay_factor,
                    max_depth=max_depth,
                    current_time=time.time()
                )

                # Collect results
                for record in result:
                    node_id = record["node_id"]
                    act = record["activation"]

                    # Update activation map (keep highest value)
                    activation[node_id] = max(activation.get(node_id, 0.0), act)

                return activation

        except Exception as e:
            self.logger.error(f"Error spreading activation (APOC): {str(e)}")
            return activation

    def _spread_activation_cypher(self, source_nodes: List[str], decay_factor: float,
                               max_depth: int) -> Dict[str, float]:
        """Spread activation using pure Cypher"""
        # Initialize activation map
        activation = {}
        for node_id in source_nodes:
            activation[node_id] = 1.0  # Full activation

        try:
            with self.driver.session(database=self.database) as session:
                # Update activation for source nodes
                current_time = time.time()
                for node_id in source_nodes:
                    session.run("""
                        MATCH (n:Node {id: $id})
                        SET n.activation = 1.0,
                            n.last_accessed = $current_time
                    """, id=node_id, current_time=current_time)

                # Spread activation level by level
                for depth in range(1, max_depth + 1):
                    # Propagation factor for this depth
                    prop_factor = (1 - decay_factor) ** depth

                    # Find nodes at exactly this depth from any source
                    cypher = """
                    UNWIND $source_nodes AS source_id
                    MATCH (source:Node {id: source_id}), (target:Node),
                          path = shortestPath((source)-[:RELATED_TO*..{depth}]->(target))
                    WHERE length(path) = $depth
                    WITH target.id AS node_id,
                         REDUCE(weight = 1.0, r IN relationships(path) |
                             weight * r.weight) AS path_weight
                    WITH node_id, MAX($prop_factor * path_weight) AS activation
                    MATCH (n:Node {id: node_id})
                    SET n.activation = CASE
                                        WHEN n.activation < activation
                                        THEN activation
                                        ELSE n.activation
                                      END,
                        n.last_accessed = CASE
                                            WHEN activation > 0.1
                                            THEN $current_time
                                            ELSE n.last_accessed
                                          END
                    RETURN node_id, activation
                    """

                    result = session.run(
                        cypher,
                        source_nodes=source_nodes,
                        depth=depth,
                        prop_factor=prop_factor,
                        current_time=current_time
                    )

                    # Update activation map
                    for record in result:
                        node_id = record["node_id"]
                        act = record["activation"]

                        # Keep highest activation
                        activation[node_id] = max(activation.get(node_id, 0.0), act)

                return activation

        except Exception as e:
            self.logger.error(f"Error spreading activation (Cypher): {str(e)}")
            return activation

    def get_activated_subgraph(self, threshold: float = 0.2) -> Tuple[List[str], List[str]]:
        """
        Get the currently activated subgraph (nodes with activation >= threshold).

        Args:
            threshold: Minimum activation threshold

        Returns:
            Tuple of (node_ids, edge_ids) lists
        """
        activated_nodes = []
        activated_edges = []

        try:
            with self.driver.session(database=self.database) as session:
                # Find activated nodes
                cypher_nodes = """
                MATCH (n:Node)
                WHERE n.activation >= $threshold
                RETURN n.id as id
                """

                result = session.run(cypher_nodes, threshold=threshold)
                activated_nodes = [record["id"] for record in result]

                # Find edges connecting activated nodes
                if activated_nodes:
                    cypher_edges = """
                    MATCH (s:Node)-[r:RELATED_TO]->(t:Node)
                    WHERE s.id IN $activated_nodes AND t.id IN $activated_nodes
                    RETURN r.id as id
                    """

                    result = session.run(cypher_edges, activated_nodes=activated_nodes)
                    activated_edges = [record["id"] for record in result]

                return activated_nodes, activated_edges

        except Exception as e:
            self.logger.error(f"Error getting activated subgraph: {str(e)}")
            return [], []

    def detect_patterns(self, min_frequency: int = 2, max_size: int = 5) -> List[GraphPattern]:
        """
        Detect frequent patterns in the graph.

        Args:
            min_frequency: Minimum number of occurrences for a pattern
            max_size: Maximum number of nodes in a pattern

        Returns:
            List of detected patterns
        """
        if self.use_apoc:
            return self._detect_patterns_apoc(min_frequency, max_size)
        else:
            return self._detect_patterns_basic(min_frequency, max_size)

    def _detect_patterns_apoc(self, min_frequency: int, max_size: int) -> List[GraphPattern]:
        """Detect patterns using APOC"""
        patterns = []

        try:
            with self.driver.session(database=self.database) as session:
                # Use APOC to find frequent subgraphs
                # This is a simplified version - real implementation would use
                # more sophisticated algorithms

                # Find frequent node label combinations
                cypher = """
                CALL apoc.periodic.iterate(
                    'MATCH (n:Node) RETURN n',
                    'WITH n
                     MATCH (n)-[r:RELATED_TO]->(m:Node)
                     WITH n.label as label1, r.type as rel_type, m.label as label2
                     RETURN label1, rel_type, label2, count(*) as freq
                     ORDER BY freq DESC',
                    {batchSize: 1000}
                )
                YIELD batches, total, timeTaken, committedOperations
                RETURN batches, total
                """

                # For now, use a simpler approach
                # Find frequent edge patterns
                cypher = """
                MATCH (n:Node)-[r:RELATED_TO]->(m:Node)
                WITH n.label as label1, r.type as rel_type, m.label as label2,
                     collect({source: n.id, target: m.id, edge: r.id}) as instances
                WHERE size(instances) >= $min_frequency
                RETURN label1, rel_type, label2, size(instances) as freq, instances[0..5] as sample
                ORDER BY freq DESC
                LIMIT 50
                """

                result = session.run(cypher, min_frequency=min_frequency)

                for record in result:
                    if record["sample"]:
                        pattern_id = str(uuid.uuid4())
                        sample = record["sample"][0]

                        patterns.append(GraphPattern(
                            pattern_id=pattern_id,
                            nodes=[sample["source"], sample["target"]],
                            edges=[sample["edge"]],
                            frequency=record["freq"],
                            significance=0.5 + (record["freq"] - min_frequency) * 0.1
                        ))

                return patterns

        except Exception as e:
            self.logger.error(f"Error detecting patterns (APOC): {str(e)}")
            return []

    def _detect_patterns_basic(self, min_frequency: int, max_size: int) -> List[GraphPattern]:
        """Basic pattern detection without APOC"""
        patterns = []

        try:
            with self.driver.session(database=self.database) as session:
                # Find frequent edge type patterns
                cypher = """
                MATCH (n:Node)-[r:RELATED_TO]->(m:Node)
                WITH n.label as source_label, r.type as edge_type, m.label as target_label,
                     count(*) as freq,
                     collect({source: n.id, target: m.id, edge: r.id})[0..5] as samples
                WHERE freq >= $min_frequency
                RETURN source_label, edge_type, target_label, freq, samples
                ORDER BY freq DESC
                LIMIT 50
                """

                result = session.run(cypher, min_frequency=min_frequency)

                for record in result:
                    if record["samples"]:
                        pattern_id = str(uuid.uuid4())
                        first_sample = record["samples"][0]

                        patterns.append(GraphPattern(
                            pattern_id=pattern_id,
                            nodes=[first_sample["source"], first_sample["target"]],
                            edges=[first_sample["edge"]],
                            frequency=record["freq"],
                            significance=0.5 + (record["freq"] - min_frequency) * 0.1
                        ))

                # Also look for star patterns (nodes with many connections)
                if max_size >= 3:
                    cypher = """
                    MATCH (center:Node)-[r:RELATED_TO]-(neighbor:Node)
                    WITH center, count(DISTINCT neighbor) as degree
                    WHERE degree >= $min_frequency
                    WITH center, degree
                    ORDER BY degree DESC
                    LIMIT 10
                    MATCH (center)-[r:RELATED_TO]-(neighbor:Node)
                    WITH center, degree, collect(neighbor.id)[0..5] as neighbors,
                         collect(r.id)[0..5] as edges
                    RETURN center.id as center_id, degree, neighbors, edges
                    """

                    result = session.run(cypher, min_frequency=min_frequency)

                    for record in result:
                        pattern_id = str(uuid.uuid4())
                        nodes = [record["center_id"]] + record["neighbors"]

                        patterns.append(GraphPattern(
                            pattern_id=pattern_id,
                            nodes=nodes[:max_size],
                            edges=record["edges"][:max_size-1],
                            frequency=record["degree"],
                            significance=0.5 + (record["degree"] - min_frequency) * 0.05
                        ))

                return patterns

        except Exception as e:
            self.logger.error(f"Error detecting patterns (basic): {str(e)}")
            return []

    def clear(self) -> None:
        """Clear all nodes and edges from the graph"""
        try:
            with self.driver.session(database=self.database) as session:
                # Delete all nodes and relationships
                session.run("MATCH (n:Node) DETACH DELETE n")

                # Clear caches
                self.node_cache.clear()
                self.edge_cache.clear()

                self.logger.info("Graph cleared")

        except Exception as e:
            self.logger.error(f"Error clearing graph: {str(e)}")
            raise GraphEngineError(f"Failed to clear graph: {str(e)}")

    def serialize(self) -> Dict[str, Any]:
        """Serialize the graph to a dictionary"""
        nodes_data = {}
        edges_data = {}

        try:
            with self.driver.session(database=self.database) as session:
                # Export nodes
                cypher_nodes = """
                MATCH (n:Node)
                RETURN n
                """

                result = session.run(cypher_nodes)

                for record in result:
                    node_data = dict(record["n"])
                    node_id = node_data.pop("id")

                    # Extract standard properties
                    nodes_data[node_id] = {
                        "label": node_data.pop("label", "Node"),
                        "properties": {k: v for k, v in node_data.items()
                                     if k not in ["activation", "last_accessed",
                                                 "importance", "creation_time"]},
                        "activation": node_data.get("activation", 0.0),
                        "last_accessed": node_data.get("last_accessed", time.time()),
                        "importance": node_data.get("importance", 0.5),
                        "creation_time": node_data.get("creation_time", time.time())
                    }

                # Export edges
                cypher_edges = """
                MATCH ()-[r:RELATED_TO]->()
                RETURN r, startNode(r).id as source_id, endNode(r).id as target_id
                """

                result = session.run(cypher_edges)

                for record in result:
                    edge_data = dict(record["r"])
                    edge_id = edge_data.pop("id")

                    # Extract standard properties
                    edges_data[edge_id] = {
                        "source_id": record["source_id"],
                        "target_id": record["target_id"],
                        "type": edge_data.pop("type", "RELATED_TO"),
                        "properties": {k: v for k, v in edge_data.items()
                                     if k not in ["weight", "bidirectional", "creation_time"]},
                        "weight": edge_data.get("weight", 1.0),
                        "bidirectional": edge_data.get("bidirectional", False),
                        "creation_time": edge_data.get("creation_time", time.time())
                    }

                return {
                    "nodes": nodes_data,
                    "edges": edges_data,
                    "metadata": {
                        "timestamp": time.time(),
                        "instance_id": self.instance_id,
                        "node_count": len(nodes_data),
                        "edge_count": len(edges_data)
                    }
                }

        except Exception as e:
            self.logger.error(f"Error serializing graph: {str(e)}")
            raise GraphEngineError(f"Failed to serialize graph: {str(e)}")

    def deserialize(self, data: Dict[str, Any]) -> None:
        """Load the graph from a serialized representation"""
        # Clear existing graph
        self.clear()

        try:
            with self.driver.session(database=self.database) as session:
                # Import nodes
                for node_id, node_data in data.get("nodes", {}).items():
                    # Prepare node properties
                    props = node_data.get("properties", {})
                    props.update({
                        "id": node_id,
                        "label": node_data.get("label", "Node"),
                        "activation": node_data.get("activation", 0.0),
                        "last_accessed": node_data.get("last_accessed", time.time()),
                        "importance": node_data.get("importance", 0.5),
                        "creation_time": node_data.get("creation_time", time.time())
                    })

                    # Create node
                    cypher = """
                    CREATE (n:Node $props)
                    """

                    session.run(cypher, props=props)

                # Import edges
                for edge_id, edge_data in data.get("edges", {}).items():
                    # Create edge
                    cypher = """
                    MATCH (s:Node {id: $source_id}), (t:Node {id: $target_id})
                    CREATE (s)-[r:RELATED_TO {
                        id: $edge_id,
                        type: $edge_type,
                        weight: $weight,
                        bidirectional: $bidirectional,
                        creation_time: $creation_time
                    }]->(t)
                    SET r += $properties
                    """

                    session.run(
                        cypher,
                        source_id=edge_data["source_id"],
                        target_id=edge_data["target_id"],
                        edge_id=edge_id,
                        edge_type=edge_data.get("type", "RELATED_TO"),
                        weight=edge_data.get("weight", 1.0),
                        bidirectional=edge_data.get("bidirectional", False),
                        creation_time=edge_data.get("creation_time", time.time()),
                        properties=edge_data.get("properties", {})
                    )

                self.logger.info(f"Loaded graph with {len(data.get('nodes', {}))} nodes and {len(data.get('edges', {}))} edges")

        except Exception as e:
            self.logger.error(f"Error deserializing graph: {str(e)}")
            raise GraphEngineError(f"Failed to deserialize graph: {str(e)}")

    def close(self) -> None:
        """Close the graph engine and release resources"""
        try:
            # Close the Neo4j driver
            self.driver.close()

            # Clear caches
            self.node_cache.clear()
            self.edge_cache.clear()

            self.logger.info("Neo4j connection closed")

        except Exception as e:
            self.logger.error(f"Error closing Neo4j connection: {str(e)}")


# -----------------------------------------------------------------------------
# Factory function to create graph engines
# -----------------------------------------------------------------------------

def create_graph_engine(engine_type: str, **kwargs) -> AbstractGraphEngine:
    """
    Factory function to create the appropriate graph engine.

    Args:
        engine_type: Type of engine to create ("inmemory" or "neo4j")
        **kwargs: Arguments to pass to the engine constructor

    Returns:
        An instance of the appropriate graph engine

    Raises:
        ValueError: If engine_type is not recognized
        ImportError: If required dependencies are not available
    """
    engine_type = engine_type.lower()

    if engine_type == "inmemory":
        return InMemoryGraphEngine(**kwargs)

    elif engine_type == "neo4j":
        if not HAS_NEO4J:
            raise ImportError("Neo4j Python driver is not installed. Install with: pip install neo4j")

        # Validate required parameters
        required_params = ["uri", "user", "password"]
        missing_params = [p for p in required_params if p not in kwargs]

        if missing_params:
            raise ValueError(f"Missing required Neo4j parameters: {missing_params}")

        return Neo4jGraphEngine(**kwargs)

    else:
        raise ValueError(f"Unknown engine type: {engine_type}. Supported types: inmemory, neo4j")

# --- fix 6.5: name expected by pattern_recognizer, reasoning_core, llm_engine
GraphEngine = AbstractGraphEngine
