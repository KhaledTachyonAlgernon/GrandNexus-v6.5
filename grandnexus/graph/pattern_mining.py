"""
pattern_mining.py - Advanced Pattern Mining & Anomaly Detection for GrandNexus

This module provides advanced pattern analysis capabilities for the graph structure:
- Frequent subgraph mining
- Pattern matching and similarity search
- Anomaly detection
- Subgraph embedding and comparison

It works in tandem with the graph_engine.py module while maintaining separation of concerns:
- graph_engine.py handles the basic graph operations and structure
- pattern_mining.py provides higher-level cognitive pattern analysis
"""

import logging
import time
import threading
import math
from typing import Dict, Any, List, Optional, Union, Tuple, Set, Callable
from dataclasses import dataclass, field
from collections import Counter, defaultdict
import uuid

from grandnexus.graph.graph_engine import ( create_graph_engine,
    AbstractGraphEngine, GraphNode, GraphEdge, GraphPattern,
    GraphQuery, NodeNotFoundError, EdgeNotFoundError
)
from grandnexus.graph.graph_pattern_interface import (
    PatternMiningInterface, PatternMatchResult, FrequentPattern
)

# Optional dependencies for vector operations
try:
    import numpy as np
    HAS_NUMPY = True
except ImportError:
    HAS_NUMPY = False

# Optional dependency for graph algorithms
try:
    import networkx as nx
    HAS_NETWORKX = True
except ImportError:
    HAS_NETWORKX = False


class PatternMiningModule(PatternMiningInterface):
    """
    Provides advanced pattern-mining capabilities for GrandNexus.

    Works with GraphEngine to discover patterns, detect anomalies,
    compare subgraphs, and extract knowledge structures from the graph.
    """

    def __init__(self,
                 graph_engine: AbstractGraphEngine,
                 use_networkx: bool = True,
                 default_similarity_threshold: float = 0.7,
                 embedding_dim: int = 128,
                 pattern_cache_size: int = 1000):
        """
        Initialize the pattern mining module.

        Args:
            graph_engine: The graph engine instance to work with
            use_networkx: Whether to use NetworkX for advanced graph algorithms
            default_similarity_threshold: Default threshold for pattern matching
            embedding_dim: Dimension for pattern embeddings
            pattern_cache_size: Maximum number of patterns to cache
        """
        self.logger = logging.getLogger("PatternMiningModule")
        self.graph_engine = graph_engine
        self.use_networkx = use_networkx and HAS_NETWORKX
        self.default_similarity_threshold = default_similarity_threshold
        self.embedding_dim = embedding_dim
        self.pattern_cache_size = pattern_cache_size

        # Cache for discovered patterns
        self._pattern_cache = {}
        self._frequent_patterns = []
        self._nx_graph = None

        # Concurrency control
        self.lock = threading.RLock()

        # Initialize NetworkX graph if enabled
        if self.use_networkx:
            self._build_networkx_graph_if_needed()

        self.logger.info(f"PatternMiningModule initialized. NetworkX={self.use_networkx}, "
                        f"similarity_threshold={self.default_similarity_threshold:.2f}")

    def _build_networkx_graph_if_needed(self) -> Optional["nx.Graph"]:
        """Build or update the NetworkX representation of the graph."""
        if not self.use_networkx:
            return None

        with self.lock:
            try:
                self.logger.debug("Building/updating NetworkX graph representation")

                # Create a new MultiDiGraph to support multiple edge types between nodes
                g = nx.MultiDiGraph()

                # Get all nodes from the graph engine
                # We'll use a graph query to retrieve all nodes
                query = GraphQuery(
                    max_results=100000  # Set a high limit to get all nodes
                )

                result = self.graph_engine.execute_query(query)
                nodes_data = result.get('nodes', [])
                edges_data = result.get('edges', [])

                # Add nodes to NetworkX graph
                for node_data in nodes_data:
                    node_id = node_data['id']
                    properties = node_data.get('properties', {})
                    label = node_data.get('label', 'Node')

                    # Prepare node attributes
                    attrs = dict(properties)
                    attrs['label'] = label
                    attrs['activation'] = node_data.get('activation', 0.0)
                    attrs['importance'] = node_data.get('importance', 0.5)

                    g.add_node(node_id, **attrs)

                # Add edges to NetworkX graph
                for edge_data in edges_data:
                    source_id = edge_data['source']
                    target_id = edge_data['target']
                    edge_id = edge_data['id']
                    edge_type = edge_data.get('type', 'RELATES')
                    properties = edge_data.get('properties', {})

                    # Prepare edge attributes
                    attrs = dict(properties)
                    attrs['edge_id'] = edge_id
                    attrs['type'] = edge_type
                    attrs['weight'] = edge_data.get('weight', 1.0)

                    g.add_edge(source_id, target_id, key=edge_id, **attrs)

                self._nx_graph = g
                self.logger.debug(f"NetworkX graph built with {g.number_of_nodes()} nodes "
                                f"and {g.number_of_edges()} edges")
                return g

            except Exception as e:
                self.logger.error(f"Error building NetworkX graph: {e}")
                return None

    def find_patterns(self,
                      pattern_spec: Dict[str, Any],
                      similarity_threshold: Optional[float] = None,
                      max_results: int = 10,
                      use_cached: bool = True) -> List[PatternMatchResult]:
        """
        Searches the graph for subgraphs matching the given pattern specification.

        Args:
            pattern_spec: Dictionary describing the pattern to match
            similarity_threshold: Minimum similarity score (defaults to self.default_similarity_threshold)
            max_results: Maximum number of results to return
            use_cached: Whether to use/update the pattern cache

        Returns:
            List of PatternMatchResult objects
        """
        with self.lock:
            # Generate a cache key for this pattern
            pattern_hash = str(hash(str(pattern_spec)))
            if use_cached and pattern_hash in self._pattern_cache:
                self.logger.debug(f"Using cached pattern results for {pattern_hash}")
                return self._pattern_cache[pattern_hash][:max_results]

            similarity_threshold = similarity_threshold or self.default_similarity_threshold
            self.logger.info(f"Finding patterns with threshold={similarity_threshold:.2f}")

            # Choose implementation based on available tools
            if self.use_networkx and self._nx_graph:
                results = self._find_patterns_networkx(pattern_spec, similarity_threshold)
            else:
                results = self._find_patterns_direct(pattern_spec, similarity_threshold)

            # Sort by similarity score (descending)
            results.sort(key=lambda x: x.similarity_score, reverse=True)

            # Cache the results if requested
            if use_cached:
                self._pattern_cache[pattern_hash] = results

                # Prune cache if it exceeds the maximum size
                if len(self._pattern_cache) > self.pattern_cache_size:
                    # Remove the oldest entries
                    keys_to_remove = sorted(self._pattern_cache.keys())[:len(self._pattern_cache) - self.pattern_cache_size]
                    for key in keys_to_remove:
                        del self._pattern_cache[key]

            return results[:max_results]

    def _find_patterns_networkx(self, pattern_spec: Dict[str, Any],
                              similarity_threshold: float) -> List[PatternMatchResult]:
        """Find patterns using NetworkX graph algorithms."""
        results = []

        # Extract pattern constraints
        pattern_nodes = pattern_spec.get("nodes", [])
        pattern_edges = pattern_spec.get("edges", [])
        structure_info = pattern_spec.get("structure", {})
        max_depth = structure_info.get("max_depth", 2)

        if not pattern_nodes:
            return []

        # For each node in the graph, try it as a starting point
        for start_node in self._nx_graph.nodes():
            # Extract local subgraph around this node
            subgraph = nx.ego_graph(self._nx_graph, start_node, radius=max_depth)

            # Skip tiny subgraphs that are too small to match the pattern
            if len(subgraph) < len(pattern_nodes):
                continue

            # Check if the subgraph matches the pattern
            match_result = self._check_subgraph_match(subgraph, pattern_spec)

            if match_result and match_result.similarity_score >= similarity_threshold:
                results.append(match_result)

        return results

    def _find_patterns_direct(self, pattern_spec: Dict[str, Any],
                           similarity_threshold: float) -> List[PatternMatchResult]:
        """Find patterns using direct GraphEngine queries."""
        results = []

        # Extract pattern constraints
        pattern_nodes = pattern_spec.get("nodes", [])
        if not pattern_nodes:
            return []

        # Find candidate starting nodes
        first_node_spec = pattern_nodes[0]
        node_type = first_node_spec.get("type", "Node")
        constraints = first_node_spec.get("constraints", {})

        # Convert to GraphQuery
        query = GraphQuery(
            node_criteria={"type": node_type, **constraints},
            max_depth=1
        )

        try:
            # Execute the query to find potential starting nodes
            query_result = self.graph_engine.execute_query(query)
            candidate_nodes = [node['id'] for node in query_result.get('nodes', [])]

            # For each candidate node, expand and check if it matches the pattern
            for node_id in candidate_nodes:
                # Create a simple pattern match with this as the starting point
                result = self._check_direct_pattern_match(node_id, pattern_spec)

                if result and result.similarity_score >= similarity_threshold:
                    results.append(result)

                    # Limit results to avoid excessive computation
                    if len(results) >= 100:  # Internal limit
                        break

        except Exception as e:
            self.logger.error(f"Error finding patterns directly: {e}")

        return results

    def _check_subgraph_match(self, subgraph: "nx.Graph",
                            pattern_spec: Dict[str, Any]) -> Optional[PatternMatchResult]:
        """Check if a subgraph matches the pattern specification."""
        # Extract pattern constraints
        pattern_nodes = pattern_spec.get("nodes", [])
        pattern_edges = pattern_spec.get("edges", [])

        # Count nodes matching each required type
        node_type_matches = defaultdict(int)
        node_ids_by_type = defaultdict(list)

        for node_id in subgraph.nodes():
            node_data = subgraph.nodes[node_id]
            node_type = node_data.get("type", node_data.get("label", "unknown"))
            node_type_matches[node_type] += 1
            node_ids_by_type[node_type].append(node_id)

        # Check if we have enough nodes of each required type
        pattern_node_types = [n.get("type", "Node") for n in pattern_nodes]
        required_type_counts = Counter(pattern_node_types)

        # If we're missing any required node type, return None
        for req_type, count in required_type_counts.items():
            if node_type_matches[req_type] < count:
                return None

        # Check edges (simplified)
        edge_type_matches = defaultdict(int)
        edge_ids_by_type = defaultdict(list)

        for u, v, k, edge_data in subgraph.edges(data=True, keys=True):
            rel_type = edge_data.get("type", edge_data.get("relation_type", "unknown"))
            edge_type_matches[rel_type] += 1
            edge_ids_by_type[rel_type].append(edge_data.get("edge_id", k))

        # Check if we have the required edges
        pattern_edge_types = [e.get("type", "RELATES") for e in pattern_edges]
        required_edge_counts = Counter(pattern_edge_types)

        # If we're missing any required edge type, return None
        for req_type, count in required_edge_counts.items():
            if edge_type_matches[req_type] < count:
                return None

        # Calculate similarity score

        # Node type similarity
        node_type_sim = 0.0
        if required_type_counts:
            node_type_sim = sum(min(node_type_matches[t], required_type_counts[t])
                             for t in required_type_counts) / sum(required_type_counts.values())

        # Edge type similarity
        edge_type_sim = 1.0
        if required_edge_counts:
            edge_type_sim = sum(min(edge_type_matches[t], required_edge_counts[t])
                             for t in required_edge_counts) / sum(required_edge_counts.values())

        # Combined similarity
        similarity = (node_type_sim + edge_type_sim) / 2.0

        # Create the match result
        all_matched_nodes = []
        for node_type in required_type_counts:
            # Take only as many nodes as required by the pattern
            all_matched_nodes.extend(node_ids_by_type[node_type][:required_type_counts[node_type]])

        all_matched_edges = []
        for edge_type in required_edge_counts:
            # Take only as many edges as required by the pattern
            all_matched_edges.extend(edge_ids_by_type[edge_type][:required_edge_counts[edge_type]])

        return PatternMatchResult(
            subgraph_nodes=all_matched_nodes,
            subgraph_edges=all_matched_edges,
            similarity_score=similarity,
            pattern_type=pattern_spec.get("pattern_type", "custom"),
            extra_info={
                "node_type_similarity": node_type_sim,
                "edge_type_similarity": edge_type_sim,
                "matched_node_types": dict(node_type_matches),
                "matched_edge_types": dict(edge_type_matches)
            }
        )

    def _check_direct_pattern_match(self, start_node_id: str,
                                 pattern_spec: Dict[str, Any]) -> Optional[PatternMatchResult]:
        """Check if a pattern matches starting from a specific node using direct graph queries."""
        # Extract pattern constraints
        pattern_nodes = pattern_spec.get("nodes", [])
        pattern_edges = pattern_spec.get("edges", [])
        max_depth = pattern_spec.get("structure", {}).get("max_depth", 2)

        # Start with the first node matching the first pattern node
        matched_nodes = [start_node_id]
        matched_edges = []

        # Get neighboring nodes to a certain depth
        try:
            # For simplicity, just get the neighbors up to max_depth
            # In a real implementation, this would be more sophisticated
            current_nodes = [start_node_id]
            next_nodes = []
            visited = {start_node_id}

            for depth in range(max_depth):
                for node_id in current_nodes:
                    neighbors = self.graph_engine.get_node_neighbors(node_id, direction="both")

                    for neighbor_id, edge_id, edge_type in neighbors:
                        if neighbor_id not in visited:
                            visited.add(neighbor_id)
                            next_nodes.append(neighbor_id)

                            # Get the node to check its type
                            try:
                                neighbor = self.graph_engine.get_node(neighbor_id)

                                # Check if this matches any required node type
                                for pattern_node in pattern_nodes[1:]:  # Skip first, we already matched it
                                    required_type = pattern_node.get("type", "Node")
                                    if neighbor.label == required_type:
                                        matched_nodes.append(neighbor_id)
                                        matched_edges.append(edge_id)
                                        break
                            except NodeNotFoundError:
                                pass

                current_nodes = next_nodes
                next_nodes = []

            # Calculate a simple similarity score based on how many pattern nodes we matched
            nodes_matched = len(matched_nodes)
            nodes_required = len(pattern_nodes)

            similarity = nodes_matched / nodes_required if nodes_required > 0 else 0.0

            if similarity > 0:
                return PatternMatchResult(
                    subgraph_nodes=matched_nodes,
                    subgraph_edges=matched_edges,
                    similarity_score=similarity,
                    pattern_type=pattern_spec.get("pattern_type", "custom"),
                    extra_info={
                        "direct_match": True,
                        "starting_node": start_node_id
                    }
                )

        except Exception as e:
            self.logger.error(f"Error checking direct pattern match: {e}")

        return None

    def extract_frequent_subgraphs(self,
                                min_support: int = 2,
                                max_size: int = 5,
                                store_embeddings: bool = True) -> List[FrequentPattern]:
        """
        Discovers frequently occurring patterns in the graph.

        Args:
            min_support: Minimum number of occurrences to be considered frequent
            max_size: Maximum size (nodes) of patterns to consider
            store_embeddings: Whether to compute and store embeddings for patterns

        Returns:
            List of FrequentPattern objects
        """
        with self.lock:
            self.logger.info(f"Extracting frequent subgraphs (min_support={min_support}, max_size={max_size})")

            if self.use_networkx and self._nx_graph:
                patterns = self._extract_frequent_subgraphs_networkx(min_support, max_size)
            else:
                patterns = self._extract_frequent_subgraphs_direct(min_support, max_size)

            # Compute embeddings if requested
            if store_embeddings and patterns:
                for pattern in patterns:
                    if pattern.pattern_embedding is None:
                        # Create a temporary PatternMatchResult to compute the embedding
                        temp_result = PatternMatchResult(
                            subgraph_nodes=pattern.nodes,
                            subgraph_edges=pattern.edges
                        )
                        pattern.pattern_embedding = self._compute_pattern_embedding(temp_result)

            # Sort by support (descending)
            patterns.sort(key=lambda x: x.support, reverse=True)

            # Update the cache
            self._frequent_patterns = patterns

            self.logger.info(f"Discovered {len(patterns)} frequent patterns")
            return patterns

    def _extract_frequent_subgraphs_networkx(self, min_support: int, max_size: int) -> List[FrequentPattern]:
        """Extract frequent subgraphs using NetworkX."""
        if not HAS_NETWORKX or not self._nx_graph:
            return []

        # This is a simplified approach using subgraph sampling
        # A real implementation would use algorithms like gSpan, Subdue, etc.

        pattern_candidates = defaultdict(lambda: {"nodes": [], "edges": [], "occurrences": []})

        # Sample random starting points
        node_list = list(self._nx_graph.nodes())
        if len(node_list) < 3:
            self.logger.warning("Graph is too small for meaningful pattern mining")
            return []

        # For small graphs, use all nodes. For large graphs, sample a subset.
        sample_size = min(len(node_list), 100)
        try:
            starting_nodes = np.random.choice(node_list, size=sample_size, replace=False)
        except Exception:
            # Fallback if numpy is not available
            starting_nodes = node_list[:min(100, len(node_list))]

        for start_node in starting_nodes:
            # For each size from 2 to max_size
            for size in range(2, max_size + 1):
                # Get a local subgraph of approximately this size
                subgraph = self._sample_local_subgraph(start_node, size)

                if len(subgraph) < 2:  # Skip trivial subgraphs
                    continue

                # Compute a canonical signature for this subgraph
                signature = self._compute_subgraph_signature(subgraph)

                # Add to candidates
                node_ids = list(subgraph.nodes())
                edge_ids = []
                for u, v, k, data in subgraph.edges(data=True, keys=True):
                    edge_id = data.get("edge_id", k)
                    edge_ids.append(edge_id)

                pattern_candidates[signature]["nodes"].extend(node_ids)
                pattern_candidates[signature]["edges"].extend(edge_ids)
                pattern_candidates[signature]["occurrences"].append((node_ids, edge_ids))

        # Filter by support and convert to FrequentPattern objects
        frequent_patterns = []
        pattern_id_counter = 0

        for signature, info in pattern_candidates.items():
            occurrences = info["occurrences"]
            if len(occurrences) >= min_support:
                # Create a FrequentPattern
                pattern_id = f"pattern_{pattern_id_counter}"
                pattern_id_counter += 1

                # Use the first occurrence as the example
                example_nodes, example_edges = occurrences[0]

                pattern = FrequentPattern(
                    pattern_id=pattern_id,
                    nodes=example_nodes,
                    edges=example_edges,
                    support=len(occurrences),
                    signature=signature
                )

                frequent_patterns.append(pattern)

        return frequent_patterns

    def _extract_frequent_subgraphs_direct(self, min_support: int, max_size: int) -> List[FrequentPattern]:
        """Extract frequent subgraphs using direct GraphEngine queries."""
        # Simplified approach as a fallback when NetworkX is not available

        # Strategy:
        # 1. Group nodes by type/label
        # 2. For each label, find pairs of connected nodes
        # 3. Expand these pairs to form larger subgraphs
        # 4. Count occurrences of similar structures

        # Get all nodes with their labels
        nodes_by_label = defaultdict(list)

        # Query for all nodes
        query = GraphQuery(max_results=100000)  # High limit to get all nodes
        result = self.graph_engine.execute_query(query)

        for node in result.get('nodes', []):
            node_id = node['id']
            label = node.get('label', 'Node')
            nodes_by_label[label].append(node_id)

        # Find frequent patterns
        patterns = []
        pattern_id_counter = 0

        # Look for star patterns (nodes with multiple connections of the same type)
        # This is a simplified approach compared to full graph mining
        for label, node_ids in nodes_by_label.items():
            for node_id in node_ids:
                # Get all neighbors
                neighbors = self.graph_engine.get_node_neighbors(node_id, direction="both")

                # Group neighbors by edge type
                neighbors_by_type = defaultdict(list)
                edges_by_type = defaultdict(list)

                for neighbor_id, edge_id, edge_type in neighbors:
                    neighbors_by_type[edge_type].append(neighbor_id)
                    edges_by_type[edge_type].append(edge_id)

                # Look for edge types with multiple occurrences
                for edge_type, neighbor_ids in neighbors_by_type.items():
                    if len(neighbor_ids) >= min_support and len(neighbor_ids) + 1 <= max_size:
                        # We found a star pattern
                        pattern_id = f"pattern_{pattern_id_counter}"
                        pattern_id_counter += 1

                        # Create a pattern with the center node and its neighbors
                        pattern_nodes = [node_id] + neighbor_ids
                        pattern_edges = edges_by_type[edge_type]

                        # Create a signature
                        signature = f"star:{label}:{edge_type}:{len(neighbor_ids)}"

                        pattern = FrequentPattern(
                            pattern_id=pattern_id,
                            nodes=pattern_nodes,
                            edges=pattern_edges,
                            support=len(neighbor_ids),
                            signature=signature
                        )

                        patterns.append(pattern)

        return patterns

    def _sample_local_subgraph(self, start_node: str, target_size: int) -> "nx.Graph":
        """
        Samples a local subgraph of approximately target_size nodes starting from start_node.

        Args:
            start_node: Starting node ID
            target_size: Desired number of nodes

        Returns:
            A NetworkX graph (subgraph of original)
        """
        if not self._nx_graph or not self.use_networkx:
            return nx.Graph()  # Empty graph as fallback

        # Use BFS to expand from start_node
        visited = {start_node}
        frontier = [start_node]

        while len(visited) < target_size and frontier:
            current = frontier.pop(0)

            # Add neighbors in random order
            neighbors = list(self._nx_graph.neighbors(current))

            # Shuffle neighbors if numpy is available
            if HAS_NUMPY:
                try:
                    np.random.shuffle(neighbors)
                except Exception:
                    pass

            for neighbor in neighbors:
                if neighbor not in visited:
                    visited.add(neighbor)
                    frontier.append(neighbor)

                    if len(visited) >= target_size:
                        break

        # Extract the subgraph
        return self._nx_graph.subgraph(visited)

    def _compute_subgraph_signature(self, subgraph: "nx.Graph") -> str:
        """
        Computes a canonical signature for a subgraph.

        This is a simplified version - a real implementation would use more sophisticated
        isomorphism-invariant signatures.

        Args:
            subgraph: A NetworkX graph

        Returns:
            A string signature
        """
        # Count node types
        node_types = Counter()
        for node, data in subgraph.nodes(data=True):
            node_type = data.get("type", data.get("label", "unknown"))
            node_types[node_type] += 1

        # Count edge types
        edge_types = Counter()
        for u, v, data in subgraph.edges(data=True):
            if isinstance(data, dict):
                edge_type = data.get("type", data.get("relation_type", "unknown"))
                edge_types[edge_type] += 1

        # Create signature parts
        signature_parts = []

        # Add sorted node types
        node_sig = ";".join(f"{t}:{c}" for t, c in sorted(node_types.items()))
        signature_parts.append(f"N[{node_sig}]")

        # Add sorted edge types
        edge_sig = ";".join(f"{t}:{c}" for t, c in sorted(edge_types.items()))
        signature_parts.append(f"E[{edge_sig}]")

        # Add basic structural info
        signature_parts.append(f"S[{len(subgraph)}:{subgraph.number_of_edges()}]")

        return "|".join(signature_parts)

    def _compute_pattern_embedding(self, pattern: PatternMatchResult) -> Optional[Any]:
        """
        Computes a vector embedding for a pattern.

        Args:
            pattern: A PatternMatchResult object

        Returns:
            Numpy array or None if embeddings not supported
        """
        if not HAS_NUMPY:
            return None

        embedding_features = []
        node_ids = pattern.subgraph_nodes

        if not node_ids:
            return np.zeros(self.embedding_dim)

        # If we don't have NetworkX or it's not enabled, use a simpler approach
        if not self.use_networkx or not self._nx_graph:
            # Create a basic embedding using node and edge counts
            try:
                # Get node types
                node_types = Counter()
                for node_id in node_ids:
                    try:
                        node = self.graph_engine.get_node(node_id)
                        node_types[node.label] += 1
                    except NodeNotFoundError:
                        pass

                # Get edge types
                edge_types = Counter()
                for edge_id in pattern.subgraph_edges:
                    try:
                        edge = self.graph_engine.get_edge(edge_id)
                        edge_types[edge.type] += 1
                    except EdgeNotFoundError:
                        pass

                # Add node type counts as features
                for node_type, count in sorted(node_types.items()):
                    embedding_features.append(count)

                # Add edge type counts as features
                for edge_type, count in sorted(edge_types.items()):
                    embedding_features.append(count)

                # Add basic size information
                embedding_features.append(len(node_ids))
                embedding_features.append(len(pattern.subgraph_edges))

            except Exception as e:
                self.logger.warning(f"Error computing embedding: {e}")
                embedding_features = [0] * 5  # Fallback

        else:
            # Use NetworkX for more sophisticated embeddings
            try:
                # Extract the subgraph
                subgraph = self._nx_graph.subgraph(node_ids)

                # Get node type distribution
                node_types = Counter()
                for node_id in node_ids:
                    if node_id in self._nx_graph:
                        node_data = self._nx_graph.nodes[node_id]
                        node_type = node_data.get("type", node_data.get("label", "unknown"))
                        node_types[node_type] += 1

                # Get edge type distribution
                edge_types = Counter()
                for edge_id in pattern.subgraph_edges:
                    # Find the edge with this ID
                    for u, v, k, data in self._nx_graph.edges(data=True, keys=True):
                        if data.get("edge_id") == edge_id or k == edge_id:
                            edge_type = data.get("type", data.get("relation_type", "unknown"))
                            edge_types[edge_type] += 1
                            break

                # Add basic graph metrics as features
                if len(node_ids) > 1:
                    density = nx.density(subgraph)
                    embedding_features.append(density)

                    # Try to compute clustering
                    try:
                        clustering = nx.average_clustering(subgraph.to_undirected())
                        embedding_features.append(clustering)
                    except Exception:
                        embedding_features.append(0.0)

                    # Add degree distribution features
                    degrees = [d for _, d in subgraph.degree()]
                    if degrees:
                        avg_degree = sum(degrees) / len(degrees)
                        max_degree = max(degrees) if degrees else 0
                        embedding_features.extend([avg_degree, max_degree])
                else:
                    # Placeholders for single-node patterns
                    embedding_features.extend([0.0, 0.0, 0.0, 0.0])

                # Add node and edge type features
                for node_type, count in sorted(node_types.items()):
                    embedding_features.append(count)

                for edge_type, count in sorted(edge_types.items()):
                    embedding_features.append(count)

            except Exception as e:
                self.logger.warning(f"Error computing topological features: {e}")
                embedding_features = [0] * 5  # Fallback

        # Convert to numpy array and pad/truncate to fixed dimension
        embedding = np.array(embedding_features, dtype=np.float32)

        # Pad or truncate to the desired dimension
        if len(embedding) < self.embedding_dim:
            embedding = np.pad(embedding, (0, self.embedding_dim - len(embedding)))
        elif len(embedding) > self.embedding_dim:
            embedding = embedding[:self.embedding_dim]

        # Normalize
        norm = np.linalg.norm(embedding)
        if norm > 0:
            embedding = embedding / norm

        return embedding

    def find_similar_patterns(self,
                            pattern: Union[PatternMatchResult, Dict[str, Any]],
                            top_n: int = 5) -> List[Tuple[PatternMatchResult, float]]:
        """
        Finds patterns similar to the provided pattern using embedding similarity.

        Args:
            pattern: Either a PatternMatchResult or a pattern specification dict
            top_n: Maximum number of results to return

        Returns:
            List of (pattern, similarity_score) tuples
        """
        with self.lock:
            # Convert pattern to PatternMatchResult if needed
            if isinstance(pattern, dict):
                # It's a pattern spec - find a matching pattern first
                matches = self.find_patterns(pattern, max_results=1)
                if not matches:
                    return []
                query_pattern = matches[0]
            else:
                query_pattern = pattern

            # Compute embedding for the query pattern
            query_embedding = self._compute_pattern_embedding(query_pattern)
            if query_embedding is None:
                return []  # Can't compute similarities without embeddings

            # Get patterns to compare against
            all_patterns = []

            # First, check the frequent patterns we've discovered
            for freq_pattern in self._frequent_patterns:
                if freq_pattern.pattern_embedding is not None:
                    # Create a PatternMatchResult from the FrequentPattern
                    pattern_result = PatternMatchResult(
                        subgraph_nodes=freq_pattern.nodes,
                        subgraph_edges=freq_pattern.edges,
                        similarity_score=1.0,  # Will be replaced
                        pattern_type="frequent",
                        extra_info={"support": freq_pattern.support, "pattern_id": freq_pattern.pattern_id}
                    )
                    all_patterns.append((pattern_result, freq_pattern.pattern_embedding))

            # Also check the pattern cache
            for patterns in self._pattern_cache.values():
                for p in patterns:
                    # Compute embedding if needed
                    embedding = self._compute_pattern_embedding(p)
                    if embedding is not None:
                        all_patterns.append((p, embedding))

            # Calculate similarities
            similarities = []
            for p, emb in all_patterns:
                # Skip if it's the exact same pattern (same nodes)
                if set(p.subgraph_nodes) == set(query_pattern.subgraph_nodes):
                    continue

                # Calculate cosine similarity
                try:
                    similarity = np.dot(query_embedding, emb) / (
                        np.linalg.norm(query_embedding) * np.linalg.norm(emb))
                    similarities.append((p, float(similarity)))
                except Exception as e:
                    self.logger.warning(f"Error calculating similarity: {e}")

            # Sort by similarity (descending) and return top N
            similarities.sort(key=lambda x: x[1], reverse=True)
            return similarities[:top_n]

    def detect_anomalies(self,
                       property_outliers: bool = True,
                       structural_anomalies: bool = True,
                       threshold: float = 2.0) -> List[Dict[str, Any]]:
        """
        Detects nodes or relationships that appear anomalous by their properties or structure.

        Args:
            property_outliers: Whether to look for anomalies based on properties
            structural_anomalies: Whether to look for structural anomalies
            threshold: Z-score threshold for anomaly detection

        Returns:
            List of anomaly dictionaries with details
        """
        with self.lock:
            anomalies = []

            if not self.use_networkx or not self._nx_graph:
                # Fallback to direct graph engine when NetworkX is not available
                return self._detect_anomalies_direct(property_outliers, structural_anomalies, threshold)

            self.logger.info(f"Detecting anomalies (property={property_outliers}, "
                           f"structural={structural_anomalies}, threshold={threshold})")

            # Property-based anomaly detection
            if property_outliers:
                # Collect statistics by node type
                node_type_stats = defaultdict(lambda: defaultdict(list))

                # For each node, collect numeric properties by type
                for node_id, attrs in self._nx_graph.nodes(data=True):
                    node_type = attrs.get("type", attrs.get("label", "unknown"))

                    # Collect numeric properties
                    for key, val in attrs.items():
                        if isinstance(val, (int, float)) and key not in ('created_at', 'updated_at'):
                            node_type_stats[node_type][key].append((node_id, val))

                # For each node type and property, detect anomalies
                for node_type, prop_data in node_type_stats.items():
                    for prop_name, values in prop_data.items():
                        if len(values) <= 1:
                            continue  # Not enough data for statistics

                        # Extract values
                        node_ids, numeric_values = zip(*values)

                        # Calculate mean and standard deviation
                        if HAS_NUMPY:
                            mean_val = np.mean(numeric_values)
                            std_val = np.std(numeric_values)
                        else:
                            # Fallback to manual calculation
                            mean_val = sum(numeric_values) / len(numeric_values)
                            squared_diffs = [(x - mean_val) ** 2 for x in numeric_values]
                            std_val = math.sqrt(sum(squared_diffs) / len(numeric_values))

                        if std_val <= 0:
                            continue  # Avoid division by zero

                        # Detect anomalies
                        for node_id, val in values:
                            z_score = abs(val - mean_val) / std_val
                            if z_score > threshold:
                                anomalies.append({
                                    "node_id": node_id,
                                    "type": "property_outlier",
                                    "node_type": node_type,
                                    "property": prop_name,
                                    "value": val,
                                    "mean": mean_val,
                                    "std": std_val,
                                    "z_score": z_score,
                                    "score": z_score  # For uniform sorting
                                })

            # Structural anomaly detection
            if structural_anomalies:
                # Degree statistics by node type
                degree_stats = defaultdict(list)
                in_degree_stats = defaultdict(list)
                out_degree_stats = defaultdict(list)

                for node_id, attrs in self._nx_graph.nodes(data=True):
                    node_type = attrs.get("type", attrs.get("label", "unknown"))

                    # Calculate degrees
                    total_degree = self._nx_graph.degree(node_id)
                    in_degree = self._nx_graph.in_degree(node_id)
                    out_degree = self._nx_graph.out_degree(node_id)

                    degree_stats[node_type].append((node_id, total_degree))
                    in_degree_stats[node_type].append((node_id, in_degree))
                    out_degree_stats[node_type].append((node_id, out_degree))

                # Analyze degree statistics by type
                for degree_type, stats_dict in [
                    ("total", degree_stats),
                    ("in", in_degree_stats),
                    ("out", out_degree_stats)
                ]:
                    for node_type, values in stats_dict.items():
                        if len(values) <= 1:
                            continue

                        node_ids, degrees = zip(*values)

                        # Calculate statistics
                        if HAS_NUMPY:
                            mean_degree = np.mean(degrees)
                            std_degree = np.std(degrees)
                        else:
                            mean_degree = sum(degrees) / len(degrees)
                            squared_diffs = [(x - mean_degree) ** 2 for x in degrees]
                            std_degree = math.sqrt(sum(squared_diffs) / len(degrees))

                        if std_degree <= 0:
                            continue

                        for node_id, degree in values:
                            z_score = abs(degree - mean_degree) / std_degree
                            if z_score > threshold:
                                anomalies.append({
                                    "node_id": node_id,
                                    "type": "structural_anomaly",
                                    "anomaly_type": f"{degree_type}_degree_outlier",
                                    "node_type": node_type,
                                    "degree": degree,
                                    "mean": mean_degree,
                                    "std": std_degree,
                                    "z_score": z_score,
                                    "score": z_score
                                })

                # Detect clustering anomalies (optional)
                try:
                    # Calculate clustering coefficients
                    clustering = nx.clustering(self._nx_graph.to_undirected())

                    clustering_stats = defaultdict(list)
                    for node_id, cluster_coef in clustering.items():
                        if node_id in self._nx_graph:
                            node_type = self._nx_graph.nodes[node_id].get("type",
                                                                       self._nx_graph.nodes[node_id].get("label", "unknown"))
                            clustering_stats[node_type].append((node_id, cluster_coef))

                    # Analyze clustering statistics by type
                    for node_type, values in clustering_stats.items():
                        if len(values) <= 1:
                            continue

                        node_ids, coeffs = zip(*values)

                        # Calculate statistics
                        if HAS_NUMPY:
                            mean_coeff = np.mean(coeffs)
                            std_coeff = np.std(coeffs)
                        else:
                            mean_coeff = sum(coeffs) / len(coeffs)
                            squared_diffs = [(x - mean_coeff) ** 2 for x in coeffs]
                            std_coeff = math.sqrt(sum(squared_diffs) / len(coeffs))

                        if std_coeff <= 0:
                            continue

                        for node_id, coeff in values:
                            z_score = abs(coeff - mean_coeff) / std_coeff
                            if z_score > threshold:
                                anomalies.append({
                                    "node_id": node_id,
                                    "type": "structural_anomaly",
                                    "anomaly_type": "clustering_outlier",
                                    "node_type": node_type,
                                    "clustering": coeff,
                                    "mean": mean_coeff,
                                    "std": std_coeff,
                                    "z_score": z_score,
                                    "score": z_score
                                })
                except Exception as e:
                    self.logger.warning(f"Error calculating clustering: {e}")

            # Sort by anomaly score (descending)
            anomalies.sort(key=lambda x: x["score"], reverse=True)

            self.logger.info(f"Detected {len(anomalies)} anomalies")
            return anomalies

    def _detect_anomalies_direct(self, property_outliers: bool,
                              structural_anomalies: bool,
                              threshold: float) -> List[Dict[str, Any]]:
        """Detect anomalies using direct GraphEngine queries (fallback)."""
        anomalies = []

        # Query for all nodes
        query = GraphQuery(max_results=100000)  # High limit to get all nodes
        result = self.graph_engine.execute_query(query)
        nodes = result.get('nodes', [])

        if not nodes:
            return []

        # Property-based anomaly detection
        if property_outliers:
            # Group nodes by type
            nodes_by_type = defaultdict(list)

            for node in nodes:
                node_type = node.get('label', 'Node')
                nodes_by_type[node_type].append(node)

            # For each node type, look for property outliers
            for node_type, type_nodes in nodes_by_type.items():
                if len(type_nodes) <= 1:
                    continue

                # Collect property statistics
                property_values = defaultdict(list)

                for node in type_nodes:
                    for prop, value in node.get('properties', {}).items():
                        if isinstance(value, (int, float)):
                            property_values[prop].append((node['id'], value))

                # Analyze each property
                for prop_name, values in property_values.items():
                    if len(values) <= 1:
                        continue

                    # Extract values
                    node_ids, numeric_values = zip(*values)

                    # Calculate mean and standard deviation
                    if HAS_NUMPY:
                        mean_val = np.mean(numeric_values)
                        std_val = np.std(numeric_values)
                    else:
                        # Fallback to manual calculation
                        mean_val = sum(numeric_values) / len(numeric_values)
                        squared_diffs = [(x - mean_val) ** 2 for x in numeric_values]
                        std_val = math.sqrt(sum(squared_diffs) / len(numeric_values))

                    if std_val <= 0:
                        continue  # Avoid division by zero

                    # Detect anomalies
                    for node_id, val in values:
                        z_score = abs(val - mean_val) / std_val
                        if z_score > threshold:
                            anomalies.append({
                                "node_id": node_id,
                                "type": "property_outlier",
                                "node_type": node_type,
                                "property": prop_name,
                                "value": val,
                                "mean": mean_val,
                                "std": std_val,
                                "z_score": z_score,
                                "score": z_score
                            })

        # Structural anomaly detection
        if structural_anomalies:
            # Count degrees for all nodes
            degrees = {}

            for node in nodes:
                node_id = node['id']
                try:
                    neighbors = self.graph_engine.get_node_neighbors(node_id, direction="both")
                    degrees[node_id] = len(neighbors)
                except Exception:
                    degrees[node_id] = 0

            # Group by node type
            degree_by_type = defaultdict(list)

            for node in nodes:
                node_id = node['id']
                node_type = node.get('label', 'Node')
                if node_id in degrees:
                    degree_by_type[node_type].append((node_id, degrees[node_id]))

            # Analyze each node type
            for node_type, values in degree_by_type.items():
                if len(values) <= 1:
                    continue

                node_ids, degree_values = zip(*values)

                # Calculate statistics
                # Calculate statistics
                if HAS_NUMPY:
                    mean_degree = np.mean(degree_values)
                    std_degree = np.std(degree_values)
                else:
                    # Fallback to manual calculation
                    mean_degree = sum(degree_values) / len(degree_values)
                    squared_diffs = [(x - mean_degree) ** 2 for x in degree_values]
                    std_degree = math.sqrt(sum(squared_diffs) / len(degree_values))

                if std_degree <= 0:
                    continue

                # Find outliers
                for node_id, degree in values:
                    z_score = abs(degree - mean_degree) / std_degree
                    if z_score > threshold:
                        anomalies.append({
                            "node_id": node_id,
                            "type": "structural_anomaly",
                            "anomaly_type": "degree_outlier",
                            "node_type": node_type,
                            "degree": degree,
                            "mean": mean_degree,
                            "std": std_degree,
                            "z_score": z_score,
                            "score": z_score
                        })

        # Sort by anomaly score (descending)
        anomalies.sort(key=lambda x: x["score"], reverse=True)

        return anomalies

    def get_pattern_statistics(self) -> Dict[str, Any]:
        """
        Returns statistics about the discovered patterns and cached patterns.

        Returns:
            Dictionary of statistics
        """
        with self.lock:
            stats = {
                "cached_patterns": sum(len(patterns) for patterns in self._pattern_cache.values()),
                "frequent_patterns": len(self._frequent_patterns),
                "pattern_types": defaultdict(int),
                "patterns_with_embeddings": 0,
                "timestamp": time.time()
            }

            # Count pattern types in cache
            for patterns in self._pattern_cache.values():
                for p in patterns:
                    if p.pattern_type:
                        stats["pattern_types"][p.pattern_type] += 1

            # Count patterns with embeddings
            for p in self._frequent_patterns:
                if p.pattern_embedding is not None:
                    stats["patterns_with_embeddings"] += 1

            # If we have NetworkX, add some graph statistics
            if self.use_networkx and self._nx_graph:
                try:
                    stats["graph_statistics"] = {
                        "nodes": self._nx_graph.number_of_nodes(),
                        "edges": self._nx_graph.number_of_edges(),
                        "density": nx.density(self._nx_graph),
                        "connected_components": nx.number_weakly_connected_components(self._nx_graph),
                        "strongly_connected_components": nx.number_strongly_connected_components(self._nx_graph)
                    }
                except Exception as e:
                    self.logger.warning(f"Error calculating graph statistics: {e}")

            return dict(stats)

    def clear_caches(self) -> None:
        """
        Clears the pattern caches to free memory and force recomputation.
        """
        with self.lock:
            self._pattern_cache.clear()
            self._frequent_patterns = []
            self.logger.info("Pattern caches cleared")

    def rebuild_networkx_graph(self) -> None:
        """
        Forces a rebuild of the NetworkX graph representation.
        """
        with self.lock:
            self._nx_graph = None
            self._build_networkx_graph_if_needed()
            self.logger.info("NetworkX graph rebuilt")

    def export_patterns_to_graph(self) -> int:
        """
        Exports discovered frequent patterns back to the GraphEngine as nodes.
        This allows patterns to be part of the graph itself, enabling meta-reasoning.

        Returns:
            Number of patterns exported
        """
        with self.lock:
            count = 0

            for pattern in self._frequent_patterns:
                pattern_id = f"pattern_{pattern.pattern_id}"

                try:
                    # Add a node representing the pattern
                    properties = {
                        "type": "Pattern",
                        "support": pattern.support,
                        "signature": pattern.signature,
                        "node_count": len(pattern.nodes),
                        "edge_count": len(pattern.edges)
                    }

                    # Add the pattern node
                    try:
                        self.graph_engine.add_node(
                            node_id=pattern_id,
                            label="Pattern",
                            properties=properties
                        )
                    except Exception:
                        # Node might already exist, try updating
                        self.graph_engine.update_node(pattern_id, properties)

                    # Connect the pattern to its nodes
                    for node_id in pattern.nodes:
                        try:
                            self.graph_engine.add_edge(
                                source_id=pattern_id,
                                target_id=node_id,
                                edge_type="CONTAINS",
                                properties={"role": "pattern_member"}
                            )
                        except Exception as e:
                            self.logger.debug(f"Couldn't connect pattern to node {node_id}: {e}")

                    count += 1

                except Exception as e:
                    self.logger.warning(f"Error exporting pattern {pattern_id}: {e}")

            self.logger.info(f"Exported {count} patterns to graph")
            return count

    def compare_subgraphs(self, subgraph1: List[str], subgraph2: List[str]) -> float:
        """
        Compares two subgraphs (lists of node IDs) and returns a similarity score.

        Args:
            subgraph1: List of node IDs for the first subgraph
            subgraph2: List of node IDs for the second subgraph

        Returns:
            Similarity score between 0 and 1
        """
        if not subgraph1 or not subgraph2:
            return 0.0

        # Create PatternMatchResult objects for the subgraphs
        result1 = PatternMatchResult(subgraph_nodes=subgraph1)
        result2 = PatternMatchResult(subgraph_nodes=subgraph2)

        # Compute embeddings
        emb1 = self._compute_pattern_embedding(result1)
        emb2 = self._compute_pattern_embedding(result2)

        if emb1 is None or emb2 is None:
            # Fallback to Jaccard similarity if embeddings not available
            set1 = set(subgraph1)
            set2 = set(subgraph2)

            if not set1 or not set2:
                return 0.0

            jaccard = len(set1.intersection(set2)) / len(set1.union(set2))
            return jaccard

        # Calculate cosine similarity
        try:
            similarity = np.dot(emb1, emb2) / (np.linalg.norm(emb1) * np.linalg.norm(emb2))
            return float(similarity)
        except Exception as e:
            self.logger.warning(f"Error calculating similarity: {e}")
            return 0.0

    def find_pattern_instances(self, pattern: FrequentPattern, min_similarity: float = 0.8) -> List[List[str]]:
        """
        Finds all instances of a pattern in the graph.

        Args:
            pattern: The pattern to find instances of
            min_similarity: Minimum similarity score for a match

        Returns:
            List of node ID lists, each representing an instance of the pattern
        """
        # Convert pattern to a query specification
        pattern_spec = {
            "pattern_type": "template",
            "nodes": [],
            "edges": [],
            "structure": {"max_depth": len(pattern.nodes) + 1}
        }

        # Add node specifications
        node_types = {}
        try:
            for node_id in pattern.nodes:
                node = self.graph_engine.get_node(node_id)
                node_types[node_id] = node.label
                pattern_spec["nodes"].append({"type": node.label})
        except Exception as e:
            self.logger.warning(f"Error getting node types for pattern: {e}")
            return []

        # Add edge specifications
        try:
            edge_types = set()
            for edge_id in pattern.edges:
                edge = self.graph_engine.get_edge(edge_id)
                edge_types.add(edge.type)

                source_idx = pattern.nodes.index(edge.source_id)
                target_idx = pattern.nodes.index(edge.target_id)

                pattern_spec["edges"].append({
                    "source_idx": source_idx,
                    "target_idx": target_idx,
                    "type": edge.type
                })
        except Exception as e:
            self.logger.warning(f"Error getting edge types for pattern: {e}")
            # Continue with node types only

        # Find matches to the pattern
        matches = self.find_patterns(
            pattern_spec=pattern_spec,
            similarity_threshold=min_similarity,
            max_results=100
        )

        # Extract instances
        instances = []
        for match in matches:
            # Skip exact same instance
            if set(match.subgraph_nodes) == set(pattern.nodes):
                continue

            instances.append(match.subgraph_nodes)

        return instances

    def analyze_pattern_evolution(self, pattern_id: str,
                               time_window: Optional[Tuple[float, float]] = None) -> Dict[str, Any]:
        """
        Analyzes how a pattern has evolved over time.

        Args:
            pattern_id: ID of the pattern to analyze
            time_window: Optional tuple of (start_time, end_time)

        Returns:
            Dictionary with evolution metrics
        """
        # Find the pattern
        pattern = None
        for p in self._frequent_patterns:
            if p.pattern_id == pattern_id:
                pattern = p
                break

        if not pattern:
            return {"error": f"Pattern {pattern_id} not found"}

        # This is a placeholder - in a real implementation, you would:
        # 1. Query the graph for historical state or snapshots
        # 2. Trace the pattern over time
        # 3. Analyze changes in structure, support, etc.

        return {
            "pattern_id": pattern_id,
            "current_support": pattern.support,
            "nodes_count": len(pattern.nodes),
            "edges_count": len(pattern.edges),
            "note": "Full pattern evolution analysis requires temporal graph data"
        }

# Utility function to create a PatternMiningModule from a GraphEngine
def create_pattern_mining(graph_engine: AbstractGraphEngine,
                        use_networkx: bool = True,
                        embedding_dim: int = 128) -> PatternMiningModule:
    """
    Factory function to create a PatternMiningModule for a given GraphEngine.

    Args:
        graph_engine: The GraphEngine to mine patterns from
        use_networkx: Whether to use NetworkX for advanced algorithms (if available)
        embedding_dim: Dimension for pattern embeddings

    Returns:
        A configured PatternMiningModule instance
    """
    return PatternMiningModule(
        graph_engine=graph_engine,
        use_networkx=use_networkx,
        default_similarity_threshold=0.7,
        embedding_dim=embedding_dim
    )