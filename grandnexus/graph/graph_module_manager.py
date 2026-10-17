"""
graph_module_manager.py - Integration point for Graph and PatternMining modules with NexusCore

This module serves as the central manager for all graph-related operations in GrandNexus.
It coordinates the GraphEngine and PatternMining modules, ensuring they work as a unified
system while maintaining clean separation of concerns.
"""

import logging
import traceback
from typing import Dict, Any, List, Optional, Union, Tuple, Set, Type
import time

from grandnexus.core.nexus_core import NexusCore
from grandnexus.core.registry import ModuleRegistry

from grandnexus.graph.graph_engine import (
    create_graph_engine, AbstractGraphEngine, GraphNode, GraphEdge, GraphPattern,
    GraphQuery, NodeNotFoundError, EdgeNotFoundError,
    create_graph_engine
)
from grandnexus.graph.pattern_mining import (
    PatternMiningModule, PatternMatchResult, FrequentPattern,
    create_pattern_mining
)
from grandnexus.graph.graph_pattern_interface import GraphPatternBridge


class GraphModuleManager:
    """
    Central manager for all graph-related functionality in GrandNexus.

    This class:
    1. Initializes and manages both GraphEngine and PatternMining
    2. Connects them to NexusCore for system-wide integration
    3. Provides a unified API for graph operations
    4. Handles messaging, events, and cross-module integration
    """

    def __init__(self, nexus_core: NexusCore, config: Dict[str, Any]):
        """
        Initialize the GraphModuleManager.

        Args:
            nexus_core: The NexusCore instance for system integration
            config: Configuration dictionary with graph settings
        """
        self.logger = logging.getLogger("GraphModuleManager")
        self.nexus_core = nexus_core
        self.config = config

        # Extract configuration
        graph_engine_type = config.get('graph_engine_type', 'inmemory')
        graph_engine_config = config.get('graph_engine_config', {})
        use_pattern_mining = config.get('use_pattern_mining', True)
        use_networkx = config.get('use_networkx', True)
        embedding_dim = config.get('embedding_dim', 128)

        # Initialize the GraphEngine
        self.logger.info(f"Initializing GraphEngine of type '{graph_engine_type}'")
        shared = config.get('graph_engine_instance')  # réparation 6.5 : graphe partagé
        self.graph_engine = shared if shared is not None else create_graph_engine(graph_engine_type, **graph_engine_config)

        # Initialize PatternMining if enabled
        self.pattern_mining = None
        if use_pattern_mining:
            self.logger.info("Initializing PatternMining module")
            self.pattern_mining = create_pattern_mining(
                graph_engine=self.graph_engine,
                use_networkx=use_networkx,
                embedding_dim=embedding_dim
            )

            # Create the bridge between GraphEngine and PatternMining
            self.bridge = GraphPatternBridge(
                graph_engine=self.graph_engine,
                pattern_mining=self.pattern_mining
            )
        else:
            self.logger.info("PatternMining module disabled")
            self.bridge = None

        # Register with NexusCore
        self._register_with_nexus_core()

        self.logger.info("GraphModuleManager initialized successfully")

    def _register_with_nexus_core(self):
        """Register handlers and callbacks with NexusCore."""
        # Register message handlers
        self.nexus_core.register_handler("graph.add_node", self._handle_add_node)
        self.nexus_core.register_handler("graph.get_node", self._handle_get_node)
        self.nexus_core.register_handler("graph.update_node", self._handle_update_node)
        self.nexus_core.register_handler("graph.delete_node", self._handle_delete_node)
        self.nexus_core.register_handler("graph.add_edge", self._handle_add_edge)
        self.nexus_core.register_handler("graph.get_edge", self._handle_get_edge)
        self.nexus_core.register_handler("graph.update_edge", self._handle_update_edge)
        self.nexus_core.register_handler("graph.delete_edge", self._handle_delete_edge)
        self.nexus_core.register_handler("graph.query", self._handle_query)
        self.nexus_core.register_handler("graph.activate", self._handle_activate)

        # Pattern mining handlers (if enabled)
        if self.pattern_mining:
            self.nexus_core.register_handler("graph.find_patterns", self._handle_find_patterns)
            self.nexus_core.register_handler("graph.extract_patterns", self._handle_extract_patterns)
            self.nexus_core.register_handler("graph.detect_anomalies", self._handle_detect_anomalies)
            self.nexus_core.register_handler("graph.similar_patterns", self._handle_similar_patterns)

        # Register for memory events
        self.nexus_core.register_handler("memory.concept_stored", self._handle_concept_stored)
        self.nexus_core.register_handler("memory.episodic_created", self._handle_episodic_created)

        # Register with ModuleRegistry
        registry = getattr(self.nexus_core, "module_registry", None)  # réparation 6.5
        if registry is None:
            registry = ModuleRegistry(self.nexus_core)
            self.nexus_core.module_registry = registry
        registry.register("graph", self)

    # --------------------------------------------------------------------------
    # Core GraphEngine API handlers
    # --------------------------------------------------------------------------

    def _handle_add_node(self, message: Dict[str, Any]) -> Dict[str, Any]:
        """Handle add_node request."""
        try:
            node_id = self.graph_engine.add_node(
                node_id=message.get('node_id'),
                label=message.get('label', 'Node'),
                properties=message.get('properties', {})
            )

            # Publish event for other modules
            self.nexus_core.publish_event("graph.node_added", {
                "node_id": node_id,
                "label": message.get('label', 'Node'),
                "source_module": message.get('source_module')
            })

            return {"success": True, "node_id": node_id}
        except Exception as e:
            self.logger.error(f"Error adding node: {e}")
            traceback.print_exc()
            return {"success": False, "error": str(e)}

    def _handle_get_node(self, message: Dict[str, Any]) -> Dict[str, Any]:
        """Handle get_node request."""
        try:
            node_id = message.get('node_id')
            if not node_id:
                return {"success": False, "error": "No node_id provided"}

            node = self.graph_engine.get_node(node_id)

            # Convert to serializable format
            node_data = {
                "node_id": node.node_id,
                "label": node.label,
                "properties": node.properties,
                "activation": node.activation,
                "importance": node.importance,
                "last_accessed": node.last_accessed,
                "creation_time": node.creation_time
            }

            return {"success": True, "node": node_data}
        except NodeNotFoundError as e:
            return {"success": False, "error": str(e), "not_found": True}
        except Exception as e:
            self.logger.error(f"Error getting node: {e}")
            return {"success": False, "error": str(e)}

    def _handle_update_node(self, message: Dict[str, Any]) -> Dict[str, Any]:
        """Handle update_node request."""
        try:
            node_id = message.get('node_id')
            properties = message.get('properties', {})

            if not node_id:
                return {"success": False, "error": "No node_id provided"}

            self.graph_engine.update_node(node_id, properties)

            # Publish event for other modules
            self.nexus_core.publish_event("graph.node_updated", {
                "node_id": node_id,
                "properties": properties,
                "source_module": message.get('source_module')
            })

            return {"success": True}
        except NodeNotFoundError as e:
            return {"success": False, "error": str(e), "not_found": True}
        except Exception as e:
            self.logger.error(f"Error updating node: {e}")
            return {"success": False, "error": str(e)}

    def _handle_delete_node(self, message: Dict[str, Any]) -> Dict[str, Any]:
        """Handle delete_node request."""
        try:
            node_id = message.get('node_id')
            if not node_id:
                return {"success": False, "error": "No node_id provided"}

            self.graph_engine.delete_node(node_id)

            # Publish event for other modules
            self.nexus_core.publish_event("graph.node_deleted", {
                "node_id": node_id,
                "source_module": message.get('source_module')
            })

            return {"success": True}
        except NodeNotFoundError as e:
            return {"success": False, "error": str(e), "not_found": True}
        except Exception as e:
            self.logger.error(f"Error deleting node: {e}")
            return {"success": False, "error": str(e)}

    def _handle_add_edge(self, message: Dict[str, Any]) -> Dict[str, Any]:
        """Handle add_edge request."""
        try:
            source_id = message.get('source_id')
            target_id = message.get('target_id')
            edge_type = message.get('edge_type', 'RELATES')

            if not source_id or not target_id:
                return {"success": False, "error": "Source and target IDs required"}

            edge_id = self.graph_engine.add_edge(
                source_id=source_id,
                target_id=target_id,
                edge_type=edge_type,
                edge_id=message.get('edge_id'),
                properties=message.get('properties', {}),
                weight=message.get('weight', 1.0),
                bidirectional=message.get('bidirectional', False)
            )

            # Publish event for other modules
            self.nexus_core.publish_event("graph.edge_added", {
                "edge_id": edge_id,
                "source_id": source_id,
                "target_id": target_id,
                "edge_type": edge_type,
                "source_module": message.get('source_module')
            })

            return {"success": True, "edge_id": edge_id}
        except NodeNotFoundError as e:
            return {"success": False, "error": str(e), "not_found": True}
        except Exception as e:
            self.logger.error(f"Error adding edge: {e}")
            return {"success": False, "error": str(e)}

    def _handle_get_edge(self, message: Dict[str, Any]) -> Dict[str, Any]:
        """Handle get_edge request."""
        try:
            edge_id = message.get('edge_id')
            if not edge_id:
                return {"success": False, "error": "No edge_id provided"}

            edge = self.graph_engine.get_edge(edge_id)

            # Convert to serializable format
            edge_data = {
                "edge_id": edge.edge_id,
                "source_id": edge.source_id,
                "target_id": edge.target_id,
                "type": edge.type,
                "properties": edge.properties,
                "weight": edge.weight,
                "bidirectional": edge.bidirectional,
                "creation_time": edge.creation_time
            }

            return {"success": True, "edge": edge_data}
        except EdgeNotFoundError as e:
            return {"success": False, "error": str(e), "not_found": True}
        except Exception as e:
            self.logger.error(f"Error getting edge: {e}")
            return {"success": False, "error": str(e)}

    def _handle_update_edge(self, message: Dict[str, Any]) -> Dict[str, Any]:
        """Handle update_edge request."""
        try:
            edge_id = message.get('edge_id')
            properties = message.get('properties', {})

            if not edge_id:
                return {"success": False, "error": "No edge_id provided"}

            self.graph_engine.update_edge(edge_id, properties)

            # Publish event for other modules
            self.nexus_core.publish_event("graph.edge_updated", {
                "edge_id": edge_id,
                "properties": properties,
                "source_module": message.get('source_module')
            })

            return {"success": True}
        except EdgeNotFoundError as e:
            return {"success": False, "error": str(e), "not_found": True}
        except Exception as e:
            self.logger.error(f"Error updating edge: {e}")
            return {"success": False, "error": str(e)}

    def _handle_delete_edge(self, message: Dict[str, Any]) -> Dict[str, Any]:
        """Handle delete_edge request."""
        try:
            edge_id = message.get('edge_id')
            if not edge_id:
                return {"success": False, "error": "No edge_id provided"}

            self.graph_engine.delete_edge(edge_id)

            # Publish event for other modules
            self.nexus_core.publish_event("graph.edge_deleted", {
                "edge_id": edge_id,
                "source_module": message.get('source_module')
            })

            return {"success": True}
        except EdgeNotFoundError as e:
            return {"success": False, "error": str(e), "not_found": True}
        except Exception as e:
            self.logger.error(f"Error deleting edge: {e}")
            return {"success": False, "error": str(e)}

    def _handle_query(self, message: Dict[str, Any]) -> Dict[str, Any]:
        """Handle query request."""
        try:
            # Convert message data to GraphQuery object
            query_data = message.get('query', {})
            query = GraphQuery(
                node_criteria=query_data.get('node_criteria', {}),
                edge_criteria=query_data.get('edge_criteria', {}),
                start_nodes=query_data.get('start_nodes'),
                max_depth=query_data.get('max_depth', 3),
                max_results=query_data.get('max_results', 100),
                include_inactive=query_data.get('include_inactive', False),
                min_activation=query_data.get('min_activation', 0.0),
                sort_by=query_data.get('sort_by', "activation")
            )

            # Execute query
            result = self.graph_engine.execute_query(query)

            return {"success": True, "result": result}
        except Exception as e:
            self.logger.error(f"Error executing query: {e}")
            return {"success": False, "error": str(e)}

    def _handle_activate(self, message: Dict[str, Any]) -> Dict[str, Any]:
        """Handle activate (spread activation) request."""
        try:
            source_nodes = message.get('source_nodes', [])
            decay_factor = message.get('decay_factor', 0.5)
            max_depth = message.get('max_depth', 3)

            if not source_nodes:
                return {"success": False, "error": "No source_nodes provided"}

            # Spread activation
            result = self.graph_engine.spread_activation(
                source_nodes=source_nodes,
                decay_factor=decay_factor,
                max_depth=max_depth
            )

            # Publish event for other modules
            self.nexus_core.publish_event("graph.activation_spread", {
                "source_nodes": source_nodes,
                "activated_nodes": list(result.keys()),
                "source_module": message.get('source_module')
            })

            return {"success": True, "activations": result}
        except Exception as e:
            self.logger.error(f"Error spreading activation: {e}")
            return {"success": False, "error": str(e)}

    # --------------------------------------------------------------------------
    # Pattern Mining API handlers
    # --------------------------------------------------------------------------

    def _handle_find_patterns(self, message: Dict[str, Any]) -> Dict[str, Any]:
        """Handle find_patterns request."""
        if not self.pattern_mining:
            return {"success": False, "error": "Pattern mining not enabled"}

        try:
            pattern_spec = message.get('pattern_spec', {})
            similarity_threshold = message.get('similarity_threshold')
            max_results = message.get('max_results', 10)

            if not pattern_spec:
                return {"success": False, "error": "No pattern_spec provided"}

            # Find patterns
            patterns = self.pattern_mining.find_patterns(
                pattern_spec=pattern_spec,
                similarity_threshold=similarity_threshold,
                max_results=max_results
            )

            # Convert to serializable format
            pattern_results = []
            for p in patterns:
                pattern_results.append({
                    "nodes": p.subgraph_nodes,
                    "edges": p.subgraph_edges,
                    "similarity": p.similarity_score,
                    "pattern_type": p.pattern_type,
                    "extra_info": p.extra_info
                })

            # Publish event for other modules
            if pattern_results:
                self.nexus_core.publish_event("graph.patterns_found", {
                    "pattern_count": len(pattern_results),
                    "pattern_types": list(set(p.get('pattern_type') for p in pattern_results if p.get('pattern_type'))),
                    "source_module": message.get('source_module')
                })

            return {"success": True, "patterns": pattern_results}
        except Exception as e:
            self.logger.error(f"Error finding patterns: {e}")
            return {"success": False, "error": str(e)}

    def _handle_extract_patterns(self, message: Dict[str, Any]) -> Dict[str, Any]:
        """Handle extract_patterns (frequent pattern mining) request."""
        if not self.pattern_mining:
            return {"success": False, "error": "Pattern mining not enabled"}

        try:
            min_support = message.get('min_support', 2)
            max_size = message.get('max_size', 5)
            store_embeddings = message.get('store_embeddings', True)

            # Extract patterns
            patterns = self.pattern_mining.extract_frequent_subgraphs(
                min_support=min_support,
                max_size=max_size,
                store_embeddings=store_embeddings
            )

            # Convert to serializable format
            pattern_results = []
            for p in patterns:
                pattern_results.append({
                    "pattern_id": p.pattern_id,
                    "nodes": p.nodes,
                    "edges": p.edges,
                    "support": p.support,
                    "signature": p.signature
                })

            # Publish event for other modules
            if pattern_results:
                self.nexus_core.publish_event("graph.patterns_extracted", {
                    "pattern_count": len(pattern_results),
                    "source_module": message.get('source_module')
                })

                # If requested, store patterns in graph
                if message.get('store_in_graph', False) and self.bridge:
                    self.bridge.store_discovered_patterns(patterns)

            return {"success": True, "patterns": pattern_results}
        except Exception as e:
            self.logger.error(f"Error extracting patterns: {e}")
            return {"success": False, "error": str(e)}

    def _handle_detect_anomalies(self, message: Dict[str, Any]) -> Dict[str, Any]:
        """Handle detect_anomalies request."""
        if not self.pattern_mining:
            return {"success": False, "error": "Pattern mining not enabled"}

        try:
            property_outliers = message.get('property_outliers', True)
            structural_anomalies = message.get('structural_anomalies', True)
            threshold = message.get('threshold', 2.0)

            # Detect anomalies
            anomalies = self.pattern_mining.detect_anomalies(
                property_outliers=property_outliers,
                structural_anomalies=structural_anomalies,
                threshold=threshold
            )

            # Publish event for other modules
            if anomalies:
                self.nexus_core.publish_event("graph.anomalies_detected", {
                    "anomaly_count": len(anomalies),
                    "source_module": message.get('source_module')
                })

            return {"success": True, "anomalies": anomalies}
        except Exception as e:
            self.logger.error(f"Error detecting anomalies: {e}")
            return {"success": False, "error": str(e)}

    def _handle_similar_patterns(self, message: Dict[str, Any]) -> Dict[str, Any]:
        """Handle similar_patterns request."""
        if not self.pattern_mining:
            return {"success": False, "error": "Pattern mining not enabled"}

        try:
            pattern = message.get('pattern')
            top_n = message.get('top_n', 5)

            if not pattern:
                return {"success": False, "error": "No pattern provided"}

            # Find similar patterns
            similar = self.pattern_mining.find_similar_patterns(
                pattern=pattern,
                top_n=top_n
            )

            # Convert to serializable format
            similar_results = []
            for pattern, similarity in similar:
                similar_results.append({
                    "nodes": pattern.subgraph_nodes,
                    "edges": pattern.subgraph_edges,
                    "similarity": similarity,
                    "pattern_type": pattern.pattern_type
                })

            return {"success": True, "similar_patterns": similar_results}
        except Exception as e:
            self.logger.error(f"Error finding similar patterns: {e}")
            return {"success": False, "error": str(e)}

    # --------------------------------------------------------------------------
    # Event handlers for integration with other modules
    # --------------------------------------------------------------------------

    def _handle_concept_stored(self, event_data: Dict[str, Any]) -> None:
        """Handle memory.concept_stored events."""
        # When a concept is stored in memory, we can automatically add it to the graph
        concept_id = event_data.get('concept_id')
        concept_type = event_data.get('concept_type', 'Concept')
        properties = event_data.get('properties', {})

        if not concept_id:
            return

        try:
            # Check if node already exists
            try:
                self.graph_engine.get_node(concept_id)
                # Node exists, update it
                self.graph_engine.update_node(concept_id, properties)
            except NodeNotFoundError:
                # Node doesn't exist, create it
                self.graph_engine.add_node(
                    node_id=concept_id,
                    label=concept_type,
                    properties=properties
                )

            # Connect to related concepts if specified
            related_concepts = event_data.get('related_concepts', [])
            for relation in related_concepts:
                related_id = relation.get('concept_id')
                relation_type = relation.get('relation_type', 'RELATED_TO')

                if related_id:
                    try:
                        self.graph_engine.add_edge(
                            source_id=concept_id,
                            target_id=related_id,
                            edge_type=relation_type
                        )
                    except NodeNotFoundError:
                        # Related node doesn't exist yet, create it
                        self.graph_engine.add_node(
                            node_id=related_id,
                            label='Concept'
                        )
                        self.graph_engine.add_edge(
                            source_id=concept_id,
                            target_id=related_id,
                            edge_type=relation_type
                        )

            # If pattern mining is enabled, check for new patterns
            if self.pattern_mining and self.config.get('auto_mine_patterns', False):
                # Schedule a task to extract patterns
                self.nexus_core.schedule_task(
                    task_id=f"extract_patterns_{time.time()}",
                    handler="graph.extract_patterns",
                    data={
                        "min_support": 2,
                        "max_size": 5,
                        "store_in_graph": True
                    },
                    delay=5.0  # Wait 5 seconds to batch multiple additions
                )
        except Exception as e:
            self.logger.error(f"Error handling concept_stored event: {e}")

    def _handle_episodic_created(self, event_data: Dict[str, Any]) -> None:
        """Handle memory.episodic_created events."""
        # When an episodic memory is created, we can represent it in the graph
        episode_id = event_data.get('episode_id')
        episode_type = event_data.get('episode_type', 'Episode')
        entities = event_data.get('entities', [])

        if not episode_id:
            return

        try:
            # Create episode node
            try:
                self.graph_engine.add_node(
                    node_id=episode_id,
                    label=episode_type,
                    properties=event_data.get('properties', {})
                )
            except Exception:
                # Node might already exist
                pass

            # Connect to entities involved in the episode
            for entity in entities:
                entity_id = entity.get('entity_id')
                role = entity.get('role', 'INVOLVED_IN')

                if entity_id:
                    try:
                        # Create relationship: Entity -[INVOLVED_IN]-> Episode
                        self.graph_engine.add_edge(
                            source_id=entity_id,
                            target_id=episode_id,
                            edge_type=role
                        )
                    except NodeNotFoundError:
                        # Entity doesn't exist yet, create it
                        entity_type = entity.get('entity_type', 'Entity')
                        self.graph_engine.add_node(
                            node_id=entity_id,
                            label=entity_type
                        )
                        self.graph_engine.add_edge(
                            source_id=entity_id,
                            target_id=episode_id,
                            edge_type=role
                        )
        except Exception as e:
            self.logger.error(f"Error handling episodic_created event: {e}")

    # --------------------------------------------------------------------------
    # Public API for direct use by other Python modules
    # --------------------------------------------------------------------------

    def add_node(self, node_id: Optional[str] = None, label: str = "Node",
                properties: Optional[Dict[str, Any]] = None) -> str:
        """
        Add a node to the graph.

        Args:
            node_id: Optional node ID (generated if None)
            label: Node type/label
            properties: Node properties

        Returns:
            The ID of the node
        """
        return self.graph_engine.add_node(node_id, label, properties)

    def add_edge(self, source_id: str, target_id: str, edge_type: str,
                edge_id: Optional[str] = None, properties: Optional[Dict[str, Any]] = None,
                weight: float = 1.0, bidirectional: bool = False) -> str:
        """
        Add an edge between nodes.

        Args:
            source_id: Source node ID
            target_id: Target node ID
            edge_type: Type of relationship
            edge_id: Optional edge ID (generated if None)
            properties: Edge properties
            weight: Edge weight
            bidirectional: Whether the edge is bidirectional

        Returns:
            The ID of the edge
        """
        return self.graph_engine.add_edge(
            source_id, target_id, edge_type,
            edge_id, properties, weight, bidirectional
        )

    def query(self, **kwargs) -> Dict[str, Any]:
        """
        Execute a graph query.

        Args:
            **kwargs: Query parameters (node_criteria, edge_criteria, etc.)

        Returns:
            Query results
        """
        query = GraphQuery(**kwargs)
        return self.graph_engine.execute_query(query)

    def find_patterns(self, pattern_spec: Dict[str, Any], **kwargs) -> List[PatternMatchResult]:
        """
        Find patterns in the graph.

        Args:
            pattern_spec: Pattern specification
            **kwargs: Additional parameters

        Returns:
            List of pattern matches
        """
        if not self.pattern_mining:
            raise ValueError("Pattern mining is not enabled")

        return self.pattern_mining.find_patterns(pattern_spec, **kwargs)

    def spread_activation(self, source_nodes: List[str], **kwargs) -> Dict[str, float]:
        """
        Spread activation from source nodes.

        Args:
            source_nodes: Starting nodes for activation
            **kwargs: Additional parameters

        Returns:
            Dictionary of node activations
        """
        return self.graph_engine.spread_activation(source_nodes, **kwargs)

    def detect_anomalies(self, **kwargs) -> List[Dict[str, Any]]:
        """
        Detect anomalies in the graph.

        Args:
            **kwargs: Detection parameters

        Returns:
            List of anomalies
        """
        if not self.pattern_mining:
            raise ValueError("Pattern mining is not enabled")

        return self.pattern_mining.detect_anomalies(**kwargs)

    def create_subgraph_from_query(self, query: Dict[str, Any]) -> Tuple[List[str], List[str]]:
        """
        Create a subgraph based on a query.

        Args:
            query: Query specification

        Returns:
            Tuple of (node_ids, edge_ids)
        """
        result = self.query(**query)

        node_ids = [node['id'] for node in result.get('nodes', [])]
        edge_ids = [edge['id'] for edge in result.get('edges', [])]

        return node_ids, edge_ids

    # --------------------------------------------------------------------------
    # Maintenance and management methods
    # --------------------------------------------------------------------------

    def get_statistics(self) -> Dict[str, Any]:
        """
        Get statistics about the graph and pattern mining modules.

        Returns:
            Dictionary of statistics
        """
        stats = {
            "timestamp": time.time(),
            "graph": {}
        }

        # Get basic graph statistics
        try:
            node_query = GraphQuery(max_results=1)
            result = self.graph_engine.execute_query(node_query)
            stats["graph"]["node_count"] = len(result.get('nodes', []))
            stats["graph"]["edge_count"] = len(result.get('edges', []))

            # Get node types distribution
            node_types = {}
            for node in result.get('nodes', []):
                label = node.get('label', 'Unknown')
                node_types[label] = node_types.get(label, 0) + 1
            stats["graph"]["node_types"] = node_types

        except Exception as e:
            self.logger.error(f"Error getting graph statistics: {e}")
            stats["graph"]["error"] = str(e)

        # Get pattern mining statistics if enabled
        if self.pattern_mining:
            try:
                pattern_stats = self.pattern_mining.get_pattern_statistics()
                stats["pattern_mining"] = pattern_stats
            except Exception as e:
                self.logger.error(f"Error getting pattern mining statistics: {e}")
                stats["pattern_mining"] = {"error": str(e)}

        return stats

    def clear_caches(self) -> None:
        """Clear all caches to free memory."""
        try:
            if self.pattern_mining:
                self.pattern_mining.clear_caches()
                if hasattr(self.pattern_mining, "rebuild_networkx_graph"):
                    self.pattern_mining.rebuild_networkx_graph()

            self.logger.info("All caches cleared")
        except Exception as e:
            self.logger.error(f"Error clearing caches: {e}")
            raise

    def export_graph(self, format: str = "json") -> Dict[str, Any]:
        """
        Export the graph in the specified format.

        Args:
            format: Export format (currently only 'json' is supported)

        Returns:
            Exported graph data
        """
        if format.lower() != "json":
            raise ValueError(f"Unsupported export format: {format}")

        try:
            # Use the graph engine's serialization method
            return self.graph_engine.serialize()
        except Exception as e:
            self.logger.error(f"Error exporting graph: {e}")
            raise

    def import_graph(self, data: Dict[str, Any]) -> None:
        """
        Import graph data.

        Args:
            data: Graph data to import
        """
        try:
            # Use the graph engine's deserialization method
            self.graph_engine.deserialize(data)

            # Rebuild pattern mining if needed
            if self.pattern_mining and hasattr(self.pattern_mining, "rebuild_networkx_graph"):
                self.pattern_mining.rebuild_networkx_graph()

            self.logger.info("Graph imported successfully")
        except Exception as e:
            self.logger.error(f"Error importing graph: {e}")
            raise

    def shutdown(self) -> None:
        """Close and clean up resources."""
        try:
            # Close the graph engine
            if hasattr(self.graph_engine, "close"):
                self.graph_engine.close()

            self.logger.info("GraphModuleManager shut down")
        except Exception as e:
            self.logger.error(f"Error during shutdown: {e}")

