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