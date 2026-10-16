"""
reasoning_core.py - Core Reasoning Module for GrandNexus

This module provides the foundation for reasoning capabilities in GrandNexus.
It facilitates different forms of reasoning including deductive, inductive,
abductive, analogical, and causal reasoning, while allowing for emergent
behavior through flexible integration of reasoning strategies.

The reasoning module serves as a substrate for higher-level cognitive processes,
providing mechanisms for drawing inferences, solving problems, making decisions,
and generating hypotheses.

Key principles:
- Support diverse reasoning strategies without forcing a particular approach
- Allow for dynamic composition of reasoning methods based on context
- Facilitate emergent reasoning patterns through interaction between strategies
- Provide mechanisms for both certain and uncertain reasoning
- Support integration with memory and perception modules
"""

import re  # fix 6.5: previously provided by the notebook's shared namespace
import logging
import threading
import time
import uuid
from typing import Dict, List, Any, Optional, Set, Union, Callable, Tuple
from dataclasses import dataclass, field
from enum import Enum, auto
import random
import math

# Will be imported from other modules
from grandnexus.cognition.contracts import (
    CognitionManagerBase as CognitionManager,  # contrat : casse l'import circulaire
    CognitiveModuleBase, CognitiveProcessType, CognitiveMode,
    CognitiveContext, CognitiveRequest, CognitiveResult,
)
from grandnexus.memory.working import WorkingMemory
from grandnexus.memory.semantic import SemanticMemory
from grandnexus.graph.graph_engine import GraphEngine
from grandnexus.graph.graph_engine import create_graph_engine, AbstractGraphEngine
from grandnexus.llm_interface.llm_engine import LLMEngine


class ReasoningType(Enum):
    """Types of reasoning processes"""
    DEDUCTIVE = auto()    # From general principles to specific conclusions
    INDUCTIVE = auto()    # From specific observations to general principles
    ABDUCTIVE = auto()    # Inference to the best explanation
    ANALOGICAL = auto()   # Mapping between similar situations
    CAUSAL = auto()       # Reasoning about cause and effect
    COUNTERFACTUAL = auto() # Reasoning about hypothetical scenarios
    STATISTICAL = auto()  # Probabilistic reasoning
    SPATIAL = auto()      # Reasoning about spatial relationships
    TEMPORAL = auto()     # Reasoning about time and sequence
    CREATIVE = auto()     # Generating novel combinations and ideas
    EMOTIONAL = auto()    # Reasoning about feelings and social context
    HYBRID = auto()       # Combination of multiple reasoning types


class ConfidenceLevel(Enum):
    """Levels of confidence in reasoning results"""
    CERTAIN = 1.0         # Logically certain (like mathematical proof)
    VERY_LIKELY = 0.9     # Strong evidence, highly probable
    LIKELY = 0.7          # Good evidence, more likely than not
    PLAUSIBLE = 0.5       # Possible explanation, balanced evidence
    UNLIKELY = 0.3        # Weak evidence, improbable but not impossible
    VERY_UNLIKELY = 0.1   # Very weak evidence, highly improbable
    UNKNOWN = 0.0         # Cannot determine confidence


@dataclass
class ReasoningStrategy:
    """
    A specific approach to reasoning that can be employed by the reasoning engine.
    Strategies can be composed and combined to create more complex reasoning patterns.
    """
    strategy_id: str
    reasoning_type: ReasoningType
    description: str
    handler: Callable  # Function that implements the strategy
    required_inputs: List[str] = field(default_factory=list)
    provided_outputs: List[str] = field(default_factory=list)
    compatibility: Set[ReasoningType] = field(default_factory=set)  # Compatible with these reasoning types
    synergy: Dict[str, float] = field(default_factory=dict)  # Strategy IDs and their synergy scores
    confidence_modifier: float = 0.0  # Adjustment to confidence when using this strategy
    complexity: int = 1  # Computational complexity (1-10)


@dataclass
class InferenceStep:
    """A single step in a reasoning process"""
    step_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    strategy_id: str = ""
    description: str = ""
    inputs: Dict[str, Any] = field(default_factory=dict)
    outputs: Dict[str, Any] = field(default_factory=dict)
    confidence: float = 1.0
    supporting_evidence: List[str] = field(default_factory=list)
    contradicting_evidence: List[str] = field(default_factory=list)
    sub_steps: List['InferenceStep'] = field(default_factory=list)
    timestamp: float = field(default_factory=time.time)


@dataclass
class ReasoningQuery:
    """A query for the reasoning engine"""
    query_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    question: Any = None  # The query/question to answer
    context: Dict[str, Any] = field(default_factory=dict)  # Additional context information
    preferred_strategies: List[str] = field(default_factory=list)
    excluded_strategies: List[str] = field(default_factory=list)
    required_reasoning_types: List[ReasoningType] = field(default_factory=list)
    min_confidence: float = 0.0
    max_steps: Optional[int] = None
    max_time: Optional[float] = None
    exploration_factor: float = 0.2  # How much to explore different reasoning paths


@dataclass
class ReasoningResult:
    """Result of a reasoning process"""
    result_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    query_id: str = ""
    conclusions: List[Dict[str, Any]] = field(default_factory=list)
    confidence: float = 0.0
    inference_steps: List[InferenceStep] = field(default_factory=list)
    reasoning_types_used: Set[ReasoningType] = field(default_factory=set)
    strategies_used: List[str] = field(default_factory=list)
    supporting_evidence: Dict[str, List[str]] = field(default_factory=dict)
    contradicting_evidence: Dict[str, List[str]] = field(default_factory=dict)
    processing_time: float = 0.0
    complete: bool = False
    annotations: Dict[str, Any] = field(default_factory=dict)


class ReasoningCore(CognitiveModuleBase):
    """
    Core class for the reasoning module in GrandNexus.

    Provides a flexible framework for different reasoning approaches,
    allowing them to be composed dynamically based on context. Acts as
    a substrate for emergent reasoning patterns rather than enforcing
    fixed reasoning pathways.
    """

    def __init__(self,
                 cognition_manager: Optional[CognitionManager] = None,
                 working_memory: Optional[WorkingMemory] = None,
                 semantic_memory: Optional[SemanticMemory] = None,
                 graph_engine: Optional[GraphEngine] = None,
                 llm_engine: Optional[LLMEngine] = None,
                 episodic_memory: Any = None,
                 nexus_core: Any = None,
                 **kwargs):
        """
        Initialize the Reasoning Core.

        Args:
            cognition_manager: Reference to the CognitionManager
            working_memory: Reference to the WorkingMemory module
            semantic_memory: Reference to the SemanticMemory module
            graph_engine: Reference to the GraphEngine module
            llm_engine: Reference to the LLM interface
        """
        self.cognition_manager = cognition_manager
        self.working_memory = working_memory
        self.semantic_memory = semantic_memory
        self.graph_engine = graph_engine
        self.llm_engine = llm_engine
        self.episodic_memory = episodic_memory
        self.nexus_core = nexus_core

        # Logger setup
        self.logger = logging.getLogger("ReasoningCore")

        # Threading and synchronization
        self.lock = threading.RLock()

        # Reasoning strategies registry
        self.strategies: Dict[str, ReasoningStrategy] = {}

        # Reasoning workspace (shared state for reasoning processes)
        self.workspace: Dict[str, Dict[str, Any]] = {}

        # Active reasoning processes
        self.active_processes: Dict[str, Dict[str, Any]] = {}

        # Initialize basic reasoning strategies
        self._register_basic_strategies()

        self.logger.info("ReasoningCore initialized")

    def _register_basic_strategies(self) -> None:
        """Register built-in basic reasoning strategies"""
        # Simple deductive strategy
        self.register_strategy(ReasoningStrategy(
            strategy_id="simple_deduction",
            reasoning_type=ReasoningType.DEDUCTIVE,
            description="Simple deductive reasoning using modus ponens",
            handler=self._simple_deductive_reasoning,
            required_inputs=["premises", "rules"],
            provided_outputs=["conclusions"],
            compatibility={ReasoningType.DEDUCTIVE, ReasoningType.HYBRID},
            complexity=2
        ))

        # Simple inductive strategy
        self.register_strategy(ReasoningStrategy(
            strategy_id="simple_induction",
            reasoning_type=ReasoningType.INDUCTIVE,
            description="Simple inductive reasoning from examples to general rules",
            handler=self._simple_inductive_reasoning,
            required_inputs=["examples"],
            provided_outputs=["patterns", "generalizations"],
            compatibility={ReasoningType.INDUCTIVE, ReasoningType.HYBRID},
            complexity=3
        ))

        # Abductive strategy
        self.register_strategy(ReasoningStrategy(
            strategy_id="simple_abduction",
            reasoning_type=ReasoningType.ABDUCTIVE,
            description="Simple abductive reasoning to find explanations",
            handler=self._simple_abductive_reasoning,
            required_inputs=["observations"],
            provided_outputs=["explanations"],
            compatibility={ReasoningType.ABDUCTIVE, ReasoningType.HYBRID},
            confidence_modifier=-0.1,  # Abduction is inherently less certain
            complexity=4
        ))

        # Analogical strategy
        self.register_strategy(ReasoningStrategy(
            strategy_id="simple_analogy",
            reasoning_type=ReasoningType.ANALOGICAL,
            description="Simple analogical reasoning between source and target domains",
            handler=self._simple_analogical_reasoning,
            required_inputs=["source_domain", "target_domain"],
            provided_outputs=["mappings", "inferences"],
            compatibility={ReasoningType.ANALOGICAL, ReasoningType.HYBRID, ReasoningType.CREATIVE},
            complexity=5
        ))

        # LLM-assisted reasoning (if LLM engine is available)
        if self.llm_engine:
            self.register_strategy(ReasoningStrategy(
                strategy_id="llm_assisted",
                reasoning_type=ReasoningType.HYBRID,
                description="Use LLM to assist with general reasoning tasks",
                handler=self._llm_assisted_reasoning,
                required_inputs=["question", "context"],
                provided_outputs=["thoughts", "conclusions"],
                compatibility=set(ReasoningType),  # Compatible with all reasoning types
                confidence_modifier=-0.2,  # LLM responses have inherent uncertainty
                complexity=3
            ))

    def register_strategy(self, strategy: ReasoningStrategy) -> bool:
        """
        Register a reasoning strategy.

        Args:
            strategy: The strategy to register

        Returns:
            bool: True if successful, False otherwise
        """
        with self.lock:
            if strategy.strategy_id in self.strategies:
                self.logger.warning(f"Strategy with ID {strategy.strategy_id} already registered")
                return False

            self.strategies[strategy.strategy_id] = strategy
            self.logger.info(f"Registered reasoning strategy {strategy.strategy_id}")
            return True

    def unregister_strategy(self, strategy_id: str) -> bool:
        """
        Unregister a reasoning strategy.

        Args:
            strategy_id: ID of the strategy to unregister

        Returns:
            bool: True if successful, False otherwise
        """
        with self.lock:
            if strategy_id not in self.strategies:
                self.logger.warning(f"Strategy with ID {strategy_id} not found")
                return False

            del self.strategies[strategy_id]
            self.logger.info(f"Unregistered reasoning strategy {strategy_id}")
            return True

    def get_strategy(self, strategy_id: str) -> Optional[ReasoningStrategy]:
        """
        Get a registered reasoning strategy by ID.

        Args:
            strategy_id: ID of the strategy to retrieve

        Returns:
            Optional[ReasoningStrategy]: The strategy if found, None otherwise
        """
        return self.strategies.get(strategy_id)

    def get_strategies_by_type(self, reasoning_type: ReasoningType) -> List[ReasoningStrategy]:
        """
        Get all strategies of a specific reasoning type.

        Args:
            reasoning_type: The type of reasoning

        Returns:
            List[ReasoningStrategy]: List of matching strategies
        """
        return [s for s in self.strategies.values() if s.reasoning_type == reasoning_type]

    def create_reasoning_workspace(self, workspace_id: Optional[str] = None) -> str:
        """
        Create a new reasoning workspace.

        Args:
            workspace_id: Optional ID for the workspace (generated if None)

        Returns:
            str: ID of the created workspace
        """
        with self.lock:
            workspace_id = workspace_id or str(uuid.uuid4())
            if workspace_id in self.workspace:
                self.logger.warning(f"Workspace {workspace_id} already exists, will be overwritten")

            self.workspace[workspace_id] = {
                "created_at": time.time(),
                "data": {},
                "history": []
            }

            return workspace_id

    def get_workspace(self, workspace_id: str) -> Dict[str, Any]:
        """
        Get a reasoning workspace.

        Args:
            workspace_id: ID of the workspace

        Returns:
            Dict[str, Any]: The workspace data
        """
        with self.lock:
            if workspace_id not in self.workspace:
                self.logger.warning(f"Workspace {workspace_id} not found, creating new one")
                workspace_id = self.create_reasoning_workspace(workspace_id)

            return self.workspace[workspace_id]["data"]

    def update_workspace(self, workspace_id: str, updates: Dict[str, Any]) -> None:
        """
        Update a reasoning workspace.

        Args:
            workspace_id: ID of the workspace
            updates: Dictionary of updates to apply
        """
        with self.lock:
            if workspace_id not in self.workspace:
                self.logger.warning(f"Workspace {workspace_id} not found, creating new one")
                workspace_id = self.create_reasoning_workspace(workspace_id)

            # Record history of changes
            history_entry = {
                "timestamp": time.time(),
                "previous_state": self.workspace[workspace_id]["data"].copy(),
                "updates": updates.copy()
            }
            self.workspace[workspace_id]["history"].append(history_entry)

            # Apply updates
            self.workspace[workspace_id]["data"].update(updates)

    @CognitionManager.cognitive_capability(
        capability_type=CognitiveProcessType.REASONING,
        description="General reasoning capability",
        priority=50,
        compatible_modes={CognitiveMode.DELIBERATIVE, CognitiveMode.CREATIVE}
    )
    def reason(self,
              request: CognitiveRequest,
              context: CognitiveContext,
              query: Any,
              **kwargs) -> Dict[str, Any]:
        """
        Main reasoning capability. Processes the query using appropriate
        reasoning strategies based on context.

        Args:
            request: The cognitive request
            context: The cognitive context
            query: The query/question to reason about
            **kwargs: Additional arguments

        Returns:
            Dict[str, Any]: Reasoning results
        """
        start_time = time.time()

        try:
            self.logger.info(f"Processing reasoning request: {query}")

            # Create reasoning query
            reasoning_query = self._create_reasoning_query(query, context)

            # Create workspace for this reasoning process
            workspace_id = self.create_reasoning_workspace()

            # Initialize workspace with query and context
            initial_data = {
                "query": reasoning_query,
                "context_data": context.parameters,
                "working_memory_snapshot": {},  # Would be populated from working memory
                "intermediate_results": []
            }
            self.update_workspace(workspace_id, initial_data)

            # Execute reasoning process
            reasoning_result = self._execute_reasoning_process(reasoning_query, workspace_id)

            # Record processing time
            end_time = time.time()
            reasoning_result.processing_time = end_time - start_time

            # Convert reasoning result to dict for return
            return self._reasoning_result_to_dict(reasoning_result)

        except Exception as e:
            self.logger.error(f"Error in reasoning process: {e}", exc_info=True)

            # Return error information
            return {
                "error": str(e),
                "processing_time": time.time() - start_time,
                "query": str(query)
            }

    def _create_reasoning_query(self, query: Any, context: CognitiveContext) -> ReasoningQuery:
        """
        Create a reasoning query from the input query and context.

        Args:
            query: The input query
            context: The cognitive context

        Returns:
            ReasoningQuery: The created reasoning query
        """
        # Extract reasoning preferences from context
        preferred_strategies = context.parameters.get("preferred_strategies", [])
        excluded_strategies = context.parameters.get("excluded_strategies", [])

        # Parse required reasoning types
        required_types = []
        if "reasoning_types" in context.parameters:
            for type_name in context.parameters["reasoning_types"]:
                try:
                    if isinstance(type_name, str):
                        required_types.append(ReasoningType[type_name.upper()])
                    else:
                        required_types.append(type_name)
                except (KeyError, ValueError):
                    self.logger.warning(f"Unknown reasoning type: {type_name}")

        # Extract other parameters
        min_confidence = context.parameters.get("min_confidence", 0.0)
        max_steps = context.parameters.get("max_steps", None)
        max_time = context.parameters.get("max_time", None)
        exploration_factor = context.parameters.get("exploration_factor", 0.2)

        # Prepare context for the reasoning query
        query_context = {}

        # Add working memory state if available
        if self.working_memory:
            # In a real implementation, this would extract relevant information
            # from working memory based on the query
            query_context["working_memory"] = {"available": True}

        # Add semantic memory state if available
        if self.semantic_memory:
            query_context["semantic_memory"] = {"available": True}

        # Create the reasoning query
        return ReasoningQuery(
            question=query,
            context=query_context,
            preferred_strategies=preferred_strategies,
            excluded_strategies=excluded_strategies,
            required_reasoning_types=required_types,
            min_confidence=min_confidence,
            max_steps=max_steps,
            max_time=max_time,
            exploration_factor=exploration_factor
        )

    def _execute_reasoning_process(self, query: ReasoningQuery, workspace_id: str) -> ReasoningResult:
        """
        Execute a reasoning process for the given query.

        This is the core reasoning loop that selects and applies appropriate
        strategies based on the query and available information.

        Args:
            query: The reasoning query
            workspace_id: ID of the workspace to use

        Returns:
            ReasoningResult: Results of the reasoning process
        """
        # Initialize result
        result = ReasoningResult(
            query_id=query.query_id
        )

        # Fetch workspace
        workspace = self.get_workspace(workspace_id)

        # Track start time for time limit
        start_time = time.time()

        # Initialize reasoning process state
        current_step = 0

        # Flag to indicate if reasoning is complete
        reasoning_complete = False

        # Main reasoning loop
        while not reasoning_complete:
            # Check time limit
            if query.max_time and (time.time() - start_time) > query.max_time:
                self.logger.info(f"Reasoning time limit reached for query {query.query_id}")
                break

            # Check step limit
            if query.max_steps and current_step >= query.max_steps:
                self.logger.info(f"Reasoning step limit reached for query {query.query_id}")
                break

            # Select next reasoning strategy
            strategy_id = self._select_next_strategy(query, result, workspace)

            if not strategy_id:
                self.logger.info(f"No suitable strategy found for query {query.query_id}")
                break

            strategy = self.strategies[strategy_id]

            # Apply the strategy
            self.logger.debug(f"Applying strategy {strategy_id}")

            try:
                inference_step = InferenceStep(
                    strategy_id=strategy_id,
                    description=f"Applying {strategy.description}"
                )

                # Prepare inputs for the strategy
                strategy_inputs = self._prepare_strategy_inputs(strategy, workspace, query)
                inference_step.inputs = strategy_inputs

                # Execute the strategy
                strategy_result = strategy.handler(
                    query=query,
                    workspace=workspace,
                    inputs=strategy_inputs,
                    current_result=result
                )

                # Update inference step with results
                inference_step.outputs = strategy_result.get("outputs", {})
                inference_step.confidence = strategy_result.get("confidence", 0.5)

                if "sub_steps" in strategy_result:
                    inference_step.sub_steps = strategy_result["sub_steps"]

                if "supporting_evidence" in strategy_result:
                    inference_step.supporting_evidence = strategy_result["supporting_evidence"]

                if "contradicting_evidence" in strategy_result:
                    inference_step.contradicting_evidence = strategy_result["contradicting_evidence"]

                # Add step to result
                result.inference_steps.append(inference_step)

                # Update reasoning types used
                result.reasoning_types_used.add(strategy.reasoning_type)

                # Add strategy to used list
                result.strategies_used.append(strategy_id)

                # Update workspace with new information
                if "workspace_updates" in strategy_result:
                    self.update_workspace(workspace_id, strategy_result["workspace_updates"])

                # Check if new conclusions were generated
                if "conclusions" in strategy_result:
                    for conclusion in strategy_result["conclusions"]:
                        result.conclusions.append(conclusion)

                # Check if reasoning is complete
                if strategy_result.get("reasoning_complete", False):
                    reasoning_complete = True

                # Update evidence
                if "supporting_evidence" in strategy_result:
                    for conclusion_id, evidence in strategy_result.get("supporting_evidence", {}).items():
                        if conclusion_id not in result.supporting_evidence:
                            result.supporting_evidence[conclusion_id] = []
                        result.supporting_evidence[conclusion_id].extend(evidence)

                if "contradicting_evidence" in strategy_result:
                    for conclusion_id, evidence in strategy_result.get("contradicting_evidence", {}).items():
                        if conclusion_id not in result.contradicting_evidence:
                            result.contradicting_evidence[conclusion_id] = []
                        result.contradicting_evidence[conclusion_id].extend(evidence)

            except Exception as e:
                self.logger.error(f"Error applying strategy {strategy_id}: {e}", exc_info=True)

                # Create an error inference step
                error_step = InferenceStep(
                    strategy_id=strategy_id,
                    description=f"Error applying {strategy.description}: {str(e)}",
                    confidence=0.0
                )
                result.inference_steps.append(error_step)

            # Increment step counter
            current_step += 1

        # Calculate overall confidence
        if result.conclusions:
            # Average confidence of all conclusions, weighted by their individual confidence
            total_confidence = sum(c.get("confidence", 0.5) for c in result.conclusions)
            result.confidence = total_confidence / len(result.conclusions)
        else:
            result.confidence = 0.0

        # Mark as complete
        result.complete = reasoning_complete

        return result

    def _select_next_strategy(self,
                           query: ReasoningQuery,
                           current_result: ReasoningResult,
                           workspace: Dict[str, Any]) -> Optional[str]:
        """
        Select the next reasoning strategy to apply.

        This is a key function that determines the reasoning flow.
        It considers:
        - Query preferences
        - Current state of reasoning
        - Availability of required inputs for strategies
        - Compatibility between strategies
        - Exploration vs. exploitation tradeoff

        Args:
            query: The reasoning query
            current_result: Current reasoning result
            workspace: The reasoning workspace

        Returns:
            Optional[str]: ID of the selected strategy, or None if no suitable strategy
        """
        candidate_strategies = []

        # Consider preferred strategies first
        if query.preferred_strategies:
            for strategy_id in query.preferred_strategies:
                if strategy_id in self.strategies and strategy_id not in query.excluded_strategies:
                    candidate_strategies.append(strategy_id)

        # If no preferred strategies available, consider all strategies
        if not candidate_strategies:
            # Filter by required reasoning types
            if query.required_reasoning_types:
                for strategy in self.strategies.values():
                    if (strategy.reasoning_type in query.required_reasoning_types
                        and strategy.strategy_id not in query.excluded_strategies
                        and strategy.strategy_id not in current_result.strategies_used):
                        candidate_strategies.append(strategy.strategy_id)
            else:
                # Consider all strategies not excluded and not already used
                for strategy in self.strategies.values():
                    if (strategy.strategy_id not in query.excluded_strategies
                        and strategy.strategy_id not in current_result.strategies_used):
                        candidate_strategies.append(strategy.strategy_id)

        # If still no candidates, consider reusing strategies (except last one)
        if not candidate_strategies and current_result.strategies_used:
            for strategy_id in current_result.strategies_used[:-1]:
                if strategy_id not in query.excluded_strategies:
                    candidate_strategies.append(strategy_id)

        # If still no candidates, return None
        if not candidate_strategies:
            return None

        # Filter candidates by input availability
        valid_candidates = []
        for strategy_id in candidate_strategies:
            strategy = self.strategies[strategy_id]

            # Check if required inputs are available in workspace
            missing_inputs = []
            for input_name in strategy.required_inputs:
                if input_name not in workspace:
                    missing_inputs.append(input_name)

            # Special case: if missing inputs are standard (e.g., "query", "context"),
            # and corresponding data exists, consider the strategy valid
            if all(input_name in ["query", "context", "examples", "premises", "observations"]
                  for input_name in missing_inputs):
                valid_candidates.append(strategy_id)
            elif not missing_inputs:
                valid_candidates.append(strategy_id)

        # If no valid candidates, try to infer missing inputs
        if not valid_candidates:
            self.logger.debug(f"No strategies with all required inputs available, trying to infer missing inputs")

            # Try each candidate and see if we can infer its missing inputs
            for strategy_id in candidate_strategies:
                strategy = self.strategies[strategy_id]

                # Check which inputs are missing
                missing_inputs = []
                for input_name in strategy.required_inputs:
                    if input_name not in workspace:
                        missing_inputs.append(input_name)

                # Try to infer missing inputs
                if self._can_infer_inputs(strategy, missing_inputs, workspace, query):
                    valid_candidates.append(strategy_id)

        # If still no valid candidates, return None
        if not valid_candidates:
            return None

        # Apply exploration vs. exploitation
        if random.random() < query.exploration_factor:
            # Exploration: Choose a random strategy from valid candidates
            return random.choice(valid_candidates)
        else:
            # Exploitation: Choose based on confidence and synergy
            strategy_scores = {}

            for strategy_id in valid_candidates:
                strategy = self.strategies[strategy_id]

                # Base score is 1.0
                score = 1.0

                # Adjust based on compatibility with reasoning types already used
                for used_type in current_result.reasoning_types_used:
                    if used_type in strategy.compatibility:
                        score += 0.2

                # Adjust based on synergy with strategies already used
                for used_strategy in current_result.strategies_used:
                    if used_strategy in strategy.synergy:
                        score += strategy.synergy[used_strategy]

                # Store score
                strategy_scores[strategy_id] = score

            # Select strategy with highest score
            if strategy_scores:
                return max(strategy_scores.items(), key=lambda x: x[1])[0]
            else:
                # Fallback to random selection
                return random.choice(valid_candidates)

    def _can_infer_inputs(self,
                        strategy: ReasoningStrategy,
                        missing_inputs: List[str],
                        workspace: Dict[str, Any],
                        query: ReasoningQuery) -> bool:
        """
        Determine if missing inputs for a strategy can be inferred.

        Args:
            strategy: The strategy being considered
            missing_inputs: List of missing input names
            workspace: The reasoning workspace
            query: The reasoning query

        Returns:
            bool: True if inputs can be inferred, False otherwise
        """
        # This is a simplified implementation
        # In a real system, this would use more sophisticated inference methods

        if "query" in missing_inputs and "question" in workspace:
            # Can infer query from question
            return True

        if "premises" in missing_inputs and "context" in workspace:
            # Might be able to extract premises from context
            return True

        if "examples" in missing_inputs and ("observations" in workspace or "context" in workspace):
            # Might be able to extract examples from observations or context
            return True

        if "observations" in missing_inputs and ("examples" in workspace or "context" in workspace):
            # Might be able to interpret examples or context as observations
            return True

        # Default: cannot infer
        return False

    def _prepare_strategy_inputs(self,
                               strategy: ReasoningStrategy,
                               workspace: Dict[str, Any],
                               query: ReasoningQuery) -> Dict[str, Any]:
        """
        Prepare inputs for a reasoning strategy.

        This function:
        1. Extracts inputs already present in the workspace
        2. Attempts to infer missing inputs
        3. Creates default values for required inputs that cannot be found

        Args:
            strategy: The strategy to prepare inputs for
            workspace: The reasoning workspace
            query: The reasoning query

        Returns:
            Dict[str, Any]: Prepared inputs for the strategy
        """
        inputs = {}

        # First, extract inputs directly available in workspace
        for input_name in strategy.required_inputs:
            if input_name in workspace:
                inputs[input_name] = workspace[input_name]

        # Handle special cases for common inputs
        if "query" in strategy.required_inputs and "query" not in inputs:
            inputs["query"] = query.question

        if "context" in strategy.required_inputs and "context" not in inputs:
            inputs["context"] = query.context

        # Attempt to infer missing inputs
        missing_inputs = [name for name in strategy.required_inputs if name not in inputs]

        for input_name in missing_inputs:
            # Try to infer the input based on what's available
            inferred_value = self._infer_input(input_name, strategy, workspace, query)

            if inferred_value is not None:
                inputs[input_name] = inferred_value
            else:
                # Create an empty default value
                if input_name in ["premises", "rules", "examples", "observations"]:
                    inputs[input_name] = []
                elif input_name in ["source_domain", "target_domain"]:
                    inputs[input_name] = {}
                else:
                    inputs[input_name] = None

        return inputs

    def _infer_input(self,
                    input_name: str,
                    strategy: ReasoningStrategy,
                    workspace: Dict[str, Any],
                    query: ReasoningQuery) -> Optional[Any]:
        """
        Attempt to infer a missing input from available information.

        Args:
            input_name: Name of the input to infer
            strategy: The strategy requesting the input
            workspace: The reasoning workspace
            query: The reasoning query

        Returns:
            Optional[Any]: The inferred input value, or None if cannot be inferred
        """
        # This is a simplified implementation
        # In a real system, this would use more sophisticated inference methods

        # Inference rules for different input types
        if input_name == "premises" and "context" in workspace:
            # Extract statements from context that could be premises
            context = workspace["context"]
            if isinstance(context, dict) and "statements" in context:
                return context["statements"]
            return []

        if input_name == "rules" and "context" in workspace:
            # Extract rules from context
            context = workspace["context"]
            if isinstance(context, dict) and "rules" in context:
                return context["rules"]
            return []

        if input_name == "examples" and "observations" in workspace:
            # Interpret observations as examples
            return workspace["observations"]

        if input_name == "observations" and "examples" in workspace:
            # Interpret examples as observations
            return workspace["examples"]

        if input_name == "source_domain" and "context" in workspace:
            # Try to extract a source domain from context
            context = workspace["context"]
            if isinstance(context, dict) and "source_domain" in context:
                return context["source_domain"]
            return {}

        if input_name == "target_domain" and "context" in workspace:
            # Try to extract a target domain from context
            context = workspace["context"]
            if isinstance(context, dict) and "target_domain" in context:
                return context["target_domain"]
            return {}

        # Use LLM to infer input if available
        if self.llm_engine and input_name in ["premises", "rules", "examples", "observations"]:
            try:
                prompt = f"""
                Extract {input_name} from the following information:

                Question: {query.question}

                Extract only the {input_name} without additional commentary.
                Format your response as a list of items.
                """

                llm_response = self.llm_engine.complete({
                    "prompt": prompt,
                    "max_tokens": 200,
                    "temperature": 0.3
                })

                # Parse the response (simplified)
                if hasattr(llm_response, 'text'):
                    response_text = llm_response.text
                else:
                    response_text = str(llm_response)

                # Basic extraction of list items
                items = []
                for line in response_text.split('\n'):
                    line = line.strip()
                    if line.startswith('- ') or line.startswith('* '):
                        items.append(line[2:].strip())
                    elif line.startswith('\d+\. '):
                        items.append(line[line.find(' ')+1:].strip())

                if items:
                    return items

            except Exception as e:
                self.logger.warning(f"Error using LLM to infer {input_name}: {e}")

        # Cannot infer
        return None

    def _reasoning_result_to_dict(self, result: ReasoningResult) -> Dict[str, Any]:
        """
        Convert a ReasoningResult to a dictionary.

        Args:
            result: The reasoning result to convert

        Returns:
            Dict[str, Any]: Dictionary representation of the result
        """
        return {
            "result_id": result.result_id,
            "query_id": result.query_id,
            "conclusions": result.conclusions,
            "confidence": result.confidence,
            "reasoning_types_used": [rt.name for rt in result.reasoning_types_used],
            "strategies_used": result.strategies_used,
            "inference_steps": [self._inference_step_to_dict(step) for step in result.inference_steps],
            "supporting_evidence": result.supporting_evidence,
            "contradicting_evidence": result.contradicting_evidence,
            "processing_time": result.processing_time,
            "complete": result.complete,
            "annotations": result.annotations
        }

    def _inference_step_to_dict(self, step: InferenceStep) -> Dict[str, Any]:
        """
        Convert an InferenceStep to a dictionary.

        Args:
            step: The inference step to convert

        Returns:
            Dict[str, Any]: Dictionary representation of the step
        """
        return {
            "step_id": step.step_id,
            "strategy_id": step.strategy_id,
            "description": step.description,
            "inputs": step.inputs,
            "outputs": step.outputs,
            "confidence": step.confidence,
            "supporting_evidence": step.supporting_evidence,
            "contradicting_evidence": step.contradicting_evidence,
            "sub_steps": [self._inference_step_to_dict(sub_step) for sub_step in step.sub_steps],
            "timestamp": step.timestamp
        }

    #------------------------------------------------------------------
    # Basic reasoning strategies
    #------------------------------------------------------------------

    def _simple_deductive_reasoning(self,
                                 query: ReasoningQuery,
                                 workspace: Dict[str, Any],
                                 inputs: Dict[str, Any],
                                 current_result: ReasoningResult) -> Dict[str, Any]:
        """
        Simple deductive reasoning strategy.

        Applies basic modus ponens: If A implies B, and A is true, then B is true.

        Args:
            query: The reasoning query
            workspace: The reasoning workspace
            inputs: Strategy inputs
            current_result: Current reasoning result

        Returns:
            Dict[str, Any]: Strategy results
        """
        premises = inputs.get("premises", [])
        rules = inputs.get("rules", [])

        # Check if we have the necessary inputs
        if not premises or not rules:
            return {
                "outputs": {
                    "message": "Insufficient inputs for deductive reasoning",
                    "conclusions": []
                },
                "confidence": 0.0,
                "workspace_updates": {},
                "conclusions": []
            }

        # Simple deductive reasoning (modus ponens)
        conclusions = []
        supporting_evidence = {}
        sub_steps = []

        # For each rule of the form "if A then B"
        for rule in rules:
            # Extract antecedent (if part) and consequent (then part)
            if "if" in rule and "then" in rule:
                antecedent = rule["if"]
                consequent = rule["then"]

                # Check if antecedent is in premises
                if antecedent in premises:
                    # Apply modus ponens
                    conclusion_id = str(uuid.uuid4())
                    conclusion = {
                        "id": conclusion_id,
                        "statement": consequent,
                        "confidence": rule.get("confidence", 0.95),
                        "derived_from": {
                            "rule": rule,
                            "premise": antecedent
                        }
                    }
                    conclusions.append(conclusion)

                    # Record supporting evidence
                    supporting_evidence[conclusion_id] = [
                        f"Premise: {antecedent}",
                        f"Rule: If {antecedent} then {consequent}"
                    ]

                    # Create sub-step
                    sub_step = InferenceStep(
                        description=f"Applied modus ponens with rule: If {antecedent} then {consequent}",
                        inputs={"premise": antecedent, "rule": rule},
                        outputs={"conclusion": consequent},
                        confidence=rule.get("confidence", 0.95)
                    )
                    sub_steps.append(sub_step)

        # Transitive reasoning: If A implies B and B implies C, then A implies C
        # This is a simple example of chaining multiple deductions
        new_conclusions = []
        for conclusion in conclusions:
            statement = conclusion["statement"]

            # Check if this statement is an antecedent in any rule
            for rule in rules:
                if "if" in rule and "then" in rule and rule["if"] == statement:
                    # Apply transitive reasoning
                    trans_conclusion_id = str(uuid.uuid4())
                    trans_conclusion = {
                        "id": trans_conclusion_id,
                        "statement": rule["then"],
                        "confidence": min(conclusion["confidence"], rule.get("confidence", 0.95)),
                        "derived_from": {
                            "rule": rule,
                            "prior_conclusion": conclusion
                        }
                    }
                    new_conclusions.append(trans_conclusion)

                    # Record supporting evidence
                    supporting_evidence[trans_conclusion_id] = [
                        f"Prior conclusion: {statement}",
                        f"Rule: If {rule['if']} then {rule['then']}"
                    ]

                    # Create sub-step
                    sub_step = InferenceStep(
                        description=f"Applied transitive reasoning with rule: If {rule['if']} then {rule['then']}",
                        inputs={"prior_conclusion": statement, "rule": rule},
                        outputs={"conclusion": rule["then"]},
                        confidence=min(conclusion["confidence"], rule.get("confidence", 0.95))
                    )
                    sub_steps.append(sub_step)

        # Add transitive conclusions
        conclusions.extend(new_conclusions)

        # Calculate overall confidence
        confidence = 0.95 if conclusions else 0.0

        # Check if reasoning is complete (if we derived any conclusions)
        reasoning_complete = len(conclusions) > 0

        return {
            "outputs": {
                "conclusions": conclusions
            },
            "workspace_updates": {
                "deductive_conclusions": conclusions
            },
            "confidence": confidence,
            "conclusions": conclusions,
            "reasoning_complete": reasoning_complete,
            "supporting_evidence": supporting_evidence,
            "sub_steps": sub_steps
        }

    def _simple_inductive_reasoning(self,
                                  query: ReasoningQuery,
                                  workspace: Dict[str, Any],
                                  inputs: Dict[str, Any],
                                  current_result: ReasoningResult) -> Dict[str, Any]:
        """
        Simple inductive reasoning strategy.

        Looks for patterns in examples and generates generalizations.

        Args:
            query: The reasoning query
            workspace: The reasoning workspace
            inputs: Strategy inputs
            current_result: Current reasoning result

        Returns:
            Dict[str, Any]: Strategy results
        """
        examples = inputs.get("examples", [])

        # Check if we have the necessary inputs
        if not examples:
            return {
                "outputs": {
                    "message": "Insufficient examples for inductive reasoning",
                    "patterns": [],
                    "generalizations": []
                },
                "confidence": 0.0,
                "workspace_updates": {},
                "conclusions": []
            }

        # Simple pattern detection in examples
        patterns = []
        generalizations = []
        supporting_evidence = {}
        sub_steps = []

        # Look for common attributes across examples
        if all(isinstance(example, dict) for example in examples):
            # For dictionaries, look for common keys and values
            common_keys = set(examples[0].keys())
            for example in examples[1:]:
                common_keys &= set(example.keys())

            # For each common key, check if values are consistent
            for key in common_keys:
                values = [example[key] for example in examples]

                # If all values are the same, this is a pattern
                if all(value == values[0] for value in values):
                    pattern_id = str(uuid.uuid4())
                    pattern = {
                        "id": pattern_id,
                        "type": "common_attribute",
                        "attribute": key,
                        "value": values[0],
                        "confidence": 0.7  # Lower confidence for inductive reasoning
                    }
                    patterns.append(pattern)

                    # Create generalization based on this pattern
                    generalization_id = str(uuid.uuid4())
                    generalization = {
                        "id": generalization_id,
                        "statement": f"All examples have {key} = {values[0]}",
                        "confidence": 0.7,
                        "derived_from": {
                            "pattern": pattern
                        }
                    }
                    generalizations.append(generalization)

                    # Record supporting evidence
                    supporting_evidence[generalization_id] = [
                        f"Example {i}: {example[key]}" for i, example in enumerate(examples)
                    ]

                    # Create sub-step
                    sub_step = InferenceStep(
                        description=f"Identified common attribute pattern: {key} = {values[0]}",
                        inputs={"examples": examples},
                        outputs={"pattern": pattern, "generalization": generalization},
                        confidence=0.7
                    )
                    sub_steps.append(sub_step)

                # If values follow a pattern (e.g., always increasing)
                elif all(isinstance(value, (int, float)) for value in values):
                    # Check for monotonic increase
                    if all(values[i] < values[i+1] for i in range(len(values)-1)):
                        pattern_id = str(uuid.uuid4())
                        pattern = {
                            "id": pattern_id,
                            "type": "increasing_attribute",
                            "attribute": key,
                            "values": values,
                            "confidence": 0.6  # Even lower confidence for trend detection
                        }
                        patterns.append(pattern)

                        # Create generalization based on this pattern
                        generalization_id = str(uuid.uuid4())
                        generalization = {
                            "id": generalization_id,
                            "statement": f"The value of {key} increases across examples",
                            "confidence": 0.6,
                            "derived_from": {
                                "pattern": pattern
                            }
                        }
                        generalizations.append(generalization)

                        # Record supporting evidence
                        supporting_evidence[generalization_id] = [
                            f"Example {i}: {key} = {example[key]}" for i, example in enumerate(examples)
                        ]

                        # Create sub-step
                        sub_step = InferenceStep(
                            description=f"Identified increasing trend in attribute: {key}",
                            inputs={"examples": examples},
                            outputs={"pattern": pattern, "generalization": generalization},
                            confidence=0.6
                        )
                        sub_steps.append(sub_step)

        # Use LLM for more sophisticated pattern detection if available
        elif self.llm_engine:
            try:
                # Prepare examples in a format suitable for LLM
                examples_text = "\n".join([f"Example {i}: {example}" for i, example in enumerate(examples)])

                prompt = f"""
                Analyze the following examples and identify patterns or generalizations:

                {examples_text}

                Provide your analysis in this format:

                Patterns:
                1. [pattern description]
                2. [pattern description]

                Generalizations:
                1. [generalization statement]
                2. [generalization statement]

                Please be specific and concrete in your analysis.
                """

                llm_response = self.llm_engine.complete({
                    "prompt": prompt,
                    "max_tokens": 300,
                    "temperature": 0.3
                })

                # Parse the response (simplified)
                if hasattr(llm_response, 'text'):
                    response_text = llm_response.text
                else:
                    response_text = str(llm_response)

                # Extract patterns and generalizations
                llm_patterns_section = re.search(r'Patterns:(.*?)(?:Generalizations:|$)', response_text, re.DOTALL)
                llm_generalizations_section = re.search(r'Generalizations:(.*?)$', response_text, re.DOTALL)

                if llm_patterns_section:
                    llm_patterns_text = llm_patterns_section.group(1).strip()
                    pattern_lines = re.findall(r'\d+\.\s+(.*?)(?:\n|$)', llm_patterns_text)

                    for i, pattern_text in enumerate(pattern_lines):
                        pattern_id = str(uuid.uuid4())
                        pattern = {
                            "id": pattern_id,
                            "type": "llm_detected",
                            "description": pattern_text,
                            "confidence": 0.65  # Lower confidence for LLM-detected patterns
                        }
                        patterns.append(pattern)

                        # Create sub-step
                        sub_step = InferenceStep(
                            description=f"LLM detected pattern: {pattern_text}",
                            inputs={"examples": examples},
                            outputs={"pattern": pattern},
                            confidence=0.65
                        )
                        sub_steps.append(sub_step)

                if llm_generalizations_section:
                    llm_generalizations_text = llm_generalizations_section.group(1).strip()
                    generalization_lines = re.findall(r'\d+\.\s+(.*?)(?:\n|$)', llm_generalizations_text)

                    for i, generalization_text in enumerate(generalization_lines):
                        generalization_id = str(uuid.uuid4())
                        generalization = {
                            "id": generalization_id,
                            "statement": generalization_text,
                            "confidence": 0.6,
                            "derived_from": {
                                "llm_analysis": True
                            }
                        }
                        generalizations.append(generalization)

                        # Record supporting evidence
                        supporting_evidence[generalization_id] = [
                            f"Example {i}: {example}" for i, example in enumerate(examples)
                        ]

                        # Create sub-step
                        sub_step = InferenceStep(
                            description=f"LLM proposed generalization: {generalization_text}",
                            inputs={"examples": examples},
                            outputs={"generalization": generalization},
                            confidence=0.6
                        )
                        sub_steps.append(sub_step)

            except Exception as e:
                self.logger.warning(f"Error using LLM for pattern detection: {e}")

        # Calculate overall confidence
        confidence = 0.7 if generalizations else 0.0

        # Check if reasoning is complete (if we derived any generalizations)
        reasoning_complete = len(generalizations) > 0

        # Convert generalizations to conclusions
        conclusions = []
        for generalization in generalizations:
            conclusion_id = str(uuid.uuid4())
            conclusion = {
                "id": conclusion_id,
                "statement": generalization["statement"],
                "confidence": generalization["confidence"],
                "derived_from": generalization.get("derived_from", {})
            }
            conclusions.append(conclusion)

            # Transfer supporting evidence
            if generalization["id"] in supporting_evidence:
                supporting_evidence[conclusion_id] = supporting_evidence[generalization["id"]]

        return {
            "outputs": {
                "patterns": patterns,
                "generalizations": generalizations
            },
            "workspace_updates": {
                "inductive_patterns": patterns,
                "inductive_generalizations": generalizations
            },
            "confidence": confidence,
            "conclusions": conclusions,
            "reasoning_complete": reasoning_complete,
            "supporting_evidence": supporting_evidence,
            "sub_steps": sub_steps
        }

    def _simple_abductive_reasoning(self,
                                  query: ReasoningQuery,
                                  workspace: Dict[str, Any],
                                  inputs: Dict[str, Any],
                                  current_result: ReasoningResult) -> Dict[str, Any]:
        """
        Simple abductive reasoning strategy.

        Generates potential explanations for observations.

        Args:
            query: The reasoning query
            workspace: The reasoning workspace
            inputs: Strategy inputs
            current_result: Current reasoning result

        Returns:
            Dict[str, Any]: Strategy results
        """
        observations = inputs.get("observations", [])

        # Check if we have the necessary inputs
        if not observations:
            return {
                "outputs": {
                    "message": "Insufficient observations for abductive reasoning",
                    "explanations": []
                },
                "confidence": 0.0,
                "workspace_updates": {},
                "conclusions": []
            }

        # Simple abductive reasoning (inference to the best explanation)
        explanations = []
        supporting_evidence = {}
        contradicting_evidence = {}
        sub_steps = []

        # If LLM is available, use it for generating explanations
        if self.llm_engine:
            try:
                # Prepare observations in a format suitable for LLM
                observations_text = "\n".join([f"Observation {i}: {obs}" for i, obs in enumerate(observations)])

                prompt = f"""
                Given the following observations:

                {observations_text}

                Generate 3-5 possible explanations that could account for these observations.
                For each explanation:
                1. Provide a clear statement of the explanation
                2. Explain how it accounts for the observations
                3. Rate your confidence in this explanation (0-100%)
                4. Identify any observations that support or contradict this explanation

                Format your response as:

                Explanation 1: [explanation statement]
                - Accounts for observations by: [explanation]
                - Confidence: [percentage]
                - Supporting evidence: [list of supporting observations]
                - Contradicting evidence: [list of contradicting observations]

                Explanation 2: ...
                """

                llm_response = self.llm_engine.complete({
                    "prompt": prompt,
                    "max_tokens": 500,
                    "temperature": 0.7  # Higher temperature for creative explanations
                })

                # Parse the response (simplified)
                if hasattr(llm_response, 'text'):
                    response_text = llm_response.text
                else:
                    response_text = str(llm_response)

                # Extract explanations
                explanation_sections = re.findall(r'Explanation \d+: (.*?)(?=Explanation \d+:|$)', response_text, re.DOTALL)

                for section in explanation_sections:
                    # Extract explanation statement
                    statement = section.split('\n')[0].strip()

                    # Extract confidence
                    confidence_match = re.search(r'Confidence:\s*(\d+)', section)
                    confidence = 0.5  # Default
                    if confidence_match:
                        confidence = int(confidence_match.group(1)) / 100  # Convert percentage to 0-1

                    # Extract supporting evidence
                    supporting_match = re.search(r'Supporting evidence:(.*?)(?=Contradicting evidence:|$)', section, re.DOTALL)
                    supporting = []
                    if supporting_match:
                        supporting_text = supporting_match.group(1)
                        supporting = [item.strip() for item in re.findall(r'- (.*?)(?=\n|$)', supporting_text)]

                    # Extract contradicting evidence
                    contradicting_match = re.search(r'Contradicting evidence:(.*?)$', section, re.DOTALL)
                    contradicting = []
                    if contradicting_match:
                        contradicting_text = contradicting_match.group(1)
                        contradicting = [item.strip() for item in re.findall(r'- (.*?)(?=\n|$)', contradicting_text)]

                    # Create explanation
                    explanation_id = str(uuid.uuid4())
                    explanation = {
                        "id": explanation_id,
                        "statement": statement,
                        "confidence": confidence,
                        "derived_from": {
                            "observations": observations
                        }
                    }
                    explanations.append(explanation)

                    # Record evidence
                    if supporting:
                        supporting_evidence[explanation_id] = supporting

                    if contradicting:
                        contradicting_evidence[explanation_id] = contradicting

                    # Create sub-step
                    sub_step = InferenceStep(
                        description=f"Generated explanation: {statement}",
                        inputs={"observations": observations},
                        outputs={"explanation": explanation},
                        confidence=confidence,
                        supporting_evidence=supporting,
                        contradicting_evidence=contradicting
                    )
                    sub_steps.append(sub_step)

            except Exception as e:
                self.logger.warning(f"Error using LLM for abductive reasoning: {e}")

        # Fallback: Generate simple explanations based on observations
        if not explanations:
            for obs in observations:
                explanation_id = str(uuid.uuid4())
                explanation = {
                    "id": explanation_id,
                    "statement": f"A possible explanation for '{obs}' is that it is a common occurrence",
                    "confidence": 0.3,
                    "derived_from": {
                        "observation": obs
                    }
                }
                explanations.append(explanation)

                # Record supporting evidence
                supporting_evidence[explanation_id] = [obs]

                # Create sub-step
                sub_step = InferenceStep(
                    description=f"Generated simple explanation for observation: {obs}",
                    inputs={"observation": obs},
                    outputs={"explanation": explanation},
                    confidence=0.3
                )
                sub_steps.append(sub_step)

        # Rank explanations by confidence
        explanations.sort(key=lambda x: x["confidence"], reverse=True)

        # Calculate overall confidence (based on best explanation)
        confidence = explanations[0]["confidence"] if explanations else 0.0

        # Check if reasoning is complete (if we derived any explanations)
        reasoning_complete = len(explanations) > 0

        # Convert explanations to conclusions
        conclusions = []
        for explanation in explanations:
            conclusion_id = str(uuid.uuid4())
            conclusion = {
                "id": conclusion_id,
                "statement": explanation["statement"],
                "confidence": explanation["confidence"],
                "derived_from": explanation.get("derived_from", {})
            }
            conclusions.append(conclusion)

            # Transfer supporting evidence
            if explanation["id"] in supporting_evidence:
                supporting_evidence[conclusion_id] = supporting_evidence[explanation["id"]]

            # Transfer contradicting evidence
            if explanation["id"] in contradicting_evidence:
                contradicting_evidence[conclusion_id] = contradicting_evidence[explanation["id"]]

        return {
            "outputs": {
                "explanations": explanations
            },
            "workspace_updates": {
                "abductive_explanations": explanations
            },
            "confidence": confidence,
            "conclusions": conclusions,
            "reasoning_complete": reasoning_complete,
            "supporting_evidence": supporting_evidence,
            "contradicting_evidence": contradicting_evidence,
            "sub_steps": sub_steps
        }

    def _simple_analogical_reasoning(self,
                                   query: ReasoningQuery,
                                   workspace: Dict[str, Any],
                                   inputs: Dict[str, Any],
                                   current_result: ReasoningResult) -> Dict[str, Any]:
        """
        Simple analogical reasoning strategy.

        Maps relationships from a source domain to a target domain.

        Args:
            query: The reasoning query
            workspace: The reasoning workspace
            inputs: Strategy inputs
            current_result: Current reasoning result

        Returns:
            Dict[str, Any]: Strategy results
        """
        source_domain = inputs.get("source_domain", {})
        target_domain = inputs.get("target_domain", {})

        # Check if we have the necessary inputs
        if not source_domain or not target_domain:
            return {
                "outputs": {
                    "message": "Insufficient domain information for analogical reasoning",
                    "mappings": [],
                    "inferences": []
                },
                "confidence": 0.0,
                "workspace_updates": {},
                "conclusions": []
            }

        # Simple analogical reasoning
        mappings = []
        inferences = []
        supporting_evidence = {}
        sub_steps = []

        # Extract entities from both domains
        source_entities = source_domain.get("entities", [])
        target_entities = target_domain.get("entities", [])

        # Extract relationships from source domain
        source_relationships = source_domain.get("relationships", [])

        # Map entities based on similarity
        entity_mappings = {}

        # If we have attributes for entities, use them for mapping
        if all("attributes" in entity for entity in source_entities) and all("attributes" in entity for entity in target_entities):
            # For each source entity
            for source_entity in source_entities:
                # Find most similar target entity based on attributes
                best_match = None
                best_score = 0.0

                for target_entity in target_entities:
                    # Calculate attribute similarity
                    score = self._calculate_attribute_similarity(
                        source_entity["attributes"],
                        target_entity["attributes"]
                    )

                    if score > best_score:
                        best_score = score
                        best_match = target_entity

                # If we found a reasonable match
                if best_match and best_score > 0.3:
                    mapping_id = str(uuid.uuid4())
                    mapping = {
                        "id": mapping_id,
                        "source_entity": source_entity["name"],
                        "target_entity": best_match["name"],
                        "confidence": best_score,
                        "mapped_attributes": self._get_shared_attributes(source_entity["attributes"], best_match["attributes"])
                    }
                    mappings.append(mapping)

                    # Record entity mapping
                    entity_mappings[source_entity["name"]] = best_match["name"]

                    # Create sub-step
                    sub_step = InferenceStep(
                        description=f"Mapped {source_entity['name']} to {best_match['name']} (similarity: {best_score:.2f})",
                        inputs={"source_entity": source_entity, "target_entity": best_match},
                        outputs={"mapping": mapping},
                        confidence=best_score
                    )
                    sub_steps.append(sub_step)

        # If no attribute-based mapping was possible, use simple name-based mapping
        if not mappings and len(source_entities) == len(target_entities):
            for i, source_entity in enumerate(source_entities):
                target_entity = target_entities[i]

                mapping_id = str(uuid.uuid4())
                mapping = {
                    "id": mapping_id,
                    "source_entity": source_entity["name"],
                    "target_entity": target_entity["name"],
                    "confidence": 0.3,  # Lower confidence for simple position-based mapping
                    "mapped_attributes": []
                }
                mappings.append(mapping)

                # Record entity mapping
                entity_mappings[source_entity["name"]] = target_entity["name"]

                # Create sub-step
                sub_step = InferenceStep(
                    description=f"Mapped {source_entity['name']} to {target_entity['name']} (positional mapping)",
                    inputs={"source_entity": source_entity, "target_entity": target_entity},
                    outputs={"mapping": mapping},
                    confidence=0.3
                )
                sub_steps.append(sub_step)

        # Transfer relationships from source to target domain
        for relationship in source_relationships:
            # Get source entities involved in this relationship
            source_entity1 = relationship.get("entity1")
            source_entity2 = relationship.get("entity2", None)  # May be None for unary relationships
            relation_type = relationship.get("type")

            # Check if we have mappings for these entities
            if source_entity1 in entity_mappings:
                target_entity1 = entity_mappings[source_entity1]
                target_entity2 = None

                if source_entity2 and source_entity2 in entity_mappings:
                    target_entity2 = entity_mappings[source_entity2]

                # Create inference
                inference_id = str(uuid.uuid4())

                if target_entity2:
                    # Binary relationship inference
                    inference = {
                        "id": inference_id,
                        "statement": f"{relation_type}({target_entity1}, {target_entity2})",
                        "confidence": 0.5,  # Lower confidence for analogical inference
                        "derived_from": {
                            "source_relationship": relationship,
                            "mapping1": next((m for m in mappings if m["source_entity"] == source_entity1), None),
                            "mapping2": next((m for m in mappings if m["source_entity"] == source_entity2), None)
                        }
                    }
                else:
                    # Unary relationship inference
                    inference = {
                        "id": inference_id,
                        "statement": f"{relation_type}({target_entity1})",
                        "confidence": 0.5,
                        "derived_from": {
                            "source_relationship": relationship,
                            "mapping": next((m for m in mappings if m["source_entity"] == source_entity1), None)
                        }
                    }

                inferences.append(inference)

                # Record supporting evidence
                if source_entity2:
                    supporting_evidence[inference_id] = [
                        f"Source relationship: {relation_type}({source_entity1}, {source_entity2})",
                        f"Mapping: {source_entity1} -> {target_entity1}",
                        f"Mapping: {source_entity2} -> {target_entity2}"
                    ]
                else:
                    supporting_evidence[inference_id] = [
                        f"Source relationship: {relation_type}({source_entity1})",
                        f"Mapping: {source_entity1} -> {target_entity1}"
                    ]

                # Create sub-step
                if target_entity2:
                    sub_step = InferenceStep(
                        description=f"Transferred relationship {relation_type} from ({source_entity1}, {source_entity2}) to ({target_entity1}, {target_entity2})",
                        inputs={"relationship": relationship, "mappings": entity_mappings},
                        outputs={"inference": inference},
                        confidence=0.5
                    )
                else:
                    sub_step = InferenceStep(
                        description=f"Transferred relationship {relation_type} from {source_entity1} to {target_entity1}",
                        inputs={"relationship": relationship, "mappings": entity_mappings},
                        outputs={"inference": inference},
                        confidence=0.5
                    )

                sub_steps.append(sub_step)

        # Use LLM for more sophisticated analogical reasoning if available
        if self.llm_engine and not inferences:
            try:
                # Prepare domain descriptions for LLM
                source_text = f"Source domain: {source_domain.get('name', 'Source')}\n"
                source_text += "Entities:\n"
                for entity in source_entities:
                    source_text += f"- {entity['name']}"
                    if "attributes" in entity:
                        attrs = ", ".join(f"{k}: {v}" for k, v in entity["attributes"].items())
                        source_text += f" ({attrs})"
                    source_text += "\n"

                source_text += "Relationships:\n"
                for rel in source_relationships:
                    if "entity2" in rel:
                        source_text += f"- {rel['type']}({rel['entity1']}, {rel['entity2']})\n"
                    else:
                        source_text += f"- {rel['type']}({rel['entity1']})\n"

                target_text = f"Target domain: {target_domain.get('name', 'Target')}\n"
                target_text += "Entities:\n"
                for entity in target_entities:
                    target_text += f"- {entity['name']}"
                    if "attributes" in entity:
                        attrs = ", ".join(f"{k}: {v}" for k, v in entity["attributes"].items())
                        target_text += f" ({attrs})"
                    target_text += "\n"

                prompt = f"""
                I'll provide information about a source domain and a target domain.
                Use analogical reasoning to map entities between domains and infer relationships in the target domain.

                {source_text}

                {target_text}

                Please provide:
                1. Entity mappings from source to target domain
                2. Inferred relationships in the target domain

                Format your response as:

                Entity mappings:
                - [source_entity] maps to [target_entity] (confidence: [0-100%])
                - ...

                Inferred relationships:
                - [relationship_type]([target_entity1], [target_entity2]) (confidence: [0-100%])
                - ...

                Explain your reasoning for each mapping and inference.
                """

                llm_response = self.llm_engine.complete({
                    "prompt": prompt,
                    "max_tokens": 500,
                    "temperature": 0.4
                })

                # Parse the response (simplified)
                if hasattr(llm_response, 'text'):
                    response_text = llm_response.text
                else:
                    response_text = str(llm_response)

                # Extract mappings
                mapping_section = re.search(r'Entity mappings:(.*?)(?=Inferred relationships:|$)', response_text, re.DOTALL)
                if mapping_section:
                    mapping_text = mapping_section.group(1).strip()
                    mapping_lines = re.findall(r'- (.*?)(?=\n|$)', mapping_text)

                    entity_mappings = {}
                    for line in mapping_lines:
                        # Extract source, target, and confidence
                        match = re.match(r'(.*?)\s+maps to\s+(.*?)\s+\(confidence:\s+(\d+)%\)', line)
                        if match:
                            source_entity = match.group(1).strip()
                            target_entity = match.group(2).strip()
                            confidence = int(match.group(3)) / 100

                            mapping_id = str(uuid.uuid4())
                            mapping = {
                                "id": mapping_id,
                                "source_entity": source_entity,
                                "target_entity": target_entity,
                                "confidence": confidence,
                                "llm_generated": True
                            }
                            mappings.append(mapping)

                            # Record entity mapping
                            entity_mappings[source_entity] = target_entity

                            # Create sub-step
                            sub_step = InferenceStep(
                                description=f"LLM mapped {source_entity} to {target_entity} (confidence: {confidence:.2f})",
                                inputs={"source_entity": source_entity, "target_entity": target_entity},
                                outputs={"mapping": mapping},
                                confidence=confidence
                            )
                            sub_steps.append(sub_step)

                # Extract inferences
                inference_section = re.search(r'Inferred relationships:(.*?)$', response_text, re.DOTALL)
                if inference_section:
                    inference_text = inference_section.group(1).strip()
                    inference_lines = re.findall(r'- (.*?)(?=\n|$)', inference_text)

                    for line in inference_lines:
                        # Extract relationship and confidence
                        match = re.match(r'(.*?)\s+\(confidence:\s+(\d+)%\)', line)
                        if match:
                            relationship = match.group(1).strip()
                            confidence = int(match.group(2)) / 100

                            # Parse relationship text
                            rel_match = re.match(r'(\w+)\((.*?)\)', relationship)
                            if rel_match:
                                rel_type = rel_match.group(1)
                                entities = [e.strip() for e in rel_match.group(2).split(',')]

                                inference_id = str(uuid.uuid4())
                                inference = {
                                    "id": inference_id,
                                    "statement": relationship,
                                    "confidence": confidence,
                                    "llm_generated": True
                                }
                                inferences.append(inference)

                                # Record supporting evidence based on entity mappings
                                supporting_evidence[inference_id] = [
                                    f"LLM-inferred relationship with confidence {confidence:.2f}"
                                ]

                                # Add any mapping evidence if we can find it
                                for entity in entities:
                                    for source, target in entity_mappings.items():
                                        if target == entity:
                                            supporting_evidence[inference_id].append(
                                                f"Mapping: {source} -> {target}"
                                            )

                                # Create sub-step
                                sub_step = InferenceStep(
                                    description=f"LLM inferred relationship: {relationship} (confidence: {confidence:.2f})",
                                    inputs={"entity_mappings": entity_mappings},
                                    outputs={"inference": inference},
                                    confidence=confidence
                                )
                                sub_steps.append(sub_step)

            except Exception as e:
                self.logger.warning(f"Error using LLM for analogical reasoning: {e}")

        # Calculate overall confidence
        confidence = 0.5 if mappings and inferences else 0.0

        # Check if reasoning is complete (if we derived any inferences)
        reasoning_complete = len(inferences) > 0

        # Convert inferences to conclusions
        conclusions = []
        for inference in inferences:
            conclusion_id = str(uuid.uuid4())
            conclusion = {
                "id": conclusion_id,
                "statement": inference["statement"],
                "confidence": inference["confidence"],
                "derived_from": inference.get("derived_from", {})
            }
            conclusions.append(conclusion)

            # Transfer supporting evidence
            if inference["id"] in supporting_evidence:
                supporting_evidence[conclusion_id] = supporting_evidence[inference["id"]]

        return {
            "outputs": {
                "mappings": mappings,
                "inferences": inferences
            },
            "workspace_updates": {
                "analogical_mappings": mappings,
                "analogical_inferences": inferences
            },
            "confidence": confidence,
            "conclusions": conclusions,
            "reasoning_complete": reasoning_complete,
            "supporting_evidence": supporting_evidence,
            "sub_steps": sub_steps
        }

    def _calculate_attribute_similarity(self, attrs1: Dict[str, Any], attrs2: Dict[str, Any]) -> float:
        """
        Calculate similarity between two sets of attributes.

        Args:
            attrs1: First set of attributes
            attrs2: Second set of attributes

        Returns:
            float: Similarity score (0-1)
        """
        if not attrs1 or not attrs2:
            return 0.0

        shared_keys = set(attrs1.keys()) & set(attrs2.keys())
        if not shared_keys:
            return 0.0

        # Calculate similarity for each shared key
        key_similarities = []

        for key in shared_keys:
            value1 = attrs1[key]
            value2 = attrs2[key]

            # Calculate value similarity based on type
            if type(value1) != type(value2):
                # Different types
                key_similarities.append(0.0)
            elif isinstance(value1, (int, float)) and isinstance(value2, (int, float)):
                # Numerical values
                max_val = max(abs(value1), abs(value2))
                if max_val == 0:
                    key_similarities.append(1.0)  # Both zero
                else:
                    diff = abs(value1 - value2) / max_val
                    key_similarities.append(max(0.0, 1.0 - diff))
            elif isinstance(value1, str) and isinstance(value2, str):
                # String values - simple exact match
                if value1.lower() == value2.lower():
                    key_similarities.append(1.0)
                else:
                    # Simple partial match
                    key_similarities.append(0.3)
            else:
                # Other types - simple equality check
                key_similarities.append(1.0 if value1 == value2 else 0.0)

        # Compute overall similarity
        return sum(key_similarities) / len(key_similarities)

    def _get_shared_attributes(self, attrs1: Dict[str, Any], attrs2: Dict[str, Any]) -> List[str]:
        """
        Get list of attributes that have similar values in both attribute sets.

        Args:
            attrs1: First set of attributes
            attrs2: Second set of attributes

        Returns:
            List[str]: List of shared attribute names
        """
        shared = []
        shared_keys = set(attrs1.keys()) & set(attrs2.keys())

        for key in shared_keys:
            value1 = attrs1[key]
            value2 = attrs2[key]

            if type(value1) == type(value2):
                if isinstance(value1, (int, float)):
                    # Consider numerical values similar if within 20%
                    max_val = max(abs(value1), abs(value2))
                    if max_val == 0 or abs(value1 - value2) / max_val < 0.2:
                        shared.append(key)
                elif value1 == value2:
                    # Exact match for other types
                    shared.append(key)

        return shared

    def _llm_assisted_reasoning(self,
                             query: ReasoningQuery,
                             workspace: Dict[str, Any],
                             inputs: Dict[str, Any],
                             current_result: ReasoningResult) -> Dict[str, Any]:
        """
        Use LLM to assist with general reasoning tasks.

        Args:
            query: The reasoning query
            workspace: The reasoning workspace
            inputs: Strategy inputs
            current_result: Current reasoning result

        Returns:
            Dict[str, Any]: Strategy results
        """
        question = inputs.get("question", query.question)
        context = inputs.get("context", {})

        # Check if LLM engine is available
        if not self.llm_engine:
            return {
                "outputs": {
                    "message": "LLM engine not available for reasoning",
                    "thoughts": [],
                    "conclusions": []
                },
                "confidence": 0.0,
                "workspace_updates": {},
                "conclusions": []
            }

        try:
            # Prepare context information
            context_text = ""

            # Include workspace data in context
            if workspace:
                for key, value in workspace.items():
                    if key in ["deductive_conclusions", "inductive_patterns",
                             "inductive_generalizations", "abductive_explanations"]:
                        context_text += f"\n{key.replace('_', ' ').title()}:\n"
                        for item in value:
                            if "statement" in item:
                                context_text += f"- {item['statement']}\n"
                            elif "description" in item:
                                context_text += f"- {item['description']}\n"

            # Include memory information if available
            if "working_memory_snapshot" in workspace:
                context_text += "\nRelevant Memory:\n"
                for item in workspace["working_memory_snapshot"]:
                    context_text += f"- {item}\n"

            # Create a reasoning prompt with chain-of-thought
            prompt = f"""
            I need your help with a reasoning task. Please think step by step.

            Question: {question}

            {context_text if context_text else ""}

            Please analyze this question carefully using the following structure:

            1. Break down the key aspects of the question
            2. Consider the available information and relevant knowledge
            3. Explore possible approaches to addressing the question
            4. Work through each step of your reasoning explicitly
            5. Evaluate the strength of your conclusions
            6. Provide your final answer with a confidence level (0-100%)

            Start your response with "Reasoning steps:" and end with "Final conclusion:"
            """

            # Call LLM for reasoning
            llm_response = self.llm_engine.complete({
                "prompt": prompt,
                "max_tokens": 800,
                "temperature": 0.3
            })

            # Parse the response
            if hasattr(llm_response, 'text'):
                response_text = llm_response.text
            else:
                response_text = str(llm_response)

            # Extract reasoning steps
            steps_match = re.search(r'Reasoning steps:(.*?)(?=Final conclusion:|$)', response_text, re.DOTALL)
            steps_text = steps_match.group(1).strip() if steps_match else ""

            # Extract steps as list
            steps = []
            step_items = re.split(r'\d+\.\s+', steps_text)
            for item in step_items:
                if item.strip():
                    steps.append(item.strip())

            # Extract final conclusion
            conclusion_match = re.search(r'Final conclusion:(.*?)$', response_text, re.DOTALL)
            conclusion_text = conclusion_match.group(1).strip() if conclusion_match else ""

            # Extract confidence if present
            confidence = 0.5  # Default
            confidence_match = re.search(r'confidence[:\s]+(\d+)%', response_text, re.IGNORECASE)
            if confidence_match:
                confidence = int(confidence_match.group(1)) / 100

            # Create conclusions
            conclusion_id = str(uuid.uuid4())
            conclusion = {
                "id": conclusion_id,
                "statement": conclusion_text,
                "confidence": confidence,
                "derived_from": {
                    "llm_reasoning": True,
                    "steps": steps
                }
            }

            # Create supporting evidence
            supporting_evidence = {
                conclusion_id: [f"Reasoning step: {step}" for step in steps]
            }

            # Create inference steps
            sub_steps = []
            for i, step in enumerate(steps):
                sub_step = InferenceStep(
                    description=f"Step {i+1}: {step[:100]}{'...' if len(step) > 100 else ''}",
                    outputs={"thought": step},
                    confidence=confidence
                )
                sub_steps.append(sub_step)

            # Overall confidence (same as conclusion confidence)
            result_confidence = confidence

            # Reasoning is complete
            reasoning_complete = True

            return {
                "outputs": {
                    "thoughts": steps,
                    "conclusion": conclusion_text
                },
                "workspace_updates": {
                    "llm_reasoning_steps": steps,
                    "llm_conclusion": conclusion_text
                },
                "confidence": result_confidence,
                "conclusions": [conclusion],
                "reasoning_complete": reasoning_complete,
                "supporting_evidence": supporting_evidence,
                "sub_steps": sub_steps
            }

        except Exception as e:
            self.logger.error(f"Error in LLM-assisted reasoning: {e}", exc_info=True)

            return {
                "outputs": {
                    "error": str(e)
                },
                "confidence": 0.0,
                "conclusions": [],
                "reasoning_complete": False
            }

    #------------------------------------------------------------------
    # Utility methods
    #------------------------------------------------------------------

    @CognitionManager.cognitive_capability(
        capability_type=CognitiveProcessType.REASONING,
        description="Solve a problem using multi-strategy approach",
        priority=40,
        compatible_modes={CognitiveMode.DELIBERATIVE}
    )
    def solve_problem(self,
                     request: CognitiveRequest,
                     context: CognitiveContext,
                     query: Any,
                     **kwargs) -> Dict[str, Any]:
        """
        Higher-level problem-solving capability that combines multiple reasoning strategies.

        Args:
            request: The cognitive request
            context: The cognitive context
            query: The problem to solve
            **kwargs: Additional arguments

        Returns:
            Dict[str, Any]: Problem-solving results
        """
        try:
            self.logger.info(f"Solving problem: {query}")

            # Create a reasoning workspace for this problem
            workspace_id = self.create_reasoning_workspace()

            # Initialize problem-solving state
            problem_state = {
                "problem": query,
                "context": context.parameters,
                "approach": "multi-strategy",
                "steps_taken": [],
                "current_state": "initial",
                "goal_state": "solved"
            }

            self.update_workspace(workspace_id, {"problem_state": problem_state})

            # Problem-solving involves multiple phases:
            # 1. Problem analysis
            # 2. Strategy selection
            # 3. Solution generation
            # 4. Solution evaluation

            # Phase 1: Problem analysis
            analysis_result = self._analyze_problem(query, workspace_id)

            self.update_workspace(workspace_id, {
                "problem_analysis": analysis_result,
                "problem_state": {**problem_state, "current_state": "analyzed"}
            })

            # Phase 2: Strategy selection
            strategies = self._select_problem_solving_strategies(analysis_result, workspace_id)

            self.update_workspace(workspace_id, {
                "selected_strategies": strategies,
                "problem_state": {**problem_state, "current_state": "strategies_selected"}
            })

            # Phase 3: Solution generation using selected strategies
            solutions = []
            for strategy in strategies:
                strategy_result = self._apply_problem_solving_strategy(strategy, workspace_id)
                if strategy_result.get("solution"):
                    solutions.append({
                        "strategy": strategy,
                        "solution": strategy_result["solution"],
                        "confidence": strategy_result.get("confidence", 0.5)
                    })

            self.update_workspace(workspace_id, {
                "generated_solutions": solutions,
                "problem_state": {**problem_state, "current_state": "solutions_generated"}
            })

            # Phase 4: Solution evaluation and selection
            final_solution = self._evaluate_solutions(solutions, workspace_id)

            self.update_workspace(workspace_id, {
                "final_solution": final_solution,
                "problem_state": {**problem_state, "current_state": "solved"}
            })

            # Prepare result
            result = {
                "problem": query,
                "analysis": analysis_result,
                "strategies_used": [s["name"] for s in strategies],
                "solution": final_solution.get("solution"),
                "confidence": final_solution.get("confidence", 0.0),
                "reasoning_trace": final_solution.get("reasoning_trace", [])
            }

            return result

        except Exception as e:
            self.logger.error(f"Error in problem solving: {e}", exc_info=True)

            return {
                "error": str(e),
                "problem": query
            }

    def _analyze_problem(self, problem: Any, workspace_id: str) -> Dict[str, Any]:
        """
        Analyze a problem to determine its characteristics.

        Args:
            problem: The problem to analyze
            workspace_id: ID of the workspace

        Returns:
            Dict[str, Any]: Problem analysis results
        """
        # This is a simplified implementation
        analysis = {
            "problem_type": "unknown",
            "key_elements": [],
            "constraints": [],
            "potential_approaches": []
        }

        # Use LLM for problem analysis if available
        if self.llm_engine:
            try:
                prompt = f"""
                Analyze the following problem:

                Problem: {problem}

                Please provide a structured analysis with the following:

                1. Problem type (e.g., logical, mathematical, decision-making, etc.)
                2. Key elements of the problem
                3. Constraints or requirements
                4. Potential approaches for solving this problem

                Format your response with clear section headers.
                """

                llm_response = self.llm_engine.complete({
                    "prompt": prompt,
                    "max_tokens": 500,
                    "temperature": 0.3
                })

                # Parse the response
                if hasattr(llm_response, 'text'):
                    response_text = llm_response.text
                else:
                    response_text = str(llm_response)

                # Extract problem type
                type_match = re.search(r'Problem type[:\s]+(.*?)(?=\n|$)', response_text)
                if type_match:
                    analysis["problem_type"] = type_match.group(1).strip()

                # Extract key elements
                elements_section = re.search(r'Key elements[:\s]+(.*?)(?=\n\d+\.\s|Constraints|Potential approaches|$)', response_text, re.DOTALL)
                if elements_section:
                    elements_text = elements_section.group(1).strip()
                    analysis["key_elements"] = [
                        item.strip() for item in re.findall(r'(?:^|\n)- (.*?)(?=\n|$)', elements_text)
                    ]

                # Extract constraints
                constraints_section = re.search(r'Constraints[:\s]+(.*?)(?=\n\d+\.\s|Potential approaches|$)', response_text, re.DOTALL)
                if constraints_section:
                    constraints_text = constraints_section.group(1).strip()
                    analysis["constraints"] = [
                        item.strip() for item in re.findall(r'(?:^|\n)- (.*?)(?=\n|$)', constraints_text)
                    ]

                # Extract potential approaches
                approaches_section = re.search(r'Potential approaches[:\s]+(.*?)$', response_text, re.DOTALL)
                if approaches_section:
                    approaches_text = approaches_section.group(1).strip()
                    analysis["potential_approaches"] = [
                        item.strip() for item in re.findall(r'(?:^|\n)- (.*?)(?=\n|$)', approaches_text)
                    ]

            except Exception as e:
                self.logger.warning(f"Error using LLM for problem analysis: {e}")

        # Fallback: Simple keyword-based analysis
        if analysis["problem_type"] == "unknown":
            problem_text = str(problem).lower()

            if any(word in problem_text for word in ["prove", "theorem", "logical", "deduce"]):
                analysis["problem_type"] = "logical"
            elif any(word in problem_text for word in ["calculate", "compute", "math", "equation"]):
                analysis["problem_type"] = "mathematical"
            elif any(word in problem_text for word in ["decide", "choose", "best", "optimal"]):
                analysis["problem_type"] = "decision-making"
            elif any(word in problem_text for word in ["why", "explain", "reason", "cause"]):
                analysis["problem_type"] = "explanatory"
            elif any(word in problem_text for word in ["compare", "contrast", "similar", "different"]):
                analysis["problem_type"] = "comparative"
            else:
                analysis["problem_type"] = "general"

            # Basic key elements extraction (just tokenize and take nouns/phrases)
            words = problem_text.split()
            if len(words) > 2:
                analysis["key_elements"] = [" ".join(words[:3]), " ".join(words[-3:])]

        return analysis

    def _select_problem_solving_strategies(self, analysis: Dict[str, Any], workspace_id: str) -> List[Dict[str, Any]]:
        """
        Select appropriate problem-solving strategies based on problem analysis.

        Args:
            analysis: Problem analysis results
            workspace_id: ID of the workspace

        Returns:
            List[Dict[str, Any]]: Selected strategies
        """
        strategies = []
        problem_type = analysis.get("problem_type", "unknown")

        # Select strategies based on problem type
        if problem_type == "logical":
            strategies.append({
                "name": "deductive_reasoning",
                "reasoning_type": ReasoningType.DEDUCTIVE,
                "description": "Use deductive reasoning to draw logical conclusions",
                "confidence": 0.8
            })
        elif problem_type == "mathematical":
            strategies.append({
                "name": "mathematical_reasoning",
                "reasoning_type": ReasoningType.DEDUCTIVE,
                "description": "Apply mathematical principles and calculations",
                "confidence": 0.8
            })
        elif problem_type == "decision-making":
            strategies.append({
                "name": "decision_analysis",
                "reasoning_type": ReasoningType.CAUSAL,
                "description": "Analyze decision factors and their implications",
                "confidence": 0.7
            })
            strategies.append({
                "name": "counterfactual_reasoning",
                "reasoning_type": ReasoningType.COUNTERFACTUAL,
                "description": "Consider alternative scenarios and outcomes",
                "confidence": 0.6
            })
        elif problem_type == "explanatory":
            strategies.append({
                "name": "abductive_reasoning",
                "reasoning_type": ReasoningType.ABDUCTIVE,
                "description": "Infer the most likely explanation",
                "confidence": 0.7
            })
            strategies.append({
                "name": "causal_analysis",
                "reasoning_type": ReasoningType.CAUSAL,
                "description": "Analyze cause-and-effect relationships",
                "confidence": 0.7
            })
        elif problem_type == "comparative":
            strategies.append({
                "name": "analogical_reasoning",
                "reasoning_type": ReasoningType.ANALOGICAL,
                "description": "Compare and map between domains",
                "confidence": 0.7
            })

        # Fallback: Add general strategies if no specific ones were selected
        if not strategies:
            strategies.append({
                "name": "general_problem_solving",
                "reasoning_type": ReasoningType.HYBRID,
                "description": "Apply general problem-solving methods",
                "confidence": 0.5
            })

        # Always consider adding LLM-assisted reasoning as a complementary strategy
        if self.llm_engine:
            strategies.append({
                "name": "llm_assisted",
                "reasoning_type": ReasoningType.HYBRID,
                "description": "Use LLM to assist with reasoning",
                "confidence": 0.6
            })

        return strategies

    def _apply_problem_solving_strategy(self, strategy: Dict[str, Any], workspace_id: str) -> Dict[str, Any]:
        """
        Apply a problem-solving strategy.

        Args:
            strategy: The strategy to apply
            workspace_id: ID of the workspace

        Returns:
            Dict[str, Any]: Strategy application results
        """
        # Get workspace data
        workspace = self.get_workspace(workspace_id)
        problem = workspace.get("problem_state", {}).get("problem", "")
        problem_analysis = workspace.get("problem_analysis", {})

        # Initialize result
        result = {
            "strategy": strategy["name"],
            "reasoning_trace": [],
            "solution": None,
            "confidence": 0.0
        }

        # Apply strategy based on its type
        if strategy["name"] == "deductive_reasoning":
            # Apply deductive reasoning strategy
            # This is a simplified implementation

            # Extract premises and rules from problem and analysis
            premises = []
            rules = []

            # Simple keyword-based extraction of premises and rules
            if isinstance(problem, str):
                sentences = problem.split(".")
                for sentence in sentences:
                    sentence = sentence.strip().lower()

                    # Look for premises (simple heuristic)
                    if sentence.startswith(("given", "assume", "consider", "suppose")):
                        premises.append(sentence)

                    # Look for rules (simple heuristic)
                    if "if" in sentence and "then" in sentence:
                        parts = sentence.split("if")[1].split("then")
                        if len(parts) == 2:
                            rule = {
                                "if": parts[0].strip(),
                                "then": parts[1].strip(),
                                "confidence": 0.9
                            }
                            rules.append(rule)

            # Add key elements from analysis as potential premises
            for element in problem_analysis.get("key_elements", []):
                if element not in premises:
                    premises.append(element)

            # If we have premises and rules, apply deductive reasoning
            if premises and rules:
                # Call simple deductive reasoning
                deduction_input = {
                    "premises": premises,
                    "rules": rules
                }

                deduction_result = self._simple_deductive_reasoning(
                    query=None,
                    workspace=workspace,
                    inputs=deduction_input,
                    current_result=None
                )

                # Extract conclusions
                conclusions = deduction_result.get("conclusions", [])

                if conclusions:
                    # Use the highest confidence conclusion as the solution
                    best_conclusion = max(conclusions, key=lambda x: x.get("confidence", 0))

                    result["solution"] = best_conclusion["statement"]
                    result["confidence"] = best_conclusion["confidence"]

                    # Add reasoning trace
                    for step in deduction_result.get("sub_steps", []):
                        result["reasoning_trace"].append(step.description)
                else:
                    result["reasoning_trace"].append("No conclusions derived from deductive reasoning")
            else:
                result["reasoning_trace"].append("Insufficient premises or rules for deductive reasoning")

        elif strategy["name"] == "abductive_reasoning":
            # Apply abductive reasoning strategy
            # This is a simplified implementation

            # Extract observations from problem and analysis
            observations = []

            # Extract observations from problem
            if isinstance(problem, str):
                sentences = problem.split(".")
                for sentence in sentences:
                    sentence = sentence.strip()
                    if sentence and not sentence.lower().startswith(("why", "how", "what")):
                        observations.append(sentence)

            # Add key elements as observations
            for element in problem_analysis.get("key_elements", []):
                if element not in observations:
                    observations.append(element)

            # If we have observations, apply abductive reasoning
            if observations:
                # Call simple abductive reasoning
                abduction_input = {
                    "observations": observations
                }

                abduction_result = self._simple_abductive_reasoning(
                    query=None,
                    workspace=workspace,
                    inputs=abduction_input,
                    current_result=None
                )

                # Extract explanations
                explanations = abduction_result.get("outputs", {}).get("explanations", [])

                if explanations:
                    # Use the highest confidence explanation as the solution
                    best_explanation = max(explanations, key=lambda x: x.get("confidence", 0))

                    result["solution"] = best_explanation["statement"]
                    result["confidence"] = best_explanation["confidence"]

                    # Add reasoning trace
                    for step in abduction_result.get("sub_steps", []):
                        result["reasoning_trace"].append(step.description)
                else:
                    result["reasoning_trace"].append("No explanations derived from abductive reasoning")
            else:
                result["reasoning_trace"].append("Insufficient observations for abductive reasoning")

        elif strategy["name"] == "llm_assisted":
            # Apply LLM-assisted reasoning strategy
            if self.llm_engine:
                llm_input = {
                    "question": problem,
                    "context": problem_analysis
                }

                llm_result = self._llm_assisted_reasoning(
                    query=None,
                    workspace=workspace,
                    inputs=llm_input,
                    current_result=None
                )

                # Extract conclusion
                conclusion = llm_result.get("outputs", {}).get("conclusion")

                if conclusion:
                    result["solution"] = conclusion
                    result["confidence"] = llm_result.get("confidence", 0.6)

                    # Add reasoning trace
                    thoughts = llm_result.get("outputs", {}).get("thoughts", [])
                    for thought in thoughts:
                        result["reasoning_trace"].append(thought)
                else:
                    result["reasoning_trace"].append("No conclusion from LLM-assisted reasoning")
            else:
                result["reasoning_trace"].append("LLM engine not available")

        elif strategy["name"] == "general_problem_solving":
            # Apply general problem-solving strategy
            # This is a placeholder implementation

            # Simply combine all available information
            description = f"General problem-solving for: {problem}"
            approaches = problem_analysis.get("potential_approaches", [])

            if approaches:
                # Choose the first suggested approach
                approach = approaches[0]
                result["solution"] = f"Using the approach: {approach}"
                result["confidence"] = 0.4
                result["reasoning_trace"].append(f"Selected approach: {approach}")
            else:
                result["reasoning_trace"].append("No specific approaches identified")

        # Return the result
        return result

    def _evaluate_solutions(self, solutions: List[Dict[str, Any]], workspace_id: str) -> Dict[str, Any]:
        """
        Evaluate and select the best solution from multiple candidates.

        Args:
            solutions: List of candidate solutions
            workspace_id: ID of the workspace

        Returns:
            Dict[str, Any]: Selected best solution
        """
        if not solutions:
            return {
                "solution": None,
                "confidence": 0.0,
                "reasoning_trace": ["No solutions to evaluate"]
            }

        # Get workspace data
        workspace = self.get_workspace(workspace_id)
        problem = workspace.get("problem_state", {}).get("problem", "")

        # Initialize evaluation result
        evaluation = {
            "evaluated_solutions": [],
            "reasoning_trace": [],
            "selected_solution": None,
            "confidence": 0.0
        }

        # Evaluate each solution
        for solution in solutions:
            solution_eval = {
                "strategy": solution["strategy"],
                "solution": solution["solution"],
                "base_confidence": solution["confidence"],
                "adjusted_confidence": solution["confidence"],  # Will be adjusted
                "strengths": [],
                "weaknesses": []
            }

            # Simple evaluation heuristics
            if solution["confidence"] > 0.7:
                solution_eval["strengths"].append("High confidence from strategy")
            elif solution["confidence"] < 0.5:
                solution_eval["weaknesses"].append("Low confidence from strategy")

            if len(solution.get("reasoning_trace", [])) > 3:
                solution_eval["strengths"].append("Detailed reasoning trace")
            else:
                solution_eval["weaknesses"].append("Limited reasoning steps")

            # Use LLM to evaluate solution if available
            if self.llm_engine:
                try:
                    prompt = f"""
                    Evaluate this solution to the given problem:

                    Problem: {problem}

                    Proposed solution: {solution["solution"]}

                    Strategy used: {solution["strategy"]}

                    Please evaluate this solution on:
                    1. Correctness (is it accurate and valid?)
                    2. Completeness (does it fully address the problem?)
                    3. Clarity (is it clearly explained?)

                    For each criterion, provide a rating (0-10) and brief explanation.
                    End with an overall assessment and confidence score (0-100%).
                    """

                    llm_response = self.llm_engine.complete({
                        "prompt": prompt,
                        "max_tokens": 300,
                        "temperature": 0.3
                    })

                    # Parse the response
                    if hasattr(llm_response, 'text'):
                        response_text = llm_response.text
                    else:
                        response_text = str(llm_response)

                    # Extract ratings
                    correctness_match = re.search(r'Correctness.*?(\d+)/10', response_text)
                    completeness_match = re.search(r'Completeness.*?(\d+)/10', response_text)
                    clarity_match = re.search(r'Clarity.*?(\d+)/10', response_text)

                    correctness = int(correctness_match.group(1)) / 10 if correctness_match else 0.5
                    completeness = int(completeness_match.group(1)) / 10 if completeness_match else 0.5
                    clarity = int(clarity_match.group(1)) / 10 if clarity_match else 0.5

                    # Extract overall confidence
                    confidence_match = re.search(r'confidence.*?(\d+)%', response_text, re.IGNORECASE)
                    overall_confidence = int(confidence_match.group(1)) / 100 if confidence_match else 0.5

                    # Adjust solution confidence based on evaluation
                    solution_eval["adjusted_confidence"] = (solution["confidence"] + overall_confidence) / 2

                    # Extract strengths and weaknesses
                    if correctness > 0.7:
                        solution_eval["strengths"].append(f"High correctness ({correctness:.1f}/1.0)")
                    elif correctness < 0.5:
                        solution_eval["weaknesses"].append(f"Low correctness ({correctness:.1f}/1.0)")

                    if completeness > 0.7:
                        solution_eval["strengths"].append(f"High completeness ({completeness:.1f}/1.0)")
                    elif completeness < 0.5:
                        solution_eval["weaknesses"].append(f"Low completeness ({completeness:.1f}/1.0)")

                    if clarity > 0.7:
                        solution_eval["strengths"].append(f"High clarity ({clarity:.1f}/1.0)")
                    elif clarity < 0.5:
                        solution_eval["weaknesses"].append(f"Low clarity ({clarity:.1f}/1.0)")

                except Exception as e:
                    self.logger.warning(f"Error using LLM for solution evaluation: {e}")
                    # Fall back to the strategy's confidence
                    solution_eval["adjusted_confidence"] = solution["confidence"]

            evaluation["evaluated_solutions"].append(solution_eval)

        # Select the best solution based on adjusted confidence
        if evaluation["evaluated_solutions"]:
            best_solution = max(evaluation["evaluated_solutions"], key=lambda x: x["adjusted_confidence"])

            evaluation["selected_solution"] = best_solution["solution"]
            evaluation["confidence"] = best_solution["adjusted_confidence"]

            # Add reasoning trace
            evaluation["reasoning_trace"].append(f"Selected solution from strategy: {best_solution['strategy']}")
            evaluation["reasoning_trace"].append(f"Confidence: {best_solution['adjusted_confidence']:.2f}")
            evaluation["reasoning_trace"].append("Strengths: " + ", ".join(best_solution["strengths"]))
            if best_solution["weaknesses"]:
                evaluation["reasoning_trace"].append("Weaknesses: " + ", ".join(best_solution["weaknesses"]))

            # Add the original reasoning trace from the selected solution
            for solution in solutions:
                if solution["strategy"] == best_solution["strategy"]:
                    evaluation["reasoning_trace"].extend(solution.get("reasoning_trace", []))
                    break

        return evaluation


# Standalone execution for testing
if __name__ == "__main__":
    # Set up logging
    logging.basicConfig(level=logging.INFO)

    # Create a ReasoningCore instance
    reasoning = ReasoningCore()

    # Create a context and request for testing
    context = CognitiveContext(
        parameters={}
    )

    # Test with a simple query
    test_query = "If all birds can fly and a penguin is a bird, can a penguin fly?"

    request = CognitiveRequest(query=test_query, context=context)
    result = reasoning.reason(request, context, test_query)

    print(f"\nReasoning result for: {test_query}")
    if "error" in result:
        print(f"Error: {result['error']}")
    else:
        print(f"Conclusions:")
        for conclusion in result.get("conclusions", []):
            print(f"- {conclusion['statement']} (confidence: {conclusion['confidence']:.2f})")

        print("\nInference steps:")
        for step in result.get("inference_steps", []):
            print(f"- {step['description']}")

    print("-" * 40)