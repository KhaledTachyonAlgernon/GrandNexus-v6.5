"""
confidence.py - Confidence Evaluation and Uncertainty Management for GrandNexus

This module provides a comprehensive framework for assessing the confidence and
uncertainty in GrandNexus's cognitive processes. It enables the system to be aware
of its own limitations, the reliability of its knowledge, and the certainty of its
inferences, providing a foundation for metacognitive awareness.

Key features:
- Multi-level confidence assessment for different cognitive processes
- Uncertainty propagation through inference chains
- Evidence-based confidence scoring
- Calibration of confidence metrics against observed outcomes
- Integration with memory systems to track confidence over time
- Decision-making support under uncertainty

The design emphasizes:
- Calibrated confidence scores that align with actual performance
- Integration with all cognitive modules (reasoning, learning, perception)
- Principled uncertainty management based on probability theory
- Transparency about confidence levels in system outputs
"""

import logging
import threading
import time
import uuid
import math
from typing import Dict, List, Any, Optional, Set, Tuple, Union, Callable
from enum import Enum, auto
from dataclasses import dataclass, field
from collections import defaultdict, deque
import statistics
import numpy as np

# Assuming these exist in GrandNexus architecture
from grandnexus.core.nexus_core import NexusCore
from grandnexus.memory.working import WorkingMemory
from grandnexus.memory.semantic import SemanticMemory
from grandnexus.memory.episodic import EpisodicMemory

# Confidence source types
class ConfidenceSource(Enum):
    """Sources of confidence information"""
    PERCEPTION = auto()      # Confidence from perceptual processes
    REASONING = auto()       # Confidence from reasoning processes
    MEMORY = auto()          # Confidence from memory retrieval
    LEARNING = auto()        # Confidence from learning processes
    PRIOR = auto()           # Prior/background confidence
    COMBINED = auto()        # Combined from multiple sources
    EXTERNAL = auto()        # Provided externally (e.g., user feedback)

@dataclass
class ConfidenceScore:
    """Represents a confidence assessment for a specific item/process"""
    score_id: str
    source: ConfidenceSource
    value: float  # 0.0 to 1.0
    timestamp: float
    item_id: Optional[str] = None  # ID of the item being assessed
    module_name: str = ""
    context: Dict[str, Any] = field(default_factory=dict)
    evidence: List[Dict[str, Any]] = field(default_factory=list)
    uncertainty_type: Optional[str] = None  # e.g., "aleatoric", "epistemic"

@dataclass
class ConfidenceThresholds:
    """Thresholds for different confidence levels"""
    very_high: float = 0.95
    high: float = 0.8
    medium: float = 0.6
    low: float = 0.4
    very_low: float = 0.2

@dataclass
class ConfidenceStats:
    """Statistics about a set of confidence assessments"""
    count: int = 0
    mean: float = 0.0
    median: float = 0.0
    std_dev: float = 0.0
    min: float = 0.0
    max: float = 0.0
    calibration_error: float = 0.0  # Difference between confidence and actual accuracy

class ConfidenceEvaluator:
    """
    Central component for evaluating confidence across GrandNexus's cognitive processes.

    This module provides mechanisms to assess, track, and manage uncertainty in the
    system's knowledge and reasoning, enabling metacognitive awareness and appropriate
    action selection based on confidence levels.
    """

    def __init__(self, nexus_core: Optional[NexusCore] = None,
                 working_memory: Optional[WorkingMemory] = None,
                 semantic_memory: Optional[SemanticMemory] = None,
                 episodic_memory: Optional[EpisodicMemory] = None,
                 config: Optional[Dict[str, Any]] = None):
        """
        Initialize the confidence evaluator.

        Args:
            nexus_core: Reference to the NexusCore for system integration
            working_memory: Reference to working memory module
            semantic_memory: Reference to semantic memory module
            episodic_memory: Reference to episodic memory module
            config: Optional configuration dictionary
        """
        self.logger = logging.getLogger("GrandNexus.Metacognition.Confidence")
        self.nexus_core = nexus_core
        self.working_memory = working_memory
        self.semantic_memory = semantic_memory
        self.episodic_memory = episodic_memory

        # Initialize configuration
        self.config = config or {}
        self.confidence_thresholds = ConfidenceThresholds(
            very_high=self.config.get("threshold_very_high", 0.95),
            high=self.config.get("threshold_high", 0.8),
            medium=self.config.get("threshold_medium", 0.6),
            low=self.config.get("threshold_low", 0.4),
            very_low=self.config.get("threshold_very_low", 0.2)
        )

        # Internal state
        self.instance_id = str(uuid.uuid4())[:8]
        self.initialized = False
        self.lock = threading.RLock()

        # Confidence history
        self.confidence_history: Dict[str, deque[ConfidenceScore]] = defaultdict(
            lambda: deque(maxlen=self.config.get("history_length", 1000))
        )

        # Aggregated confidence by module/source
        self.aggregated_confidence: Dict[str, Dict[str, Any]] = {}

        # Calibration data (predicted confidence vs. actual correctness)
        self.calibration_data: Dict[str, List[Tuple[float, bool]]] = defaultdict(list)

        # Confidence models for different modules/sources
        self.confidence_models: Dict[str, Any] = {}

        # Custom confidence evaluators (module_name -> list of evaluator functions)
        self.custom_evaluators: Dict[str, List[Callable[..., float]]] = defaultdict(list)

        self.logger.info(f"ConfidenceEvaluator initialized with ID={self.instance_id}")

    def initialize(self) -> bool:
        """Initialize the confidence evaluator"""
        if self.initialized:
            return True

        with self.lock:
            try:
                # Connect to other modules
                if self.nexus_core:
                    self._register_with_nexus_core()

                # Initialize baseline confidence models
                self._initialize_confidence_models()

                self.initialized = True
                self.logger.info("ConfidenceEvaluator initialization complete")
                return True
            except Exception as e:
                self.logger.error(f"ConfidenceEvaluator initialization failed: {str(e)}")
                return False

    def record_confidence(self, source: ConfidenceSource, value: float,
                         module_name: str, item_id: Optional[str] = None,
                         context: Optional[Dict[str, Any]] = None,
                         evidence: Optional[List[Dict[str, Any]]] = None,
                         uncertainty_type: Optional[str] = None) -> str:
        """
        Record a confidence assessment.

        Args:
            source: Source of the confidence assessment
            value: Confidence value (0.0 to 1.0)
            module_name: Name of the module making the assessment
            item_id: Optional ID of the item being assessed
            context: Optional contextual information
            evidence: Optional evidence supporting the assessment
            uncertainty_type: Optional type of uncertainty

        Returns:
            ID of the recorded confidence score
        """
        if not self.initialized:
            self.initialize()

        # Validate confidence value
        value = max(0.0, min(1.0, value))  # Ensure between 0 and 1

        # Create confidence score object
        score_id = str(uuid.uuid4())
        score = ConfidenceScore(
            score_id=score_id,
            source=source,
            value=value,
            timestamp=time.time(),
            item_id=item_id,
            module_name=module_name,
            context=context or {},
            evidence=evidence or [],
            uncertainty_type=uncertainty_type
        )

        # Store the score
        with self.lock:
            # Add to history
            self.confidence_history[module_name].append(score)

            # Update aggregated confidence
            self._update_aggregated_confidence(score)

        return score_id

    def get_confidence(self, item_id: str) -> Optional[float]:
        """
        Get the most recent confidence score for an item.

        Args:
            item_id: ID of the item

        Returns:
            Confidence value (0.0 to 1.0), or None if not found
        """
        if not self.initialized:
            self.initialize()

        with self.lock:
            # Search for the most recent confidence score for this item
            most_recent = None
            most_recent_time = 0.0

            for module_history in self.confidence_history.values():
                for score in module_history:
                    if score.item_id == item_id and score.timestamp > most_recent_time:
                        most_recent = score
                        most_recent_time = score.timestamp

            return most_recent.value if most_recent else None

    def combine_confidence_scores(self, scores: List[ConfidenceScore],
                                 method: str = "weighted_average") -> float:
        """
        Combine multiple confidence scores into a single assessment.

        Args:
            scores: List of confidence scores to combine
            method: Method for combining scores ("weighted_average", "min", "max", etc.)

        Returns:
            Combined confidence value
        """
        if not scores:
            return 0.0

        if method == "weighted_average":
            # Calculate weighted average based on recency and source reliability
            weights = []
            values = []

            for score in scores:
                # Base weight
                weight = 1.0

                # Adjust weight based on recency (newer scores have higher weight)
                age = time.time() - score.timestamp
                recency_factor = max(0.5, math.exp(-age / 3600.0))  # Decay over hour
                weight *= recency_factor

                # Adjust weight based on source reliability
                source_reliability = self._get_source_reliability(score.source, score.module_name)
                weight *= source_reliability

                weights.append(weight)
                values.append(score.value)

            # Compute weighted average
            if sum(weights) > 0:
                return sum(w * v for w, v in zip(weights, values)) / sum(weights)
            else:
                return statistics.mean(values)

        elif method == "min":
            # Return the minimum confidence (most conservative)
            return min(score.value for score in scores)

        elif method == "max":
            # Return the maximum confidence (most optimistic)
            return max(score.value for score in scores)

        elif method == "geometric_mean":
            # Geometric mean (good for probabilities)
            return math.pow(math.prod(score.value for score in scores), 1.0 / len(scores))

        else:
            # Default to simple average
            return statistics.mean(score.value for score in scores)

    def evaluate_perception_confidence(self, perception_result: Any,
                                     context: Optional[Dict[str, Any]] = None) -> float:
        """
        Evaluate confidence in a perception result.

        Args:
            perception_result: Result from a perception process
            context: Optional contextual information

        Returns:
            Confidence value (0.0 to 1.0)
        """
        if not self.initialized:
            self.initialize()

        # Initialize context if needed
        if context is None:
            context = {}

        # Check if perception_result has a confidence attribute
        if hasattr(perception_result, "confidence"):
            return float(perception_result.confidence)

        # If it's a dictionary with confidence key
        if isinstance(perception_result, dict) and "confidence" in perception_result:
            return float(perception_result["confidence"])

        # Apply custom evaluators for perception
        for evaluator in self.custom_evaluators.get("perception", []):
            try:
                confidence = evaluator(perception_result, context)
                if confidence is not None:
                    return confidence
            except Exception as e:
                self.logger.warning(f"Error in custom perception evaluator: {str(e)}")

        # Default confidence model
        return self._default_perception_confidence(perception_result, context)

    def evaluate_reasoning_confidence(self, reasoning_result: Any,
                                    context: Optional[Dict[str, Any]] = None) -> float:
        """
        Evaluate confidence in a reasoning result.

        Args:
            reasoning_result: Result from a reasoning process
            context: Optional contextual information

        Returns:
            Confidence value (0.0 to 1.0)
        """
        if not self.initialized:
            self.initialize()

        # Initialize context if needed
        if context is None:
            context = {}

        # Check if reasoning_result has a confidence attribute
        if hasattr(reasoning_result, "confidence"):
            return float(reasoning_result.confidence)

        # If it's a dictionary with confidence key
        if isinstance(reasoning_result, dict) and "confidence" in reasoning_result:
            return float(reasoning_result["confidence"])

        # Apply custom evaluators for reasoning
        for evaluator in self.custom_evaluators.get("reasoning", []):
            try:
                confidence = evaluator(reasoning_result, context)
                if confidence is not None:
                    return confidence
            except Exception as e:
                self.logger.warning(f"Error in custom reasoning evaluator: {str(e)}")

        # Default confidence model
        return self._default_reasoning_confidence(reasoning_result, context)

    def evaluate_memory_confidence(self, memory_item: Any,
                                 context: Optional[Dict[str, Any]] = None) -> float:
        """
        Evaluate confidence in a memory retrieval.

        Args:
            memory_item: Retrieved memory item
            context: Optional contextual information

        Returns:
            Confidence value (0.0 to 1.0)
        """
        if not self.initialized:
            self.initialize()

        # Initialize context if needed
        if context is None:
            context = {}

        # Check if memory_item has a confidence or importance attribute
        if hasattr(memory_item, "confidence"):
            return float(memory_item.confidence)

        if hasattr(memory_item, "importance"):
            # Scale importance to confidence-like value
            return min(1.0, float(memory_item.importance) * 1.25)

        # If it's a dictionary with confidence key
        if isinstance(memory_item, dict):
            if "confidence" in memory_item:
                return float(memory_item["confidence"])
            if "importance" in memory_item:
                return min(1.0, float(memory_item["importance"]) * 1.25)

        # Apply custom evaluators for memory
        for evaluator in self.custom_evaluators.get("memory", []):
            try:
                confidence = evaluator(memory_item, context)
                if confidence is not None:
                    return confidence
            except Exception as e:
                self.logger.warning(f"Error in custom memory evaluator: {str(e)}")

        # Default confidence model
        return self._default_memory_confidence(memory_item, context)

    def propagate_confidence(self, input_confidences: List[float],
                           operation: str = "chain_rule") -> float:
        """
        Propagate confidence through a multi-step process.

        Args:
            input_confidences: List of confidence values for inputs
            operation: Type of confidence propagation operation

        Returns:
            Propagated confidence value
        """
        if not input_confidences:
            return 0.0

        if operation == "chain_rule":
            # Probabilistic chain rule (product of probabilities)
            return math.prod(input_confidences)

        elif operation == "noisy_or":
            # Noisy-OR model (1 - Product(1 - p))
            return 1.0 - math.prod(1.0 - p for p in input_confidences)

        elif operation == "min":
            # Weakest link model
            return min(input_confidences)

        elif operation == "max":
            # Strongest evidence model
            return max(input_confidences)

        elif operation == "average":
            # Simple average
            return statistics.mean(input_confidences)

        else:
            # Default to chain rule
            return math.prod(input_confidences)

    def register_calibration_data(self, module_name: str, predicted_confidence: float,
                                was_correct: bool) -> None:
        """
        Register actual outcomes to calibrate confidence assessments.

        Args:
            module_name: Name of the module that made the prediction
            predicted_confidence: The confidence that was assigned
            was_correct: Whether the prediction was actually correct
        """
        with self.lock:
            self.calibration_data[module_name].append((predicted_confidence, was_correct))

            # Update calibration statistics if we have enough data
            if len(self.calibration_data[module_name]) >= 10:
                self._update_calibration_stats(module_name)

    def get_calibration_stats(self, module_name: Optional[str] = None) -> Dict[str, Any]:
        """
        Get calibration statistics for confidence assessments.

        Args:
            module_name: Optional module name to get stats for (if None, get overall stats)

        Returns:
            Dictionary with calibration statistics
        """
        with self.lock:
            if module_name:
                # Return stats for specific module
                if module_name in self.aggregated_confidence:
                    return self.aggregated_confidence[module_name].get("calibration", {})
                return {}

            # Return overall stats
            overall_stats = {}

            # Combine calibration data from all modules
            all_data = []
            for module_data in self.calibration_data.values():
                all_data.extend(module_data)

            if all_data:
                # Calculate calibration metrics
                overall_stats = self._calculate_calibration_metrics(all_data)

            return overall_stats

    def register_custom_evaluator(self, module_name: str,
                                evaluator_func: Callable[..., float]) -> bool:
        """
        Register a custom confidence evaluator function.

        Args:
            module_name: Name of the module (e.g., "perception", "reasoning")
            evaluator_func: Function that takes (result, context) and returns confidence

        Returns:
            True if registration succeeded, False otherwise
        """
        with self.lock:
            self.custom_evaluators[module_name].append(evaluator_func)
            self.logger.info(f"Registered custom evaluator for module: {module_name}")
            return True

    def get_confidence_level(self, confidence_value: float) -> str:
        """
        Map a confidence value to a descriptive level.

        Args:
            confidence_value: Confidence value (0.0 to 1.0)

        Returns:
            String describing the confidence level
        """
        if confidence_value >= self.confidence_thresholds.very_high:
            return "very_high"
        elif confidence_value >= self.confidence_thresholds.high:
            return "high"
        elif confidence_value >= self.confidence_thresholds.medium:
            return "medium"
        elif confidence_value >= self.confidence_thresholds.low:
            return "low"
        else:
            return "very_low"

    def get_uncertainty_interval(self, confidence_value: float,
                               distribution: str = "gaussian") -> Tuple[float, float]:
        """
        Calculate an uncertainty interval around an estimate.

        Args:
            confidence_value: Confidence value (0.0 to 1.0)
            distribution: Probability distribution model

        Returns:
            Tuple of (lower_bound, upper_bound) representing uncertainty interval
        """
        # Convert confidence to standard deviation
        if distribution == "gaussian":
            # Convert confidence to standard deviation
            # Confidence of 1.0 -> std_dev of 0.01
            # Confidence of 0.5 -> std_dev of 0.25
            # Confidence of 0.0 -> std_dev of 0.5
            std_dev = 0.5 - (confidence_value * 0.49)

            # 95% confidence interval is approximately ±2 standard deviations
            interval = std_dev * 2.0

            # Limit the interval to [0, 1]
            lower_bound = max(0.0, 0.5 - interval)
            upper_bound = min(1.0, 0.5 + interval)

            return (lower_bound, upper_bound)

        elif distribution == "beta":
            # Use a Beta distribution shaped by confidence
            # High confidence -> concentrated distribution
            # Low confidence -> spread out distribution
            if confidence_value > 0.5:
                alpha = 2.0 + (confidence_value - 0.5) * 18.0  # 2 to 20
                beta = 2.0
            else:
                alpha = 2.0
                beta = 2.0 + (0.5 - confidence_value) * 18.0  # 2 to 20

            # Calculate 95% confidence interval
            # For a true implementation, use scipy.stats.beta.ppf
            # This is a rough approximation
            width = 2.0 / (alpha + beta)
            lower_bound = max(0.0, 0.5 - width)
            upper_bound = min(1.0, 0.5 + width)

            return (lower_bound, upper_bound)

        else:
            # Simple heuristic for unknown distributions
            interval = 1.0 - confidence_value
            lower_bound = max(0.0, 0.5 - interval / 2.0)
            upper_bound = min(1.0, 0.5 + interval / 2.0)

            return (lower_bound, upper_bound)

    def should_defer_decision(self, confidence_value: float,
                            importance: float = 0.5,
                            threshold: Optional[float] = None) -> bool:
        """
        Determine if a decision should be deferred due to low confidence.

        Args:
            confidence_value: Confidence value (0.0 to 1.0)
            importance: Importance of the decision (0.0 to 1.0)
            threshold: Optional explicit threshold (if None, calculated based on importance)

        Returns:
            True if the decision should be deferred, False otherwise
        """
        # If threshold is provided, use it directly
        if threshold is not None:
            return confidence_value < threshold

        # Calculate threshold based on importance
        # Higher importance -> higher confidence required
        adaptive_threshold = 0.5 + (importance * 0.3)

        return confidence_value < adaptive_threshold

    # -----------------------------------------------------------------------
    # Internal methods
    # -----------------------------------------------------------------------

    def _register_with_nexus_core(self) -> None:
        """Register with NexusCore for system-wide coordination"""
        if not self.nexus_core:
            self.logger.warning("No NexusCore available for registration")
            return

        try:
            # Register as a module with NexusCore
            dependencies = []
            if self.working_memory:
                dependencies.append("working_memory")
            if self.semantic_memory:
                dependencies.append("semantic_memory")
            if self.episodic_memory:
                dependencies.append("episodic_memory")

            self.nexus_core.register_module(
                name="confidence_evaluator",
                module=self,
                dependencies=dependencies
            )

            self.logger.info("Successfully registered with NexusCore")
        except Exception as e:
            self.logger.error(f"Failed to register with NexusCore: {str(e)}")

    def _initialize_confidence_models(self) -> None:
        """Initialize baseline confidence models for different sources"""
        # This would typically load or configure confidence estimation models
        # For now, we use simple heuristic models
        self.confidence_models["perception"] = {"type": "heuristic"}
        self.confidence_models["reasoning"] = {"type": "heuristic"}
        self.confidence_models["memory"] = {"type": "heuristic"}
        self.confidence_models["combined"] = {"type": "weighted_average"}

    def _update_aggregated_confidence(self, score: ConfidenceScore) -> None:
        """
        Update aggregated confidence statistics for a module/source.

        Args:
            score: The confidence score to incorporate
        """
        module_name = score.module_name
        source_name = score.source.name

        # Initialize if not exists
        if module_name not in self.aggregated_confidence:
            self.aggregated_confidence[module_name] = {
                "sources": {},
                "overall": {
                    "count": 0,
                    "sum": 0.0,
                    "sum_sq": 0.0,
                    "min": 1.0,
                    "max": 0.0,
                    "last_update": 0.0
                },
                "calibration": {}
            }

        # Initialize source if not exists
        if source_name not in self.aggregated_confidence[module_name]["sources"]:
            self.aggregated_confidence[module_name]["sources"][source_name] = {
                "count": 0,
                "sum": 0.0,
                "sum_sq": 0.0,
                "min": 1.0,
                "max": 0.0,
                "last_update": 0.0
            }

        # Update source statistics
        source_stats = self.aggregated_confidence[module_name]["sources"][source_name]
        source_stats["count"] += 1
        source_stats["sum"] += score.value
        source_stats["sum_sq"] += score.value ** 2
        source_stats["min"] = min(source_stats["min"], score.value)
        source_stats["max"] = max(source_stats["max"], score.value)
        source_stats["last_update"] = score.timestamp

        # Update overall statistics
        overall_stats = self.aggregated_confidence[module_name]["overall"]
        overall_stats["count"] += 1
        overall_stats["sum"] += score.value
        overall_stats["sum_sq"] += score.value ** 2
        overall_stats["min"] = min(overall_stats["min"], score.value)
        overall_stats["max"] = max(overall_stats["max"], score.value)
        overall_stats["last_update"] = score.timestamp

        # Calculate derived statistics
        self._calculate_derived_statistics(source_stats)
        self._calculate_derived_statistics(overall_stats)

    def _calculate_derived_statistics(self, stats: Dict[str, Any]) -> None:
        """
        Calculate derived statistics from raw counts/sums.

        Args:
            stats: Dictionary of statistics to update
        """
        if stats["count"] > 0:
            stats["mean"] = stats["sum"] / stats["count"]

            # Variance
            if stats["count"] > 1:
                variance = (stats["sum_sq"] / stats["count"]) - (stats["mean"] ** 2)
                stats["std_dev"] = math.sqrt(max(0.0, variance))
            else:
                stats["std_dev"] = 0.0

    def _update_calibration_stats(self, module_name: str) -> None:
        """
        Update calibration statistics for a module.

        Args:
            module_name: Name of the module to update
        """
        if module_name not in self.calibration_data:
            return

        # Get calibration data
        data = self.calibration_data[module_name]
        if not data:
            return

        # Calculate calibration metrics
        calibration_metrics = self._calculate_calibration_metrics(data)

        # Update module's calibration stats
        if module_name in self.aggregated_confidence:
            self.aggregated_confidence[module_name]["calibration"] = calibration_metrics

    def _calculate_calibration_metrics(self, data: List[Tuple[float, bool]]) -> Dict[str, Any]:
        """
        Calculate calibration metrics from prediction-outcome pairs.

        Args:
            data: List of (predicted_confidence, was_correct) pairs

        Returns:
            Dictionary with calibration metrics
        """
        if not data:
            return {}

        # Basic metrics
        total = len(data)
        correct = sum(1 for _, correct in data if correct)
        accuracy = correct / total if total > 0 else 0.0

        # Group by confidence bins
        bins = {}
        bin_size = 0.1  # 10 bins (0.0-0.1, 0.1-0.2, etc.)

        for confidence, correct in data:
            bin_index = min(int(confidence / bin_size), 9)  # Ensure bin_index in [0, 9]
            if bin_index not in bins:
                bins[bin_index] = {"count": 0, "correct": 0}
            bins[bin_index]["count"] += 1
            if correct:
                bins[bin_index]["correct"] += 1

        # Calculate calibration error
        calibration_error = 0.0
        bin_stats = []

        for bin_index, bin_data in bins.items():
            if bin_data["count"] > 0:
                bin_confidence = (bin_index + 0.5) * bin_size  # Midpoint of bin
                bin_accuracy = bin_data["correct"] / bin_data["count"]
                bin_error = abs(bin_confidence - bin_accuracy)
                calibration_weight = bin_data["count"] / total
                calibration_error += bin_error * calibration_weight

                bin_stats.append({
                    "bin": bin_index,
                    "confidence_range": f"{bin_index * bin_size:.1f}-{(bin_index + 1) * bin_size:.1f}",
                    "count": bin_data["count"],
                    "accuracy": bin_accuracy,
                    "error": bin_error
                })

        # Calculate Brier score (mean squared error)
        brier_score = sum((confidence - (1.0 if correct else 0.0)) ** 2
                         for confidence, correct in data) / total

        return {
            "total_predictions": total,
            "accuracy": accuracy,
            "calibration_error": calibration_error,
            "brier_score": brier_score,
            "bins": bin_stats
        }

    def _get_source_reliability(self, source: ConfidenceSource, module_name: str) -> float:
        """
        Get the reliability factor for a confidence source.

        Args:
            source: The confidence source
            module_name: Name of the module

        Returns:
            Reliability factor (0.0 to 1.0)
        """
        # Check if we have calibration data for this source/module
        source_name = source.name

        if (module_name in self.aggregated_confidence and
            "sources" in self.aggregated_confidence[module_name] and
            source_name in self.aggregated_confidence[module_name]["sources"]):

            # If we have calibration data, use it to determine reliability
            if "calibration" in self.aggregated_confidence[module_name]:
                calibration = self.aggregated_confidence[module_name]["calibration"]
                if "calibration_error" in calibration:
                    # Convert calibration error to reliability
                    # Lower error = higher reliability
                    error = calibration["calibration_error"]
                    reliability = max(0.5, 1.0 - error)  # Minimum reliability of 0.5
                    return reliability

        # Default reliability factors based on source type
        default_reliability = {
            ConfidenceSource.PERCEPTION: 0.8,
            ConfidenceSource.REASONING: 0.85,
            ConfidenceSource.MEMORY: 0.9,
            ConfidenceSource.LEARNING: 0.75,
            ConfidenceSource.PRIOR: 0.7,
            ConfidenceSource.COMBINED: 0.85,
            ConfidenceSource.EXTERNAL: 0.95  # High trust in external sources
        }

        return default_reliability.get(source, 0.8)

    def _default_perception_confidence(self, perception_result: Any,
                                     context: Dict[str, Any]) -> float:
        """
        Default heuristic for estimating perception confidence.

        Args:
            perception_result: Result from a perception process
            context: Contextual information

        Returns:
            Confidence value (0.0 to 1.0)
        """
        # Default moderate confidence
        base_confidence = 0.7

        # Simple heuristics based on perception_result type
        if hasattr(perception_result, "score"):
            # If it has a score attribute, use that
            return min(1.0, float(perception_result.score))

        # Check for similarity score in context
        if "similarity_score" in context:
            sim_score = context["similarity_score"]
            if isinstance(sim_score, (int, float)):
                return min(1.0, float(sim_score))

        # Check for ambiguity in context
        if "ambiguity" in context:
            ambiguity = context["ambiguity"]
            if isinstance(ambiguity, (int, float)):
                return max(0.0, 1.0 - float(ambiguity))

        # Check for noise level in context
        if "noise_level" in context:
            noise = context["noise_level"]
            if isinstance(noise, (int, float)):
                return max(0.0, 1.0 - float(noise))

        # Return base confidence
        return base_confidence

    def _default_reasoning_confidence(self, reasoning_result: Any,
                                    context: Dict[str, Any]) -> float:
        """
        Default heuristic for estimating reasoning confidence.

        Args:
            reasoning_result: Result from a reasoning process
            context: Contextual information

        Returns:
            Confidence value (0.0 to 1.0)
        """
        # Default moderate confidence
        base_confidence = 0.75

        # Check for explicit confidence indicators
        if hasattr(reasoning_result, "certainty"):
            return min(1.0, float(reasoning_result.certainty))

        if hasattr(reasoning_result, "probability"):
            return min(1.0, float(reasoning_result.probability))

        # Check for inference steps in reasoning_result
        if hasattr(reasoning_result, "inference_steps"):
            steps = reasoning_result.inference_steps
            if steps:
                # More steps can mean more uncertainty
                step_count = len(steps)
                step_factor = max(0.6, 1.0 / (1.0 + 0.1 * step_count))
                base_confidence *= step_factor

        # Check for contradictions in context
        if "contradictions" in context:
            contradictions = context["contradictions"]
            if contradictions:
                contradiction_factor = max(0.3, 1.0 - (0.2 * len(contradictions)))
                base_confidence *= contradiction_factor

        # Check for assumptions in context
        if "assumptions" in context:
            assumptions = context["assumptions"]
            if assumptions:
                assumption_factor = max(0.5, 1.0 - (0.1 * len(assumptions)))
                base_confidence *= assumption_factor

        # Return adjusted confidence
        return base_confidence

    def _default_memory_confidence(self, memory_item: Any,
                                 context: Dict[str, Any]) -> float:
        """
        Default heuristic for estimating memory retrieval confidence.

        Args:
            memory_item: Retrieved memory item
            context: Contextual information

        Returns:
            Confidence value (0.0 to 1.0)
        """
        # Default moderate confidence
        base_confidence = 0.8

        # Check for age/recency of memory
        if hasattr(memory_item, "timestamp"):
            age = time.time() - memory_item.timestamp
            age_hours = age / 3600.0

            # Memories decay over time
            recency_factor = 1.0 / (1.0 + (age_hours / 24.0))  # Half confidence after 24 hours
            base_confidence *= recency_factor

        # Check for reinforcement count
        if hasattr(memory_item, "access_count"):
            count = memory_item.access_count

            # More access = stronger memory
            access_factor = min(1.5, 1.0 + (0.1 * count))  # Up to 50% boost
            base_confidence *= access_factor

            # But cap at 1.0
            base_confidence = min(1.0, base_confidence)

        # Check for retrieval conditions
        if "retrieval_score" in context:
            retrieval_score = context["retrieval_score"]
            if isinstance(retrieval_score, (int, float)):
                retrieval_factor = min(1.0, float(retrieval_score))
                base_confidence *= retrieval_factor

        # Return adjusted confidence
        return base_confidence

# --- réparation 6.5 : alias du nom attendu par les autres modules
ConfidenceEstimator = ConfidenceEvaluator
