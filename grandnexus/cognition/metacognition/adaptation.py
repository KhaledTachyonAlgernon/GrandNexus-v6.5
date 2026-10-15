"""
adaptation.py - Dynamic Parameter Adaptation Module for GrandNexus

This module implements meta-cognitive capabilities to dynamically adjust
system parameters based on performance monitoring, environmental changes,
and operational needs. It serves as a key component of the system's
self-optimization capability.

Key features:
- Parameter tracking and change history management
- Adaptive strategies based on performance feedback
- Multi-dimensional parameter space navigation
- Interference detection between parameter adjustments
- Reinforcement learning based parameter optimization
- Hierarchical parameter dependency modeling
"""

import logging
import time
import threading
import uuid
from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional, Callable, Union, Tuple, Set
from enum import Enum, auto
import math
import json
import numpy as np
from collections import deque, defaultdict

# Assuming these modules exist in the GrandNexus framework
from grandnexus.core.nexus_core import NexusCore
from grandnexus.memory.working import WorkingMemory
from grandnexus.memory.episodic import EpisodicMemory
from grandnexus.metacognition.monitor import MetaMonitor
from grandnexus.metacognition.confidence import ConfidenceEstimator

# Parameter adjustment strategies
class AdaptationStrategy(Enum):
    """Strategies for parameter adaptation"""
    GRADIENT_ASCENT = auto()     # Adjust parameters to maximize performance metric
    GRADIENT_DESCENT = auto()    # Adjust parameters to minimize some metric (like error)
    BINARY_SEARCH = auto()       # Try midpoint between known good/bad values
    RANDOM_SEARCH = auto()       # Try random values within bounds
    BAYESIAN_OPT = auto()        # Bayesian optimization approach
    DISCRETE_SWEEP = auto()      # Sweep across discrete values
    EVOLUTIONARY = auto()        # Genetic algorithm-like approach
    REINFORCEMENT = auto()       # RL-based parameter tuning
    HEURISTIC = auto()           # Manual rule-based adjustments
    MULTI_ARMED_BANDIT = auto()  # Exploration vs exploitation approach

# Parameter change types
class ChangeType(Enum):
    """Types of parameter changes"""
    AUTOMATIC = auto()     # System-initiated change
    FEEDBACK = auto()      # Change based on explicit feedback
    SCHEDULED = auto()     # Pre-scheduled change
    EMERGENCY = auto()     # Change due to system emergency
    EXPERIMENTAL = auto()  # Change to experiment with new settings
    RESET = auto()         # Reset to default/previous value
    CALIBRATION = auto()   # Calibration-related adjustment

@dataclass
class ParameterChange:
    """Record of a parameter change"""
    change_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    parameter_name: str = ""
    previous_value: Any = None
    new_value: Any = None
    timestamp: float = field(default_factory=time.time)
    strategy: AdaptationStrategy = AdaptationStrategy.HEURISTIC
    change_type: ChangeType = ChangeType.AUTOMATIC
    reason: str = ""
    performance_before: Optional[Dict[str, float]] = None
    performance_after: Optional[Dict[str, float]] = None
    confidence: float = 0.5  # Confidence in this change (0-1)
    status: str = "pending"  # pending, applied, reverted, superseded
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class AdaptableParameter:
    """
    A parameter that can be dynamically adjusted.
    Includes constraints, history tracking, and metadata.
    """
    name: str
    value: Any
    default_value: Any
    min_value: Optional[Any] = None
    max_value: Optional[Any] = None
    step_size: Optional[Any] = None
    param_type: str = "float"  # float, int, bool, str, enum, etc.
    description: str = ""
    tags: List[str] = field(default_factory=list)
    affects: List[str] = field(default_factory=list)  # Other parameters this influences
    depends_on: List[str] = field(default_factory=list)  # Parameters this depends on
    is_critical: bool = False
    acceptable_range: Optional[Tuple[Any, Any]] = None  # Narrower than min/max for safety
    change_history: List[str] = field(default_factory=list)  # List of change IDs
    performance_correlation: Dict[str, float] = field(default_factory=dict)  # Metric -> correlation
    change_sensitivity: float = 1.0  # How aggressively to change this parameter (0-1)
    exposure_level: str = "internal"  # internal, exposed, controlled, locked

    def validate_value(self, new_value: Any) -> bool:
        """Check if a proposed value is valid for this parameter"""
        # Check type compatibility
        if self.param_type == "float":
            if not isinstance(new_value, (int, float)):
                return False
        elif self.param_type == "int":
            if not isinstance(new_value, int):
                return False
        elif self.param_type == "bool":
            if not isinstance(new_value, bool):
                return False

        # Check bounds
        if self.min_value is not None and new_value < self.min_value:
            return False
        if self.max_value is not None and new_value > self.max_value:
            return False

        return True

    def update_value(self, new_value: Any, change_record: ParameterChange) -> bool:
        """Update the parameter value and record the change"""
        if not self.validate_value(new_value):
            return False

        self.value = new_value
        self.change_history.append(change_record.change_id)
        return True

    def reset_to_default(self) -> Any:
        """Reset the parameter to its default value"""
        old_value = self.value
        self.value = self.default_value
        return old_value

    def suggest_adjustment(self,
                          performance_metrics: Dict[str, float],
                          strategy: AdaptationStrategy,
                          step_multiplier: float = 1.0) -> Optional[Tuple[Any, float]]:
        """
        Suggest an adjustment to this parameter based on performance metrics.

        Args:
            performance_metrics: Current performance metrics
            strategy: Adaptation strategy to use
            step_multiplier: Multiply the step size by this factor

        Returns:
            Tuple of (suggested_value, confidence) or None if no adjustment recommended
        """
        # This is a simplified implementation - in a real system this would be more sophisticated

        if strategy == AdaptationStrategy.GRADIENT_ASCENT:
            # Try to increase parameter if it's positively correlated with performance
            target_metric = max(self.performance_correlation.items(), key=lambda x: x[1])[0]
            if target_metric in performance_metrics and self.performance_correlation[target_metric] > 0.2:
                if self.param_type == "float" or self.param_type == "int":
                    if self.max_value is not None and self.value >= self.max_value:
                        return None  # Already at max

                    # Calculate step size
                    if self.step_size is not None:
                        step = self.step_size * step_multiplier * self.change_sensitivity
                    else:
                        # Default step is 5% of range or 0.1 of current value
                        if self.min_value is not None and self.max_value is not None:
                            step = (self.max_value - self.min_value) * 0.05 * step_multiplier * self.change_sensitivity
                        else:
                            step = max(0.1 * abs(self.value), 0.01) * step_multiplier * self.change_sensitivity

                    if self.param_type == "int":
                        step = max(1, int(step))

                    new_value = self.value + step
                    if self.max_value is not None:
                        new_value = min(new_value, self.max_value)

                    # Confidence based on correlation strength
                    confidence = min(0.9, abs(self.performance_correlation[target_metric]))
                    return (new_value, confidence)

        elif strategy == AdaptationStrategy.GRADIENT_DESCENT:
            # Try to decrease parameter if it's negatively correlated with performance
            target_metric = min(self.performance_correlation.items(), key=lambda x: x[1])[0]
            if target_metric in performance_metrics and self.performance_correlation[target_metric] < -0.2:
                if self.param_type == "float" or self.param_type == "int":
                    if self.min_value is not None and self.value <= self.min_value:
                        return None  # Already at min

                    # Calculate step size
                    if self.step_size is not None:
                        step = self.step_size * step_multiplier * self.change_sensitivity
                    else:
                        # Default step is 5% of range or 0.1 of current value
                        if self.min_value is not None and self.max_value is not None:
                            step = (self.max_value - self.min_value) * 0.05 * step_multiplier * self.change_sensitivity
                        else:
                            step = max(0.1 * abs(self.value), 0.01) * step_multiplier * self.change_sensitivity

                    if self.param_type == "int":
                        step = max(1, int(step))

                    new_value = self.value - step
                    if self.min_value is not None:
                        new_value = max(new_value, self.min_value)

                    # Confidence based on correlation strength
                    confidence = min(0.9, abs(self.performance_correlation[target_metric]))
                    return (new_value, confidence)

        elif strategy == AdaptationStrategy.BINARY_SEARCH:
            # This would require knowledge of previous good/bad values
            # Simplified implementation:
            if hasattr(self, '_binary_search_upper') and hasattr(self, '_binary_search_lower'):
                midpoint = (self._binary_search_upper + self._binary_search_lower) / 2
                if self.param_type == "int":
                    midpoint = int(midpoint)
                return (midpoint, 0.7)
            else:
                # Initialize binary search bounds
                if self.min_value is not None and self.max_value is not None:
                    self._binary_search_lower = self.min_value
                    self._binary_search_upper = self.max_value
                    midpoint = (self._binary_search_upper + self._binary_search_lower) / 2
                    if self.param_type == "int":
                        midpoint = int(midpoint)
                    return (midpoint, 0.5)

        elif strategy == AdaptationStrategy.RANDOM_SEARCH:
            import random

            if self.param_type == "float":
                if self.min_value is not None and self.max_value is not None:
                    # Random value within bounds
                    new_value = self.min_value + random.random() * (self.max_value - self.min_value)
                    return (new_value, 0.3)  # Low confidence due to randomness

            elif self.param_type == "int":
                if self.min_value is not None and self.max_value is not None:
                    # Random integer within bounds
                    new_value = random.randint(self.min_value, self.max_value)
                    return (new_value, 0.3)

            elif self.param_type == "bool":
                # Toggle the value
                return (not self.value, 0.5)

        # Default: no adjustment
        return None

@dataclass
class ParameterGroup:
    """Group of related parameters that should be considered together"""
    group_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    name: str = ""
    description: str = ""
    parameters: List[str] = field(default_factory=list)  # List of parameter names
    joint_constraints: List[str] = field(default_factory=list)  # E.g., "param1 + param2 <= 1.0"
    tags: List[str] = field(default_factory=list)
    priority: int = 0  # Higher means more important to optimize

class AdaptationEngine:
    """
    Core engine for dynamically adapting system parameters based on
    performance metrics, environmental conditions, and operational feedback.

    This module lies at the heart of GrandNexus's self-optimization and
    metacognitive capabilities, allowing the system to intelligently
    adjust its own operations.
    """

    def __init__(self, nexus_core: Optional[NexusCore] = None,
                 meta_monitor: Optional[MetaMonitor] = None,
                 confidence_estimator: Optional[ConfidenceEstimator] = None,
                 working_memory: Optional[WorkingMemory] = None,
                 episodic_memory: Optional[EpisodicMemory] = None):
        """
        Initialize the adaptation engine.

        Args:
            nexus_core: Reference to the NexusCore
            meta_monitor: Reference to the metacognitive monitoring module
            confidence_estimator: Reference to the confidence estimator
            working_memory: Reference to working memory
            episodic_memory: Reference to episodic memory
        """
        self.logger = logging.getLogger("GrandNexus.Metacognition.Adaptation")
        self.nexus_core = nexus_core
        self.meta_monitor = meta_monitor
        self.confidence_estimator = confidence_estimator
        self.working_memory = working_memory
        self.episodic_memory = episodic_memory

        # Core state
        self.instance_id = str(uuid.uuid4())[:8]
        self.running = False
        self.initialized = False
        self.lock = threading.RLock()

        # Parameters registry
        self.parameters: Dict[str, AdaptableParameter] = {}

        # Parameter groups
        self.parameter_groups: Dict[str, ParameterGroup] = {}

        # Parameter change history
        self.change_history: Dict[str, ParameterChange] = {}
        self.recent_changes = deque(maxlen=100)  # IDs of recent changes

        # Performance metrics history
        self.performance_history: Dict[str, deque] = defaultdict(lambda: deque(maxlen=100))

        # Adaptation strategies
        self.default_strategy = AdaptationStrategy.GRADIENT_ASCENT
        self.strategy_history: Dict[AdaptationStrategy, float] = {
            strategy: 0.0 for strategy in AdaptationStrategy
        }  # Strategy -> success rate

        # Adaptation schedule and state
        self.adaptation_interval = 60  # seconds
        self.last_adaptation_time = 0
        self.adaptation_enabled = True
        self.current_experiment = None

        self.logger.info(f"AdaptationEngine initialized with ID={self.instance_id}")

    def initialize(self) -> bool:
        """Initialize the adaptation engine"""
        if self.initialized:
            return True

        with self.lock:
            try:
                # Register with NexusCore if available
                if self.nexus_core:
                    self._register_with_nexus_core()

                # Initialize parameter registry
                self._initialize_default_parameters()

                self.initialized = True
                self.logger.info("AdaptationEngine initialization complete")
                return True
            except Exception as e:
                self.logger.error(f"AdaptationEngine initialization failed: {str(e)}")
                return False

    def start(self) -> bool:
        """Start the adaptation engine"""
        if not self.initialized:
            if not self.initialize():
                return False

        with self.lock:
            if self.running:
                return True

            try:
                # Start the adaptation background thread
                self.running = True
                self.adaptation_thread = threading.Thread(
                    target=self._adaptation_loop,
                    daemon=True
                )
                self.adaptation_thread.start()

                self.logger.info("AdaptationEngine started successfully")
                return True
            except Exception as e:
                self.logger.error(f"Failed to start AdaptationEngine: {str(e)}")
                return False

    def stop(self) -> bool:
        """Stop the adaptation engine"""
        with self.lock:
            if not self.running:
                return True

            try:
                # Signal the adaptation thread to stop
                self.running = False

                # Wait for the thread to terminate (with timeout)
                if hasattr(self, 'adaptation_thread'):
                    self.adaptation_thread.join(timeout=5.0)

                self.logger.info("AdaptationEngine stopped successfully")
                return True
            except Exception as e:
                self.logger.error(f"Error stopping AdaptationEngine: {str(e)}")
                return False

    def register_parameter(self, parameter: AdaptableParameter) -> bool:
        """
        Register a parameter for adaptation.

        Args:
            parameter: The AdaptableParameter to register

        Returns:
            True if registration succeeded, False otherwise
        """
        with self.lock:
            if parameter.name in self.parameters:
                self.logger.warning(f"Parameter {parameter.name} already exists, will be overwritten")

            self.parameters[parameter.name] = parameter

            # If this parameter has tags that match any group, add it to those groups
            for group_id, group in self.parameter_groups.items():
                if any(tag in parameter.tags for tag in group.tags):
                    if parameter.name not in group.parameters:
                        group.parameters.append(parameter.name)
                        self.logger.info(f"Added parameter {parameter.name} to group {group.name} based on tags")

            self.logger.info(f"Registered parameter: {parameter.name}")
            return True

    def register_parameter_group(self, group: ParameterGroup) -> bool:
        """
        Register a group of related parameters.

        Args:
            group: The ParameterGroup to register

        Returns:
            True if registration succeeded, False otherwise
        """
        with self.lock:
            if group.group_id in self.parameter_groups:
                self.logger.warning(f"Parameter group {group.group_id} already exists, will be overwritten")

            self.parameter_groups[group.group_id] = group

            # Validate that all parameters exist
            missing_params = [p for p in group.parameters if p not in self.parameters]
            if missing_params:
                self.logger.warning(f"Group {group.name} references non-existent parameters: {missing_params}")

            self.logger.info(f"Registered parameter group: {group.name} with {len(group.parameters)} parameters")
            return True

    def set_parameter(self, name: str, value: Any,
                     reason: str = "",
                     change_type: ChangeType = ChangeType.AUTOMATIC) -> Optional[str]:
        """
        Set a parameter to a new value.

        Args:
            name: Parameter name
            value: New value
            reason: Reason for the change
            change_type: Type of change

        Returns:
            Change ID if successful, None otherwise
        """
        with self.lock:
            if name not in self.parameters:
                self.logger.error(f"Parameter {name} not found")
                return None

            parameter = self.parameters[name]

            # Validate the new value
            if not parameter.validate_value(value):
                self.logger.error(f"Invalid value {value} for parameter {name}")
                return None

            # Record performance before change
            performance_before = {}
            if self.meta_monitor:
                try:
                    # Get current performance metrics
                    metrics = self.meta_monitor.get_current_metrics()
                    if metrics:
                        performance_before = metrics
                except Exception as e:
                    self.logger.warning(f"Failed to get performance metrics: {str(e)}")

            # Create change record
            change = ParameterChange(
                parameter_name=name,
                previous_value=parameter.value,
                new_value=value,
                timestamp=time.time(),
                strategy=AdaptationStrategy.HEURISTIC,  # Manual/explicit change
                change_type=change_type,
                reason=reason,
                performance_before=performance_before,
                status="applied",
                confidence=1.0  # High confidence for explicit changes
            )

            # Store the change record
            self.change_history[change.change_id] = change
            self.recent_changes.append(change.change_id)

            # Update the parameter
            success = parameter.update_value(value, change)
            if not success:
                self.logger.error(f"Failed to update parameter {name}")
                change.status = "failed"
                return None

            self.logger.info(f"Parameter {name} changed from {change.previous_value} to {value}")

            # Schedule performance after check
            threading.Timer(5.0, self._check_performance_after, args=[change.change_id]).start()

            return change.change_id

    def get_parameter(self, name: str) -> Optional[Any]:
        """
        Get the current value of a parameter.

        Args:
            name: Parameter name

        Returns:
            Current parameter value or None if not found
        """
        with self.lock:
            if name not in self.parameters:
                self.logger.error(f"Parameter {name} not found")
                return None

            return self.parameters[name].value

    def get_parameter_info(self, name: str) -> Optional[Dict[str, Any]]:
        """
        Get detailed information about a parameter.

        Args:
            name: Parameter name

        Returns:
            Dictionary with parameter information or None if not found
        """
        with self.lock:
            if name not in self.parameters:
                self.logger.error(f"Parameter {name} not found")
                return None

            param = self.parameters[name]

            # Prepare change history information
            changes = []
            for change_id in param.change_history[-10:]:  # Get last 10 changes
                if change_id in self.change_history:
                    change = self.change_history[change_id]
                    changes.append({
                        "change_id": change_id,
                        "timestamp": change.timestamp,
                        "previous_value": change.previous_value,
                        "new_value": change.new_value,
                        "reason": change.reason,
                        "status": change.status
                    })

            # Prepare parameter information
            info = {
                "name": param.name,
                "value": param.value,
                "default_value": param.default_value,
                "type": param.param_type,
                "description": param.description,
                "is_critical": param.is_critical,
                "min_value": param.min_value,
                "max_value": param.max_value,
                "step_size": param.step_size,
                "tags": param.tags,
                "affects": param.affects,
                "depends_on": param.depends_on,
                "change_sensitivity": param.change_sensitivity,
                "exposure_level": param.exposure_level,
                "performance_correlation": param.performance_correlation,
                "recent_changes": changes
            }

            return info

    def list_parameters(self, tags: Optional[List[str]] = None) -> List[Dict[str, Any]]:
        """
        List all registered parameters, optionally filtered by tags.

        Args:
            tags: Optional list of tags to filter by

        Returns:
            List of parameter information dictionaries
        """
        with self.lock:
            result = []

            for name, param in self.parameters.items():
                # Skip if we're filtering by tags and this parameter doesn't match
                if tags and not any(tag in param.tags for tag in tags):
                    continue

                result.append({
                    "name": param.name,
                    "value": param.value,
                    "type": param.param_type,
                    "description": param.description,
                    "tags": param.tags
                })

            return result

    def list_parameter_groups(self) -> List[Dict[str, Any]]:
        """
        List all registered parameter groups.

        Returns:
            List of group information dictionaries
        """
        with self.lock:
            result = []

            for group_id, group in self.parameter_groups.items():
                result.append({
                    "group_id": group_id,
                    "name": group.name,
                    "description": group.description,
                    "parameters": group.parameters,
                    "tags": group.tags,
                    "priority": group.priority
                })

            return result

    def update_performance_metrics(self, metrics: Dict[str, float]) -> None:
        """
        Update the system with current performance metrics.

        Args:
            metrics: Dictionary of metric name -> value
        """
        with self.lock:
            # Store metrics in history
            for name, value in metrics.items():
                self.performance_history[name].append((time.time(), value))

            # Update parameter correlations
            self._update_parameter_correlations(metrics)

    def suggest_parameter_adjustments(self) -> List[Dict[str, Any]]:
        """
        Suggest parameter adjustments based on current performance.

        Returns:
            List of suggested adjustments with confidence levels
        """
        with self.lock:
            if not self.adaptation_enabled:
                return []

            # Get current performance metrics
            current_metrics = {}
            if self.meta_monitor:
                try:
                    metrics = self.meta_monitor.get_current_metrics()
                    if metrics:
                        current_metrics = metrics
                except Exception as e:
                    self.logger.warning(f"Failed to get performance metrics: {str(e)}")

            # If no metrics, use recent history
            if not current_metrics:
                for metric, history in self.performance_history.items():
                    if history:
                        current_metrics[metric] = history[-1][1]

            # If still no metrics, we can't make good suggestions
            if not current_metrics:
                self.logger.warning("No performance metrics available for parameter suggestions")
                return []

            suggestions = []

            # Select adaptation strategy - prefer strategies that worked well in the past
            # This is a simple weighted random selection
            strategy_weights = {}
            for strategy, success_rate in self.strategy_history.items():
                # Base weight on success rate, but ensure some chance for all strategies
                strategy_weights[strategy] = 0.1 + success_rate

            # Normalize weights
            total_weight = sum(strategy_weights.values())
            if total_weight > 0:
                strategy_weights = {s: w/total_weight for s, w in strategy_weights.items()}

            import random
            r = random.random()
            cum_weight = 0
            selected_strategy = self.default_strategy
            for strategy, weight in strategy_weights.items():
                cum_weight += weight
                if r <= cum_weight:
                    selected_strategy = strategy
                    break

            # Let each parameter suggest an adjustment
            for name, param in self.parameters.items():
                # Skip parameters with no correlation data
                if not param.performance_correlation:
                    continue

                suggestion = param.suggest_adjustment(
                    current_metrics,
                    selected_strategy
                )

                if suggestion:
                    new_value, confidence = suggestion
                    suggestions.append({
                        "parameter": name,
                        "current_value": param.value,
                        "suggested_value": new_value,
                        "confidence": confidence,
                        "strategy": selected_strategy.name
                    })

            # Sort by confidence (highest first)
            suggestions.sort(key=lambda x: x["confidence"], reverse=True)

            return suggestions

    def apply_suggestions(self, max_changes: int = 1,
                         min_confidence: float = 0.5) -> List[str]:
        """
        Automatically apply parameter adjustment suggestions.

        Args:
            max_changes: Maximum number of changes to apply
            min_confidence: Minimum confidence threshold for applying changes

        Returns:
            List of change IDs for applied changes
        """
        with self.lock:
            if not self.adaptation_enabled:
                return []

            suggestions = self.suggest_parameter_adjustments()

            # Filter by confidence
            suggestions = [s for s in suggestions if s["confidence"] >= min_confidence]

            applied_changes = []

            # Apply up to max_changes suggestions
            for suggestion in suggestions[:max_changes]:
                change_id = self.set_parameter(
                    suggestion["parameter"],
                    suggestion["suggested_value"],
                    reason=f"Automatic adjustment using {suggestion['strategy']} strategy",
                    change_type=ChangeType.AUTOMATIC
                )

                if change_id:
                    # Update the change record with strategy information
                    change = self.change_history[change_id]
                    change.strategy = getattr(AdaptationStrategy, suggestion["strategy"])
                    change.confidence = suggestion["confidence"]

                    applied_changes.append(change_id)

            return applied_changes

    def revert_change(self, change_id: str,
                     reason: str = "Manual revert") -> Optional[str]:
        """
        Revert a parameter change.

        Args:
            change_id: ID of the change to revert
            reason: Reason for the revert

        Returns:
            New change ID if successful, None otherwise
        """
        with self.lock:
            if change_id not in self.change_history:
                self.logger.error(f"Change {change_id} not found")
                return None

            change = self.change_history[change_id]

            # Skip if already reverted or failed
            if change.status in ["reverted", "failed"]:
                self.logger.warning(f"Change {change_id} is already {change.status}")
                return None

            # Get the affected parameter
            param_name = change.parameter_name
            if param_name not in self.parameters:
                self.logger.error(f"Parameter {param_name} no longer exists")
                return None

            # Create a new change that reverts to the previous value
            revert_id = self.set_parameter(
                param_name,
                change.previous_value,
                reason=reason,
                change_type=ChangeType.RESET
            )

            if revert_id:
                # Update status of the original change
                change.status = "reverted"

                # Link the revert change to the original
                revert_change = self.change_history[revert_id]
                revert_change.metadata["reverts"] = change_id

                self.logger.info(f"Change {change_id} reverted by change {revert_id}")

            return revert_id

    def reset_parameter(self, name: str) -> Optional[str]:
        """
        Reset a parameter to its default value.

        Args:
            name: Parameter name

        Returns:
            Change ID if successful, None otherwise
        """
        with self.lock:
            if name not in self.parameters:
                self.logger.error(f"Parameter {name} not found")
                return None

            parameter = self.parameters[name]

            # Create a change record for the reset
            change_id = self.set_parameter(
                name,
                parameter.default_value,
                reason="Reset to default value",
                change_type=ChangeType.RESET
            )

            return change_id

    def disable_adaptation(self) -> None:
        """Temporarily disable automatic parameter adaptation."""
        with self.lock:
            self.adaptation_enabled = False
            self.logger.info("Parameter adaptation disabled")

    def enable_adaptation(self) -> None:
        """Re-enable automatic parameter adaptation."""
        with self.lock:
            self.adaptation_enabled = True
            self.logger.info("Parameter adaptation enabled")

    def run_experiment(self, experiment_name: str,
                     parameters: Dict[str, Any],
                     duration: float = 300.0,
                     reason: str = "") -> str:
        """
        Run an experiment with temporary parameter values.

        Args:
            experiment_name: Name of the experiment
            parameters: Dictionary of parameter name -> value
            duration: How long to run the experiment in seconds
            reason: Reason for the experiment

        Returns:
            Experiment ID (actually a UUID)
        """
        with self.lock:
            # Create an experiment record
            experiment_id = str(uuid.uuid4())

            # Store original values
            original_values = {}
            change_ids = []

            # Set the experimental values
            for name, value in parameters.items():
                if name in self.parameters:
                    original_values[name] = self.parameters[name].value

                    change_id = self.set_parameter(
                        name,
                        value,
                        reason=f"Experiment: {experiment_name} - {reason}",
                        change_type=ChangeType.EXPERIMENTAL
                    )

                    if change_id:
                        change_ids.append(change_id)
                else:
                    self.logger.warning(f"Experiment parameter {name} not found")

            # Store experiment data
            self.current_experiment = {
                "id": experiment_id,
                "name": experiment_name,
                "start_time": time.time(),
                "end_time": time.time() + duration,
                "original_values": original_values,
                "change_ids": change_ids,
                "reason": reason
            }

            # Schedule the end of the experiment
            threading.Timer(duration, self._end_experiment, args=[experiment_id]).start()

            self.logger.info(f"Started experiment {experiment_name} with ID {experiment_id}")

            return experiment_id

    def export_parameter_config(self) -> Dict[str, Any]:
        """
        Export the current parameter configuration.

        Returns:
            Dictionary of parameter configurations
        """
        with self.lock:
            config = {}

            for name, param in self.parameters.items():
                config[name] = {
                    "value": param.value,
                    "type": param.param_type,
                    "description": param.description,
                    "is_critical": param.is_critical,
                    "min_value": param.min_value,
                    "max_value": param.max_value
                }

            return config

    def import_parameter_config(self, config: Dict[str, Dict[str, Any]],
                               reason: str = "Imported configuration") -> List[str]:
        """
        Import parameter configuration from a dictionary.

        Args:
            config: Dictionary of parameter name -> configuration
            reason: Reason for the import

        Returns:
            List of change IDs for applied changes
        """
        with self.lock:
            change_ids = []

            for name, param_config in config.items():
                if name in self.parameters and "value" in param_config:
                    change_id = self.set_parameter(
                        name,
                        param_config["value"],
                        reason=reason,
                        change_type=ChangeType.AUTOMATIC
                    )

                    if change_id:
                        change_ids.append(change_id)

            return change_ids

    def get_change_history(self, limit: int = 100) -> List[Dict[str, Any]]:
        """
        Get the history of parameter changes.

        Args:
            limit: Maximum number of changes to return

        Returns:
            List of change records (most recent first)
        """
        with self.lock:
            changes = []

            # Get the most recent change IDs
            recent_ids = list(self.recent_changes)[-limit:]

            for change_id in reversed(recent_ids):
                if change_id in self.change_history:
                    change = self.change_history[change_id]
                    changes.append({
                        "change_id": change_id,
                        "parameter_name": change.parameter_name,
                        "previous_value": change.previous_value,
                        "new_value": change.new_value,
                        "timestamp": change.timestamp,
                        "reason": change.reason,
                        "strategy": change.strategy.name,
                        "change_type": change.change_type.name,
                        "status": change.status,
                        "confidence": change.confidence
                    })

            return changes

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

            if self.meta_monitor:
                dependencies.append("meta_monitor")
            if self.confidence_estimator:
                dependencies.append("confidence_estimator")
            if self.working_memory:
                dependencies.append("working_memory")
            if self.episodic_memory:
                dependencies.append("episodic_memory")

            self.nexus_core.register_module(
                name="adaptation_engine",
                module=self,
                dependencies=dependencies
            )

            self.logger.info("Successfully registered with NexusCore")
        except Exception as e:
            self.logger.error(f"Failed to register with NexusCore: {str(e)}")

    def _initialize_default_parameters(self) -> None:
        """Initialize default adaptable parameters"""
        # System-wide parameters
        self.register_parameter(AdaptableParameter(
            name="system.logging_level",
            value="INFO",
            default_value="INFO",
            param_type="enum",
            description="System-wide logging level",
            tags=["system", "logging"],
            acceptable_range=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
            is_critical=True,
            exposure_level="exposed"
        ))

        self.register_parameter(AdaptableParameter(
            name="system.thread_pool_size",
            value=8,
            default_value=8,
            param_type="int",
            min_value=1,
            max_value=32,
            step_size=1,
            description="Number of worker threads in the thread pool",
            tags=["system", "performance", "resources"],
            is_critical=True,
            exposure_level="internal"
        ))

        # Metacognition parameters
        self.register_parameter(AdaptableParameter(
            name="metacog.confidence_threshold",
            value=0.7,
            default_value=0.7,
            param_type="float",
            min_value=0.0,
            max_value=1.0,
            step_size=0.05,
            description="Threshold for high-confidence decisions",
            tags=["metacognition", "confidence"],
            exposure_level="exposed"
        ))

        self.register_parameter(AdaptableParameter(
            name="metacog.monitoring_frequency",
            value=1.0,
            default_value=1.0,
            param_type="float",
            min_value=0.1,
            max_value=10.0,
            step_size=0.1,
            description="Frequency of metacognitive monitoring in Hz",
            tags=["metacognition", "monitoring"],
            exposure_level="internal"
        ))

        # Adaptation engine's own parameters
        self.register_parameter(AdaptableParameter(
            name="adaptation.interval",
            value=self.adaptation_interval,
            default_value=60.0,
            param_type="float",
            min_value=5.0,
            max_value=3600.0,
            step_size=5.0,
            description="Interval between adaptation cycles in seconds",
            tags=["adaptation", "timing"],
            is_critical=False,
            exposure_level="internal"
        ))

        self.register_parameter(AdaptableParameter(
            name="adaptation.max_changes_per_cycle",
            value=1,
            default_value=1,
            param_type="int",
            min_value=0,
            max_value=10,
            step_size=1,
            description="Maximum parameter changes per adaptation cycle",
            tags=["adaptation", "limits"],
            is_critical=True,
            exposure_level="exposed"
        ))

        # Create a parameter group
        self.register_parameter_group(ParameterGroup(
            name="System Resources",
            description="Parameters affecting system resource utilization",
            parameters=["system.thread_pool_size"],
            tags=["system", "resources", "performance"],
            priority=5
        ))

        self.register_parameter_group(ParameterGroup(
            name="Metacognition",
            description="Parameters for metacognitive processes",
            parameters=["metacog.confidence_threshold", "metacog.monitoring_frequency"],
            tags=["metacognition"],
            priority=3
        ))

        # Add more default parameters and groups as needed

    def _adaptation_loop(self) -> None:
        """Background thread for periodic parameter adaptation"""
        self.logger.info("Adaptation loop started")

        while self.running:
            try:
                # Check if adaptation is enabled
                if self.adaptation_enabled:
                    # Check if it's time for adaptation
                    current_time = time.time()

                    # Get the current interval setting (may have been changed)
                    interval = self.get_parameter("adaptation.interval")
                    if interval is None:
                        interval = self.adaptation_interval

                    if current_time - self.last_adaptation_time >= interval:
                        # Get max changes setting
                        max_changes = self.get_parameter("adaptation.max_changes_per_cycle")
                        if max_changes is None:
                            max_changes = 1

                        # Apply parameter suggestions
                        changes = self.apply_suggestions(max_changes=max_changes)

                        if changes:
                            self.logger.info(f"Applied {len(changes)} parameter changes")

                        self.last_adaptation_time = current_time

                # Check for expired experiment
                if self.current_experiment and current_time >= self.current_experiment["end_time"]:
                    self._end_experiment(self.current_experiment["id"])

                # Sleep a bit to avoid busy-waiting
                # We use small sleeps and check self.running frequently to ensure quick shutdown
                time.sleep(1.0)

            except Exception as e:
                self.logger.error(f"Error in adaptation loop: {str(e)}")
                time.sleep(5.0)  # Sleep a bit longer on error

    def _update_parameter_correlations(self, metrics: Dict[str, float]) -> None:
        """
        Update the correlation between parameters and performance metrics.

        Args:
            metrics: Current performance metrics
        """
        # This is a simplistic implementation
        # In a real system, this would use more sophisticated statistical methods

        # For each parameter, calculate correlation with each metric
        for param_name, param in self.parameters.items():
            # Skip if no change history
            if not param.change_history:
                continue

            # Get the most recent changes
            recent_changes = []
            for change_id in param.change_history[-10:]:  # Last 10 changes
                if change_id in self.change_history:
                    change = self.change_history[change_id]
                    if change.performance_before and change.performance_after:
                        recent_changes.append(change)

            # Skip if not enough data
            if len(recent_changes) < 3:
                continue

            # Calculate direction of correlation
            for metric_name, current_value in metrics.items():
                correlations = []

                for change in recent_changes:
                    if metric_name in change.performance_before and metric_name in change.performance_after:
                        before = change.performance_before[metric_name]
                        after = change.performance_after[metric_name]

                        # Calculate the change in parameter and metric
                        param_change = change.new_value - change.previous_value
                        metric_change = after - before

                        # Skip if no change in parameter
                        if param_change == 0:
                            continue

                        # Calculate the sign of correlation: +1 if same direction, -1 if opposite
                        correlation = 1 if (param_change > 0) == (metric_change > 0) else -1

                        correlations.append(correlation)

                # Calculate average correlation
                if correlations:
                    avg_correlation = sum(correlations) / len(correlations)
                    param.performance_correlation[metric_name] = avg_correlation

    def _check_performance_after(self, change_id: str) -> None:
        """
        Check performance metrics after a parameter change.

        Args:
            change_id: ID of the change to check
        """
        with self.lock:
            if change_id not in self.change_history:
                return

            change = self.change_history[change_id]

            # Skip if already has performance_after
            if change.performance_after:
                return

            # Get current performance metrics
            performance_after = {}
            if self.meta_monitor:
                try:
                    metrics = self.meta_monitor.get_current_metrics()
                    if metrics:
                        performance_after = metrics
                        change.performance_after = performance_after

                        # Calculate impact on performance
                        impact = {}
                        if change.performance_before:
                            for metric, after_value in performance_after.items():
                                if metric in change.performance_before:
                                    before_value = change.performance_before[metric]
                                    impact[metric] = after_value - before_value

                            # Update strategy success history
                            if impact:
                                # Count this as a success if any important metric improved
                                # This is a simplistic approach
                                success = any(v > 0 for k, v in impact.items()
                                             if k in ["throughput", "accuracy", "reward"])

                                # Update strategy success rate
                                strategy = change.strategy
                                current_rate = self.strategy_history[strategy]
                                # Weighted update of success rate (30% new, 70% history)
                                self.strategy_history[strategy] = 0.7 * current_rate + 0.3 * (1.0 if success else 0.0)
                except Exception as e:
                    self.logger.warning(f"Failed to get performance metrics after change: {str(e)}")

    def _end_experiment(self, experiment_id: str) -> None:
        """
        End an experiment by restoring original parameter values.

        Args:
            experiment_id: ID of the experiment to end
        """
        with self.lock:
            if not self.current_experiment or self.current_experiment["id"] != experiment_id:
                return

            experiment = self.current_experiment

            # Restore original values
            for name, value in experiment["original_values"].items():
                self.set_parameter(
                    name,
                    value,
                    reason=f"End of experiment: {experiment['name']}",
                    change_type=ChangeType.RESET
                )

            self.logger.info(f"Ended experiment {experiment['name']} with ID {experiment_id}")

            # Clear current experiment
            self.current_experiment = None

# --- réparation 6.5 : alias du nom attendu par les autres modules
AdaptationManager = AdaptationEngine
