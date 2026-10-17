"""
intrinsic_motivation.py - Intrinsic Motivation and Curiosity Module for GrandNexus

This module implements computational curiosity and intrinsic motivation mechanisms,
enabling GrandNexus to explore and learn in the absence of external rewards.
It draws inspiration from both neuroscience and computational theories of curiosity,
providing a flexible foundation for self-motivated learning.

Key features:
- Multiple intrinsic reward generators (novelty, surprise, uncertainty, competence)
- Integration with both memory systems and learning modules
- Adaptive curiosity mechanisms that evolve with experience
- Attention direction based on information-theoretic principles
- Support for both symbolic and neural approaches to curiosity
"""

import logging
import threading
import time
import uuid
import random
import math
from enum import Enum, auto
from typing import Dict, List, Any, Optional, Callable, Tuple, Union, Set
from dataclasses import dataclass, field
from collections import deque, defaultdict
import copy
import numpy as np

# Assuming these exist in the architecture
from grandnexus.core.nexus_core import NexusCore
from grandnexus.memory.episodic import EpisodicMemory
from grandnexus.memory.working import WorkingMemory
from grandnexus.memory.semantic import SemanticMemory
from grandnexus.learning.learning_core import LearningCore

# Optional imports for advanced functionality
try:
    from sklearn.neighbors import NearestNeighbors
    from sklearn.cluster import KMeans
    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

# Intrinsic motivation types
class MotivationType(Enum):
    """Types of intrinsic motivation mechanisms"""
    NOVELTY = auto()       # Driven by encountering new states/experiences
    SURPRISE = auto()      # Driven by prediction errors
    UNCERTAINTY = auto()   # Driven by epistemic uncertainty
    COMPETENCE = auto()    # Driven by mastery (learning progress)
    EMPOWERMENT = auto()   # Driven by control over environment (options/choices)
    CURIOSITY = auto()     # General/mixed curiosity drive
    CUSTOM = auto()        # Custom motivation type

class AttentionFocusType(Enum):
    """Types of attention focus mechanisms"""
    ENTROPY = auto()       # Focus on high entropy areas
    UNCERTAINTY = auto()   # Focus on uncertain areas
    PREDICTION_ERROR = auto() # Focus on areas with high prediction error
    LEARNING_PROGRESS = auto() # Focus on areas with high learning progress
    RANDOM = auto()        # Random focus (baseline)
    CUSTOM = auto()        # Custom focus type

@dataclass
class CuriosityState:
    """
    Represents the current state of curiosity and focus of attention.
    This serves as a high-level aggregation of different intrinsic drives.
    """
    interest_level: float = 0.5         # Overall interest/curiosity level (0-1)
    attention_focus: Optional[Any] = None  # Current focus of attention
    focus_strength: float = 0.5         # Strength of current focus (0-1)
    active_drives: List[MotivationType] = field(default_factory=list)
    novelty_seeking: float = 0.5        # Current novelty seeking level (0-1)
    exploration_rate: float = 0.5       # Current exploration bias (0-1)
    timestamp: float = field(default_factory=time.time)

@dataclass
class ObservationData:
    """
    Container for an observation that can be evaluated for intrinsic motivation.
    Flexible to support both symbolic and neural representations.
    """
    content: Any                        # The actual observation
    context: Dict[str, Any] = field(default_factory=dict)  # Contextual information
    source: str = "unknown"             # Source of the observation
    timestamp: float = field(default_factory=time.time)
    embedding: Optional[np.ndarray] = None  # Vector representation if available
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class MotivationSignal:
    """
    Represents an intrinsic motivation signal generated from an observation.
    """
    observation: ObservationData
    motivation_type: MotivationType
    strength: float                     # Strength of the signal (0-1)
    explanation: str = ""               # Human-readable explanation
    target: Optional[Any] = None        # What aspect should be explored/attended to
    timestamp: float = field(default_factory=time.time)
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class IntrinsicMotivationConfig:
    """Configuration for the intrinsic motivation module"""
    # General settings
    enabled_types: List[MotivationType] = field(default_factory=lambda: [
        MotivationType.NOVELTY, MotivationType.SURPRISE, MotivationType.CURIOSITY
    ])
    default_weights: Dict[MotivationType, float] = field(default_factory=lambda: {
        MotivationType.NOVELTY: 1.0,
        MotivationType.SURPRISE: 1.0,
        MotivationType.UNCERTAINTY: 0.8,
        MotivationType.COMPETENCE: 0.6,
        MotivationType.EMPOWERMENT: 0.5,
        MotivationType.CURIOSITY: 1.0,
    })

    # Memory and integration
    embedding_dimension: int = 128      # Dimension for embeddings
    memory_capacity: int = 1000         # How many observation embeddings to store
    memory_decay: float = 0.995         # Rate of memory decay (forgetting)

    # Novelty detection
    novelty_threshold: float = 0.8      # Minimum distance to be considered novel
    novelty_k_neighbors: int = 5        # k for nearest neighbor novelty detection

    # Surprise model parameters
    surprise_window_size: int = 50      # Window size for surprise detection
    surprise_min_count: int = 10        # Minimum observations before surprise detection

    # Uncertainty estimation
    uncertainty_ensemble_size: int = 5  # Number of models in uncertainty ensemble
    uncertainty_threshold: float = 0.3  # Threshold for high uncertainty

    # Competence progress tracking
    competence_window_size: int = 100   # Window size for competence tracking
    competence_smoothing: float = 0.1   # Smoothing factor for competence progress

    # Empowerment estimation
    empowerment_horizon: int = 3        # Horizon for empowerment calculation
    empowerment_samples: int = 10       # Number of samples for empowerment estimation

    # Attention mechanism
    attention_decay: float = 0.9        # Decay of attention strength
    attention_focus_type: AttentionFocusType = AttentionFocusType.UNCERTAINTY
    attention_switch_threshold: float = 0.2  # Threshold for switching attention

    # Adaptation parameters
    adaptive_weights: bool = True       # Whether weights adapt based on success
    learning_rate: float = 0.01         # Rate for adapting weights

    # Integration with other modules
    use_episodic_memory: bool = True
    use_semantic_memory: bool = True
    use_working_memory: bool = True

    # Custom parameters
    custom_parameters: Dict[str, Any] = field(default_factory=dict)


class NoveltyDetector:
    """
    Detects novelty in observations based on distance to previous observations.
    Uses a nearest-neighbor approach for efficient novelty calculation.
    """

    def __init__(self, config: IntrinsicMotivationConfig):
        """
        Initialize novelty detector.

        Args:
            config: Configuration parameters
        """
        self.config = config
        self.embedding_memory = []
        self.timestamps = []
        self.k = config.novelty_k_neighbors
        self.threshold = config.novelty_threshold
        self.nn_model = None

        # Logging for debugging
        self.logger = logging.getLogger("GrandNexus.Learning.IntrinsicMotivation.Novelty")

    def compute_novelty(self, observation: ObservationData) -> float:
        """
        Compute novelty score for an observation.

        Args:
            observation: The observation to evaluate

        Returns:
            Novelty score in [0, 1] range
        """
        # Ensure we have an embedding
        if observation.embedding is None:
            self.logger.warning("Cannot compute novelty for observation without embedding")
            return 0.5  # Default mid-range novelty

        # If memory is empty, this is maximally novel
        if not self.embedding_memory:
            self._add_to_memory(observation)
            return 1.0

        # Compute distance to existing memories
        if HAS_SKLEARN and len(self.embedding_memory) >= self.k:
            # Use efficient nearest neighbors for large memories
            novelty = self._compute_novelty_nn(observation.embedding)
        else:
            # Manual calculation for small memories
            novelty = self._compute_novelty_manual(observation.embedding)

        # Add to memory if novel or randomly (to ensure diversity)
        if novelty > self.threshold or random.random() < 0.1:
            self._add_to_memory(observation)

        return novelty

    def _compute_novelty_nn(self, embedding: np.ndarray) -> float:
        """
        Compute novelty using nearest neighbors algorithm.

        Args:
            embedding: Vector representation of the observation

        Returns:
            Novelty score in [0, 1] range
        """
        # Ensure embedding is 2D
        if embedding.ndim == 1:
            embedding = embedding.reshape(1, -1)

        # Initialize or update model if needed
        if self.nn_model is None or len(self.embedding_memory) > len(self.timestamps):
            self._update_nn_model()

        # Get distances to k nearest neighbors
        distances, _ = self.nn_model.kneighbors(embedding)

        # Compute mean distance to k nearest neighbors, normalized to [0, 1]
        mean_distance = np.mean(distances)

        # Apply sigmoid-like normalization
        novelty = 2.0 / (1.0 + np.exp(-5.0 * mean_distance)) - 1.0

        return float(min(1.0, max(0.0, novelty)))

    def _compute_novelty_manual(self, embedding: np.ndarray) -> float:
        """
        Compute novelty manually for small memory sizes.

        Args:
            embedding: Vector representation of the observation

        Returns:
            Novelty score in [0, 1] range
        """
        # Compute distances to all memories
        distances = []
        for mem_embedding in self.embedding_memory:
            # Euclidean distance
            distance = np.sqrt(np.sum((embedding - mem_embedding) ** 2))
            distances.append(distance)

        # Sort distances and take mean of k smallest
        distances.sort()
        k = min(self.k, len(distances))
        if k == 0:
            return 1.0

        mean_distance = np.mean(distances[:k])

        # Apply sigmoid-like normalization
        novelty = 2.0 / (1.0 + np.exp(-5.0 * mean_distance)) - 1.0

        return float(min(1.0, max(0.0, novelty)))

    def _add_to_memory(self, observation: ObservationData) -> None:
        """
        Add observation to memory.

        Args:
            observation: Observation to add
        """
        # Add embedding to memory
        self.embedding_memory.append(observation.embedding)
        self.timestamps.append(observation.timestamp)

        # Limit memory size
        if len(self.embedding_memory) > self.config.memory_capacity:
            # Remove oldest or randomly (with preference for low-novelty items)
            if random.random() < 0.7:  # 70% chance to remove oldest
                self.embedding_memory.pop(0)
                self.timestamps.pop(0)
            else:
                # Remove a random item, weighted by age (older more likely)
                now = time.time()
                ages = [now - ts for ts in self.timestamps]
                weights = [age for age in ages]
                if sum(weights) > 0:
                    weights = [w / sum(weights) for w in weights]
                    idx = random.choices(range(len(weights)), weights=weights)[0]
                else:
                    idx = random.randint(0, len(self.embedding_memory) - 1)

                self.embedding_memory.pop(idx)
                self.timestamps.pop(idx)

            # Clear model to force retraining
            self.nn_model = None

    def _update_nn_model(self) -> None:
        """Update the nearest neighbors model with current memory"""
        if not HAS_SKLEARN:
            return

        # Convert memory to numpy array
        memory_array = np.array(self.embedding_memory)

        # Create and fit model
        self.nn_model = NearestNeighbors(n_neighbors=min(self.k, len(memory_array)))
        self.nn_model.fit(memory_array)

    def reset(self) -> None:
        """Clear memory and reset the detector"""
        self.embedding_memory = []
        self.timestamps = []
        self.nn_model = None


class SurpriseDetector:
    """
    Detects surprise (prediction errors) in observations.
    Maintains prediction models and tracks their errors over time.
    """

    def __init__(self, config: IntrinsicMotivationConfig):
        """
        Initialize surprise detector.

        Args:
            config: Configuration parameters
        """
        self.config = config
        self.window_size = config.surprise_window_size
        self.min_count = config.surprise_min_count

        # History of observations and predictions
        self.observation_history = deque(maxlen=self.window_size)
        self.prediction_errors = deque(maxlen=self.window_size)

        # Simple prediction models
        self.prediction_models = {}

        # Logging
        self.logger = logging.getLogger("GrandNexus.Learning.IntrinsicMotivation.Surprise")

    def compute_surprise(self, observation: ObservationData) -> float:
        """
        Compute surprise score for an observation.

        Args:
            observation: The observation to evaluate

        Returns:
            Surprise score in [0, 1] range
        """
        # Make prediction
        prediction = self._predict(observation)

        # Compute error between prediction and actual
        error = self._compute_prediction_error(observation, prediction)

        # Update history
        self.observation_history.append(observation)
        self.prediction_errors.append(error)

        # Update model
        self._update_model(observation)

        # Compute surprise based on how unusual this error is
        if len(self.prediction_errors) < self.min_count:
            # Not enough data yet, return moderate surprise
            surprise = 0.5
        else:
            # Compute how many standard deviations away from mean
            mean_error = np.mean(self.prediction_errors)
            std_error = np.std(self.prediction_errors) + 1e-5  # Avoid division by zero
            z_score = (error - mean_error) / std_error

            # Convert to [0, 1] range using sigmoid-like function
            surprise = 1.0 / (1.0 + np.exp(-z_score))

        return float(min(1.0, max(0.0, surprise)))

    def _predict(self, observation: ObservationData) -> Any:
        """
        Make a prediction for an observation.

        Args:
            observation: Observation to predict

        Returns:
            Prediction of observation content
        """
        # If we have no history, return the observation itself (no surprise)
        if not self.observation_history:
            return observation.content

        # Get context keys to use for prediction
        context_keys = set(observation.context.keys())

        # Find a suitable model based on context
        model_key = None
        for key in self.prediction_models:
            if set(key.split('|')) <= context_keys:
                model_key = key
                break

        if model_key is not None and self.prediction_models[model_key].get('count', 0) >= self.min_count:
            # Use existing model
            model = self.prediction_models[model_key]

            # Very simple prediction: most frequent value seen in this context
            counts = model.get('counts', {})
            if counts:
                prediction = max(counts.items(), key=lambda x: x[1])[0]
                return prediction

        # Fallback: use most recent observation
        return self.observation_history[-1].content

    def _compute_prediction_error(self, observation: ObservationData, prediction: Any) -> float:
        """
        Compute error between prediction and actual observation.

        Args:
            observation: Actual observation
            prediction: Predicted observation

        Returns:
            Error value
        """
        # Handle different types of content
        if observation.embedding is not None and hasattr(prediction, 'embedding'):
            # Use embedding distance if available
            error = np.sqrt(np.sum((observation.embedding - prediction.embedding) ** 2))
        elif isinstance(observation.content, (int, float)) and isinstance(prediction, (int, float)):
            # Numerical value
            error = abs(observation.content - prediction)
        elif isinstance(observation.content, str) and isinstance(prediction, str):
            # String
            error = 1.0 if observation.content != prediction else 0.0
        else:
            # Default: binary match/no-match
            error = 1.0 if observation.content != prediction else 0.0

        return error

    def _update_model(self, observation: ObservationData) -> None:
        """
        Update prediction model with new observation.

        Args:
            observation: New observation
        """
        # Create/update model for this context
        # Use relevant context keys
        context_keys = sorted(observation.context.keys())
        model_key = '|'.join(context_keys)

        if model_key not in self.prediction_models:
            self.prediction_models[model_key] = {
                'counts': {},
                'count': 0
            }

        model = self.prediction_models[model_key]

        # Update counts of observed values
        content_key = str(observation.content)
        if content_key not in model['counts']:
            model['counts'][content_key] = 0
        model['counts'][content_key] += 1
        model['count'] += 1

        # Limit number of models
        if len(self.prediction_models) > 100:  # Arbitrary limit
            # Remove model with fewest observations
            min_model = min(self.prediction_models.items(), key=lambda x: x[1]['count'])
            del self.prediction_models[min_model[0]]

    def reset(self) -> None:
        """Clear history and reset the detector"""
        self.observation_history.clear()
        self.prediction_errors.clear()
        self.prediction_models = {}


class UncertaintyEstimator:
    """
    Estimates epistemic uncertainty for observations.
    Uses ensemble or Bayesian methods for uncertainty quantification.
    """

    def __init__(self, config: IntrinsicMotivationConfig):
        """
        Initialize uncertainty estimator.

        Args:
            config: Configuration parameters
        """
        self.config = config
        self.ensemble_size = config.uncertainty_ensemble_size
        self.threshold = config.uncertainty_threshold

        # Knowledge coverage map (for symbolic approach)
        self.knowledge_map = {}

        # Ensemble models (for neural approach)
        self.ensemble_models = []

        # Logging
        self.logger = logging.getLogger("GrandNexus.Learning.IntrinsicMotivation.Uncertainty")

    def compute_uncertainty(self, observation: ObservationData) -> float:
        """
        Compute uncertainty score for an observation.

        Args:
            observation: The observation to evaluate

        Returns:
            Uncertainty score in [0, 1] range
        """
        # If neural approach is available and observation has embedding
        if HAS_TORCH and observation.embedding is not None and self.ensemble_models:
            uncertainty = self._compute_neural_uncertainty(observation)
        else:
            # Fallback to symbolic approach
            uncertainty = self._compute_symbolic_uncertainty(observation)

        return uncertainty

    def _compute_neural_uncertainty(self, observation: ObservationData) -> float:
        """
        Compute uncertainty using neural ensemble.

        Args:
            observation: Observation to evaluate

        Returns:
            Uncertainty score in [0, 1] range
        """
        if not self.ensemble_models:
            # No models yet, maximum uncertainty
            return 1.0

        # Get predictions from all models
        predictions = []
        for model in self.ensemble_models:
            if HAS_TORCH:
                with torch.no_grad():
                    embedding = torch.FloatTensor(observation.embedding).unsqueeze(0)
                    pred = model(embedding).squeeze(0).numpy()
                    predictions.append(pred)

        if not predictions:
            return 1.0

        # Compute variance across models
        predictions = np.array(predictions)
        variance = np.mean(np.var(predictions, axis=0))

        # Normalize to [0, 1]
        uncertainty = 2.0 / (1.0 + np.exp(-5.0 * variance)) - 1.0

        return float(min(1.0, max(0.0, uncertainty)))

    def _compute_symbolic_uncertainty(self, observation: ObservationData) -> float:
        """
        Compute uncertainty using symbolic knowledge map.

        Args:
            observation: Observation to evaluate

        Returns:
            Uncertainty score in [0, 1] range
        """
        # Generate a key for this observation
        key = self._get_observation_key(observation)

        # Check if we have seen similar observations before
        if key in self.knowledge_map:
            count = self.knowledge_map[key]['count']

            # Uncertainty decreases with count, but never reaches zero
            uncertainty = 1.0 / (1.0 + count)

            # Update count
            self.knowledge_map[key]['count'] += 1
        else:
            # Never seen before, maximum uncertainty
            uncertainty = 1.0

            # Add to knowledge map
            self.knowledge_map[key] = {
                'count': 1,
                'first_seen': time.time()
            }

        return uncertainty

    def _get_observation_key(self, observation: ObservationData) -> str:
        """
        Generate a key for the observation to use in the knowledge map.

        Args:
            observation: Observation to generate key for

        Returns:
            String key
        """
        # Use context and content to generate key
        context_key = '|'.join(f"{k}:{v}" for k, v in sorted(observation.context.items()))

        # Handle different content types
        if isinstance(observation.content, (int, float)):
            # Discretize numerical value
            bin_size = 0.1
            binned = int(observation.content / bin_size) * bin_size
            content_key = f"num:{binned}"
        elif isinstance(observation.content, str):
            content_key = f"str:{observation.content}"
        else:
            content_key = f"obj:{type(observation.content).__name__}"

        return f"{context_key}|{content_key}"

    def update_models(self, observations: List[ObservationData]) -> None:
        """
        Update uncertainty models with new observations.

        Args:
            observations: List of observations to learn from
        """
        # Only update neural models if available
        if not HAS_TORCH or not observations:
            return

        # Ensure all observations have embeddings
        embeddings = [obs.embedding for obs in observations if obs.embedding is not None]
        if not embeddings:
            return

        # Create ensemble if not already created
        if not self.ensemble_models:
            self._create_ensemble(embeddings[0].shape[0])

        # Convert to tensors
        X = torch.FloatTensor(embeddings)

        # Simple "targets" - just predict the embedding itself
        y = X.clone()

        # Train each model slightly differently
        for i, model in enumerate(self.ensemble_models):
            # Add random noise to targets for diversity
            noisy_y = y + torch.randn_like(y) * 0.1 * (i + 1)

            # Simple training (in practice, would use proper training loop)
            optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
            for _ in range(5):  # Just a few iterations for demonstration
                optimizer.zero_grad()
                pred = model(X)
                loss = F.mse_loss(pred, noisy_y)
                loss.backward()
                optimizer.step()

    def _create_ensemble(self, input_dim: int) -> None:
        """
        Create ensemble of models for uncertainty estimation.

        Args:
            input_dim: Dimension of input embeddings
        """
        if not HAS_TORCH:
            return

        # Clear any existing models
        self.ensemble_models = []

        # Create ensemble
        for _ in range(self.ensemble_size):
            # Simple MLP model
            model = nn.Sequential(
                nn.Linear(input_dim, 64),
                nn.ReLU(),
                nn.Linear(64, 64),
                nn.ReLU(),
                nn.Linear(64, input_dim)
            )

            # Initialize weights differently for each model
            for param in model.parameters():
                nn.init.normal_(param, mean=0.0, std=0.02)

            self.ensemble_models.append(model)

    def reset(self) -> None:
        """Clear knowledge and reset the estimator"""
        self.knowledge_map = {}
        self.ensemble_models = []


class CompetenceProgressTracker:
    """
    Tracks learning progress (competence) in different areas.
    Intrinsic motivation arises from improving performance.
    """

    def __init__(self, config: IntrinsicMotivationConfig):
        """
        Initialize competence progress tracker.

        Args:
            config: Configuration parameters
        """
        self.config = config
        self.window_size = config.competence_window_size
        self.smoothing = config.competence_smoothing

        # Track performance history for different skills/areas
        self.performance_history = {}

        # Logging
        self.logger = logging.getLogger("GrandNexus.Learning.IntrinsicMotivation.Competence")

    def compute_competence_progress(self, observation: ObservationData,
                                   performance: float, skill_area: str) -> float:
        """
        Compute competence progress for a skill area.

        Args:
            observation: The context observation
            performance: Performance measure in [0, 1]
            skill_area: Identifier for the skill/area

        Returns:
            Competence progress score in [0, 1] range
        """
        # Initialize history for this skill if needed
        if skill_area not in self.performance_history:
            self.performance_history[skill_area] = {
                'values': deque(maxlen=self.window_size),
                'timestamps': deque(maxlen=self.window_size),
                'smoothed': 0.0,
                'previous_smoothed': 0.0
            }

        history = self.performance_history[skill_area]

        # Update history
        history['values'].append(performance)
        history['timestamps'].append(observation.timestamp)

        # Update smoothed average
        if not history['values']:
            history['smoothed'] = performance
        else:
            # Exponential moving average
            history['previous_smoothed'] = history['smoothed']
            history['smoothed'] = (1 - self.smoothing) * history['smoothed'] + self.smoothing * performance

        # Compute progress (derivative of performance)
        if len(history['values']) < 2:
            # Not enough data yet
            progress = 0.5  # Neutral
        else:
            # Compare current smoothed to previous
            progress_raw = history['smoothed'] - history['previous_smoothed']

            # Normalize to [-1, 1], then to [0, 1]
            progress = (np.tanh(5.0 * progress_raw) + 1.0) / 2.0

        return float(progress)

    def get_high_progress_areas(self, threshold: float = 0.7) -> List[Tuple[str, float]]:
        """
        Get skill areas with high learning progress.

        Args:
            threshold: Minimum progress to consider high

        Returns:
            List of (skill_area, progress) tuples
        """
        high_progress = []

        for skill_area, history in self.performance_history.items():
            if len(history['values']) < 2:
                continue

            # Compute current progress
            progress_raw = history['smoothed'] - history['previous_smoothed']
            progress = (np.tanh(5.0 * progress_raw) + 1.0) / 2.0

            if progress >= threshold:
                high_progress.append((skill_area, float(progress)))

        # Sort by progress (highest first)
        high_progress.sort(key=lambda x: x[1], reverse=True)

        return high_progress

    def get_low_progress_areas(self, threshold: float = 0.3) -> List[Tuple[str, float]]:
        """
        Get skill areas with low learning progress (plateaued).

        Args:
            threshold: Maximum progress to consider low

        Returns:
            List of (skill_area, progress) tuples
        """
        low_progress = []

        for skill_area, history in self.performance_history.items():
            if len(history['values']) < self.window_size // 2:
                # Not enough data yet
                continue

            # Compute current progress
            progress_raw = history['smoothed'] - history['previous_smoothed']
            progress = (np.tanh(5.0 * progress_raw) + 1.0) / 2.0

            if progress <= threshold:
                low_progress.append((skill_area, float(progress)))

        # Sort by progress (lowest first)
        low_progress.sort(key=lambda x: x[1])

        return low_progress

    def reset(self) -> None:
        """Clear history and reset the tracker"""
        self.performance_history = {}


class EmpowermentEstimator:
    """
    Estimates empowerment (information-theoretic measure of control/options).
    This is a simplified implementation of empowerment principles.
    """

    def __init__(self, config: IntrinsicMotivationConfig):
        """
        Initialize empowerment estimator.

        Args:
            config: Configuration parameters
        """
        self.config = config
        self.horizon = config.empowerment_horizon
        self.samples = config.empowerment_samples

        # Model of state transitions (for empowerment simulation)
        self.transition_model = {}

        # Logging
        self.logger = logging.getLogger("GrandNexus.Learning.IntrinsicMotivation.Empowerment")

    def compute_empowerment(self, observation: ObservationData,
                           available_actions: List[Any]) -> float:
        """
        Compute empowerment score for an observation.

        Args:
            observation: The observation to evaluate
            available_actions: List of actions available in this state

        Returns:
            Empowerment score in [0, 1] range
        """
        # If no actions, no empowerment
        if not available_actions:
            return 0.0

        # First-order empowerment: simply number of available actions
        action_count = len(available_actions)
        first_order = min(1.0, action_count / 10.0)  # Normalize to [0, 1]

        # If we have a transition model, estimate higher-order empowerment
        state_key = self._get_state_key(observation)
        if state_key in self.transition_model and self.horizon > 1:
            # Sample action sequences and count unique outcomes
            unique_outcomes = set()

            for _ in range(self.samples):
                # Sample a random action sequence
                current_state = state_key
                for _ in range(self.horizon):
                    if current_state not in self.transition_model:
                        break

                    transitions = self.transition_model[current_state]
                    if not transitions:
                        break

                    # Sample a random action
                    action = random.choice(list(transitions.keys()))

                    # Sample next state from possible outcomes
                    next_states = transitions[action]
                    if not next_states:
                        break

                    # Weight by probability
                    total = sum(prob for _, prob in next_states)
                    if total <= 0:
                        break

                    rand_val = random.random() * total
                    cumulative = 0
                    next_state = None

                    for state, prob in next_states:
                        cumulative += prob
                        if cumulative >= rand_val:
                            next_state = state
                            break

                    if next_state is None:
                        break

                    current_state = next_state

                # Add final state to set of unique outcomes
                unique_outcomes.add(current_state)

            # Empowerment is log of number of unique reachable states
            if unique_outcomes:
                # Normalize to [0, 1] assuming maximum of 2^horizon states
                higher_order = min(1.0, len(unique_outcomes) / (2.0 ** self.horizon))

                # Combine first and higher order
                empowerment = 0.3 * first_order + 0.7 * higher_order
            else:
                empowerment = first_order
        else:
            empowerment = first_order

        return empowerment

    def update_model(self, current_state: ObservationData,
                    action: Any, next_state: ObservationData) -> None:
        """
        Update transition model with observed transition.

        Args:
            current_state: Starting state
            action: Action taken
            next_state: Resulting state
        """
        # Get state keys
        current_key = self._get_state_key(current_state)
        next_key = self._get_state_key(next_state)

        # Initialize model for this state if needed
        if current_key not in self.transition_model:
            self.transition_model[current_key] = {}

        # Initialize actions if needed
        action_key = str(action)
        if action_key not in self.transition_model[current_key]:
            self.transition_model[current_key][action_key] = []

        # Check if we've seen this transition before
        for i, (state, count) in enumerate(self.transition_model[current_key][action_key]):
            if state == next_key:
                # Increment count
                self.transition_model[current_key][action_key][i] = (state, count + 1)
                break
        else:
            # New transition
            self.transition_model[current_key][action_key].append((next_key, 1))

        # Limit model size
        if len(self.transition_model) > 1000:  # Arbitrary limit
            # Remove least recently used states
            # In a real implementation, would track usage timestamps
            keys = list(self.transition_model.keys())
            to_remove = keys[:len(keys)//5]  # Remove 20%
            for key in to_remove:
                del self.transition_model[key]

    def _get_state_key(self, observation: ObservationData) -> str:
        """
        Generate a key for the state.

        Args:
            observation: Observation to generate key for

        Returns:
            String key
        """
        # If embedding available, use a discretized version
        if observation.embedding is not None:
            # Simple discretization: round each dimension to 1 decimal place
            discretized = np.round(observation.embedding * 10) / 10
            key = str(discretized.tolist())
        else:
            # Use content and context
            context_str = '|'.join(f"{k}:{v}" for k, v in sorted(observation.context.items()))
            key = f"{observation.content}|{context_str}"

        return key

    def reset(self) -> None:
        """Clear model and reset the estimator"""
        self.transition_model = {}


class IntrinsicMotivation:
    """
    Main module for intrinsic motivation in GrandNexus.

    Coordinates different intrinsic motivation mechanisms to generate
    curiosity signals that drive exploration and learning.
    """

    def __init__(self, config: Optional[IntrinsicMotivationConfig] = None,
                nexus_core: Optional[NexusCore] = None,
                episodic_memory: Optional[EpisodicMemory] = None,
                working_memory: Optional[WorkingMemory] = None,
                semantic_memory: Optional[SemanticMemory] = None,
                learning_core: Optional[LearningCore] = None):
        """
        Initialize the intrinsic motivation module.

        Args:
            config: Configuration for the module
            nexus_core: Reference to the NexusCore
            episodic_memory: Reference to episodic memory
            working_memory: Reference to working memory
            semantic_memory: Reference to semantic memory
            learning_core: Reference to learning core
        """
        self.logger = logging.getLogger("GrandNexus.Learning.IntrinsicMotivation")

        # Store configuration
        self.config = config or IntrinsicMotivationConfig()

        # Store module references
        self.nexus_core = nexus_core
        self.episodic_memory = episodic_memory
        self.working_memory = working_memory
        self.semantic_memory = semantic_memory
        self.learning_core = learning_core

        # Internal state
        self.instance_id = str(uuid.uuid4())[:8]
        self.lock = threading.RLock()
        self.initialized = False
        self.running = False

        # Current curiosity state
        self.curiosity_state = CuriosityState()

        # Curiosity mechanisms
        self.novelty_detector = None
        self.surprise_detector = None
        self.uncertainty_estimator = None
        self.competence_tracker = None
        self.empowerment_estimator = None

        # Motivation history
        self.recent_motivations = deque(maxlen=100)

        # Weights for different motivation types
        self.motivation_weights = dict(self.config.default_weights)

        # Performance metrics
        self.metrics = defaultdict(lambda: deque(maxlen=100))

        self.logger.info(f"IntrinsicMotivation initialized with ID={self.instance_id}")

    def initialize(self) -> bool:
        """
        Initialize the intrinsic motivation module.

        Returns:
            True if initialization succeeded, False otherwise
        """
        if self.initialized:
            return True

        with self.lock:
            try:
                # Initialize motivation mechanisms
                self._initialize_motivation_mechanisms()

                # Register with NexusCore if available
                if self.nexus_core:
                    self._register_with_nexus_core()

                self.initialized = True
                self.logger.info("IntrinsicMotivation initialization complete")
                return True
            except Exception as e:
                self.logger.error(f"IntrinsicMotivation initialization failed: {str(e)}")
                return False

    def process_observation(self, observation: ObservationData) -> List[MotivationSignal]:
        """
        Process an observation and generate intrinsic motivation signals.

        Args:
            observation: The observation to evaluate

        Returns:
            List of motivation signals
        """
        if not self.initialized:
            self.initialize()

        with self.lock:
            # Generate embeddings if needed
            if observation.embedding is None:
                observation = self._generate_embedding(observation)

            # Generate motivation signals
            signals = []

            # Novelty
            if MotivationType.NOVELTY in self.config.enabled_types:
                novelty = self.novelty_detector.compute_novelty(observation)
                if novelty > 0.3:  # Threshold to generate signal
                    signals.append(MotivationSignal(
                        observation=observation,
                        motivation_type=MotivationType.NOVELTY,
                        strength=novelty,
                        explanation=f"Novel observation (novelty={novelty:.2f})"
                    ))
                self.metrics["novelty"].append(novelty)

            # Surprise
            if MotivationType.SURPRISE in self.config.enabled_types:
                surprise = self.surprise_detector.compute_surprise(observation)
                if surprise > 0.3:  # Threshold to generate signal
                    signals.append(MotivationSignal(
                        observation=observation,
                        motivation_type=MotivationType.SURPRISE,
                        strength=surprise,
                        explanation=f"Surprising observation (surprise={surprise:.2f})"
                    ))
                self.metrics["surprise"].append(surprise)

            # Uncertainty
            if MotivationType.UNCERTAINTY in self.config.enabled_types:
                uncertainty = self.uncertainty_estimator.compute_uncertainty(observation)
                if uncertainty > 0.3:  # Threshold to generate signal
                    signals.append(MotivationSignal(
                        observation=observation,
                        motivation_type=MotivationType.UNCERTAINTY,
                        strength=uncertainty,
                        explanation=f"Uncertain knowledge (uncertainty={uncertainty:.2f})"
                    ))
                self.metrics["uncertainty"].append(uncertainty)

            # Store in memory (both intrinsic and working memory)
            self._store_in_memory(observation, signals)

            # Update curiosity state based on signals
            self._update_curiosity_state(signals)

            # Add to history
            for signal in signals:
                self.recent_motivations.append(signal)

            # Update metrics
            self.metrics["signal_count"].append(len(signals))

            # Log strong signals
            for signal in signals:
                if signal.strength > 0.7:
                    self.logger.info(f"Strong {signal.motivation_type.name} signal: {signal.explanation}")

            return signals

    def update_competence(self, skill_area: str, performance: float,
                         observation: ObservationData) -> Optional[MotivationSignal]:
        """
        Update competence tracking for a skill area and generate motivation signal.

        Args:
            skill_area: Identifier for the skill/area
            performance: Performance measure in [0, 1]
            observation: Contextual observation

        Returns:
            Motivation signal if significant progress detected, None otherwise
        """
        if not self.initialized:
            self.initialize()

        with self.lock:
            if MotivationType.COMPETENCE not in self.config.enabled_types:
                return None

            # Compute competence progress
            progress = self.competence_tracker.compute_competence_progress(
                observation, performance, skill_area)

            self.metrics["competence_progress"].append(progress)

            # Generate signal if significant progress
            if progress > 0.6:  # Threshold for significant progress
                signal = MotivationSignal(
                    observation=observation,
                    motivation_type=MotivationType.COMPETENCE,
                    strength=progress,
                    explanation=f"Learning progress in {skill_area} (progress={progress:.2f})",
                    target=skill_area
                )

                self.recent_motivations.append(signal)
                return signal

            return None

    def update_empowerment(self, observation: ObservationData,
                          actions: List[Any], next_state: Optional[ObservationData] = None,
                          action_taken: Optional[Any] = None) -> Optional[MotivationSignal]:
        """
        Update empowerment model and generate motivation signal.

        Args:
            observation: Current state observation
            actions: Available actions in this state
            next_state: Optional next state (if action was taken)
            action_taken: Optional action that was taken

        Returns:
            Motivation signal if high empowerment detected, None otherwise
        """
        if not self.initialized:
            self.initialize()

        with self.lock:
            if MotivationType.EMPOWERMENT not in self.config.enabled_types:
                return None

            # Update transition model if action was taken
            if next_state is not None and action_taken is not None:
                self.empowerment_estimator.update_model(
                    observation, action_taken, next_state)

            # Compute empowerment
            empowerment = self.empowerment_estimator.compute_empowerment(
                observation, actions)

            self.metrics["empowerment"].append(empowerment)

            # Generate signal if high empowerment
            if empowerment > 0.7:  # Threshold for high empowerment
                signal = MotivationSignal(
                    observation=observation,
                    motivation_type=MotivationType.EMPOWERMENT,
                    strength=empowerment,
                    explanation=f"High control/options (empowerment={empowerment:.2f})"
                )

                self.recent_motivations.append(signal)
                return signal

            return None

    def get_curiosity_state(self) -> CuriosityState:
        """
        Get the current curiosity state.

        Returns:
            Current curiosity state
        """
        with self.lock:
            return copy.deepcopy(self.curiosity_state)

    def get_attention_focus(self) -> Tuple[Any, float]:
        """
        Get the current attention focus and its strength.

        Returns:
            Tuple of (focus_target, strength)
        """
        with self.lock:
            return (self.curiosity_state.attention_focus,
                    self.curiosity_state.focus_strength)

    def get_motivation_weights(self) -> Dict[MotivationType, float]:
        """
        Get the current weights for different motivation types.

        Returns:
            Dictionary mapping motivation types to weights
        """
        with self.lock:
            return dict(self.motivation_weights)

    def set_motivation_weights(self, weights: Dict[MotivationType, float]) -> None:
        """
        Set weights for different motivation types.

        Args:
            weights: Dictionary mapping motivation types to weights
        """
        with self.lock:
            for motivation_type, weight in weights.items():
                if motivation_type in self.motivation_weights:
                    self.motivation_weights[motivation_type] = max(0.0, min(1.0, weight))

    def reset(self) -> None:
        """
        Reset all intrinsic motivation mechanisms.

        This clears all history and model state, but keeps configuration.
        """
        with self.lock:
            # Reset mechanisms
            if self.novelty_detector:
                self.novelty_detector.reset()

            if self.surprise_detector:
                self.surprise_detector.reset()

            if self.uncertainty_estimator:
                self.uncertainty_estimator.reset()

            if self.competence_tracker:
                self.competence_tracker.reset()

            if self.empowerment_estimator:
                self.empowerment_estimator.reset()

            # Reset state
            self.curiosity_state = CuriosityState()
            self.recent_motivations.clear()

            # Reset metrics
            for key in list(self.metrics.keys()):
                self.metrics[key].clear()

            self.logger.info("IntrinsicMotivation reset complete")

    def get_recent_motivations(self) -> List[MotivationSignal]:
        """
        Get recent motivation signals.

        Returns:
            List of recent motivation signals
        """
        with self.lock:
            return list(self.recent_motivations)

    def get_motivation_metrics(self) -> Dict[str, Any]:
        """
        Get metrics for intrinsic motivation performance.

        Returns:
            Dictionary with metrics
        """
        with self.lock:
            metrics = {}

            # Process metric queues
            for key, values in self.metrics.items():
                if values:
                    metrics[key] = {
                        "current": values[-1],
                        "mean": float(np.mean(values)),
                        "max": float(np.max(values)),
                        "min": float(np.min(values))
                    }

            # Add current state info
            metrics["curiosity_state"] = {
                "interest_level": self.curiosity_state.interest_level,
                "focus_strength": self.curiosity_state.focus_strength,
                "novelty_seeking": self.curiosity_state.novelty_seeking,
                "exploration_rate": self.curiosity_state.exploration_rate,
                "active_drives": [drive.name for drive in self.curiosity_state.active_drives]
            }

            # Add weights
            metrics["motivation_weights"] = {k.name: v for k, v in self.motivation_weights.items()}

            return metrics

    # -----------------------------------------------------------------------
    # Internal methods
    # -----------------------------------------------------------------------

    def _initialize_motivation_mechanisms(self) -> None:
        """Initialize the various intrinsic motivation mechanisms"""
        # Novelty detection
        self.novelty_detector = NoveltyDetector(self.config)

        # Surprise detection
        self.surprise_detector = SurpriseDetector(self.config)

        # Uncertainty estimation
        self.uncertainty_estimator = UncertaintyEstimator(self.config)

        # Competence tracking
        self.competence_tracker = CompetenceProgressTracker(self.config)

        # Empowerment estimation
        self.empowerment_estimator = EmpowermentEstimator(self.config)

        self.logger.info("Intrinsic motivation mechanisms initialized")

    def _register_with_nexus_core(self) -> None:
        """Register with NexusCore for system-wide coordination"""
        if not self.nexus_core:
            self.logger.warning("No NexusCore available for registration")
            return

        try:
            # Register as a module with NexusCore
            dependencies = []
            if self.episodic_memory:
                dependencies.append("episodic_memory")
            if self.working_memory:
                dependencies.append("working_memory")
            if self.semantic_memory:
                dependencies.append("semantic_memory")
            if self.learning_core:
                dependencies.append("learning_core")

            self.nexus_core.register_module(
                name="intrinsic_motivation",
                module=self,
                dependencies=dependencies
            )

            self.logger.info("Successfully registered with NexusCore")
        except Exception as e:
            self.logger.error(f"Failed to register with NexusCore: {str(e)}")

    def _generate_embedding(self, observation: ObservationData) -> ObservationData:
        """
        Generate embedding for an observation if not already present.

        Args:
            observation: Observation to generate embedding for

        Returns:
            Observation with embedding added (if possible)
        """
        # If already has embedding, return as is
        if observation.embedding is not None:
            return observation

        # Try to generate embedding
        # This would typically use a neural network or other embedding model
        # For simplicity, we'll just create a random embedding if we can't do better

        # Check if semantic memory can provide an embedding
        if self.semantic_memory is not None and self.config.use_semantic_memory:
            # Try to get embedding from semantic memory
            # This is a placeholder - actual implementation would depend on SemanticMemory API
            try:
                if isinstance(observation.content, str):
                    concept = self.semantic_memory.get_concept(observation.content)
                    if concept and hasattr(concept, 'embedding') and concept.embedding is not None:
                        new_observation = copy.copy(observation)
                        new_observation.embedding = concept.embedding
                        return new_observation
            except Exception as e:
                self.logger.warning(f"Error getting embedding from semantic memory: {str(e)}")

        # Fallback: create simple random embedding (for demonstration only)
        # In a real system, would use a proper embedding model
        dim = self.config.embedding_dimension
        embedding = np.random.randn(dim)
        embedding = embedding / np.linalg.norm(embedding)  # Normalize

        # Create new observation with embedding
        new_observation = copy.copy(observation)
        new_observation.embedding = embedding

        return new_observation

    def _store_in_memory(self, observation: ObservationData,
                        signals: List[MotivationSignal]) -> None:
        """
        Store observation and signals in memory systems.

        Args:
            observation: Observation to store
            signals: Motivation signals generated
        """
        # Store in working memory if available
        if self.working_memory is not None and self.config.use_working_memory:
            try:
                # Store the observation
                item_id = self.working_memory.store_item(
                    content=observation.content,
                    source="intrinsic_motivation",
                    metadata={
                        "context": observation.context,
                        "signals": [
                            {
                                "type": signal.motivation_type.name,
                                "strength": signal.strength,
                                "explanation": signal.explanation
                            }
                            for signal in signals
                        ]
                    },
                    importance=max([0.5] + [signal.strength for signal in signals])
                )
            except Exception as e:
                self.logger.warning(f"Error storing in working memory: {str(e)}")

        # Store in episodic memory if available and signals are significant
        if (self.episodic_memory is not None and self.config.use_episodic_memory and
            any(signal.strength > 0.6 for signal in signals)):
            try:
                # Create an episodic memory event
                event_id = self.episodic_memory.add_event(
                    episode_id="intrinsic_motivation",  # This should be a real episode ID
                    content={
                        "observation": observation.content,
                        "signals": [
                            {
                                "type": signal.motivation_type.name,
                                "strength": signal.strength,
                                "explanation": signal.explanation
                            }
                            for signal in signals
                        ]
                    },
                    metadata={
                        "source": "intrinsic_motivation",
                        "context": observation.context,
                        "importance": max([0.5] + [signal.strength for signal in signals])
                    }
                )
            except Exception as e:
                self.logger.warning(f"Error storing in episodic memory: {str(e)}")

    def _update_curiosity_state(self, signals: List[MotivationSignal]) -> None:
        """
        Update the curiosity state based on new motivation signals.

        Args:
            signals: New motivation signals
        """
        # Default decay of current state
        self.curiosity_state.interest_level *= self.config.attention_decay
        self.curiosity_state.focus_strength *= self.config.attention_decay

        # If no signals, gradually return to baseline
        if not signals:
            baseline = 0.3
            self.curiosity_state.interest_level = (
                0.9 * self.curiosity_state.interest_level + 0.1 * baseline
            )
            return

        # Combine signals according to current weights
        total_strength = 0.0
        weighted_strength = 0.0
        active_drives = set()

        for signal in signals:
            weight = self.motivation_weights.get(signal.motivation_type, 1.0)
            weighted_strength += signal.strength * weight
            total_strength += signal.strength
            active_drives.add(signal.motivation_type)

        # Update interest level (with smoothing)
        if total_strength > 0:
            avg_weighted_strength = weighted_strength / total_strength
            self.curiosity_state.interest_level = (
                0.7 * self.curiosity_state.interest_level +
                0.3 * avg_weighted_strength
            )

        # Update active drives
        self.curiosity_state.active_drives = list(active_drives)

        # Update novelty seeking based on novelty signals
        novelty_signals = [s for s in signals if s.motivation_type == MotivationType.NOVELTY]
        if novelty_signals:
            avg_novelty = sum(s.strength for s in novelty_signals) / len(novelty_signals)
            self.curiosity_state.novelty_seeking = (
                0.8 * self.curiosity_state.novelty_seeking + 0.2 * avg_novelty
            )

        # Update exploration rate based on multiple factors
        # Higher for novelty and uncertainty, lower for competence
        exploration_factors = []

        for signal in signals:
            if signal.motivation_type in [MotivationType.NOVELTY, MotivationType.UNCERTAINTY]:
                exploration_factors.append(signal.strength)
            elif signal.motivation_type == MotivationType.COMPETENCE:
                exploration_factors.append(0.3)  # Lower when making progress (exploiting)

        if exploration_factors:
            avg_factor = sum(exploration_factors) / len(exploration_factors)
            self.curiosity_state.exploration_rate = (
                0.8 * self.curiosity_state.exploration_rate + 0.2 * avg_factor
            )

        # Update attention focus
        self._update_attention_focus(signals)

        # Update timestamp
        self.curiosity_state.timestamp = time.time()

    def _update_attention_focus(self, signals: List[MotivationSignal]) -> None:
        """
        Update the attention focus based on signals.

        Args:
            signals: New motivation signals
        """
        # If no signals, maintain current focus with decay
        if not signals:
            return

        # Find the strongest signal
        strongest = max(signals, key=lambda s: s.strength)

        # If much stronger than current focus, switch
        if (strongest.strength > self.curiosity_state.focus_strength +
            self.config.attention_switch_threshold):
            # Switch focus
            self.curiosity_state.attention_focus = strongest.target or strongest.observation.content
            self.curiosity_state.focus_strength = strongest.strength

            self.logger.info(f"Attention focus switched to {self.curiosity_state.attention_focus} "
                            f"({strongest.motivation_type.name}, "
                            f"strength={strongest.strength:.2f})")
        else:
            # Slightly update focus strength
            self.curiosity_state.focus_strength = (
                0.8 * self.curiosity_state.focus_strength + 0.2 * strongest.strength
            )

    def _adapt_weights(self, feedback: float,
                      motivation_type: MotivationType) -> None:
        """
        Adapt weights based on feedback.

        Args:
            feedback: Feedback value (-1 to 1)
            motivation_type: Type of motivation that led to the action
        """
        if not self.config.adaptive_weights:
            return

        # Update weight for this motivation type
        if motivation_type in self.motivation_weights:
            current = self.motivation_weights[motivation_type]
            # Adjust weight (bounded)
            new_weight = current + self.config.learning_rate * feedback
            self.motivation_weights[motivation_type] = max(0.1, min(1.0, new_weight))

            # Log significant changes
            if abs(new_weight - current) > 0.1:
                self.logger.info(f"Adapted weight for {motivation_type.name}: "
                                f"{current:.2f} -> {self.motivation_weights[motivation_type]:.2f}")

