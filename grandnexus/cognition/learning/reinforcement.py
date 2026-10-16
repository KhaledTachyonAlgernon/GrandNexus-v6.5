"""
reinforcement.py - Reinforcement Learning Module for GrandNexus

This module implements reinforcement learning capabilities for the
GrandNexus cognitive architecture. It provides a flexible foundation
for learning from experience through reward signals, supporting various
policy types and learning algorithms.

Key features:
- Support for multiple RL algorithms (Q-Learning, DQN, PPO, etc.)
- Integration with both symbolic and neural components
- Experience replay for efficient learning
- Policy abstraction for flexible agent behavior
- Exploration strategies with adaptive parameters
- Integration with episodic and semantic memory modules
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

# Try importing PyTorch for deep RL methods, but make it optional
try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

    class nn:  # fix 6.5: fallback without PyTorch (nn.Module annotations)
        Module = object

# Algorithm types
class RLAlgorithm(Enum):
    """Types of reinforcement learning algorithms supported"""
    Q_LEARNING = auto()
    SARSA = auto()
    DQN = auto()
    A2C = auto()
    PPO = auto()
    SAC = auto()
    CUSTOM = auto()

# Exploration strategies
class ExplorationStrategy(Enum):
    """Types of exploration strategies"""
    EPSILON_GREEDY = auto()
    BOLTZMANN = auto()
    UCB = auto()
    THOMPSON = auto()
    NOISY_NETS = auto()
    PARAMETER_NOISE = auto()
    CUSTOM = auto()

@dataclass
class State:
    """
    Representation of an environment state.

    This is deliberately flexible to support both discrete and continuous
    state spaces, as well as both numeric and symbolic representations.
    """
    features: Any  # Can be vector, tensor, dictionary, symbolic state, etc.
    metadata: Dict[str, Any] = field(default_factory=dict)

    # Hash implementation for dictionary key usage
    def __hash__(self):
        if isinstance(self.features, (tuple, list, np.ndarray)):
            return hash(tuple(self.features))
        return hash(str(self.features))

    # Equality for dictionary comparison
    def __eq__(self, other):
        if not isinstance(other, State):
            return False
        return hash(self) == hash(other)

@dataclass
class Action:
    """
    Representation of an agent action.

    Supports both discrete and continuous action spaces,
    as well as both numeric and symbolic representations.
    """
    value: Any  # Can be int, float, vector, symbolic action, etc.
    metadata: Dict[str, Any] = field(default_factory=dict)

    # Hash implementation for dictionary key usage
    def __hash__(self):
        if isinstance(self.value, (tuple, list, np.ndarray)):
            return hash(tuple(self.value))
        return hash(str(self.value))

    # Equality for dictionary comparison
    def __eq__(self, other):
        if not isinstance(other, Action):
            return False
        return hash(self) == hash(other)

@dataclass
class Experience:
    """
    Representation of a single reinforcement learning experience
    (state, action, reward, next_state).
    """
    state: State
    action: Action
    reward: float
    next_state: State
    done: bool = False
    info: Dict[str, Any] = field(default_factory=dict)
    timestamp: float = field(default_factory=time.time)
    importance: float = 0.5  # Priority for experience replay

@dataclass
class RLEnvironment:
    """Abstract representation of an environment for RL"""
    name: str
    state_space: Any  # Description of possible states
    action_space: Any  # Description of possible actions
    max_steps: int = 1000
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class RLPolicy:
    """
    Policy for determining actions based on states.

    A policy can be represented in many ways:
    - Q-table for discrete state/action spaces
    - Neural network for continuous spaces
    - Rule-based for symbolic approaches
    """
    policy_id: str
    policy_type: str  # "q_table", "neural_network", "rule_based", etc.
    parameters: Dict[str, Any] = field(default_factory=dict)
    model: Any = None  # Specific implementation (Q-table, neural net, etc.)
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class RLConfig:
    """Configuration for the reinforcement learning module"""
    algorithm: RLAlgorithm = RLAlgorithm.Q_LEARNING
    learning_rate: float = 0.01
    discount_factor: float = 0.99
    exploration_strategy: ExplorationStrategy = ExplorationStrategy.EPSILON_GREEDY
    initial_exploration_rate: float = 1.0
    final_exploration_rate: float = 0.01
    exploration_decay: float = 0.995
    batch_size: int = 32
    update_frequency: int = 4
    target_update_frequency: int = 1000  # For DQN-like algorithms
    replay_buffer_size: int = 10000
    prioritized_replay: bool = False
    double_q: bool = False  # Whether to use double Q-learning
    use_episodic_memory: bool = True
    use_semantic_memory: bool = True
    neural_network_config: Dict[str, Any] = field(default_factory=lambda: {
        "hidden_layers": [128, 128],
        "activation": "relu"
    })
    custom_parameters: Dict[str, Any] = field(default_factory=dict)

class ExperienceBuffer:
    """
    Buffer for storing and sampling experiences for experience replay.

    Supports both uniform and prioritized sampling.
    """

    def __init__(self, capacity: int = 10000, use_priorities: bool = False,
                alpha: float = 0.6, beta: float = 0.4, beta_increment: float = 0.001):
        """
        Initialize experience buffer.

        Args:
            capacity: Maximum number of experiences to store
            use_priorities: Whether to use prioritized experience replay
            alpha: Priority exponent (how much prioritization to use)
            beta: Importance sampling exponent
            beta_increment: How much to increase beta over time
        """
        self.capacity = capacity
        self.use_priorities = use_priorities
        self.alpha = alpha
        self.beta = beta
        self.beta_increment = beta_increment

        # Buffer storage
        self.buffer = deque(maxlen=capacity)
        self.priorities = np.zeros(capacity, dtype=np.float32)
        self.position = 0
        self.size = 0

        # Stats
        self.add_count = 0
        self.sample_count = 0

    def add(self, experience: Experience) -> None:
        """
        Add an experience to the buffer.

        Args:
            experience: The experience to add
        """
        if self.size < self.capacity:
            self.buffer.append(experience)
            self.size += 1
        else:
            self.buffer[self.position] = experience

        # Set maximum priority for new experiences
        if self.use_priorities:
            max_priority = self.priorities.max() if self.size > 0 else 1.0
            self.priorities[self.position] = max_priority

        # Update position
        self.position = (self.position + 1) % self.capacity
        self.add_count += 1

    def sample(self, batch_size: int) -> Tuple[List[Experience], List[int], np.ndarray]:
        """
        Sample a batch of experiences from the buffer.

        Args:
            batch_size: Number of experiences to sample

        Returns:
            Tuple of (experiences, indices, weights)
        """
        # Ensure we have enough experiences
        batch_size = min(batch_size, self.size)

        # Sample according to priorities if enabled
        if self.use_priorities:
            # Calculate sampling probabilities
            priorities = self.priorities[:self.size]
            probabilities = priorities ** self.alpha
            probabilities /= probabilities.sum()

            # Sample indices
            indices = np.random.choice(self.size, batch_size, replace=False, p=probabilities)

            # Calculate importance sampling weights
            weights = (self.size * probabilities[indices]) ** -self.beta
            weights /= weights.max()

            # Incrementally increase beta
            self.beta = min(1.0, self.beta + self.beta_increment)
        else:
            # Uniform sampling
            indices = np.random.choice(self.size, batch_size, replace=False)
            weights = np.ones(batch_size, dtype=np.float32)

        # Get experiences from buffer
        experiences = [self.buffer[i] for i in indices]

        self.sample_count += 1
        return experiences, indices, weights

    def update_priorities(self, indices: List[int], priorities: List[float]) -> None:
        """
        Update priorities for experiences.

        Args:
            indices: Indices of experiences to update
            priorities: New priorities for these experiences
        """
        if not self.use_priorities:
            return

        for idx, priority in zip(indices, priorities):
            # Ensure positive priority
            self.priorities[idx] = max(1e-6, priority)

    def clear(self) -> None:
        """Clear the buffer"""
        self.buffer.clear()
        self.priorities = np.zeros(self.capacity, dtype=np.float32)
        self.position = 0
        self.size = 0

    def get_stats(self) -> Dict[str, Any]:
        """Get buffer statistics"""
        return {
            "size": self.size,
            "capacity": self.capacity,
            "utilization": self.size / max(1, self.capacity),
            "add_count": self.add_count,
            "sample_count": self.sample_count,
            "beta": self.beta if self.use_priorities else None,
            "use_priorities": self.use_priorities
        }

class ReinforcementLearning:
    """
    Core reinforcement learning module for GrandNexus.

    This module provides the functionality for agents to learn from
    experience through interaction with environments, using various
    reinforcement learning algorithms.
    """

    def __init__(self, config: Optional[RLConfig] = None,
                nexus_core: Optional[NexusCore] = None,
                episodic_memory: Optional[EpisodicMemory] = None,
                working_memory: Optional[WorkingMemory] = None,
                semantic_memory: Optional[SemanticMemory] = None,
                learning_core: Any = None):
        """
        Initialize the reinforcement learning module.

        Args:
            config: Configuration for the RL module
            nexus_core: Reference to the NexusCore
            episodic_memory: Reference to episodic memory (for storing experiences)
            working_memory: Reference to working memory (for active learning processes)
            semantic_memory: Reference to semantic memory (for storing learned policies)
        """
        self.learning_core = learning_core  # fix 6.5
        self.logger = logging.getLogger("GrandNexus.Learning.Reinforcement")

        # Store configuration
        self.config = config or RLConfig()

        # Store module references
        self.nexus_core = nexus_core
        self.episodic_memory = episodic_memory
        self.working_memory = working_memory
        self.semantic_memory = semantic_memory

        # Internal state
        self.instance_id = str(uuid.uuid4())[:8]
        self.lock = threading.RLock()
        self.initialized = False
        self.running = False

        # Storage for environments, policies, and experiences
        self.environments = {}
        self.policies = {}
        self.experience_buffers = {}
        self.active_learning = {}  # Tracks active learning processes

        # Exploration parameters
        self.exploration_rate = self.config.initial_exploration_rate

        # Learning metrics
        self.metrics = defaultdict(lambda: deque(maxlen=100))

        # Create default experience buffer
        self.default_buffer = ExperienceBuffer(
            capacity=self.config.replay_buffer_size,
            use_priorities=self.config.prioritized_replay
        )

        self.logger.info(f"ReinforcementLearning initialized with ID={self.instance_id}")

    def initialize(self) -> bool:
        """
        Initialize the reinforcement learning module.

        Returns:
            True if initialization succeeded, False otherwise
        """
        if self.initialized:
            return True

        with self.lock:
            try:
                # Register with NexusCore if available
                if self.nexus_core:
                    self._register_with_nexus_core()

                # Initialize algorithm-specific components
                self._initialize_algorithm()

                self.initialized = True
                self.logger.info("ReinforcementLearning initialization complete")
                return True
            except Exception as e:
                self.logger.error(f"ReinforcementLearning initialization failed: {str(e)}")
                return False

    def register_environment(self, environment: RLEnvironment) -> None:
        """
        Register an environment for reinforcement learning.

        Args:
            environment: The environment to register
        """
        with self.lock:
            self.environments[environment.name] = environment
            self.logger.info(f"Registered environment: {environment.name}")

    def create_policy(self, name: str, policy_type: str, environment_name: str,
                     parameters: Optional[Dict[str, Any]] = None) -> str:
        """
        Create a new policy for a registered environment.

        Args:
            name: Name for the policy
            policy_type: Type of policy ("q_table", "neural_network", etc.)
            environment_name: Name of the environment this policy is for
            parameters: Additional parameters for the policy

        Returns:
            Policy ID string
        """
        with self.lock:
            if environment_name not in self.environments:
                raise ValueError(f"Environment {environment_name} not registered")

            # Generate policy ID
            policy_id = f"{name}_{str(uuid.uuid4())[:8]}"

            # Create policy based on type
            if policy_type == "q_table":
                model = self._create_q_table(self.environments[environment_name])
            elif policy_type == "neural_network":
                if not HAS_TORCH:
                    raise ImportError("PyTorch is required for neural network policies")
                model = self._create_neural_network(self.environments[environment_name])
            elif policy_type == "rule_based":
                model = {}  # Placeholder for rule-based policy
            else:
                raise ValueError(f"Unsupported policy type: {policy_type}")

            # Create policy object
            policy = RLPolicy(
                policy_id=policy_id,
                policy_type=policy_type,
                parameters=parameters or {},
                model=model,
                metadata={"environment_name": environment_name}
            )

            # Store policy
            self.policies[policy_id] = policy

            # Create experience buffer for this policy if one doesn't exist
            if policy_id not in self.experience_buffers:
                self.experience_buffers[policy_id] = ExperienceBuffer(
                    capacity=self.config.replay_buffer_size,
                    use_priorities=self.config.prioritized_replay
                )

            self.logger.info(f"Created policy {name} ({policy_id}) for environment {environment_name}")
            return policy_id

    def select_action(self, policy_id: str, state: State) -> Action:
        """
        Select an action according to the given policy and state.

        Args:
            policy_id: ID of the policy to use
            state: Current state of the environment

        Returns:
            Selected action
        """
        if policy_id not in self.policies:
            raise ValueError(f"Policy {policy_id} not found")

        policy = self.policies[policy_id]

        # Check if exploration should be used
        if random.random() < self.exploration_rate:
            # Explore: select random action
            environment_name = policy.metadata["environment_name"]
            environment = self.environments[environment_name]

            # Handle different action space types
            if isinstance(environment.action_space, list):
                action_value = random.choice(environment.action_space)
            elif isinstance(environment.action_space, range):
                action_value = random.choice(list(environment.action_space))
            elif hasattr(environment.action_space, 'sample'):
                # OpenAI Gym-like interface
                action_value = environment.action_space.sample()
            else:
                # Fallback: assume it's a number of discrete actions
                try:
                    num_actions = int(environment.action_space)
                    action_value = random.randint(0, num_actions - 1)
                except (ValueError, TypeError):
                    raise ValueError(f"Unsupported action space: {environment.action_space}")

            action = Action(value=action_value, metadata={"exploration": True})
        else:
            # Exploit: select best action according to policy
            if policy.policy_type == "q_table":
                action = self._select_action_q_table(policy, state)
            elif policy.policy_type == "neural_network":
                action = self._select_action_neural_network(policy, state)
            elif policy.policy_type == "rule_based":
                action = self._select_action_rule_based(policy, state)
            else:
                raise ValueError(f"Unsupported policy type: {policy.policy_type}")

        return action

    def store_experience(self, policy_id: str, experience: Experience) -> None:
        """
        Store an experience for later learning.

        Args:
            policy_id: ID of the policy that generated the experience
            experience: The experience to store
        """
        with self.lock:
            # Store in policy-specific buffer
            if policy_id in self.experience_buffers:
                self.experience_buffers[policy_id].add(experience)

            # Also store in default buffer
            self.default_buffer.add(experience)

            # Store in episodic memory if available
            if self.episodic_memory is not None and self.config.use_episodic_memory:
                self._store_in_episodic_memory(experience)

    def update_policy(self, policy_id: str, experiences: Optional[List[Experience]] = None) -> Dict[str, Any]:
        """
        Update a policy based on stored experiences.

        Args:
            policy_id: ID of the policy to update
            experiences: Optional specific experiences to learn from
                         (if None, samples from buffer)

        Returns:
            Dictionary with update metrics
        """
        with self.lock:
            if policy_id not in self.policies:
                raise ValueError(f"Policy {policy_id} not found")

            policy = self.policies[policy_id]

            # Get experiences
            if experiences is None:
                # Sample from buffer
                buffer = self.experience_buffers.get(policy_id, self.default_buffer)
                if buffer.size < self.config.batch_size:
                    # Not enough experiences yet
                    return {"updated": False, "reason": "insufficient_data"}

                experiences, indices, weights = buffer.sample(self.config.batch_size)
            else:
                # Use provided experiences
                indices = None
                weights = np.ones(len(experiences))

            # Update policy based on type
            if policy.policy_type == "q_table":
                metrics = self._update_q_table(policy, experiences, weights)
            elif policy.policy_type == "neural_network":
                metrics = self._update_neural_network(policy, experiences, weights)
            elif policy.policy_type == "rule_based":
                metrics = self._update_rule_based(policy, experiences, weights)
            else:
                raise ValueError(f"Unsupported policy type: {policy.policy_type}")

            # Update priorities if using prioritized replay
            if self.config.prioritized_replay and indices is not None:
                buffer = self.experience_buffers.get(policy_id, self.default_buffer)
                if "td_errors" in metrics:
                    priorities = np.abs(metrics["td_errors"]) + 1e-6  # Avoid zero priority
                    buffer.update_priorities(indices, priorities)

            # Update metrics
            self._update_metrics(policy_id, metrics)

            # Update exploration rate
            self._update_exploration_rate()

            # Store updated policy in semantic memory if available
            if self.semantic_memory is not None and self.config.use_semantic_memory:
                self._store_policy_in_semantic_memory(policy)

            return {"updated": True, **metrics}

    def train(self, policy_id: str, environment_name: str, episodes: int = 100,
             max_steps_per_episode: Optional[int] = None,
             step_callback: Optional[Callable[[int, int, Dict[str, Any]], None]] = None,
             episode_callback: Optional[Callable[[int, Dict[str, Any]], None]] = None) -> Dict[str, Any]:
        """
        Train a policy on an environment for a specified number of episodes.

        Args:
            policy_id: ID of the policy to train
            environment_name: Name of the environment to train on
            episodes: Number of episodes to train for
            max_steps_per_episode: Maximum steps per episode (overrides environment default)
            step_callback: Optional callback after each step
            episode_callback: Optional callback after each episode

        Returns:
            Dictionary with training metrics
        """
        if not self.initialized:
            self.initialize()

        if policy_id not in self.policies:
            raise ValueError(f"Policy {policy_id} not found")

        if environment_name not in self.environments:
            raise ValueError(f"Environment {environment_name} not registered")

        environment = self.environments[environment_name]

        # Check if this environment has a step function
        if not hasattr(environment, 'step') or not callable(environment.step):
            raise ValueError(f"Environment {environment_name} does not have a step method")

        # Initialize learning process
        process_id = f"{policy_id}_{environment_name}_{str(uuid.uuid4())[:8]}"
        self.active_learning[process_id] = {
            "policy_id": policy_id,
            "environment_name": environment_name,
            "episodes": episodes,
            "current_episode": 0,
            "start_time": time.time(),
            "status": "running",
            "metrics": {}
        }

        # Set maximum steps
        max_steps = max_steps_per_episode or environment.max_steps

        # Training metrics
        total_rewards = []
        episode_lengths = []

        try:
            # Training loop
            for episode in range(episodes):
                # Reset environment
                state = environment.reset()

                # Convert to State object if necessary
                if not isinstance(state, State):
                    state = State(features=state)

                # Episode variables
                episode_reward = 0
                episode_steps = 0
                done = False

                # Episode loop
                while not done and episode_steps < max_steps:
                    # Select action
                    action = self.select_action(policy_id, state)

                    # Take action in environment
                    next_state, reward, done, info = environment.step(action.value)

                    # Convert to State object if necessary
                    if not isinstance(next_state, State):
                        next_state = State(features=next_state)

                    # Create experience
                    experience = Experience(
                        state=state,
                        action=action,
                        reward=reward,
                        next_state=next_state,
                        done=done,
                        info=info
                    )

                    # Store experience
                    self.store_experience(policy_id, experience)

                    # Update policy
                    if episode_steps % self.config.update_frequency == 0:
                        self.update_policy(policy_id)

                    # Update state
                    state = next_state

                    # Update episode metrics
                    episode_reward += reward
                    episode_steps += 1

                    # Call step callback if provided
                    if step_callback:
                        step_callback(episode, episode_steps, {
                            "reward": reward,
                            "done": done,
                            "policy_id": policy_id,
                            "environment_name": environment_name
                        })

                # Update episode metrics
                total_rewards.append(episode_reward)
                episode_lengths.append(episode_steps)

                # Update training process status
                self.active_learning[process_id]["current_episode"] = episode + 1
                self.active_learning[process_id]["metrics"] = {
                    "mean_reward": np.mean(total_rewards[-100:]),
                    "mean_episode_length": np.mean(episode_lengths[-100:]),
                    "last_reward": episode_reward,
                    "last_episode_length": episode_steps,
                    "exploration_rate": self.exploration_rate
                }

                # Call episode callback if provided
                if episode_callback:
                    episode_callback(episode, {
                        "reward": episode_reward,
                        "steps": episode_steps,
                        "policy_id": policy_id,
                        "environment_name": environment_name
                    })

            # Mark training process as completed
            self.active_learning[process_id]["status"] = "completed"
            self.active_learning[process_id]["end_time"] = time.time()

            # Return training metrics
            return {
                "process_id": process_id,
                "status": "completed",
                "episodes_completed": episodes,
                "total_steps": sum(episode_lengths),
                "mean_reward": np.mean(total_rewards),
                "mean_episode_length": np.mean(episode_lengths),
                "rewards": total_rewards,
                "episode_lengths": episode_lengths
            }

        except Exception as e:
            # Mark training process as failed
            self.active_learning[process_id]["status"] = "failed"
            self.active_learning[process_id]["error"] = str(e)
            self.active_learning[process_id]["end_time"] = time.time()

            self.logger.error(f"Training failed: {str(e)}")

            # Re-raise exception
            raise

    def evaluate(self, policy_id: str, environment_name: str, episodes: int = 10,
                max_steps_per_episode: Optional[int] = None) -> Dict[str, Any]:
        """
        Evaluate a policy on an environment.

        Args:
            policy_id: ID of the policy to evaluate
            environment_name: Name of the environment to evaluate on
            episodes: Number of episodes to evaluate for
            max_steps_per_episode: Maximum steps per episode (overrides environment default)

        Returns:
            Dictionary with evaluation metrics
        """
        if not self.initialized:
            self.initialize()

        if policy_id not in self.policies:
            raise ValueError(f"Policy {policy_id} not found")

        if environment_name not in self.environments:
            raise ValueError(f"Environment {environment_name} not registered")

        environment = self.environments[environment_name]

        # Check if this environment has a step function
        if not hasattr(environment, 'step') or not callable(environment.step):
            raise ValueError(f"Environment {environment_name} does not have a step method")

        # Set maximum steps
        max_steps = max_steps_per_episode or environment.max_steps

        # Temporarily disable exploration
        original_exploration_rate = self.exploration_rate
        self.exploration_rate = 0.0

        # Evaluation metrics
        total_rewards = []
        episode_lengths = []

        try:
            # Evaluation loop
            for episode in range(episodes):
                # Reset environment
                state = environment.reset()

                # Convert to State object if necessary
                if not isinstance(state, State):
                    state = State(features=state)

                # Episode variables
                episode_reward = 0
                episode_steps = 0
                done = False

                # Episode loop
                while not done and episode_steps < max_steps:
                    # Select action (greedy policy)
                    action = self.select_action(policy_id, state)

                    # Take action in environment
                    next_state, reward, done, info = environment.step(action.value)

                    # Convert to State object if necessary
                    if not isinstance(next_state, State):
                        next_state = State(features=next_state)

                    # Update state
                    state = next_state

                    # Update episode metrics
                    episode_reward += reward
                    episode_steps += 1

                # Update evaluation metrics
                total_rewards.append(episode_reward)
                episode_lengths.append(episode_steps)

            # Restore exploration rate
            self.exploration_rate = original_exploration_rate

            # Return evaluation metrics
            return {
                "episodes": episodes,
                "total_steps": sum(episode_lengths),
                "mean_reward": np.mean(total_rewards),
                "mean_episode_length": np.mean(episode_lengths),
                "rewards": total_rewards,
                "episode_lengths": episode_lengths,
                "min_reward": np.min(total_rewards),
                "max_reward": np.max(total_rewards),
                "std_reward": np.std(total_rewards)
            }

        except Exception as e:
            # Restore exploration rate
            self.exploration_rate = original_exploration_rate

            self.logger.error(f"Evaluation failed: {str(e)}")

            # Re-raise exception
            raise

    def save_policy(self, policy_id: str, filepath: str) -> bool:
        """
        Save a policy to disk.

        Args:
            policy_id: ID of the policy to save
            filepath: Path to save the policy to

        Returns:
            True if successful, False otherwise
        """
        with self.lock:
            if policy_id not in self.policies:
                raise ValueError(f"Policy {policy_id} not found")

            try:
                policy = self.policies[policy_id]

                # Handle different policy types
                if policy.policy_type == "q_table":
                    # For Q-table, we can just save the table
                    np.save(filepath, policy.model)
                elif policy.policy_type == "neural_network" and HAS_TORCH:
                    # For neural network, save the state dict
                    torch.save(policy.model.state_dict(), filepath)
                elif policy.policy_type == "rule_based":
                    # For rule-based, save rules as JSON
                    import json
                    with open(filepath, 'w') as f:
                        json.dump(policy.model, f)
                else:
                    raise ValueError(f"Saving not implemented for policy type: {policy.policy_type}")

                self.logger.info(f"Saved policy {policy_id} to {filepath}")
                return True
            except Exception as e:
                self.logger.error(f"Failed to save policy {policy_id}: {str(e)}")
                return False

    def load_policy(self, policy_id: str, filepath: str) -> bool:
        """
        Load a policy from disk.

        Args:
            policy_id: ID of the policy to load into
            filepath: Path to load the policy from

        Returns:
            True if successful, False otherwise
        """
        with self.lock:
            if policy_id not in self.policies:
                raise ValueError(f"Policy {policy_id} not found")

            try:
                policy = self.policies[policy_id]

                # Handle different policy types
                if policy.policy_type == "q_table":
                    # For Q-table, load the table
                    policy.model = np.load(filepath)
                elif policy.policy_type == "neural_network" and HAS_TORCH:
                    # For neural network, load the state dict
                    policy.model.load_state_dict(torch.load(filepath))
                    policy.model.eval()  # Set to evaluation mode
                elif policy.policy_type == "rule_based":
                    # For rule-based, load rules from JSON
                    import json
                    with open(filepath, 'r') as f:
                        policy.model = json.load(f)
                else:
                    raise ValueError(f"Loading not implemented for policy type: {policy.policy_type}")

                self.logger.info(f"Loaded policy {policy_id} from {filepath}")
                return True
            except Exception as e:
                self.logger.error(f"Failed to load policy {policy_id}: {str(e)}")
                return False

    def get_policy_stats(self, policy_id: str) -> Dict[str, Any]:
        """
        Get statistics for a policy.

        Args:
            policy_id: ID of the policy to get statistics for

        Returns:
            Dictionary with policy statistics
        """
        with self.lock:
            if policy_id not in self.policies:
                raise ValueError(f"Policy {policy_id} not found")

            policy = self.policies[policy_id]

            # Basic stats for all policy types
            stats = {
                "policy_id": policy_id,
                "policy_type": policy.policy_type,
                "environment": policy.metadata.get("environment_name", "unknown")
            }

            # Policy-specific stats
            if policy.policy_type == "q_table":
                # Stats for Q-table
                q_table = policy.model
                stats.update({
                    "states": q_table.shape[0],
                    "actions": q_table.shape[1],
                    "min_q": float(np.min(q_table)),
                    "max_q": float(np.max(q_table)),
                    "mean_q": float(np.mean(q_table)),
                    "nonzero_states": int(np.sum(np.any(q_table != 0, axis=1)))
                })
            elif policy.policy_type == "neural_network" and HAS_TORCH:
                # Stats for neural network
                stats.update({
                    "model_structure": str(policy.model),
                    "parameter_count": sum(p.numel() for p in policy.model.parameters())
                })
            elif policy.policy_type == "rule_based":
                # Stats for rule-based
                stats.update({
                    "rule_count": len(policy.model) if isinstance(policy.model, dict) else 0
                })

            # Add buffer stats if available
            if policy_id in self.experience_buffers:
                stats["buffer"] = self.experience_buffers[policy_id].get_stats()

            return stats

    def get_active_learning_status(self, process_id: Optional[str] = None) -> Dict[str, Any]:
        """
        Get status of active learning processes.

        Args:
            process_id: Optional specific process ID to get status for

        Returns:
            Dictionary with learning process status
        """
        with self.lock:
            if process_id is not None:
                if process_id not in self.active_learning:
                    raise ValueError(f"Learning process {process_id} not found")
                return self.active_learning[process_id]
            else:
                return {pid: status for pid, status in self.active_learning.items()}

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
            if self.episodic_memory:
                dependencies.append("episodic_memory")
            if self.working_memory:
                dependencies.append("working_memory")
            if self.semantic_memory:
                dependencies.append("semantic_memory")

            self.nexus_core.register_module(
                name="reinforcement",
                module=self,
                dependencies=dependencies
            )

            self.logger.info("Successfully registered with NexusCore")
        except Exception as e:
            self.logger.error(f"Failed to register with NexusCore: {str(e)}")

    def _initialize_algorithm(self) -> None:
        """Initialize components specific to the selected algorithm"""
        algorithm = self.config.algorithm

        # Algorithm-specific initialization
        if algorithm == RLAlgorithm.DQN and HAS_TORCH:
            # Initialize target network for DQN
            self.target_network = None
            self.logger.info("Initialized DQN components")
        elif algorithm == RLAlgorithm.PPO and HAS_TORCH:
            # Initialize PPO components
            self.logger.info("Initialized PPO components")

        self.logger.info(f"Initialized algorithm: {algorithm.name}")

    def _create_q_table(self, environment: RLEnvironment) -> np.ndarray:
        """
        Create a Q-table for an environment.

        Args:
            environment: The environment to create a Q-table for

        Returns:
            Numpy array representing the Q-table
        """
        # Determine state space size
        if isinstance(environment.state_space, int):
            # Discrete state space
            state_size = environment.state_space
        elif isinstance(environment.state_space, list) and len(environment.state_space) > 0:
            # List of discrete states
            state_size = len(environment.state_space)
        elif isinstance(environment.state_space, tuple) and len(environment.state_space) == 2:
            # Continuous range (min, max) - discretize
            state_size = 10  # Default discretization
        else:
            # Fallback
            state_size = 10
            self.logger.warning(f"Using default state size of {state_size} for Q-table")

        # Determine action space size
        if isinstance(environment.action_space, int):
            # Discrete action space
            action_size = environment.action_space
        elif isinstance(environment.action_space, list):
            # List of discrete actions
            action_size = len(environment.action_space)
        else:
            # Fallback
            action_size = 4  # Common default
            self.logger.warning(f"Using default action size of {action_size} for Q-table")

        # Create Q-table
        q_table = np.zeros((state_size, action_size))

        return q_table

    def _create_neural_network(self, environment: RLEnvironment) -> nn.Module:
        """
        Create a neural network policy for an environment.

        Args:
            environment: The environment to create a neural network for

        Returns:
            PyTorch neural network
        """
        if not HAS_TORCH:
            raise ImportError("PyTorch is required for neural network policies")

        # Determine input size (state dimensions)
        if hasattr(environment.state_space, 'shape'):
            # Gym-like interface
            input_size = np.prod(environment.state_space.shape)
        elif isinstance(environment.state_space, int):
            # Discrete state space
            input_size = environment.state_space
        elif isinstance(environment.state_space, tuple) and len(environment.state_space) > 0:
            # Tuple of dimensions
            input_size = np.prod(environment.state_space)
        else:
            # Fallback
            input_size = 4  # Common default
            self.logger.warning(f"Using default input size of {input_size} for neural network")

        # Determine output size (action dimensions)
        if hasattr(environment.action_space, 'n'):
            # Gym-like discrete action space
            output_size = environment.action_space.n
        elif isinstance(environment.action_space, int):
            # Discrete action space
            output_size = environment.action_space
        elif isinstance(environment.action_space, list):
            # List of discrete actions
            output_size = len(environment.action_space)
        else:
            # Fallback
            output_size = 2  # Common default
            self.logger.warning(f"Using default output size of {output_size} for neural network")

        # Get neural network configuration
        nn_config = self.config.neural_network_config
        hidden_layers = nn_config.get("hidden_layers", [128, 128])
        activation_name = nn_config.get("activation", "relu")

        # Map activation name to function
        activation_map = {
            "relu": nn.ReLU(),
            "tanh": nn.Tanh(),
            "sigmoid": nn.Sigmoid(),
            "leaky_relu": nn.LeakyReLU()
        }
        activation = activation_map.get(activation_name, nn.ReLU())

        # Create model
        layers = []

        # Input layer
        prev_size = int(input_size)

        # Hidden layers
        for size in hidden_layers:
            layers.append(nn.Linear(prev_size, size))
            layers.append(activation)
            prev_size = size

        # Output layer
        layers.append(nn.Linear(prev_size, output_size))

        # Create model
        model = nn.Sequential(*layers)

        return model

    def _select_action_q_table(self, policy: RLPolicy, state: State) -> Action:
        """
        Select action using Q-table policy.

        Args:
            policy: The policy to use
            state: Current state

        Returns:
            Selected action
        """
        q_table = policy.model

        # Map state to Q-table index
        if isinstance(state.features, (int, np.integer)):
            # Direct index
            state_idx = state.features
        elif isinstance(state.features, (list, tuple, np.ndarray)):
            # For simplicity, just take the first element
            # In a real implementation, you'd need a proper state discretization method
            state_idx = int(state.features[0]) if len(state.features) > 0 else 0
        else:
            # Fallback
            state_idx = 0

        # Ensure index is within bounds
        state_idx = max(0, min(state_idx, q_table.shape[0] - 1))

        # Get action with highest Q-value
        action_idx = np.argmax(q_table[state_idx])

        # Get corresponding action value
        environment_name = policy.metadata["environment_name"]
        environment = self.environments[environment_name]

        if isinstance(environment.action_space, list):
            action_value = environment.action_space[action_idx]
        else:
            action_value = action_idx

        return Action(value=action_value, metadata={"q_value": float(q_table[state_idx][action_idx])})

    def _select_action_neural_network(self, policy: RLPolicy, state: State) -> Action:
        """
        Select action using neural network policy.

        Args:
            policy: The policy to use
            state: Current state

        Returns:
            Selected action
        """
        if not HAS_TORCH:
            raise ImportError("PyTorch is required for neural network policies")

        # Convert state to tensor
        if isinstance(state.features, (list, tuple, np.ndarray)):
            # Convert to tensor
            state_tensor = torch.FloatTensor(state.features)
        elif isinstance(state.features, (int, float)):
            # Single value
            state_tensor = torch.FloatTensor([state.features])
        else:
            # Fallback
            state_tensor = torch.zeros(1)
            self.logger.warning(f"Unable to convert state to tensor: {state.features}")

        # Get action values from model
        model = policy.model
        model.eval()  # Set to evaluation mode
        with torch.no_grad():
            action_values = model(state_tensor)

        # Get action with highest value
        action_idx = torch.argmax(action_values).item()

        # Get corresponding action value
        environment_name = policy.metadata["environment_name"]
        environment = self.environments[environment_name]

        if isinstance(environment.action_space, list):
            action_value = environment.action_space[action_idx]
        else:
            action_value = action_idx

        return Action(value=action_value, metadata={"q_value": float(action_values[action_idx])})

    def _select_action_rule_based(self, policy: RLPolicy, state: State) -> Action:
        """
        Select action using rule-based policy.

        Args:
            policy: The policy to use
            state: Current state

        Returns:
            Selected action
        """
        rules = policy.model

        # This is a simplified implementation
        # In a real system, you'd have a more sophisticated rule engine

        environment_name = policy.metadata["environment_name"]
        environment = self.environments[environment_name]

        # Default action (fallback)
        if isinstance(environment.action_space, list):
            default_action = environment.action_space[0]
        else:
            default_action = 0

        # Check if there are rules for this state
        state_key = str(state.features)
        if state_key in rules:
            action_value = rules[state_key]
        else:
            action_value = default_action

        return Action(value=action_value, metadata={"rule_based": True})

    def _update_q_table(self, policy: RLPolicy, experiences: List[Experience],
                      weights: np.ndarray) -> Dict[str, Any]:
        """
        Update Q-table policy based on experiences.

        Args:
            policy: The policy to update
            experiences: List of experiences to learn from
            weights: Importance weights for experiences

        Returns:
            Dictionary with update metrics
        """
        q_table = policy.model
        learning_rate = self.config.learning_rate
        discount_factor = self.config.discount_factor

        # Metrics
        td_errors = []

        # Process each experience
        for i, experience in enumerate(experiences):
            # Extract components
            state = experience.state
            action = experience.action
            reward = experience.reward
            next_state = experience.next_state
            done = experience.done
            weight = weights[i]

            # Map states to Q-table indices
            if isinstance(state.features, (int, np.integer)):
                state_idx = state.features
            elif isinstance(state.features, (list, tuple, np.ndarray)):
                state_idx = int(state.features[0]) if len(state.features) > 0 else 0
            else:
                state_idx = 0

            if isinstance(next_state.features, (int, np.integer)):
                next_state_idx = next_state.features
            elif isinstance(next_state.features, (list, tuple, np.ndarray)):
                next_state_idx = int(next_state.features[0]) if len(next_state.features) > 0 else 0
            else:
                next_state_idx = 0

            # Ensure indices are within bounds
            state_idx = max(0, min(state_idx, q_table.shape[0] - 1))
            next_state_idx = max(0, min(next_state_idx, q_table.shape[0] - 1))

            # Map action to Q-table index
            if isinstance(action.value, (int, np.integer)):
                action_idx = action.value
            elif isinstance(action.value, (list, tuple, np.ndarray)):
                action_idx = int(action.value[0]) if len(action.value) > 0 else 0
            else:
                # Try to map to index in environment action space
                environment_name = policy.metadata["environment_name"]
                environment = self.environments[environment_name]

                if isinstance(environment.action_space, list):
                    try:
                        action_idx = environment.action_space.index(action.value)
                    except ValueError:
                        action_idx = 0
                else:
                    action_idx = 0

            # Ensure action index is within bounds
            action_idx = max(0, min(action_idx, q_table.shape[1] - 1))

            # Get current Q-value
            current_q = q_table[state_idx, action_idx]

            # Compute target Q-value
            if done:
                target_q = reward
            else:
                # Standard Q-learning target
                target_q = reward + discount_factor * np.max(q_table[next_state_idx])

            # Compute TD error
            td_error = target_q - current_q
            td_errors.append(td_error)

            # Update Q-value with weighted learning
            q_table[state_idx, action_idx] += learning_rate * weight * td_error

        # Return metrics
        return {
            "td_errors": np.array(td_errors),
            "mean_td_error": float(np.mean(np.abs(td_errors))),
            "max_td_error": float(np.max(np.abs(td_errors))),
            "q_table_mean": float(np.mean(q_table)),
            "q_table_max": float(np.max(q_table)),
            "q_table_min": float(np.min(q_table))
        }

    def _update_neural_network(self, policy: RLPolicy, experiences: List[Experience],
                             weights: np.ndarray) -> Dict[str, Any]:
        """
        Update neural network policy based on experiences.

        Args:
            policy: The policy to update
            experiences: List of experiences to learn from
            weights: Importance weights for experiences

        Returns:
            Dictionary with update metrics
        """
        if not HAS_TORCH:
            raise ImportError("PyTorch is required for neural network policies")

        model = policy.model
        learning_rate = self.config.learning_rate
        discount_factor = self.config.discount_factor

        # Create optimizer if not already present
        if not hasattr(policy, 'optimizer'):
            policy.optimizer = optim.Adam(model.parameters(), lr=learning_rate)

        # Create target network for DQN if needed and not already present
        if self.config.algorithm == RLAlgorithm.DQN and not hasattr(policy, 'target_network'):
            policy.target_network = copy.deepcopy(model)
            policy.target_update_counter = 0

        # Prepare batches
        batch_states = []
        batch_actions = []
        batch_rewards = []
        batch_next_states = []
        batch_dones = []

        # Process each experience
        for experience in experiences:
            # Extract components
            state = experience.state
            action = experience.action
            reward = experience.reward
            next_state = experience.next_state
            done = experience.done

            # Convert states to tensors
            if isinstance(state.features, (list, tuple, np.ndarray)):
                state_tensor = state.features
            else:
                state_tensor = [state.features]

            if isinstance(next_state.features, (list, tuple, np.ndarray)):
                next_state_tensor = next_state.features
            else:
                next_state_tensor = [next_state.features]

            # Convert action to index
            if isinstance(action.value, (int, np.integer)):
                action_idx = action.value
            else:
                # Try to map to index in environment action space
                environment_name = policy.metadata["environment_name"]
                environment = self.environments[environment_name]

                if isinstance(environment.action_space, list):
                    try:
                        action_idx = environment.action_space.index(action.value)
                    except ValueError:
                        action_idx = 0
                else:
                    action_idx = 0

            # Add to batches
            batch_states.append(state_tensor)
            batch_actions.append(action_idx)
            batch_rewards.append(reward)
            batch_next_states.append(next_state_tensor)
            batch_dones.append(float(done))

        # Convert batches to tensors
        states = torch.FloatTensor(batch_states)
        actions = torch.LongTensor(batch_actions)
        rewards = torch.FloatTensor(batch_rewards)
        next_states = torch.FloatTensor(batch_next_states)
        dones = torch.FloatTensor(batch_dones)
        weights = torch.FloatTensor(weights)

        # Get current Q values
        model.train()  # Set to training mode
        current_q = model(states)
        current_q = current_q.gather(1, actions.unsqueeze(1)).squeeze(1)

        # Compute target Q values
        with torch.no_grad():
            if self.config.algorithm == RLAlgorithm.DQN:
                # DQN target
                if self.config.double_q:
                    # Double Q-learning
                    next_actions = model(next_states).max(1)[1].unsqueeze(1)
                    next_q = policy.target_network(next_states).gather(1, next_actions).squeeze(1)
                else:
                    # Standard DQN
                    next_q = policy.target_network(next_states).max(1)[0]
            else:
                # Standard Q-learning
                next_q = model(next_states).max(1)[0]

            target_q = rewards + (1 - dones) * discount_factor * next_q

        # Compute loss (weighted MSE)
        td_errors = target_q - current_q
        loss = (weights * td_errors.pow(2)).mean()

        # Update model
        policy.optimizer.zero_grad()
        loss.backward()
        policy.optimizer.step()

        # Update target network if needed
        if self.config.algorithm == RLAlgorithm.DQN:
            policy.target_update_counter += 1
            if policy.target_update_counter >= self.config.target_update_frequency:
                policy.target_network.load_state_dict(model.state_dict())
                policy.target_update_counter = 0

        # Return metrics
        return {
            "loss": float(loss.item()),
            "td_errors": td_errors.detach().numpy(),
            "mean_td_error": float(td_errors.abs().mean().item()),
            "max_td_error": float(td_errors.abs().max().item())
        }

    def _update_rule_based(self, policy: RLPolicy, experiences: List[Experience],
                         weights: np.ndarray) -> Dict[str, Any]:
        """
        Update rule-based policy based on experiences.

        Args:
            policy: The policy to update
            experiences: List of experiences to learn from
            weights: Importance weights for experiences

        Returns:
            Dictionary with update metrics
        """
        # This is a simplified implementation
        # In a real system, you'd have a more sophisticated rule learning mechanism

        rules = policy.model

        # Process each experience
        updated_rules = 0
        for experience in experiences:
            # Extract components
            state = experience.state
            action = experience.action
            reward = experience.reward

            # Only update rules for positive rewards
            if reward > 0:
                # Create rule for this state
                state_key = str(state.features)
                rules[state_key] = action.value
                updated_rules += 1

        # Return metrics
        return {
            "updated_rules": updated_rules,
            "total_rules": len(rules)
        }

    def _store_in_episodic_memory(self, experience: Experience) -> None:
        """
        Store an experience in episodic memory.

        Args:
            experience: The experience to store
        """
        if self.episodic_memory is None:
            return

        try:
            # Create an episodic memory event
            self.episodic_memory.add_event(
                episode_id="reinforcement_learning",  # This should be a real episode ID
                content={
                    "type": "experience",
                    "state": experience.state.features,
                    "action": experience.action.value,
                    "reward": experience.reward,
                    "next_state": experience.next_state.features,
                    "done": experience.done
                },
                metadata={
                    "source": "reinforcement_learning",
                    "timestamp": experience.timestamp,
                    "importance": experience.importance
                }
            )
        except Exception as e:
            self.logger.error(f"Failed to store experience in episodic memory: {str(e)}")

    def _store_policy_in_semantic_memory(self, policy: RLPolicy) -> None:
        """
        Store a policy in semantic memory.

        Args:
            policy: The policy to store
        """
        if self.semantic_memory is None:
            return

        try:
            # Store policy metadata
            self.semantic_memory.add_concept({
                "name": f"policy_{policy.policy_id}",
                "description": f"Reinforcement learning policy of type {policy.policy_type}",
                "properties": {
                    "policy_id": policy.policy_id,
                    "policy_type": policy.policy_type,
                    "environment": policy.metadata.get("environment_name", "unknown"),
                    "last_updated": time.time()
                }
            })
        except Exception as e:
            self.logger.error(f"Failed to store policy in semantic memory: {str(e)}")

    def _update_metrics(self, policy_id: str, update_metrics: Dict[str, Any]) -> None:
        """
        Update learning metrics.

        Args:
            policy_id: ID of the policy being updated
            update_metrics: Metrics from the update
        """
        # Update global metrics
        for key, value in update_metrics.items():
            if isinstance(value, (int, float)):
                metric_key = f"{policy_id}_{key}"
                self.metrics[metric_key].append(value)

        # Also track exploration rate
        self.metrics["exploration_rate"].append(self.exploration_rate)

    def _update_exploration_rate(self) -> None:
        """Update the exploration rate according to the decay schedule"""
        # Simple exponential decay
        self.exploration_rate = max(
            self.config.final_exploration_rate,
            self.exploration_rate * self.config.exploration_decay
        )

# --- fix 6.5: alias of the name expected by other modules
ReinforcementLearner = ReinforcementLearning
