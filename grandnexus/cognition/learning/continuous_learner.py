"""
continuous_learner.py - Continuous Learning Module for GrandNexus

This module provides continuous learning capabilities for GrandNexus, allowing the system
to learn from a stream of data over time while mitigating catastrophic forgetting.
It leverages the Avalanche framework for continual learning strategies.

The module supports:
- Sequential task learning with various continual learning strategies
- Dynamic model adaptation to new data without catastrophic forgetting
- Experience replay from episodic memory
- Knowledge consolidation into semantic memory
- Performance monitoring across tasks

Key design principles:
- Adaptivity: Models maintain performance on previous tasks while learning new ones
- Efficiency: Strategies balance computational resources with forgetting prevention
- Integration: Deep connection with memory systems for experience storage and retrieval
- Emergence: Complex learning behaviors emerge from simple learning rules
"""

import logging
import threading
import time
import uuid
from typing import Dict, List, Any, Optional, Tuple, Set, Union, Callable
from dataclasses import dataclass, field
from enum import Enum, auto
from collections import defaultdict, deque
import copy
import random

# Import from GrandNexus core modules
from grandnexus.core.nexus_core import NexusCore
from grandnexus.memory.working import WorkingMemory
from grandnexus.memory.semantic import SemanticMemory
from grandnexus.memory.episodic import EpisodicMemory
from grandnexus.learning.learning_core import LearningCore, LearningParadigm, LearningTask, LearningResult

# Try to import Avalanche
try:
    import torch
    import numpy as np
    from torch.utils.data import Dataset, DataLoader

    # Avalanche imports
    import avalanche
    from avalanche.benchmarks import NCScenario, ni_benchmark
    from avalanche.training.strategies import Naive, CWRStar, GEM, AGEM, EWC, LwF, SynapticIntelligence
    from avalanche.models import SimpleMLP, MTSimpleMLP, SimpleCNN
    from avalanche.evaluation.metrics import forgetting_metrics, accuracy_metrics, loss_metrics

    HAS_AVALANCHE = True
except ImportError:
    HAS_AVALANCHE = False

# Define continual learning strategies
class CLStrategy(Enum):
    """Continual learning strategies available in the system"""
    NAIVE = auto()          # Naive fine-tuning (baseline, prone to forgetting)
    EWC = auto()            # Elastic Weight Consolidation
    LWF = auto()            # Learning without Forgetting
    SI = auto()             # Synaptic Intelligence
    GEM = auto()            # Gradient Episodic Memory
    AGEM = auto()           # Averaged GEM
    REPLAY = auto()         # Experience Replay
    CWR_STAR = auto()       # CWR*
    DER = auto()            # Dark Experience Replay
    CUSTOM = auto()         # Custom strategy

# Define task types
class TaskType(Enum):
    """Types of continual learning tasks"""
    CLASSIFICATION = auto()  # Standard classification task
    REGRESSION = auto()      # Regression task
    EMBEDDING = auto()       # Learning embeddings
    GENERATION = auto()      # Generative task
    CUSTOM = auto()          # Custom task type

@dataclass
class CLExperience:
    """
    Represents a learning experience (task) in continual learning.

    An experience is a set of data for a particular task or time period
    that the model learns from before moving to the next experience.
    """
    experience_id: str
    task_id: int  # Numeric task identifier used by Avalanche
    name: str
    data: Any  # The actual data for this experience
    metadata: Dict[str, Any] = field(default_factory=dict)
    timestamp: float = field(default_factory=time.time)

    # Metrics for this experience
    metrics: Dict[str, float] = field(default_factory=dict)

    # For tracking train/eval history
    train_history: List[Dict[str, float]] = field(default_factory=list)
    eval_history: List[Dict[str, float]] = field(default_factory=list)

    def add_train_metric(self, metrics: Dict[str, float]) -> None:
        """Add training metrics for this experience"""
        self.train_history.append({
            'timestamp': time.time(),
            **metrics
        })

        # Update latest metrics
        self.metrics.update(metrics)

    def add_eval_metric(self, metrics: Dict[str, float]) -> None:
        """Add evaluation metrics for this experience"""
        self.eval_history.append({
            'timestamp': time.time(),
            **metrics
        })

        # Update latest metrics
        self.metrics.update(metrics)

@dataclass
class CLScenario:
    """
    Represents a continual learning scenario.

    A scenario consists of a sequence of experiences (tasks) that the
    model encounters, along with configuration for how learning should
    proceed through these experiences.
    """
    scenario_id: str
    name: str
    task_type: TaskType
    experiences: List[CLExperience] = field(default_factory=list)
    avalanche_scenario: Optional[Any] = None  # Avalanche's NCScenario
    strategy: CLStrategy = CLStrategy.EWC
    strategy_params: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)

    # State tracking
    current_experience_idx: int = -1
    model: Any = None  # PyTorch model
    cl_strategy: Any = None  # Avalanche strategy instance

    creation_time: float = field(default_factory=time.time)
    last_updated: float = field(default_factory=time.time)

    @property
    def current_experience(self) -> Optional[CLExperience]:
        """Get the current experience being learned"""
        if 0 <= self.current_experience_idx < len(self.experiences):
            return self.experiences[self.current_experience_idx]
        return None

    def advance_to_next_experience(self) -> bool:
        """Move to the next experience in the scenario"""
        if self.current_experience_idx + 1 < len(self.experiences):
            self.current_experience_idx += 1
            self.last_updated = time.time()
            return True
        return False

    def has_more_experiences(self) -> bool:
        """Check if there are more experiences to learn"""
        return self.current_experience_idx + 1 < len(self.experiences)

class GrandNexusDataset(Dataset):
    """
    Custom PyTorch Dataset for GrandNexus data.

    This dataset adapts data from GrandNexus memory systems
    to a format usable by PyTorch and Avalanche.
    """
    def __init__(self, data: List[Tuple[Any, Any]], transform=None, target_transform=None):
        """
        Initialize the dataset.

        Args:
            data: List of (input, target) tuples
            transform: Optional transform for inputs
            target_transform: Optional transform for targets
        """
        self.data = data
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self) -> int:
        """Get the number of samples in the dataset"""
        return len(self.data)

    def __getitem__(self, idx: int) -> Tuple[Any, Any]:
        """
        Get a sample from the dataset.

        Args:
            idx: Index of the sample

        Returns:
            Tuple of (input, target)
        """
        x, y = self.data[idx]

        if self.transform:
            x = self.transform(x)

        if self.target_transform:
            y = self.target_transform(y)

        return x, y

class ContinuousLearner:
    """
    ContinuousLearner module for GrandNexus.

    This module implements continual learning using the Avalanche framework,
    allowing the system to learn from a stream of data while mitigating
    catastrophic forgetting. It integrates with GrandNexus memory systems
    for experience storage and retrieval.
    """

    def __init__(self, learning_core: Optional[LearningCore] = None,
                 nexus_core: Optional[NexusCore] = None,
                 working_memory: Optional[WorkingMemory] = None,
                 semantic_memory: Optional[SemanticMemory] = None,
                 episodic_memory: Optional[EpisodicMemory] = None):
        """
        Initialize the ContinuousLearner module.

        Args:
            learning_core: Reference to the main LearningCore module
            nexus_core: Reference to the NexusCore for system integration
            working_memory: Reference to working memory for context
            semantic_memory: Reference to semantic memory for knowledge storage
            episodic_memory: Reference to episodic memory for experience storage
        """
        self.logger = logging.getLogger("GrandNexus.Learning.ContinuousLearner")

        if not HAS_AVALANCHE:
            self.logger.warning("Avalanche framework not available. Limited functionality.")

        # Store references to other modules
        self.learning_core = learning_core
        self.nexus_core = nexus_core
        self.working_memory = working_memory
        self.semantic_memory = semantic_memory
        self.episodic_memory = episodic_memory

        # Internal state
        self.instance_id = str(uuid.uuid4())[:8]
        self.running = False
        self.initialized = False
        self.lock = threading.RLock()

        # Scenarios and models
        self.scenarios: Dict[str, CLScenario] = {}
        self.active_scenarios: Set[str] = set()

        # Strategy factories
        self.strategy_factories = {}

        # Background learning thread
        self.learning_thread = None
        self.stop_learning = threading.Event()

        # Performance metrics
        self.metrics = defaultdict(lambda: deque(maxlen=100))

        self.logger.info(f"ContinuousLearner initialized with ID={self.instance_id}")

        # Device for PyTorch (CPU/GPU)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu") if HAS_AVALANCHE else None

    def initialize(self) -> bool:
        """Initialize the continuous learner module"""
        if self.initialized:
            return True

        with self.lock:
            try:
                # Register strategy factories
                if HAS_AVALANCHE:
                    self._register_strategy_factories()

                # Register with NexusCore if available
                if self.nexus_core:
                    self._register_with_nexus_core()

                self.initialized = True
                self.logger.info("ContinuousLearner initialization complete")
                return True
            except Exception as e:
                self.logger.error(f"ContinuousLearner initialization failed: {str(e)}")
                return False

    def start(self) -> bool:
        """Start the continuous learner module"""
        if not self.initialized:
            if not self.initialize():
                return False

        with self.lock:
            if self.running:
                return True

            # Start background learning thread
            self.stop_learning.clear()
            self.learning_thread = threading.Thread(target=self._learning_loop, daemon=True)
            self.learning_thread.start()

            self.running = True
            self.logger.info("ContinuousLearner started")
            return True

    def stop(self) -> bool:
        """Stop the continuous learner module"""
        with self.lock:
            if not self.running:
                return True

            # Stop background learning thread
            self.stop_learning.set()
            if self.learning_thread:
                self.learning_thread.join(timeout=5.0)
                self.learning_thread = None

            self.running = False
            self.logger.info("ContinuousLearner stopped")
            return True

    def create_scenario(self, name: str, task_type: TaskType,
                      strategy: CLStrategy = CLStrategy.EWC,
                      strategy_params: Optional[Dict[str, Any]] = None,
                      metadata: Optional[Dict[str, Any]] = None) -> str:
        """
        Create a new continual learning scenario.

        Args:
            name: Name of the scenario
            task_type: Type of task (classification, regression, etc.)
            strategy: Continual learning strategy to use
            strategy_params: Parameters for the strategy
            metadata: Additional information about the scenario

        Returns:
            Scenario ID
        """
        if not HAS_AVALANCHE:
            self.logger.error("Cannot create scenario: Avalanche framework not available")
            return ""

        with self.lock:
            scenario_id = f"scenario_{uuid.uuid4().hex[:8]}"

            if strategy_params is None:
                strategy_params = {}

            if metadata is None:
                metadata = {}

            scenario = CLScenario(
                scenario_id=scenario_id,
                name=name,
                task_type=task_type,
                strategy=strategy,
                strategy_params=strategy_params,
                metadata=metadata
            )

            self.scenarios[scenario_id] = scenario
            self.logger.info(f"Created scenario {scenario_id}: {name}")

            return scenario_id

    def add_experience(self, scenario_id: str, name: str, data: Any,
                     task_id: Optional[int] = None,
                     metadata: Optional[Dict[str, Any]] = None) -> Optional[str]:
        """
        Add a learning experience to a scenario.

        Args:
            scenario_id: ID of the scenario to add experience to
            name: Name of the experience
            data: Data for the experience (X_train, y_train, X_val, y_val)
            task_id: Task ID for the experience (auto-assigned if None)
            metadata: Additional information about the experience

        Returns:
            Experience ID if successful, None otherwise
        """
        if not HAS_AVALANCHE:
            self.logger.error("Cannot add experience: Avalanche framework not available")
            return None

        with self.lock:
            scenario = self.scenarios.get(scenario_id)
            if not scenario:
                self.logger.warning(f"Scenario {scenario_id} not found")
                return None

            # Assign task ID if not provided
            if task_id is None:
                task_id = len(scenario.experiences)

            # Create experience
            experience_id = f"exp_{scenario_id}_{uuid.uuid4().hex[:6]}"

            if metadata is None:
                metadata = {}

            experience = CLExperience(
                experience_id=experience_id,
                task_id=task_id,
                name=name,
                data=data,
                metadata=metadata
            )

            # Add to scenario
            scenario.experiences.append(experience)
            scenario.last_updated = time.time()

            # If this is the first experience, set current_experience_idx to 0
            if scenario.current_experience_idx == -1 and len(scenario.experiences) == 1:
                scenario.current_experience_idx = 0

            # Reset Avalanche scenario to force regeneration
            scenario.avalanche_scenario = None

            self.logger.info(f"Added experience {experience_id} to scenario {scenario_id}")
            return experience_id

    def initialize_model(self, scenario_id: str, model_config: Optional[Dict[str, Any]] = None) -> bool:
        """
        Initialize a model for a scenario.

        Args:
            scenario_id: ID of the scenario to initialize model for
            model_config: Configuration for the model

        Returns:
            True if successful, False otherwise
        """
        if not HAS_AVALANCHE:
            self.logger.error("Cannot initialize model: Avalanche framework not available")
            return False

        with self.lock:
            scenario = self.scenarios.get(scenario_id)
            if not scenario:
                self.logger.warning(f"Scenario {scenario_id} not found")
                return False

            if not scenario.experiences:
                self.logger.warning(f"Scenario {scenario_id} has no experiences")
                return False

            # Default model config
            if model_config is None:
                model_config = {}

            # Get example data from first experience to determine input/output shapes
            first_exp = scenario.experiences[0]
            example_data = first_exp.data

            try:
                # Extract input shape and output size from example data
                if isinstance(example_data, dict):
                    X = example_data.get("X_train") or example_data.get("inputs")
                    y = example_data.get("y_train") or example_data.get("targets")
                elif isinstance(example_data, tuple) and len(example_data) >= 2:
                    X, y = example_data[:2]
                else:
                    raise ValueError("Invalid data format in experience")

                # Determine input shape
                if isinstance(X, torch.Tensor):
                    input_shape = tuple(X.shape[1:])  # Remove batch dimension
                elif isinstance(X, np.ndarray):
                    input_shape = tuple(X.shape[1:])
                else:
                    # Try to convert to numpy
                    X = np.array(X)
                    input_shape = tuple(X.shape[1:])

                # Determine output size
                if isinstance(y, torch.Tensor):
                    if len(y.shape) > 1 and y.shape[1] > 1:
                        # One-hot encoded
                        output_size = y.shape[1]
                    else:
                        # Class indices
                        output_size = len(torch.unique(y))
                elif isinstance(y, np.ndarray):
                    if len(y.shape) > 1 and y.shape[1] > 1:
                        output_size = y.shape[1]
                    else:
                        output_size = len(np.unique(y))
                else:
                    # Try to determine output size
                    unique_classes = set(y)
                    output_size = len(unique_classes)

                # Create appropriate model based on task type and data
                if scenario.task_type == TaskType.CLASSIFICATION:
                    # Determine if we need multi-head
                    multi_task = model_config.get("multi_task", False)

                    # For images (assuming channel-first)
                    if len(input_shape) == 3:
                        # Use CNN for image data
                        channels = input_shape[0]
                        model = SimpleCNN(channels, output_size)
                    else:
                        # Use MLP for other data
                        input_size = int(np.prod(input_shape))
                        hidden_size = model_config.get("hidden_size", 256)
                        hidden_layers = model_config.get("hidden_layers", 2)

                        if multi_task:
                            scenario.model = MTSimpleMLP(
                                input_size=input_size,
                                hidden_size=hidden_size,
                                num_layers=hidden_layers,
                                output_size=output_size
                            )
                        else:
                            scenario.model = SimpleMLP(
                                input_size=input_size,
                                hidden_size=hidden_size,
                                num_layers=hidden_layers,
                                output_size=output_size
                            )

                elif scenario.task_type == TaskType.REGRESSION:
                    # MLP for regression
                    input_size = int(np.prod(input_shape))
                    hidden_size = model_config.get("hidden_size", 256)
                    hidden_layers = model_config.get("hidden_layers", 2)

                    scenario.model = SimpleMLP(
                        input_size=input_size,
                        hidden_size=hidden_size,
                        num_layers=hidden_layers,
                        output_size=output_size
                    )

                # Move model to device
                scenario.model = scenario.model.to(self.device)

                self.logger.info(f"Initialized model for scenario {scenario_id}")
                return True

            except Exception as e:
                self.logger.error(f"Error initializing model for scenario {scenario_id}: {str(e)}")
                return False

    def setup_avalanche_scenario(self, scenario_id: str) -> bool:
        """
        Set up an Avalanche scenario from a GrandNexus scenario.

        Args:
            scenario_id: ID of the scenario to set up

        Returns:
            True if successful, False otherwise
        """
        if not HAS_AVALANCHE:
            self.logger.error("Cannot set up Avalanche scenario: Avalanche framework not available")
            return False

        with self.lock:
            scenario = self.scenarios.get(scenario_id)
            if not scenario:
                self.logger.warning(f"Scenario {scenario_id} not found")
                return False

            if not scenario.experiences:
                self.logger.warning(f"Scenario {scenario_id} has no experiences")
                return False

            if scenario.avalanche_scenario is not None:
                # Already set up
                return True

            try:
                # Create train/eval datasets for each experience
                train_datasets = []
                eval_datasets = []
                task_labels = []

                for exp in scenario.experiences:
                    data = exp.data

                    # Extract train/eval data
                    if isinstance(data, dict):
                        X_train = data.get("X_train") or data.get("inputs")
                        y_train = data.get("y_train") or data.get("targets")
                        X_val = data.get("X_val") or data.get("validation_inputs")
                        y_val = data.get("y_val") or data.get("validation_targets")
                    elif isinstance(data, tuple) and len(data) >= 2:
                        if len(data) >= 4:
                            X_train, y_train, X_val, y_val = data[:4]
                        else:
                            X_train, y_train = data[:2]
                            X_val, y_val = None, None
                    else:
                        raise ValueError(f"Invalid data format in experience {exp.experience_id}")

                    # Convert to tensors if needed
                    if not isinstance(X_train, torch.Tensor):
                        X_train = torch.tensor(X_train, dtype=torch.float32)
                    if not isinstance(y_train, torch.Tensor):
                        if scenario.task_type == TaskType.CLASSIFICATION:
                            y_train = torch.tensor(y_train, dtype=torch.long)
                        else:
                            y_train = torch.tensor(y_train, dtype=torch.float32)

                    # Create train dataset
                    train_data = [(x, y) for x, y in zip(X_train, y_train)]
                    train_dataset = GrandNexusDataset(train_data)
                    train_datasets.append(train_dataset)

                    # Create eval dataset if available
                    if X_val is not None and y_val is not None:
                        if not isinstance(X_val, torch.Tensor):
                            X_val = torch.tensor(X_val, dtype=torch.float32)
                        if not isinstance(y_val, torch.Tensor):
                            if scenario.task_type == TaskType.CLASSIFICATION:
                                y_val = torch.tensor(y_val, dtype=torch.long)
                            else:
                                y_val = torch.tensor(y_val, dtype=torch.float32)

                        eval_data = [(x, y) for x, y in zip(X_val, y_val)]
                        eval_dataset = GrandNexusDataset(eval_data)
                        eval_datasets.append(eval_dataset)
                    else:
                        # Use train data for eval if no separate eval data
                        eval_datasets.append(train_dataset)

                    # Task labels
                    task_labels.append(exp.task_id)

                # Create Avalanche scenario
                scenario.avalanche_scenario = ni_benchmark(
                    train_datasets=train_datasets,
                    test_datasets=eval_datasets,
                    task_labels=task_labels,
                    complete_test_set_only=False
                )

                self.logger.info(f"Set up Avalanche scenario for {scenario_id} with {len(scenario.experiences)} experiences")
                return True

            except Exception as e:
                self.logger.error(f"Error setting up Avalanche scenario for {scenario_id}: {str(e)}")
                return False

    def setup_cl_strategy(self, scenario_id: str) -> bool:
        """
        Set up a continual learning strategy for a scenario.

        Args:
            scenario_id: ID of the scenario to set up strategy for

        Returns:
            True if successful, False otherwise
        """
        if not HAS_AVALANCHE:
            self.logger.error("Cannot set up CL strategy: Avalanche framework not available")
            return False

        with self.lock:
            scenario = self.scenarios.get(scenario_id)
            if not scenario:
                self.logger.warning(f"Scenario {scenario_id} not found")
                return False

            if scenario.model is None:
                self.logger.warning(f"Scenario {scenario_id} has no model initialized")
                return False

            if scenario.avalanche_scenario is None:
                if not self.setup_avalanche_scenario(scenario_id):
                    return False

            try:
                # Get strategy factory
                strategy_factory = self.strategy_factories.get(scenario.strategy)
                if strategy_factory is None:
                    self.logger.error(f"No factory for strategy {scenario.strategy}")
                    return False

                # Create strategy
                strategy_params = scenario.strategy_params.copy()

                # Add common parameters
                strategy_params["model"] = scenario.model
                strategy_params["optimizer"] = torch.optim.Adam(scenario.model.parameters(), lr=0.001)

                if scenario.task_type == TaskType.CLASSIFICATION:
                    strategy_params["criterion"] = torch.nn.CrossEntropyLoss()
                else:  # Regression
                    strategy_params["criterion"] = torch.nn.MSELoss()

                # Create strategy
                scenario.cl_strategy = strategy_factory(**strategy_params)

                self.logger.info(f"Set up {scenario.strategy} strategy for scenario {scenario_id}")
                return True

            except Exception as e:
                self.logger.error(f"Error setting up CL strategy for {scenario_id}: {str(e)}")
                return False

    def train_current_experience(self, scenario_id: str,
                               train_epochs: int = 1) -> Dict[str, float]:
        """
        Train the model on the current experience in a scenario.

        Args:
            scenario_id: ID of the scenario to train
            train_epochs: Number of epochs to train for

        Returns:
            Training metrics
        """
        if not HAS_AVALANCHE:
            self.logger.error("Cannot train: Avalanche framework not available")
            return {}

        with self.lock:
            scenario = self.scenarios.get(scenario_id)
            if not scenario:
                self.logger.warning(f"Scenario {scenario_id} not found")
                return {}

            if scenario.current_experience_idx < 0 or scenario.current_experience_idx >= len(scenario.experiences):
                self.logger.warning(f"No current experience for scenario {scenario_id}")
                return {}

            if scenario.model is None:
                self.logger.warning(f"No model initialized for scenario {scenario_id}")
                return {}

            if scenario.cl_strategy is None:
                if not self.setup_cl_strategy(scenario_id):
                    return {}

            try:
                # Get current experience
                current_exp = scenario.current_experience
                avalanche_exp = scenario.avalanche_scenario.train_stream[scenario.current_experience_idx]

                self.logger.info(f"Training on experience {current_exp.name} ({scenario_id})")

                # Train
                metrics = {}
                scenario.cl_strategy.train(
                    experiences=avalanche_exp,
                    eval_streams=[],
                    num_workers=0,
                    eval_every=-1
                )

                # Extract metrics
                for name, value in scenario.cl_strategy.evaluator.get_last_metrics().items():
                    metrics[name] = float(value)

                # Add training metrics to experience
                current_exp.add_train_metric(metrics)

                # Update global metrics
                for name, value in metrics.items():
                    self.metrics[f"train_{name}"].append(value)

                self.logger.info(f"Completed training on experience {current_exp.name} ({scenario_id})")
                return metrics

            except Exception as e:
                self.logger.error(f"Error training experience for {scenario_id}: {str(e)}")
                return {}

    def evaluate_on_experiences(self, scenario_id: str,
                              experience_indices: Optional[List[int]] = None) -> Dict[str, Dict[str, float]]:
        """
        Evaluate the model on specific experiences in a scenario.

        Args:
            scenario_id: ID of the scenario to evaluate
            experience_indices: Indices of experiences to evaluate on (None for all)

        Returns:
            Evaluation metrics for each experience
        """
        if not HAS_AVALANCHE:
            self.logger.error("Cannot evaluate: Avalanche framework not available")
            return {}

        with self.lock:
            scenario = self.scenarios.get(scenario_id)
            if not scenario:
                self.logger.warning(f"Scenario {scenario_id} not found")
                return {}

            if scenario.model is None:
                self.logger.warning(f"No model initialized for scenario {scenario_id}")
                return {}

            if scenario.cl_strategy is None:
                self.logger.warning(f"No strategy initialized for scenario {scenario_id}")
                return {}

            try:
                # Determine which experiences to evaluate on
                if experience_indices is None:
                    # Evaluate on all experiences up to the current one
                    experience_indices = list(range(scenario.current_experience_idx + 1))

                # Filter valid indices
                experience_indices = [idx for idx in experience_indices if 0 <= idx < len(scenario.experiences)]

                if not experience_indices:
                    self.logger.warning(f"No valid experience indices for scenario {scenario_id}")
                    return {}

                results = {}

                # Evaluate on each experience
                for idx in experience_indices:
                    exp = scenario.experiences[idx]
                    avalanche_exp = scenario.avalanche_scenario.test_stream[idx]

                    self.logger.info(f"Evaluating on experience {exp.name} ({scenario_id})")

                    # Evaluate
                    metrics = scenario.cl_strategy.eval(avalanche_exp).metrics

                    # Extract metrics
                    exp_metrics = {}
                    for name, value in metrics.items():
                        if isinstance(value, torch.Tensor):
                            exp_metrics[name] = float(value.item())
                        else:
                            exp_metrics[name] = float(value)

                    # Add evaluation metrics to experience
                    exp.add_eval_metric(exp_metrics)

                    # Add to results
                    results[exp.experience_id] = exp_metrics

                    # Update global metrics
                    for name, value in exp_metrics.items():
                        self.metrics[f"eval_{name}"].append(value)

                self.logger.info(f"Completed evaluation for scenario {scenario_id}")
                return results

            except Exception as e:
                self.logger.error(f"Error evaluating experiences for {scenario_id}: {str(e)}")
                return {}

    def train_full_scenario(self, scenario_id: str, train_epochs: int = 1,
                          evaluate_after_each: bool = True) -> Dict[str, Any]:
        """
        Train through all experiences in a scenario sequentially.

        Args:
            scenario_id: ID of the scenario to train
            train_epochs: Number of epochs to train for each experience
            evaluate_after_each: Whether to evaluate after each experience

        Returns:
            Training summary
        """
        if not HAS_AVALANCHE:
            self.logger.error("Cannot train: Avalanche framework not available")
            return {}

        with self.lock:
            scenario = self.scenarios.get(scenario_id)
            if not scenario:
                self.logger.warning(f"Scenario {scenario_id} not found")
                return {}

            if not scenario.experiences:
                self.logger.warning(f"Scenario {scenario_id} has no experiences")
                return {}

            if scenario.model is None:
                # Initialize model
                if not self.initialize_model(scenario_id):
                    return {}

            # Reset current experience
            scenario.current_experience_idx = 0

            # Set up strategy
            if scenario.cl_strategy is None:
                if not self.setup_cl_strategy(scenario_id):
                    return {}

            self.logger.info(f"Starting training for full scenario {scenario_id}")

            # Add to active scenarios
            self.active_scenarios.add(scenario_id)

            summary = {
                "scenario_id": scenario_id,
                "experiences": [],
                "forgetting": {},
                "final_accuracy": {}
            }

            try:
                # Train each experience
                while scenario.current_experience_idx < len(scenario.experiences):
                    exp = scenario.current_experience

                    # Train on current experience
                    train_metrics = self.train_current_experience(scenario_id, train_epochs)

                    # Evaluate if requested
                    eval_metrics = {}
                    if evaluate_after_each:
                        eval_results = self.evaluate_on_experiences(scenario_id)

                        # Extract metrics for current experience
                        if exp.experience_id in eval_results:
                            eval_metrics = eval_results[exp.experience_id]

                            # Update summary with accuracy
                            if "Top1_Acc_Stream/eval_phase/test_stream" in eval_metrics:
                                summary["final_accuracy"][exp.experience_id] = eval_metrics["Top1_Acc_Stream/eval_phase/test_stream"]

                    # Record experience results
                    summary["experiences"].append({
                        "experience_id": exp.experience_id,
                        "name": exp.name,
                        "train_metrics": train_metrics,
                        "eval_metrics": eval_metrics
                    })

                    # Move to next experience
                    if not scenario.advance_to_next_experience():
                        break

                # Evaluate forgetting
                if evaluate_after_each and len(scenario.experiences) > 1:
                    # Evaluate on all experiences to measure forgetting
                    all_eval_results = self.evaluate_on_experiences(scenario_id)

                    # Calculate forgetting for each previously learned experience
                    for i in range(len(scenario.experiences) - 1):
                        exp = scenario.experiences[i]

                        if exp.experience_id in all_eval_results:
                            current_acc = all_eval_results[exp.experience_id].get("Top1_Acc_Stream/eval_phase/test_stream", 0.0)
                            max_acc = max(
                                metrics.get("Top1_Acc_Stream/eval_phase/test_stream", 0.0)
                                for metrics in exp.eval_history
                            )
                            forgetting = max(0.0, max_acc - current_acc)

                            summary["forgetting"][exp.experience_id] = forgetting

                self.logger.info(f"Completed training for full scenario {scenario_id}")

                # Remove from active scenarios
                self.active_scenarios.remove(scenario_id)

                return summary

            except Exception as e:
                self.logger.error(f"Error training full scenario {scenario_id}: {str(e)}")

                # Remove from active scenarios
                if scenario_id in self.active_scenarios:
                    self.active_scenarios.remove(scenario_id)

                return {"error": str(e)}

    def get_scenario(self, scenario_id: str) -> Optional[CLScenario]:
        """
        Get a scenario by ID.

        Args:
            scenario_id: Scenario ID

        Returns:
            Scenario object or None if not found
        """
        with self.lock:
            return self.scenarios.get(scenario_id)

    def get_experience(self, scenario_id: str, experience_id: str) -> Optional[CLExperience]:
        """
        Get an experience by ID within a scenario.

        Args:
            scenario_id: Scenario ID
            experience_id: Experience ID

        Returns:
            Experience object or None if not found
        """
        with self.lock:
            scenario = self.scenarios.get(scenario_id)
            if not scenario:
                return None

            for exp in scenario.experiences:
                if exp.experience_id == experience_id:
                    return exp

            return None

    def get_forgetting_metrics(self, scenario_id: str) -> Dict[str, float]:
        """
        Calculate forgetting metrics for a scenario.

        Args:
            scenario_id: Scenario ID

        Returns:
            Dictionary of forgetting metrics
        """
        with self.lock:
            scenario = self.scenarios.get(scenario_id)
            if not scenario:
                return {}

            forgetting = {}

            # Only calculate forgetting if we have multiple experiences with evaluation history
            if len(scenario.experiences) <= 1:
                return {}

            for i, exp in enumerate(scenario.experiences[:-1]):  # Exclude last experience
                if not exp.eval_history:
                    continue

                # Get maximum accuracy achieved for this experience
                max_acc = max(
                    metrics.get("Top1_Acc_Stream/eval_phase/test_stream", 0.0)
                    for metrics in exp.eval_history
                )

                # Get most recent accuracy
                current_acc = exp.metrics.get("Top1_Acc_Stream/eval_phase/test_stream", 0.0)

                # Calculate forgetting
                forgetting[exp.experience_id] = max(0.0, max_acc - current_acc)

            return forgetting

    def get_learning_metrics(self) -> Dict[str, Any]:
        """
        Get performance metrics for continuous learning.

        Returns:
            Dictionary of learning metrics
        """
        with self.lock:
            metrics_summary = {}

            # Process each metric
            for key, values in self.metrics.items():
                if values:
                    metrics_summary[key] = {
                        "current": values[-1],
                        "average": sum(values) / len(values),
                        "min": min(values),
                        "max": max(values)
                    }

            return metrics_summary

    def save_model(self, scenario_id: str, path: str) -> bool:
        """
        Save a trained model to disk.

        Args:
            scenario_id: Scenario ID
            path: Path to save the model

        Returns:
            True if successful, False otherwise
        """
        if not HAS_AVALANCHE:
            self.logger.error("Cannot save model: PyTorch not available")
            return False

        with self.lock:
            scenario = self.scenarios.get(scenario_id)
            if not scenario or scenario.model is None:
                self.logger.warning(f"No model available for scenario {scenario_id}")
                return False

            try:
                # Save model state
                torch.save(scenario.model.state_dict(), path)
                self.logger.info(f"Saved model for scenario {scenario_id} to {path}")
                return True
            except Exception as e:
                self.logger.error(f"Error saving model for scenario {scenario_id}: {str(e)}")
                return False

    def load_model(self, scenario_id: str, path: str) -> bool:
        """
        Load a saved model from disk.

        Args:
            scenario_id: Scenario ID
            path: Path to load the model from

        Returns:
            True if successful, False otherwise
        """
        if not HAS_AVALANCHE:
            self.logger.error("Cannot load model: PyTorch not available")
            return False

        with self.lock:
            scenario = self.scenarios.get(scenario_id)
            if not scenario:
                self.logger.warning(f"Scenario {scenario_id} not found")
                return False

            if scenario.model is None:
                # Initialize model first
                if not self.initialize_model(scenario_id):
                    return False

            try:
                # Load model state
                scenario.model.load_state_dict(torch.load(path))
                self.logger.info(f"Loaded model for scenario {scenario_id} from {path}")
                return True
            except Exception as e:
                self.logger.error(f"Error loading model for scenario {scenario_id}: {str(e)}")
                return False

    def store_experience_in_memory(self, scenario_id: str, experience_id: str) -> bool:
        """
        Store an experience in episodic memory for future replay.

        Args:
            scenario_id: Scenario ID
            experience_id: Experience ID

        Returns:
            True if successful, False otherwise
        """
        with self.lock:
            # Check if we have episodic memory
            if not self.episodic_memory:
                self.logger.warning("No episodic memory available")
                return False

            # Get experience
            experience = self.get_experience(scenario_id, experience_id)
            if not experience:
                self.logger.warning(f"Experience {experience_id} not found in scenario {scenario_id}")
                return False

            try:
                # Store in episodic memory
                memory_id = self.episodic_memory.store(
                    content={
                        "type": "learning_experience",
                        "scenario_id": scenario_id,
                        "experience_id": experience_id,
                        "name": experience.name,
                        "task_id": experience.task_id,
                        "data_summary": {
                            "size": self._get_data_size(experience.data),
                            "timestamp": experience.timestamp
                        },
                        "metrics": experience.metrics
                    },
                    metadata={
                        "scenario_id": scenario_id,
                        "experience_id": experience_id,
                        "type": "learning_experience"
                    },
                    relevance=0.7  # Medium-high relevance
                )

                self.logger.info(f"Stored experience {experience_id} in episodic memory as {memory_id}")
                return True
            except Exception as e:
                self.logger.error(f"Error storing experience in memory: {str(e)}")
                return False

    def create_replay_experience(self, scenario_id: str,
                               num_experiences: int = 3,
                               experience_filter: Optional[Callable[[CLExperience], bool]] = None) -> Optional[str]:
        """
        Create a replay experience from previous experiences in episodic memory.

        Args:
            scenario_id: Scenario ID
            num_experiences: Number of past experiences to combine
            experience_filter: Function to filter experiences

        Returns:
            Experience ID if successful, None otherwise
        """
        with self.lock:
            # Check if we have episodic memory
            if not self.episodic_memory:
                self.logger.warning("No episodic memory available")
                return None

            # Get scenario
            scenario = self.scenarios.get(scenario_id)
            if not scenario:
                self.logger.warning(f"Scenario {scenario_id} not found")
                return None

            try:
                # Query episodic memory for learning experiences
                if hasattr(self.episodic_memory, "query"):
                    memories = self.episodic_memory.query(
                        metadata_filter={
                            "type": "learning_experience",
                            "scenario_id": scenario_id
                        },
                        max_results=50  # Get a good sample
                    )
                else:
                    self.logger.warning("Episodic memory does not support querying")
                    return None

                if not memories:
                    self.logger.warning(f"No experiences found in memory for scenario {scenario_id}")
                    return None

                # Retrieve full memory items
                memory_items = []
                for memory_id in memories:
                    if isinstance(memory_id, tuple):
                        memory_id = memory_id[0]  # Extract ID if tuple
                    memory = self.episodic_memory.retrieve(memory_id)
                    if memory:
                        memory_items.append(memory)

                if not memory_items:
                    self.logger.warning("Could not retrieve memory items")
                    return None

                # Extract experiences from memory
                experiences = []
                for memory in memory_items:
                    content = memory.content
                    if content and isinstance(content, dict):
                        exp_id = content.get("experience_id")
                        if exp_id:
                            exp = self.get_experience(scenario_id, exp_id)
                            if exp:
                                experiences.append(exp)

                # Apply filter if provided
                if experience_filter:
                    experiences = [exp for exp in experiences if experience_filter(exp)]

                if not experiences:
                    self.logger.warning("No valid experiences found after filtering")
                    return None

                # Sort by task_id (to get a range of tasks)
                experiences.sort(key=lambda exp: exp.task_id)

                # Select experiences (evenly distributed if possible)
                selected = []
                if len(experiences) <= num_experiences:
                    selected = experiences
                else:
                    # Try to get a diverse set
                    step = max(1, len(experiences) // num_experiences)
                    indices = [i * step for i in range(num_experiences)]
                    indices = [min(i, len(experiences) - 1) for i in indices]  # Ensure valid indices
                    selected = [experiences[i] for i in indices]

                # Combine data from selected experiences
                combined_data = self._combine_experience_data(selected)

                # Create new replay experience
                return self.add_experience(
                    scenario_id=scenario_id,
                    name=f"Replay({', '.join(exp.name for exp in selected)})",
                    data=combined_data,
                    metadata={
                        "type": "replay",
                        "source_experiences": [exp.experience_id for exp in selected]
                    }
                )

            except Exception as e:
                self.logger.error(f"Error creating replay experience: {str(e)}")
                return None

    def process_learning_task(self, task: LearningTask) -> LearningResult:
        """
        Process a learning task from the learning core.

        Args:
            task: Learning task to process

        Returns:
            Learning result
        """
        if task.paradigm != LearningParadigm.CONTINUOUS:
            return LearningResult(
                success=False,
                logs=["Task paradigm is not CONTINUOUS"],
                metrics={}
            )

        try:
            # Extract task parameters
            task_type_str = task.parameters.get("task_type", "CLASSIFICATION")
            task_type = getattr(TaskType, task_type_str, TaskType.CLASSIFICATION)

            strategy_str = task.parameters.get("strategy", "EWC")
            strategy = getattr(CLStrategy, strategy_str, CLStrategy.EWC)

            scenario_name = task.parameters.get("scenario_name", f"Scenario-{uuid.uuid4().hex[:6]}")

            # Create scenario
            scenario_id = self.create_scenario(
                name=scenario_name,
                task_type=task_type,
                strategy=strategy,
                strategy_params=task.parameters.get("strategy_params", {})
            )

            if not scenario_id:
                return LearningResult(
                    success=False,
                    logs=["Failed to create scenario"],
                    metrics={}
                )

            # Add experiences
            experiences = task.data.get("experiences", [])
            for exp_data in experiences:
                exp_name = exp_data.get("name", f"Experience-{uuid.uuid4().hex[:6]}")
                exp_data_obj = exp_data.get("data")

                if exp_data_obj:
                    self.add_experience(
                        scenario_id=scenario_id,
                        name=exp_name,
                        data=exp_data_obj,
                        metadata=exp_data.get("metadata", {})
                    )

            # Train if requested
            results = {}
            if task.parameters.get("train", False):
                # Initialize model
                if not self.initialize_model(scenario_id):
                    return LearningResult(
                        success=False,
                        logs=["Failed to initialize model"],
                        metrics={}
                    )

                # Train full scenario
                train_epochs = task.parameters.get("train_epochs", 1)
                eval_after = task.parameters.get("evaluate_after_each", True)

                results = self.train_full_scenario(
                    scenario_id=scenario_id,
                    train_epochs=train_epochs,
                    evaluate_after_each=eval_after
                )

            return LearningResult(
                success=True,
                logs=[f"Created scenario {scenario_id} with {len(experiences)} experiences"],
                metrics=results,
                learned_model=self.get_scenario(scenario_id)
            )

        except Exception as e:
            return LearningResult(
                success=False,
                logs=[f"Error processing learning task: {str(e)}"],
                metrics={}
            )

    # -----------------------------------------------------------------------
    # Internal methods
    # -----------------------------------------------------------------------

    def _register_strategy_factories(self) -> None:
        """Register factories for creating continual learning strategies"""
        if not HAS_AVALANCHE:
            return

        # Naive (baseline, fine-tuning)
        self.strategy_factories[CLStrategy.NAIVE] = lambda **kwargs: Naive(**kwargs)

        # EWC (Elastic Weight Consolidation)
        self.strategy_factories[CLStrategy.EWC] = lambda **kwargs: EWC(
            ewc_lambda=kwargs.pop("ewc_lambda", 100.0),
            **kwargs
        )

        # LwF (Learning without Forgetting)
        self.strategy_factories[CLStrategy.LWF] = lambda **kwargs: LwF(
            alpha=kwargs.pop("alpha", 1.0),
            temperature=kwargs.pop("temperature", 2.0),
            **kwargs
        )

        # SI (Synaptic Intelligence)
        self.strategy_factories[CLStrategy.SI] = lambda **kwargs: SynapticIntelligence(
            si_lambda=kwargs.pop("si_lambda", 1.0),
            **kwargs
        )

        # GEM (Gradient Episodic Memory)
        self.strategy_factories[CLStrategy.GEM] = lambda **kwargs: GEM(
            patterns_per_experience=kwargs.pop("patterns_per_experience", 100),
            memory_strength=kwargs.pop("memory_strength", 0.5),
            **kwargs
        )

        # AGEM (Averaged GEM)
        self.strategy_factories[CLStrategy.AGEM] = lambda **kwargs: AGEM(
            patterns_per_experience=kwargs.pop("patterns_per_experience", 100),
            sample_size=kwargs.pop("sample_size", 64),
            **kwargs
        )

        # CWR* (CWR Star)
        self.strategy_factories[CLStrategy.CWR_STAR] = lambda **kwargs: CWRStar(**kwargs)

    def _register_with_nexus_core(self) -> None:
        """Register with NexusCore for system-wide coordination"""
        if not self.nexus_core:
            self.logger.warning("No NexusCore available for registration")
            return

        try:
            # Register as a module
            dependencies = []
            if self.learning_core:
                dependencies.append("learning_core")
            if self.working_memory:
                dependencies.append("working_memory")
            if self.semantic_memory:
                dependencies.append("semantic_memory")
            if self.episodic_memory:
                dependencies.append("episodic_memory")

            self.nexus_core.register_module(
                name="continuous_learner",
                module=self,
                dependencies=dependencies
            )

            self.logger.info("Successfully registered with NexusCore")
        except Exception as e:
            self.logger.error(f"Failed to register with NexusCore: {str(e)}")

    def _learning_loop(self) -> None:
        """Background thread for continuous learning"""
        self.logger.info("Starting continuous learning background thread")

        while not self.stop_learning.is_set():
            try:
                # Process active scenarios
                with self.lock:
                    active_scenarios = list(self.active_scenarios)

                # Process each active scenario
                for scenario_id in active_scenarios:
                    try:
                        self._process_scenario(scenario_id)
                    except Exception as e:
                        self.logger.error(f"Error processing scenario {scenario_id}: {str(e)}")

                # Sleep to avoid hogging CPU
                time.sleep(1.0)

            except Exception as e:
                self.logger.error(f"Error in learning loop: {str(e)}")

        self.logger.info("Continuous learning background thread stopped")

    def _process_scenario(self, scenario_id: str) -> None:
        """
        Process a single active scenario.

        Args:
            scenario_id: Scenario ID
        """
        with self.lock:
            scenario = self.scenarios.get(scenario_id)
            if not scenario:
                if scenario_id in self.active_scenarios:
                    self.active_scenarios.remove(scenario_id)
                return

            # Skip if the scenario is not properly initialized
            if scenario.model is None or scenario.cl_strategy is None:
                return

            # Check if there are more experiences to process
            if not scenario.has_more_experiences():
                # Remove from active scenarios
                self.active_scenarios.remove(scenario_id)
                self.logger.info(f"Scenario {scenario_id} completed all experiences")
                return

    def _get_data_size(self, data: Any) -> Dict[str, int]:
        """
        Get the size of data in an experience.

        Args:
            data: Experience data

        Returns:
            Dictionary with data size information
        """
        size_info = {"total": 0}

        try:
            if isinstance(data, dict):
                for key, value in data.items():
                    if isinstance(value, (torch.Tensor, np.ndarray)):
                        size_info[key] = value.shape[0]
                        size_info["total"] += value.shape[0]
            elif isinstance(data, tuple) and len(data) >= 2:
                X, y = data[:2]
                if isinstance(X, (torch.Tensor, np.ndarray)):
                    size_info["X"] = X.shape[0]
                    size_info["total"] += X.shape[0]
                if isinstance(y, (torch.Tensor, np.ndarray)):
                    size_info["y"] = y.shape[0]
        except Exception:
            pass

        return size_info

    def _combine_experience_data(self, experiences: List[CLExperience]) -> Any:
        """
        Combine data from multiple experiences.

        Args:
            experiences: List of experiences to combine

        Returns:
            Combined data
        """
        if not experiences:
            return None

        try:
            # Determine data format (dict or tuple)
            first_data = experiences[0].data
            is_dict_format = isinstance(first_data, dict)

            if is_dict_format:
                # Dictionary format
                combined = {}

                # Identify keys to combine
                keys_to_combine = set()
                for exp in experiences:
                    if isinstance(exp.data, dict):
                        keys_to_combine.update(exp.data.keys())

                # Combine data for each key
                for key in keys_to_combine:
                    data_list = []
                    for exp in experiences:
                        if isinstance(exp.data, dict) and key in exp.data:
                            data = exp.data[key]
                            if data is not None:
                                data_list.append(data)

                    if data_list:
                        # Combine data (assuming tensors or arrays)
                        if isinstance(data_list[0], torch.Tensor):
                            combined[key] = torch.cat(data_list, dim=0)
                        elif isinstance(data_list[0], np.ndarray):
                            combined[key] = np.concatenate(data_list, axis=0)
                        else:
                            # Try to convert to numpy first
                            try:
                                np_list = [np.array(d) for d in data_list]
                                combined[key] = np.concatenate(np_list, axis=0)
                            except:
                                # Just store the list
                                combined[key] = data_list

                return combined

            else:
                # Tuple format (X_train, y_train, X_val, y_val)
                X_train_list = []
                y_train_list = []
                X_val_list = []
                y_val_list = []

                for exp in experiences:
                    data = exp.data
                    if isinstance(data, tuple):
                        if len(data) >= 1 and data[0] is not None:
                            X_train_list.append(data[0])
                        if len(data) >= 2 and data[1] is not None:
                            y_train_list.append(data[1])
                        if len(data) >= 3 and data[2] is not None:
                            X_val_list.append(data[2])
                        if len(data) >= 4 and data[3] is not None:
                            y_val_list.append(data[3])

                # Combine data
                combined_data = []

                # X_train
                if X_train_list:
                    if isinstance(X_train_list[0], torch.Tensor):
                        combined_data.append(torch.cat(X_train_list, dim=0))
                    elif isinstance(X_train_list[0], np.ndarray):
                        combined_data.append(np.concatenate(X_train_list, axis=0))
                    else:
                        combined_data.append(X_train_list)
                else:
                    combined_data.append(None)

                # y_train
                if y_train_list:
                    if isinstance(y_train_list[0], torch.Tensor):
                        combined_data.append(torch.cat(y_train_list, dim=0))
                    elif isinstance(y_train_list[0], np.ndarray):
                        combined_data.append(np.concatenate(y_train_list, axis=0))
                    else:
                        combined_data.append(y_train_list)
                else:
                    combined_data.append(None)

                # X_val
                if X_val_list:
                    if isinstance(X_val_list[0], torch.Tensor):
                        combined_data.append(torch.cat(X_val_list, dim=0))
                    elif isinstance(X_val_list[0], np.ndarray):
                        combined_data.append(np.concatenate(X_val_list, axis=0))
                    else:
                        combined_data.append(X_val_list)
                else:
                    combined_data.append(None)

                # y_val
                if y_val_list:
                    if isinstance(y_val_list[0], torch.Tensor):
                        combined_data.append(torch.cat(y_val_list, dim=0))
                    elif isinstance(y_val_list[0], np.ndarray):
                        combined_data.append(np.concatenate(y_val_list, axis=0))
                    else:
                        combined_data.append(y_val_list)
                else:
                    combined_data.append(None)

                return tuple(combined_data)

        except Exception as e:
            self.logger.error(f"Error combining experience data: {str(e)}")
            return None