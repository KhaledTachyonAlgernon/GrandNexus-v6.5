"""
uncertainty.py - Uncertainty Management and Probabilistic Reasoning Module for GrandNexus

This module provides capabilities for managing uncertainty, representing probabilistic beliefs,
and performing probabilistic reasoning. It enables GrandNexus to reason about uncertain information,
model confidence levels, and make decisions in the face of incomplete or contradictory evidence.

Key features:
- Multiple uncertainty representation methods (probabilities, intervals, fuzzy logic)
- Bayesian belief networks for causal reasoning under uncertainty
- Monte Carlo sampling for complex probability distributions
- Belief revision and evidence integration
- Conflict resolution for contradictory information
- Decision-making under uncertainty with various policies
- Integration with memory and reasoning modules for consistent belief management
"""

import logging
import threading
import time
import uuid
import random
import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple, Union, Set, Callable
from enum import Enum, auto
from collections import defaultdict, deque

# Try importing optional dependencies with fallbacks
try:
    import numpy as np
    HAS_NUMPY = True
except ImportError:
    HAS_NUMPY = False

try:
    from scipy import stats
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False

# Assuming these exist or will exist
from grandnexus.core.nexus_core import NexusCore
from grandnexus.memory.working import WorkingMemory
from grandnexus.memory.semantic import SemanticMemory
from grandnexus.reasoning import reasoning_core

# Uncertainty representation types
class UncertaintyType(Enum):
    """Types of uncertainty representations supported by the module"""
    PROBABILITY = auto()       # Single probability value
    INTERVAL = auto()          # Probability interval
    DISTRIBUTION = auto()      # Probability distribution
    FUZZY = auto()             # Fuzzy membership
    DEMPSTER_SHAFER = auto()   # Belief and plausibility
    POSSIBILITY = auto()       # Possibility theory
    VERBAL = auto()            # Verbal/linguistic uncertainty

@dataclass
class UncertainBelief:
    """
    Represents a belief with associated uncertainty information.
    Acts as a container for various uncertainty representations.
    """
    belief_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    content: Any = None  # The actual belief content (statement, proposition, etc.)
    uncertainty_type: UncertaintyType = UncertaintyType.PROBABILITY
    # Different uncertainty representations based on type
    probability: Optional[float] = None  # Single probability value
    interval: Optional[Tuple[float, float]] = None  # Min and max probability
    distribution_params: Optional[Dict[str, Any]] = None  # Parameters for a distribution
    fuzzy_membership: Optional[float] = None  # Fuzzy logic membership
    belief_mass: Optional[Dict[str, float]] = None  # Dempster-Shafer belief masses
    possibility: Optional[float] = None  # Possibility value
    verbal_confidence: Optional[str] = None  # Verbal expression of uncertainty

    # Metadata and tracking
    source: Optional[str] = None  # Where this belief came from
    timestamp: float = field(default_factory=time.time)
    last_updated: float = field(default_factory=time.time)
    update_count: int = 0
    evidence: List[Dict[str, Any]] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        """Validate the belief after initialization"""
        self._validate()

    def _validate(self):
        """Validate that appropriate uncertainty values are provided for the type"""
        if self.uncertainty_type == UncertaintyType.PROBABILITY and self.probability is None:
            raise ValueError("Probability value must be provided for PROBABILITY type")
        elif self.uncertainty_type == UncertaintyType.INTERVAL and self.interval is None:
            raise ValueError("Interval must be provided for INTERVAL type")
        elif self.uncertainty_type == UncertaintyType.DISTRIBUTION and self.distribution_params is None:
            raise ValueError("Distribution parameters must be provided for DISTRIBUTION type")
        elif self.uncertainty_type == UncertaintyType.FUZZY and self.fuzzy_membership is None:
            raise ValueError("Fuzzy membership must be provided for FUZZY type")
        elif self.uncertainty_type == UncertaintyType.DEMPSTER_SHAFER and self.belief_mass is None:
            raise ValueError("Belief mass must be provided for DEMPSTER_SHAFER type")
        elif self.uncertainty_type == UncertaintyType.POSSIBILITY and self.possibility is None:
            raise ValueError("Possibility value must be provided for POSSIBILITY type")
        elif self.uncertainty_type == UncertaintyType.VERBAL and self.verbal_confidence is None:
            raise ValueError("Verbal confidence must be provided for VERBAL type")

    def update(self,
               new_uncertainty_value: Any = None,
               evidence: Optional[Dict[str, Any]] = None) -> None:
        """
        Update the belief with new uncertainty value and optional evidence.

        Args:
            new_uncertainty_value: New value for the uncertain belief
            evidence: Optional evidence for the update
        """
        self.last_updated = time.time()
        self.update_count += 1

        if evidence:
            evidence['timestamp'] = time.time()
            self.evidence.append(evidence)

        if new_uncertainty_value is not None:
            if self.uncertainty_type == UncertaintyType.PROBABILITY:
                self.probability = new_uncertainty_value
            elif self.uncertainty_type == UncertaintyType.INTERVAL:
                self.interval = new_uncertainty_value
            elif self.uncertainty_type == UncertaintyType.DISTRIBUTION:
                self.distribution_params = new_uncertainty_value
            elif self.uncertainty_type == UncertaintyType.FUZZY:
                self.fuzzy_membership = new_uncertainty_value
            elif self.uncertainty_type == UncertaintyType.DEMPSTER_SHAFER:
                self.belief_mass = new_uncertainty_value
            elif self.uncertainty_type == UncertaintyType.POSSIBILITY:
                self.possibility = new_uncertainty_value
            elif self.uncertainty_type == UncertaintyType.VERBAL:
                self.verbal_confidence = new_uncertainty_value

    def to_probability(self) -> float:
        """Convert the belief to a single probability value for comparison"""
        if self.uncertainty_type == UncertaintyType.PROBABILITY:
            return self.probability
        elif self.uncertainty_type == UncertaintyType.INTERVAL:
            # Use interval midpoint
            return (self.interval[0] + self.interval[1]) / 2
        elif self.uncertainty_type == UncertaintyType.DISTRIBUTION:
            # For distributions, return mean or expected value
            dist_type = self.distribution_params.get('type', 'normal')
            if dist_type == 'normal':
                return self.distribution_params.get('mean', 0.5)
            elif dist_type == 'beta':
                alpha = self.distribution_params.get('alpha', 1)
                beta = self.distribution_params.get('beta', 1)
                return alpha / (alpha + beta)
            else:
                return 0.5  # Default
        elif self.uncertainty_type == UncertaintyType.FUZZY:
            return self.fuzzy_membership
        elif self.uncertainty_type == UncertaintyType.DEMPSTER_SHAFER:
            # Sum belief mass for the positive case
            return sum(self.belief_mass.get(k, 0) for k in self.belief_mass if 'true' in k.lower())
        elif self.uncertainty_type == UncertaintyType.POSSIBILITY:
            # Convert possibility to probability (simplified)
            return min(1.0, self.possibility * 0.5)
        elif self.uncertainty_type == UncertaintyType.VERBAL:
            # Map verbal expressions to probabilities
            verbal_map = {
                "impossible": 0.0,
                "highly unlikely": 0.1,
                "unlikely": 0.2,
                "somewhat unlikely": 0.3,
                "uncertain": 0.5,
                "somewhat likely": 0.7,
                "likely": 0.8,
                "highly likely": 0.9,
                "certain": 1.0
            }
            return verbal_map.get(self.verbal_confidence.lower(), 0.5)
        else:
            return 0.5  # Default

    def to_dict(self) -> Dict[str, Any]:
        """Convert the belief to a dictionary representation"""
        result = {
            "belief_id": self.belief_id,
            "uncertainty_type": self.uncertainty_type.name,
            "timestamp": self.timestamp,
            "last_updated": self.last_updated,
            "update_count": self.update_count,
            "source": self.source,
            "metadata": self.metadata
        }

        # Include the content if it's a simple type
        if isinstance(self.content, (str, int, float, bool, type(None))):
            result["content"] = self.content
        else:
            # Try to get a string representation
            result["content"] = str(self.content)

        # Include the appropriate uncertainty value based on type
        if self.uncertainty_type == UncertaintyType.PROBABILITY:
            result["probability"] = self.probability
        elif self.uncertainty_type == UncertaintyType.INTERVAL:
            result["interval"] = self.interval
        elif self.uncertainty_type == UncertaintyType.DISTRIBUTION:
            result["distribution_params"] = self.distribution_params
        elif self.uncertainty_type == UncertaintyType.FUZZY:
            result["fuzzy_membership"] = self.fuzzy_membership
        elif self.uncertainty_type == UncertaintyType.DEMPSTER_SHAFER:
            result["belief_mass"] = self.belief_mass
        elif self.uncertainty_type == UncertaintyType.POSSIBILITY:
            result["possibility"] = self.possibility
        elif self.uncertainty_type == UncertaintyType.VERBAL:
            result["verbal_confidence"] = self.verbal_confidence

        # Include recent evidence
        if self.evidence:
            result["recent_evidence"] = self.evidence[-5:]  # Last 5 pieces

        return result

@dataclass
class BayesianNode:
    """Node in a Bayesian network for probabilistic reasoning"""
    node_id: str
    name: str
    states: List[str]  # Possible states for this variable
    cpt: Dict[str, Dict[str, float]]  # Conditional probability table
    parents: List[str] = field(default_factory=list)  # Parent node IDs
    children: List[str] = field(default_factory=list)  # Child node IDs
    description: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def get_probability(self, state: str, parent_states: Dict[str, str]) -> float:
        """
        Get probability for this node being in given state given parent states.

        Args:
            state: State to get probability for
            parent_states: Dictionary mapping parent node IDs to their states

        Returns:
            Probability value
        """
        if not self.parents:
            # If no parents, use unconditional probability
            return self.cpt.get("unconditional", {}).get(state, 0.0)

        # Create CPT lookup key from parent states
        lookup_key = "_".join(parent_states.get(parent, "") for parent in self.parents)

        # Get probability from CPT
        return self.cpt.get(lookup_key, {}).get(state, 0.0)

@dataclass
class BayesianNetwork:
    """Bayesian network for probabilistic reasoning with causal relationships"""
    network_id: str
    name: str
    nodes: Dict[str, BayesianNode] = field(default_factory=dict)
    description: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def add_node(self, node: BayesianNode) -> None:
        """Add a node to the network"""
        self.nodes[node.node_id] = node

    def add_edge(self, parent_id: str, child_id: str) -> None:
        """Add a directed edge from parent to child"""
        if parent_id not in self.nodes or child_id not in self.nodes:
            raise ValueError(f"Both nodes must exist in the network")

        # Update parent and child lists
        if parent_id not in self.nodes[child_id].parents:
            self.nodes[child_id].parents.append(parent_id)

        if child_id not in self.nodes[parent_id].children:
            self.nodes[parent_id].children.append(child_id)

    def get_node(self, node_id: str) -> Optional[BayesianNode]:
        """Get a node by ID"""
        return self.nodes.get(node_id)

    def get_all_nodes(self) -> List[BayesianNode]:
        """Get all nodes in the network"""
        return list(self.nodes.values())

class UncertaintyManager:
    """
    Core uncertainty management module for GrandNexus.

    This module provides capabilities for representing and reasoning with
    uncertain information, enabling GrandNexus to manage beliefs with
    various forms of uncertainty and perform probabilistic reasoning.
    """

    def __init__(self, nexus_core: Optional[NexusCore] = None,
                 working_memory: Optional[WorkingMemory] = None,
                 semantic_memory: Optional[SemanticMemory] = None,
                 reasoning_core: Optional[Any] = None):
        """
        Initialize the uncertainty manager.

        Args:
            nexus_core: Reference to the NexusCore for system integration
            working_memory: Reference to working memory
            semantic_memory: Reference to semantic memory
            reasoning_core: Reference to reasoning core
        """
        self.logger = logging.getLogger("GrandNexus.Reasoning.Uncertainty")
        self.nexus_core = nexus_core
        self.working_memory = working_memory
        self.semantic_memory = semantic_memory
        self.reasoning_core = reasoning_core

        # Internal state
        self.instance_id = str(uuid.uuid4())[:8]
        self.initialized = False
        self.lock = threading.RLock()

        # Belief storage
        self.beliefs = {}  # belief_id -> UncertainBelief
        self.belief_index = defaultdict(list)  # Various indices for faster lookup

        # Bayesian networks
        self.networks = {}  # network_id -> BayesianNetwork

        # Reasoning history
        self.reasoning_history = deque(maxlen=100)

        # Performance metrics
        self.metrics = defaultdict(lambda: deque(maxlen=100))

        self.logger.info(f"UncertaintyManager initialized with ID={self.instance_id}")

    def initialize(self) -> bool:
        """Initialize the uncertainty manager"""
        if self.initialized:
            return True

        with self.lock:
            try:
                # Connect to other modules
                if self.nexus_core:
                    self._register_with_nexus_core()

                self.initialized = True
                self.logger.info("UncertaintyManager initialization complete")
                return True
            except Exception as e:
                self.logger.error(f"UncertaintyManager initialization failed: {str(e)}")
                return False

    def create_belief(self, content: Any,
                     uncertainty_type: UncertaintyType = UncertaintyType.PROBABILITY,
                     uncertainty_value: Any = None,
                     source: Optional[str] = None,
                     metadata: Optional[Dict[str, Any]] = None,
                     evidence: Optional[Dict[str, Any]] = None) -> str:
        """
        Create a new uncertain belief.

        Args:
            content: The content of the belief
            uncertainty_type: Type of uncertainty representation
            uncertainty_value: Value for the uncertainty representation
            source: Source of the belief
            metadata: Additional metadata
            evidence: Evidence supporting the belief

        Returns:
            ID of the created belief
        """
        if not self.initialized:
            self.initialize()

        if metadata is None:
            metadata = {}

        # Prepare the appropriate uncertainty representation
        kwargs = {
            "content": content,
            "uncertainty_type": uncertainty_type,
            "source": source,
            "metadata": metadata
        }

        if uncertainty_type == UncertaintyType.PROBABILITY:
            kwargs["probability"] = uncertainty_value
        elif uncertainty_type == UncertaintyType.INTERVAL:
            kwargs["interval"] = uncertainty_value
        elif uncertainty_type == UncertaintyType.DISTRIBUTION:
            kwargs["distribution_params"] = uncertainty_value
        elif uncertainty_type == UncertaintyType.FUZZY:
            kwargs["fuzzy_membership"] = uncertainty_value
        elif uncertainty_type == UncertaintyType.DEMPSTER_SHAFER:
            kwargs["belief_mass"] = uncertainty_value
        elif uncertainty_type == UncertaintyType.POSSIBILITY:
            kwargs["possibility"] = uncertainty_value
        elif uncertainty_type == UncertaintyType.VERBAL:
            kwargs["verbal_confidence"] = uncertainty_value

        # Add evidence if provided
        if evidence:
            evidence['timestamp'] = time.time()
            kwargs["evidence"] = [evidence]

        # Create the belief
        belief = UncertainBelief(**kwargs)

        with self.lock:
            # Store the belief
            self.beliefs[belief.belief_id] = belief

            # Update indices for faster lookup
            self._update_belief_indices(belief)

            # Store in working memory if available
            if self.working_memory:
                self.working_memory.store_item(
                    content=belief.to_dict(),
                    source="uncertainty_manager",
                    metadata={
                        "type": "uncertain_belief",
                        "uncertainty_type": uncertainty_type.name
                    }
                )

        self.logger.info(f"Created uncertain belief: {belief.belief_id}")
        return belief.belief_id

    def get_belief(self, belief_id: str) -> Optional[UncertainBelief]:
        """Get a belief by ID"""
        with self.lock:
            return self.beliefs.get(belief_id)

    def query_beliefs(self, content: Optional[Any] = None,
                     uncertainty_type: Optional[UncertaintyType] = None,
                     min_probability: Optional[float] = None,
                     max_probability: Optional[float] = None,
                     source: Optional[str] = None,
                     metadata_filter: Optional[Dict[str, Any]] = None) -> List[UncertainBelief]:
        """
        Query beliefs matching given criteria.

        Args:
            content: Content to match (exact match)
            uncertainty_type: Type of uncertainty to filter for
            min_probability: Minimum probability (converted if needed)
            max_probability: Maximum probability (converted if needed)
            source: Source to filter for
            metadata_filter: Metadata key-value pairs to match

        Returns:
            List of matching beliefs
        """
        if not self.initialized:
            self.initialize()

        with self.lock:
            # Start with all beliefs
            belief_ids = set(self.beliefs.keys())

            # Apply filters
            if content is not None:
                # Filter by content (exact match for now)
                content_filter = set()
                for belief_id, belief in self.beliefs.items():
                    if belief.content == content:
                        content_filter.add(belief_id)
                belief_ids &= content_filter

            if uncertainty_type is not None:
                # Filter by uncertainty type
                type_key = f"type:{uncertainty_type.name}"
                if type_key in self.belief_index:
                    belief_ids &= set(self.belief_index[type_key])

            if min_probability is not None or max_probability is not None:
                # Filter by probability (converting if needed)
                probability_filter = set()
                for belief_id in belief_ids:
                    belief = self.beliefs[belief_id]
                    prob = belief.to_probability()

                    if min_probability is not None and prob < min_probability:
                        continue
                    if max_probability is not None and prob > max_probability:
                        continue

                    probability_filter.add(belief_id)

                belief_ids = probability_filter

            if source is not None:
                # Filter by source
                source_key = f"source:{source}"
                if source_key in self.belief_index:
                    belief_ids &= set(self.belief_index[source_key])

            if metadata_filter is not None:
                # Filter by metadata
                for key, value in metadata_filter.items():
                    meta_key = f"metadata:{key}:{value}"
                    if meta_key in self.belief_index:
                        belief_ids &= set(self.belief_index[meta_key])

            # Return matching beliefs
            return [self.beliefs[belief_id] for belief_id in belief_ids]

    def update_belief(self, belief_id: str,
                     new_uncertainty_value: Any,
                     evidence: Optional[Dict[str, Any]] = None,
                     update_method: str = "replace") -> bool:
        """
        Update an existing belief with new uncertainty information.

        Args:
            belief_id: ID of the belief to update
            new_uncertainty_value: New value for the uncertainty
            evidence: Optional evidence for the update
            update_method: How to combine with existing value ('replace', 'bayesian', 'weighted')

        Returns:
            True if update was successful, False otherwise
        """
        if not self.initialized:
            self.initialize()

        with self.lock:
            belief = self.beliefs.get(belief_id)
            if not belief:
                self.logger.warning(f"Belief {belief_id} not found for update")
                return False

            try:
                old_value = None

                # Get the old value based on uncertainty type
                if belief.uncertainty_type == UncertaintyType.PROBABILITY:
                    old_value = belief.probability
                elif belief.uncertainty_type == UncertaintyType.INTERVAL:
                    old_value = belief.interval
                elif belief.uncertainty_type == UncertaintyType.DISTRIBUTION:
                    old_value = belief.distribution_params
                elif belief.uncertainty_type == UncertaintyType.FUZZY:
                    old_value = belief.fuzzy_membership
                elif belief.uncertainty_type == UncertaintyType.DEMPSTER_SHAFER:
                    old_value = belief.belief_mass
                elif belief.uncertainty_type == UncertaintyType.POSSIBILITY:
                    old_value = belief.possibility
                elif belief.uncertainty_type == UncertaintyType.VERBAL:
                    old_value = belief.verbal_confidence

                # Apply the update based on the method
                updated_value = self._combine_uncertainty_values(
                    old_value, new_uncertainty_value,
                    belief.uncertainty_type, update_method
                )

                # Update the belief
                belief.update(updated_value, evidence)

                # Log update
                self.logger.info(f"Updated belief: {belief_id}")

                # Update in working memory if available
                if self.working_memory:
                    self.working_memory.store_item(
                        content=belief.to_dict(),
                        source="uncertainty_manager",
                        metadata={
                            "type": "updated_belief",
                            "uncertainty_type": belief.uncertainty_type.name
                        }
                    )

                return True
            except Exception as e:
                self.logger.error(f"Error updating belief {belief_id}: {str(e)}")
                return False

    def create_bayesian_network(self, name: str,
                              description: Optional[str] = None,
                              metadata: Optional[Dict[str, Any]] = None) -> str:
        """
        Create a new Bayesian network for causal probabilistic reasoning.

        Args:
            name: Name of the network
            description: Optional description
            metadata: Additional metadata

        Returns:
            ID of the created network
        """
        if not self.initialized:
            self.initialize()

        network_id = str(uuid.uuid4())

        if metadata is None:
            metadata = {}

        # Create the network
        network = BayesianNetwork(
            network_id=network_id,
            name=name,
            description=description,
            metadata=metadata
        )

        with self.lock:
            # Store the network
            self.networks[network_id] = network

        self.logger.info(f"Created Bayesian network: {network_id} ({name})")
        return network_id

    def add_node_to_network(self, network_id: str, node: BayesianNode) -> bool:
        """
        Add a node to a Bayesian network.

        Args:
            network_id: ID of the network
            node: Node to add

        Returns:
            True if addition was successful, False otherwise
        """
        if not self.initialized:
            self.initialize()

        with self.lock:
            network = self.networks.get(network_id)
            if not network:
                self.logger.warning(f"Network {network_id} not found")
                return False

            try:
                network.add_node(node)
                self.logger.info(f"Added node {node.node_id} to network {network_id}")
                return True
            except Exception as e:
                self.logger.error(f"Error adding node to network: {str(e)}")
                return False

    def add_edge_to_network(self, network_id: str,
                          parent_id: str, child_id: str) -> bool:
        """
        Add a directed edge between nodes in a Bayesian network.

        Args:
            network_id: ID of the network
            parent_id: ID of the parent node
            child_id: ID of the child node

        Returns:
            True if addition was successful, False otherwise
        """
        if not self.initialized:
            self.initialize()

        with self.lock:
            network = self.networks.get(network_id)
            if not network:
                self.logger.warning(f"Network {network_id} not found")
                return False

            try:
                network.add_edge(parent_id, child_id)
                self.logger.info(f"Added edge {parent_id} -> {child_id} to network {network_id}")
                return True
            except Exception as e:
                self.logger.error(f"Error adding edge to network: {str(e)}")
                return False

    def query_bayesian_network(self, network_id: str,
                             query_nodes: List[str],
                             evidence: Dict[str, str],
                             method: str = "exact") -> Dict[str, Dict[str, float]]:
        """
        Query a Bayesian network to compute probabilities given evidence.

        Args:
            network_id: ID of the network
            query_nodes: List of node IDs to query
            evidence: Dictionary mapping node IDs to their observed states
            method: Inference method ('exact', 'approximate', 'sampling')

        Returns:
            Dictionary mapping node IDs to probability distributions over states
        """
        if not self.initialized:
            self.initialize()

        with self.lock:
            network = self.networks.get(network_id)
            if not network:
                self.logger.warning(f"Network {network_id} not found")
                return {}

            try:
                if method == "exact":
                    return self._exact_inference(network, query_nodes, evidence)
                elif method == "approximate":
                    return self._approximate_inference(network, query_nodes, evidence)
                elif method == "sampling":
                    return self._sampling_inference(network, query_nodes, evidence)
                else:
                    self.logger.warning(f"Unknown inference method: {method}")
                    return {}
            except Exception as e:
                self.logger.error(f"Error querying network: {str(e)}")
                return {}

    def monte_carlo_simulation(self,
                              variables: Dict[str, Dict[str, Any]],
                              function: Callable,
                              num_samples: int = 1000) -> Dict[str, Any]:
        """
        Perform Monte Carlo simulation to propagate uncertainty through a function.

        Args:
            variables: Dictionary mapping variable names to distribution parameters
            function: Function that takes variable values and returns a result
            num_samples: Number of Monte Carlo samples

        Returns:
            Dictionary with simulation results (mean, std dev, quantiles, etc.)
        """
        if not self.initialized:
            self.initialize()

        if not HAS_NUMPY:
            self.logger.warning("NumPy is required for Monte Carlo simulation")
            return {"error": "NumPy is required for Monte Carlo simulation"}

        try:
            # Generate samples for each variable
            samples = {}
            for var_name, var_params in variables.items():
                dist_type = var_params.get("type", "normal")

                if dist_type == "normal":
                    mean = var_params.get("mean", 0.0)
                    std = var_params.get("std", 1.0)
                    samples[var_name] = np.random.normal(mean, std, num_samples)
                elif dist_type == "uniform":
                    low = var_params.get("low", 0.0)
                    high = var_params.get("high", 1.0)
                    samples[var_name] = np.random.uniform(low, high, num_samples)
                elif dist_type == "beta":
                    alpha = var_params.get("alpha", 1.0)
                    beta = var_params.get("beta", 1.0)
                    samples[var_name] = np.random.beta(alpha, beta, num_samples)
                else:
                    self.logger.warning(f"Unknown distribution type: {dist_type}")
                    return {"error": f"Unknown distribution type: {dist_type}"}

            # Run the function on each sample
            results = []
            for i in range(num_samples):
                sample_values = {var: samples[var][i] for var in variables}
                result = function(**sample_values)
                results.append(result)

            # Compute statistics
            results_array = np.array(results)

            return {
                "mean": float(np.mean(results_array)),
                "std": float(np.std(results_array)),
                "min": float(np.min(results_array)),
                "max": float(np.max(results_array)),
                "quantiles": {
                    "0.05": float(np.quantile(results_array, 0.05)),
                    "0.25": float(np.quantile(results_array, 0.25)),
                    "0.5": float(np.quantile(results_array, 0.5)),
                    "0.75": float(np.quantile(results_array, 0.75)),
                    "0.95": float(np.quantile(results_array, 0.95))
                },
                "num_samples": num_samples
            }
        except Exception as e:
            self.logger.error(f"Error in Monte Carlo simulation: {str(e)}")
            return {"error": str(e)}

    def reconcile_conflicting_beliefs(self,
                                     belief_ids: List[str],
                                     reconciliation_strategy: str = "weighted_average") -> Optional[str]:
        """
        Reconcile multiple potentially conflicting beliefs into a single belief.

        Args:
            belief_ids: List of belief IDs to reconcile
            reconciliation_strategy: Strategy for reconciliation

        Returns:
            ID of the reconciled belief, or None if reconciliation failed
        """
        if not self.initialized:
            self.initialize()

        with self.lock:
            # Get all beliefs
            beliefs = []
            for belief_id in belief_ids:
                belief = self.beliefs.get(belief_id)
                if belief:
                    beliefs.append(belief)

            if not beliefs:
                self.logger.warning("No beliefs found for reconciliation")
                return None

            # Check if all beliefs have the same content
            content = beliefs[0].content
            if not all(belief.content == content for belief in beliefs):
                self.logger.warning("Cannot reconcile beliefs with different content")
                return None

            try:
                # Apply reconciliation strategy
                if reconciliation_strategy == "weighted_average":
                    # Convert all to probabilities
                    probs = [belief.to_probability() for belief in beliefs]

                    # Use update count as weight (more updates = more evidence)
                    weights = [belief.update_count + 1 for belief in beliefs]  # +1 to avoid zero weight

                    # Calculate weighted average
                    weighted_prob = sum(p * w for p, w in zip(probs, weights)) / sum(weights)

                    # Create new belief with reconciled probability
                    reconciled_id = self.create_belief(
                        content=content,
                        uncertainty_type=UncertaintyType.PROBABILITY,
                        uncertainty_value=weighted_prob,
                        source="reconciliation",
                        metadata={
                            "reconciliation_strategy": reconciliation_strategy,
                            "source_beliefs": belief_ids
                        },
                        evidence={
                            "source": "reconciliation",
                            "description": f"Reconciled from {len(beliefs)} beliefs using {reconciliation_strategy}"
                        }
                    )

                    return reconciled_id

                elif reconciliation_strategy == "dempster_shafer":
                    # Implement Dempster-Shafer combination rule
                    # This is a simplified version
                    belief_masses = {}

                    for belief in beliefs:
                        if belief.uncertainty_type == UncertaintyType.DEMPSTER_SHAFER:
                            # Use existing belief masses
                            for state, mass in belief.belief_mass.items():
                                if state not in belief_masses:
                                    belief_masses[state] = mass
                                else:
                                    # Basic combination (simplified)
                                    belief_masses[state] = belief_masses[state] + mass - belief_masses[state] * mass
                        else:
                            # Convert to simple belief mass
                            prob = belief.to_probability()
                            if "true" not in belief_masses:
                                belief_masses["true"] = prob
                            else:
                                belief_masses["true"] = belief_masses["true"] + prob - belief_masses["true"] * prob

                            if "false" not in belief_masses:
                                belief_masses["false"] = 1 - prob
                            else:
                                belief_masses["false"] = belief_masses["false"] + (1 - prob) - belief_masses["false"] * (1 - prob)

                    # Create new belief with combined belief masses
                    reconciled_id = self.create_belief(
                        content=content,
                        uncertainty_type=UncertaintyType.DEMPSTER_SHAFER,
                        uncertainty_value=belief_masses,
                        source="reconciliation",
                        metadata={
                            "reconciliation_strategy": reconciliation_strategy,
                            "source_beliefs": belief_ids
                        },
                        evidence={
                            "source": "reconciliation",
                            "description": f"Reconciled from {len(beliefs)} beliefs using {reconciliation_strategy}"
                        }
                    )

                    return reconciled_id

                elif reconciliation_strategy == "highest_confidence":
                    # Select the belief with highest confidence
                    highest_conf_belief = max(beliefs, key=lambda b: b.to_probability())

                    # Create new belief based on the highest confidence one
                    reconciled_id = self.create_belief(
                        content=content,
                        uncertainty_type=highest_conf_belief.uncertainty_type,
                        uncertainty_value=self._get_uncertainty_value(highest_conf_belief),
                        source="reconciliation",
                        metadata={
                            "reconciliation_strategy": reconciliation_strategy,
                            "source_beliefs": belief_ids
                        },
                        evidence={
                            "source": "reconciliation",
                            "description": f"Selected from {len(beliefs)} beliefs using {reconciliation_strategy}"
                        }
                    )

                    return reconciled_id

                else:
                    self.logger.warning(f"Unknown reconciliation strategy: {reconciliation_strategy}")
                    return None

            except Exception as e:
                self.logger.error(f"Error reconciling beliefs: {str(e)}")
                return None

    def calculate_certainty_factor(self, evidence_factors: List[float],
                                 combination_rule: str = "cf_model") -> float:
        """
        Calculate a certainty factor from multiple pieces of evidence.

        Args:
            evidence_factors: List of certainty factors from evidence
            combination_rule: Rule for combining certainty factors

        Returns:
            Combined certainty factor
        """
        if not evidence_factors:
            return 0.0

        try:
            if combination_rule == "cf_model":
                # MYCIN certainty factor model
                cf = evidence_factors[0]

                for i in range(1, len(evidence_factors)):
                    cf2 = evidence_factors[i]

                    if cf >= 0 and cf2 >= 0:
                        # Both positive
                        cf = cf + cf2 - cf * cf2
                    elif cf < 0 and cf2 < 0:
                        # Both negative
                        cf = cf + cf2 + cf * cf2
                    else:
                        # One positive, one negative
                        cf = (cf + cf2) / (1 - min(abs(cf), abs(cf2)))

                return cf

            elif combination_rule == "min_max":
                # Conservative combination using min/max
                pos_factors = [f for f in evidence_factors if f > 0]
                neg_factors = [f for f in evidence_factors if f < 0]

                pos_cf = max(pos_factors) if pos_factors else 0
                neg_cf = min(neg_factors) if neg_factors else 0

                return pos_cf + neg_cf  # Sum of positive and negative CFs

            elif combination_rule == "average":
                # Simple average
                return sum(evidence_factors) / len(evidence_factors)

            else:
                self.logger.warning(f"Unknown combination rule: {combination_rule}")
                return sum(evidence_factors) / len(evidence_factors)

        except Exception as e:
            self.logger.error(f"Error calculating certainty factor: {str(e)}")
            return 0.0

    def fuzzy_inference(self, fuzzy_rules: List[Dict[str, Any]],
                       inputs: Dict[str, float]) -> Dict[str, float]:
        """
        Perform fuzzy logic inference based on fuzzy rules.

        Args:
            fuzzy_rules: List of fuzzy rules
            inputs: Input values for fuzzy variables

        Returns:
            Dictionary of output fuzzy variable values
        """
        if not self.initialized:
            self.initialize()

        try:
            # Simple Mamdani-style fuzzy inference
            rule_outputs = []

            for rule in fuzzy_rules:
                # Evaluate antecedent (if-part)
                if 'antecedent' not in rule:
                    continue

                antecedent_value = self._evaluate_fuzzy_expression(rule['antecedent'], inputs)

                # Apply consequent (then-part) if antecedent has nonzero value
                if antecedent_value > 0 and 'consequent' in rule:
                    consequent = rule['consequent']
                    rule_outputs.append({
                        'variable': consequent['variable'],
                        'value': consequent['value'],
                        'strength': antecedent_value
                    })

            # Combine rule outputs for each output variable
            output_values = {}

            for output in rule_outputs:
                var_name = output['variable']

                if var_name not in output_values:
                    output_values[var_name] = 0.0

                # Take maximum for each output variable (standard Mamdani approach)
                output_values[var_name] = max(output_values[var_name], output['strength'])

            return output_values

        except Exception as e:
            self.logger.error(f"Error in fuzzy inference: {str(e)}")
            return {}

    def get_metrics(self) -> Dict[str, Any]:
        """Get metrics about uncertainty management performance"""
        with self.lock:
            metrics_snapshot = {}

            # Overall belief metrics
            metrics_snapshot["total_beliefs"] = len(self.beliefs)
            metrics_snapshot["total_networks"] = len(self.networks)

            # Uncertainty type distribution
            type_counts = defaultdict(int)
            for belief in self.beliefs.values():
                type_counts[belief.uncertainty_type.name] += 1
            metrics_snapshot["uncertainty_type_distribution"] = dict(type_counts)

            # Performance metrics
            for key, values in self.metrics.items():
                if values:
                    metrics_snapshot[key] = {
                        'current': values[-1],
                        'average': sum(values) / len(values),
                        'min': min(values),
                        'max': max(values)
                    }

            return metrics_snapshot

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
            if self.reasoning_core:
                dependencies.append("reasoning_core")

            self.nexus_core.register_module(
                name="uncertainty",
                module=self,
                dependencies=dependencies
            )

            self.logger.info("Successfully registered with NexusCore")
        except Exception as e:
            self.logger.error(f"Failed to register with NexusCore: {str(e)}")

    def _update_belief_indices(self, belief: UncertainBelief) -> None:
        """
        Update indices for faster belief lookup.

        Args:
            belief: The belief to index
        """
        # Index by uncertainty type
        self.belief_index[f"type:{belief.uncertainty_type.name}"].append(belief.belief_id)

        # Index by source if available
        if belief.source:
            self.belief_index[f"source:{belief.source}"].append(belief.belief_id)

        # Index by probability range (converted if needed)
        prob = belief.to_probability()
        prob_range = int(prob * 10) / 10  # Round to nearest 0.1
        self.belief_index[f"probability:{prob_range}"].append(belief.belief_id)

        # Add metadata indices if available
        if belief.metadata:
            for key, value in belief.metadata.items():
                if isinstance(value, (str, int, float, bool)):
                    self.belief_index[f"metadata:{key}:{value}"].append(belief.belief_id)

    def _get_uncertainty_value(self, belief: UncertainBelief) -> Any:
        """Get the appropriate uncertainty value from a belief based on its type"""
        if belief.uncertainty_type == UncertaintyType.PROBABILITY:
            return belief.probability
        elif belief.uncertainty_type == UncertaintyType.INTERVAL:
            return belief.interval
        elif belief.uncertainty_type == UncertaintyType.DISTRIBUTION:
            return belief.distribution_params
        elif belief.uncertainty_type == UncertaintyType.FUZZY:
            return belief.fuzzy_membership
        elif belief.uncertainty_type == UncertaintyType.DEMPSTER_SHAFER:
            return belief.belief_mass
        elif belief.uncertainty_type == UncertaintyType.POSSIBILITY:
            return belief.possibility
        elif belief.uncertainty_type == UncertaintyType.VERBAL:
            return belief.verbal_confidence
        else:
            return None

    def _combine_uncertainty_values(self, old_value: Any, new_value: Any,
                                  uncertainty_type: UncertaintyType,
                                  method: str) -> Any:
        """
        Combine old and new uncertainty values using specified method.

        Args:
            old_value: Existing uncertainty value
            new_value: New uncertainty value
            uncertainty_type: Type of uncertainty
            method: Combination method ('replace', 'bayesian', 'weighted')

        Returns:
            Combined uncertainty value
        """
        if method == "replace":
            # Simple replacement
            return new_value

        elif method == "bayesian":
            # Bayesian update (only for probability)
            if uncertainty_type == UncertaintyType.PROBABILITY:
                # Simplified Bayesian update (for binary hypothesis)
                # P(h|e) = P(e|h) * P(h) / [P(e|h) * P(h) + P(e|~h) * P(~h)]
                # Assuming P(e|h) = new_value and P(e|~h) = 1 - new_value
                prior = old_value
                likelihood = new_value

                numerator = likelihood * prior
                denominator = likelihood * prior + (1 - likelihood) * (1 - prior)

                if denominator == 0:
                    return prior  # Avoid division by zero

                return numerator / denominator

            elif uncertainty_type == UncertaintyType.INTERVAL:
                # For interval, use Bayesian update on endpoints
                prior_min, prior_max = old_value
                likelihood_min, likelihood_max = new_value

                # Update min
                num_min = likelihood_min * prior_min
                denom_min = likelihood_min * prior_min + (1 - likelihood_min) * (1 - prior_min)
                if denom_min == 0:
                    posterior_min = prior_min
                else:
                    posterior_min = num_min / denom_min

                # Update max
                num_max = likelihood_max * prior_max
                denom_max = likelihood_max * prior_max + (1 - likelihood_max) * (1 - prior_max)
                if denom_max == 0:
                    posterior_max = prior_max
                else:
                    posterior_max = num_max / denom_max

                return (posterior_min, posterior_max)

            else:
                # For other types, fall back to replacement
                self.logger.warning(f"Bayesian update not supported for {uncertainty_type.name}")
                return new_value

        elif method == "weighted":
            # Weighted combination
            if uncertainty_type == UncertaintyType.PROBABILITY:
                # Weighted average of probabilities
                return (old_value + new_value) / 2

            elif uncertainty_type == UncertaintyType.INTERVAL:
                # Weighted average of interval endpoints
                old_min, old_max = old_value
                new_min, new_max = new_value

                return ((old_min + new_min) / 2, (old_max + new_max) / 2)

            elif uncertainty_type == UncertaintyType.DISTRIBUTION:
                # For distribution, try to average parameters
                result = {}

                # Merge keys
                all_keys = set(old_value.keys()) | set(new_value.keys())

                for key in all_keys:
                    if key in old_value and key in new_value:
                        # Average numeric values
                        if isinstance(old_value[key], (int, float)) and isinstance(new_value[key], (int, float)):
                            result[key] = (old_value[key] + new_value[key]) / 2
                        else:
                            # For non-numeric values, keep the new value
                            result[key] = new_value[key]
                    elif key in new_value:
                        result[key] = new_value[key]
                    else:
                        result[key] = old_value[key]

                return result

            elif uncertainty_type == UncertaintyType.FUZZY:
                # Average of fuzzy memberships
                return (old_value + new_value) / 2

            elif uncertainty_type == UncertaintyType.DEMPSTER_SHAFER:
                # Simplified combination of belief masses
                result = {}

                # Merge keys
                all_keys = set(old_value.keys()) | set(new_value.keys())

                for key in all_keys:
                    if key in old_value and key in new_value:
                        # Simple average for masses
                        result[key] = (old_value[key] + new_value[key]) / 2
                    elif key in new_value:
                        result[key] = new_value[key] / 2  # Reduce weight for new-only keys
                    else:
                        result[key] = old_value[key] / 2  # Reduce weight for old-only keys

                # Normalize masses
                total_mass = sum(result.values())
                if total_mass > 0:
                    for key in result:
                        result[key] /= total_mass

                return result

            elif uncertainty_type == UncertaintyType.POSSIBILITY:
                # Average of possibility values
                return (old_value + new_value) / 2

            elif uncertainty_type == UncertaintyType.VERBAL:
                # For verbal, prefer the new value
                return new_value

            else:
                return new_value

        else:
            # Unknown method
            self.logger.warning(f"Unknown update method: {method}")
            return new_value

    def _exact_inference(self, network: BayesianNetwork,
                       query_nodes: List[str],
                       evidence: Dict[str, str]) -> Dict[str, Dict[str, float]]:
        """
        Perform exact inference on a Bayesian network.

        Args:
            network: The Bayesian network
            query_nodes: Nodes to query
            evidence: Observed evidence

        Returns:
            Dictionary mapping node IDs to probability distributions
        """
        # This is a simplified implementation for small networks
        # For real applications, use libraries like pgmpy or pyAgrum

        # Get all nodes
        all_nodes = list(network.nodes.keys())

        # Find nodes that are not in evidence or query
        hidden_nodes = [n for n in all_nodes if n not in evidence and n not in query_nodes]

        # For each query node, compute marginal distribution
        result = {}

        for query_node in query_nodes:
            node = network.get_node(query_node)
            if not node:
                continue

            # Get possible states for this node
            states = node.states

            # Initialize distribution
            distribution = {state: 0.0 for state in states}

            # For a small network, we can enumerate all possible hidden node states
            # This is inefficient for larger networks
            hidden_state_combinations = self._generate_state_combinations(network, hidden_nodes)

            for hidden_states in hidden_state_combinations:
                # Combine with evidence
                full_assignment = {**evidence, **hidden_states}

                # Compute probability of this assignment
                assignment_prob = 1.0

                for node_id, node_obj in network.nodes.items():
                    state = full_assignment.get(node_id)
                    if not state:
                        continue

                    # Get parent states
                    parent_states = {p: full_assignment.get(p) for p in node_obj.parents}

                    # Skip if parent states are incomplete
                    if any(v is None for v in parent_states.values()):
                        continue

                    # Get conditional probability
                    state_prob = node_obj.get_probability(state, parent_states)

                    # Update assignment probability
                    assignment_prob *= state_prob

                # For each possible state of the query node
                for state in states:
                    # Create a new assignment with this state
                    state_assignment = {**full_assignment, query_node: state}

                    # Compute probability with this state
                    state_prob = 1.0

                    for node_id, node_obj in network.nodes.items():
                        node_state = state_assignment.get(node_id)
                        if not node_state:
                            continue

                        # Get parent states
                        parent_states = {p: state_assignment.get(p) for p in node_obj.parents}

                        # Skip if parent states are incomplete
                        if any(v is None for v in parent_states.values()):
                            continue

                        # Get conditional probability
                        node_prob = node_obj.get_probability(node_state, parent_states)

                        # Update state probability
                        state_prob *= node_prob

                    # Add to state distribution
                    distribution[state] += state_prob

            # Normalize distribution
            total_prob = sum(distribution.values())
            if total_prob > 0:
                for state in distribution:
                    distribution[state] /= total_prob

            # Store result
            result[query_node] = distribution

        return result

    def _approximate_inference(self, network: BayesianNetwork,
                             query_nodes: List[str],
                             evidence: Dict[str, str]) -> Dict[str, Dict[str, float]]:
        """
        Perform approximate inference on a Bayesian network.

        Args:
            network: The Bayesian network
            query_nodes: Nodes to query
            evidence: Observed evidence

        Returns:
            Dictionary mapping node IDs to probability distributions
        """
        # For simple implementation, use sampling inference
        return self._sampling_inference(network, query_nodes, evidence, num_samples=10000)

    def _sampling_inference(self, network: BayesianNetwork,
                          query_nodes: List[str],
                          evidence: Dict[str, str],
                          num_samples: int = 1000) -> Dict[str, Dict[str, float]]:
        """
        Perform inference via sampling.

        Args:
            network: The Bayesian network
            query_nodes: Nodes to query
            evidence: Observed evidence
            num_samples: Number of samples to draw

        Returns:
            Dictionary mapping node IDs to probability distributions
        """
        # Initialize result dictionary
        result = {}
        for query_node in query_nodes:
            node = network.get_node(query_node)
            if node:
                result[query_node] = {state: 0.0 for state in node.states}

        # Rejection sampling
        valid_samples = 0

        for _ in range(num_samples):
            # Generate a random sample
            sample = self._generate_random_sample(network)

            # Check if sample is consistent with evidence
            if all(sample.get(node) == state for node, state in evidence.items()):
                valid_samples += 1

                # Update counts for query nodes
                for query_node in query_nodes:
                    state = sample.get(query_node)
                    if state is not None and query_node in result:
                        result[query_node][state] += 1

        # Normalize counts to probabilities
        if valid_samples > 0:
            for query_node in query_nodes:
                if query_node in result:
                    for state in result[query_node]:
                        result[query_node][state] /= valid_samples

        return result

    def _generate_random_sample(self, network: BayesianNetwork) -> Dict[str, str]:
        """
        Generate a random sample from a Bayesian network.

        Args:
            network: The Bayesian network

        Returns:
            Dictionary mapping node IDs to sampled states
        """
        # Topologically sort nodes
        sorted_nodes = self._topological_sort(network)

        # Generate sample
        sample = {}

        for node_id in sorted_nodes:
            node = network.get_node(node_id)
            if not node:
                continue

            # Get parent states
            parent_states = {p: sample.get(p) for p in node.parents}

            # Skip if parent states are incomplete
            if any(v is None for v in parent_states.values()) and node.parents:
                continue

            # Get probability distribution for each state
            state_probs = {}
            total_prob = 0.0

            for state in node.states:
                prob = node.get_probability(state, parent_states)
                state_probs[state] = prob
                total_prob += prob

            # Normalize if needed
            if total_prob > 0:
                for state in state_probs:
                    state_probs[state] /= total_prob

            # Sample a state
            r = random.random()
            cumulative_prob = 0.0

            for state, prob in state_probs.items():
                cumulative_prob += prob
                if r <= cumulative_prob:
                    sample[node_id] = state
                    break

            # Fallback if no state was selected
            if node_id not in sample and node.states:
                sample[node_id] = random.choice(node.states)

        return sample

    def _topological_sort(self, network: BayesianNetwork) -> List[str]:
        """
        Topologically sort nodes in a Bayesian network.

        Args:
            network: The Bayesian network

        Returns:
            List of node IDs in topological order
        """
        # Get all nodes
        nodes = list(network.nodes.keys())

        # Initialize
        visited = set()
        temp_mark = set()
        result = []

        def visit(node_id):
            if node_id in temp_mark:
                # Cycle detected, just continue
                return
            if node_id in visited:
                return

            temp_mark.add(node_id)

            # Visit all parents
            node = network.get_node(node_id)
            if node:
                for parent in node.parents:
                    visit(parent)

            temp_mark.remove(node_id)
            visited.add(node_id)
            result.append(node_id)

        # Visit all nodes
        for node_id in nodes:
            if node_id not in visited:
                visit(node_id)

        # Reverse to get correct order
        return list(reversed(result))

    def _generate_state_combinations(self, network: BayesianNetwork,
                                   nodes: List[str]) -> List[Dict[str, str]]:
        """
        Generate all possible state combinations for a list of nodes.

        Args:
            network: The Bayesian network
            nodes: List of node IDs

        Returns:
            List of dictionaries mapping node IDs to states
        """
        if not nodes:
            return [{}]

        # Start with first node
        node_id = nodes[0]
        node = network.get_node(node_id)

        if not node:
            return self._generate_state_combinations(network, nodes[1:])

        # Get states for this node
        states = node.states

        # Generate combinations for remaining nodes
        remaining_combinations = self._generate_state_combinations(network, nodes[1:])

        # Combine with states for this node
        result = []
        for state in states:
            for combination in remaining_combinations:
                result.append({**{node_id: state}, **combination})

        return result

    def _evaluate_fuzzy_expression(self, expression: Dict[str, Any],
                                 inputs: Dict[str, float]) -> float:
        """
        Evaluate a fuzzy logic expression.

        Args:
            expression: The fuzzy expression
            inputs: Input values

        Returns:
            Fuzzy value
        """
        if 'type' not in expression:
            return 0.0

        expr_type = expression['type']

        if expr_type == 'input':
            # Direct input reference
            var_name = expression.get('variable')
            if var_name in inputs:
                return inputs[var_name]
            return 0.0

        elif expr_type == 'and':
            # Fuzzy AND (minimum)
            operands = expression.get('operands', [])
            if not operands:
                return 0.0

            values = [self._evaluate_fuzzy_expression(op, inputs) for op in operands]
            return min(values)

        elif expr_type == 'or':
            # Fuzzy OR (maximum)
            operands = expression.get('operands', [])
            if not operands:
                return 0.0

            values = [self._evaluate_fuzzy_expression(op, inputs) for op in operands]
            return max(values)

        elif expr_type == 'not':
            # Fuzzy NOT (complement)
            operand = expression.get('operand')
            if not operand:
                return 0.0

            value = self._evaluate_fuzzy_expression(operand, inputs)
            return 1.0 - value

        elif expr_type == 'membership':
            # Fuzzy membership function
            var_name = expression.get('variable')
            if var_name not in inputs:
                return 0.0

            value = inputs[var_name]
            membership_type = expression.get('membership_type', 'triangle')

            if membership_type == 'triangle':
                a = expression.get('a', 0.0)
                b = expression.get('b', 0.5)
                c = expression.get('c', 1.0)

                if value <= a or value >= c:
                    return 0.0
                elif a < value <= b:
                    return (value - a) / (b - a)
                else:  # b < value < c
                    return (c - value) / (c - b)

            elif membership_type == 'trapezoid':
                a = expression.get('a', 0.0)
                b = expression.get('b', 0.25)
                c = expression.get('c', 0.75)
                d = expression.get('d', 1.0)

                if value <= a or value >= d:
                    return 0.0
                elif a < value <= b:
                    return (value - a) / (b - a)
                elif b < value <= c:
                    return 1.0
                else:  # c < value < d
                    return (d - value) / (d - c)

            elif membership_type == 'gaussian':
                mean = expression.get('mean', 0.5)
                std = expression.get('std', 0.2)

                return math.exp(-((value - mean) ** 2) / (2 * std ** 2))

            else:
                return 0.0

        else:
            return 0.0