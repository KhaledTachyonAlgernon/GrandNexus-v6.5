"""
attention_filter.py - Attentional Filter Module for GrandNexus

This module provides a framework for filtering and highlighting information based on
attentional mechanisms, allowing GrandNexus to focus on relevant data while ignoring
distractions. It implements various attentional strategies inspired by cognitive sciences,
including:

- Bottom-up attention: Driven by inherent properties of stimuli (salience, novelty)
- Top-down attention: Guided by current goals, expectations, and context
- Divided attention: Managing multiple sources of information simultaneously
- Sustained attention: Maintaining focus over time despite distractions

The module is designed to work closely with perception and reasoning components, enhancing
the system's ability to extract meaningful patterns from complex data streams.
"""

import logging
import threading
import time
import uuid
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Dict, List, Any, Optional, Callable, Union, Set, Tuple
import numpy as np
from collections import defaultdict, deque
import re
import random

# Assuming these exist or will exist
from grandnexus.core.nexus_core import NexusCore
from grandnexus.memory.working import WorkingMemory
from grandnexus.memory.semantic import SemanticMemory
from grandnexus.cognition.perception.perception_core import PerceptionCore


class AttentionMode(Enum):
    """Different modes of attentional filtering"""
    BOTTOM_UP = auto()    # Salience-driven attention
    TOP_DOWN = auto()     # Goal-directed attention
    DIVIDED = auto()      # Multi-focus attention
    SUSTAINED = auto()    # Long-term focus maintenance
    HYBRID = auto()       # Combination of strategies
    EXPLORATORY = auto()  # Novelty-seeking attention
    URGENT = auto()       # Urgent signal prioritization


@dataclass
class AttentionalFocus:
    """Representation of the current attentional focus"""
    focus_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    mode: AttentionMode = AttentionMode.HYBRID
    target_concepts: List[str] = field(default_factory=list)
    target_patterns: List[str] = field(default_factory=list)
    importance_threshold: float = 0.3
    activation_level: float = 1.0
    start_time: float = field(default_factory=time.time)
    duration: Optional[float] = None  # None means indefinite
    decay_rate: float = 0.05
    context: Dict[str, Any] = field(default_factory=dict)

    def is_active(self, current_time: Optional[float] = None) -> bool:
        """Check if the attentional focus is still active"""
        if current_time is None:
            current_time = time.time()

        if self.duration is not None:
            return (current_time - self.start_time) < self.duration
        return True

    def get_current_activation(self, current_time: Optional[float] = None) -> float:
        """Get the current activation level, accounting for decay"""
        if current_time is None:
            current_time = time.time()

        time_elapsed = current_time - self.start_time

        # Apply exponential decay
        decay_factor = np.exp(-self.decay_rate * time_elapsed)
        return self.activation_level * decay_factor

    def matches_content(self, content: Union[str, Dict[str, Any]]) -> Tuple[bool, float]:
        """
        Check if content matches this attentional focus.

        Args:
            content: String or dictionary content to check

        Returns:
            Tuple of (matches, score) where score indicates strength of match
        """
        if isinstance(content, str):
            text = content
            metadata = {}
        elif isinstance(content, dict):
            text = content.get("text", "")
            metadata = content
        else:
            # Unsupported content type
            return False, 0.0

        # Check for concept matches
        concept_matches = 0
        for concept in self.target_concepts:
            if concept.lower() in text.lower():
                concept_matches += 1

        # Check for pattern matches
        pattern_matches = 0
        for pattern in self.target_patterns:
            try:
                if re.search(pattern, text, re.IGNORECASE):
                    pattern_matches += 1
            except re.error:
                # If pattern is not a valid regex, treat it as a literal string
                if pattern.lower() in text.lower():
                    pattern_matches += 1

        # Calculate match score
        total_targets = len(self.target_concepts) + len(self.target_patterns)
        if total_targets == 0:
            return True, 1.0  # No targets means everything matches

        match_ratio = (concept_matches + pattern_matches) / total_targets

        # Check metadata match if available
        metadata_match = 0.0
        if metadata and self.context:
            matching_keys = set(metadata.keys()) & set(self.context.keys())
            if matching_keys:
                for key in matching_keys:
                    if metadata[key] == self.context[key]:
                        metadata_match += 1.0 / len(matching_keys)

        # Combine scores
        combined_score = 0.7 * match_ratio + 0.3 * metadata_match

        # Apply activation as a weight
        current_activation = self.get_current_activation()
        weighted_score = combined_score * current_activation

        return weighted_score >= self.importance_threshold, weighted_score


@dataclass
class FilteredPerception:
    """Result of applying attentional filters to perceptual input"""
    original_input: Any
    filtered_output: Any
    attention_score: float = 0.0
    matching_focus: Optional[str] = None  # ID of matching attentional focus
    filtered_by: List[str] = field(default_factory=list)  # IDs of applied filters
    metadata: Dict[str, Any] = field(default_factory=dict)
    timestamp: float = field(default_factory=time.time)


@dataclass
class SaliencyMap:
    """
    Represents a map of content importance based on various factors.
    This is used for bottom-up attention to highlight inherently salient information.
    """
    content_id: str
    base_map: Dict[str, float] = field(default_factory=dict)  # Element -> salience
    temporal_factors: Dict[str, float] = field(default_factory=dict)  # Time-based adjustments
    contextual_factors: Dict[str, float] = field(default_factory=dict)  # Context-based adjustments

    def get_combined_salience(self, element: str) -> float:
        """Get the combined salience value for an element"""
        base = self.base_map.get(element, 0.0)
        temporal = self.temporal_factors.get(element, 1.0)
        contextual = self.contextual_factors.get(element, 1.0)

        return base * temporal * contextual

    def sort_by_salience(self) -> List[Tuple[str, float]]:
        """Sort elements by their combined salience"""
        elements = list(set(self.base_map.keys()) |
                       set(self.temporal_factors.keys()) |
                       set(self.contextual_factors.keys()))

        salience_scores = [(elem, self.get_combined_salience(elem)) for elem in elements]
        return sorted(salience_scores, key=lambda x: x[1], reverse=True)


class AttentionFilter:
    """
    Main class for attentional filtering in GrandNexus.

    This module filters perceptual input based on current attentional focus,
    goals, and salience of the information itself. It helps GrandNexus focus
    on relevant information while preventing distraction from less important data.
    """

    def __init__(self, nexus_core: Optional[NexusCore] = None,
                 perception_core: Optional[PerceptionCore] = None,
                 working_memory: Optional[WorkingMemory] = None,
                 semantic_memory: Optional[SemanticMemory] = None):
        """
        Initialize the attention filter module.

        Args:
            nexus_core: Reference to the NexusCore for system-wide coordination
            perception_core: Reference to the perception module for input
            working_memory: Reference to working memory
            semantic_memory: Reference to semantic memory for concept lookups
        """
        self.logger = logging.getLogger("GrandNexus.Perception.AttentionFilter")
        self.nexus_core = nexus_core
        self.perception_core = perception_core
        self.working_memory = working_memory
        self.semantic_memory = semantic_memory

        # Internal state
        self.instance_id = str(uuid.uuid4())[:8]
        self.running = False
        self.initialized = False
        self.lock = threading.RLock()

        # Attentional focus management
        self.attentional_foci = {}  # ID -> AttentionalFocus
        self.default_focus = None  # Default focus when no specific focus matches

        # Salience tracking
        self.saliency_maps = {}  # Content ID -> SaliencyMap

        # Filter history and metrics
        self.filter_history = deque(maxlen=100)
        self.metrics = defaultdict(lambda: deque(maxlen=100))

        # Filter registry
        self.filters = {}

        self.logger.info(f"AttentionFilter initialized with ID={self.instance_id}")

    def initialize(self) -> bool:
        """Initialize the attention filter and register default filters"""
        if self.initialized:
            return True

        with self.lock:
            try:
                # Register built-in filters
                self._register_default_filters()

                # Create default attentional focus
                self._create_default_focus()

                # Connect to other modules
                if self.nexus_core:
                    self._register_with_nexus_core()

                self.initialized = True
                self.logger.info("AttentionFilter initialization complete")
                return True
            except Exception as e:
                self.logger.error(f"AttentionFilter initialization failed: {str(e)}")
                return False

    def start(self) -> bool:
        """Start the attention filter"""
        if not self.initialized:
            if not self.initialize():
                return False

        with self.lock:
            if self.running:
                return True

            try:
                self.running = True
                self.logger.info("AttentionFilter started successfully")
                return True
            except Exception as e:
                self.logger.error(f"Failed to start AttentionFilter: {str(e)}")
                return False

    def stop(self) -> bool:
        """Stop the attention filter"""
        with self.lock:
            if not self.running:
                return True

            try:
                self.running = False
                self.logger.info("AttentionFilter stopped successfully")
                return True
            except Exception as e:
                self.logger.error(f"Error stopping AttentionFilter: {str(e)}")
                return False

    def filter(self, input_data: Any, context: Optional[Dict[str, Any]] = None) -> FilteredPerception:
        """
        Apply attentional filtering to input data.

        Args:
            input_data: The input data to filter
            context: Optional context information for filtering

        Returns:
            FilteredPerception with filtered output and metadata
        """
        if not self.initialized:
            self.initialize()

        if context is None:
            context = {}

        # Create a working copy of the input data
        filtered_data = self._clone_input(input_data)

        # Apply each registered filter in sequence
        applied_filters = []
        attention_score = 0.0
        matching_focus_id = None

        # Determine active attentional foci
        active_foci = self._get_active_foci()

        # If input_data is a dictionary with a 'content_id', create/update saliency map
        if isinstance(input_data, dict) and 'content_id' in input_data:
            content_id = input_data['content_id']
            self._update_saliency_map(content_id, input_data, context)

        # First check if the input matches any attentional focus
        best_match_score = 0.0
        best_match_id = None

        for focus_id, focus in active_foci.items():
            matches, score = focus.matches_content(input_data)
            if matches and score > best_match_score:
                best_match_score = score
                best_match_id = focus_id

        # If no specific focus matches, use default focus
        if best_match_id is None and self.default_focus is not None:
            best_match_id = self.default_focus.focus_id
            best_match_score = 0.5  # Moderate score for default focus

        # Apply filters based on matching focus and context
        if best_match_id is not None:
            matching_focus = active_foci[best_match_id]
            matching_focus_id = best_match_id
            attention_score = best_match_score

            # Determine which filters to apply based on focus mode
            filter_keys = self._get_filters_for_mode(matching_focus.mode)

            for filter_key in filter_keys:
                if filter_key in self.filters:
                    filter_func = self.filters[filter_key]
                    try:
                        filter_result = filter_func(filtered_data, matching_focus, context)
                        if isinstance(filter_result, tuple) and len(filter_result) == 2:
                            # Filter returns (filtered_data, changed)
                            filtered_data, changed = filter_result
                            if changed:
                                applied_filters.append(filter_key)
                        else:
                            # Filter just returns filtered_data
                            filtered_data = filter_result
                            applied_filters.append(filter_key)
                    except Exception as e:
                        self.logger.error(f"Error applying filter {filter_key}: {str(e)}")

        # Update metrics
        self.metrics["attention_score"].append(attention_score)
        self.metrics["filters_applied"].append(len(applied_filters))

        # Create result
        result = FilteredPerception(
            original_input=input_data,
            filtered_output=filtered_data,
            attention_score=attention_score,
            matching_focus=matching_focus_id,
            filtered_by=applied_filters,
            metadata={"context": context}
        )

        # Update history
        self.filter_history.append(result)

        return result

    def create_focus(self, mode: AttentionMode,
                     target_concepts: Optional[List[str]] = None,
                     target_patterns: Optional[List[str]] = None,
                     importance_threshold: float = 0.3,
                     duration: Optional[float] = None,
                     context: Optional[Dict[str, Any]] = None) -> str:
        """
        Create a new attentional focus.

        Args:
            mode: Mode of attention to use
            target_concepts: Concepts to focus on
            target_patterns: Patterns to look for
            importance_threshold: Minimum importance for focus activation
            duration: Optional duration in seconds (None=indefinite)
            context: Additional contextual information

        Returns:
            ID of the created attentional focus
        """
        if target_concepts is None:
            target_concepts = []
        if target_patterns is None:
            target_patterns = []
        if context is None:
            context = {}

        focus = AttentionalFocus(
            mode=mode,
            target_concepts=target_concepts,
            target_patterns=target_patterns,
            importance_threshold=importance_threshold,
            duration=duration,
            context=context
        )

        with self.lock:
            self.attentional_foci[focus.focus_id] = focus
            self.logger.info(f"Created attentional focus {focus.focus_id} with mode {mode}")

        return focus.focus_id

    def update_focus(self, focus_id: str,
                    target_concepts: Optional[List[str]] = None,
                    target_patterns: Optional[List[str]] = None,
                    importance_threshold: Optional[float] = None,
                    duration: Optional[float] = None,
                    context: Optional[Dict[str, Any]] = None) -> bool:
        """
        Update an existing attentional focus.

        Args:
            focus_id: ID of the focus to update
            target_concepts: New concepts to focus on (None to keep existing)
            target_patterns: New patterns to look for (None to keep existing)
            importance_threshold: New threshold (None to keep existing)
            duration: New duration (None to keep existing)
            context: Additional contextual information (None to keep existing)

        Returns:
            True if update succeeded, False otherwise
        """
        with self.lock:
            if focus_id not in self.attentional_foci:
                self.logger.warning(f"Attentional focus {focus_id} not found")
                return False

            focus = self.attentional_foci[focus_id]

            if target_concepts is not None:
                focus.target_concepts = target_concepts
            if target_patterns is not None:
                focus.target_patterns = target_patterns
            if importance_threshold is not None:
                focus.importance_threshold = importance_threshold
            if duration is not None:
                focus.duration = duration
            if context is not None:
                focus.context = context

            self.logger.info(f"Updated attentional focus {focus_id}")
            return True

    def remove_focus(self, focus_id: str) -> bool:
        """
        Remove an attentional focus.

        Args:
            focus_id: ID of the focus to remove

        Returns:
            True if removal succeeded, False otherwise
        """
        with self.lock:
            if focus_id not in self.attentional_foci:
                self.logger.warning(f"Attentional focus {focus_id} not found")
                return False

            # Don't remove the default focus
            if self.default_focus is not None and focus_id == self.default_focus.focus_id:
                self.logger.warning("Cannot remove default attentional focus")
                return False

            del self.attentional_foci[focus_id]
            self.logger.info(f"Removed attentional focus {focus_id}")
            return True

    def get_focus(self, focus_id: str) -> Optional[AttentionalFocus]:
        """
        Get information about an attentional focus.

        Args:
            focus_id: ID of the focus to get

        Returns:
            AttentionalFocus if found, None otherwise
        """
        with self.lock:
            return self.attentional_foci.get(focus_id)

    def set_default_focus(self, focus_id: str) -> bool:
        """
        Set the default attentional focus.

        Args:
            focus_id: ID of the focus to set as default

        Returns:
            True if setting succeeded, False otherwise
        """
        with self.lock:
            if focus_id not in self.attentional_foci:
                self.logger.warning(f"Attentional focus {focus_id} not found")
                return False

            self.default_focus = self.attentional_foci[focus_id]
            self.logger.info(f"Set default attentional focus to {focus_id}")
            return True

    def register_filter(self, filter_key: str,
                        filter_func: Callable[[Any, AttentionalFocus, Dict[str, Any]], Any]) -> None:
        """
        Register a custom filter function.

        Args:
            filter_key: Unique key for the filter
            filter_func: Function that takes (data, focus, context) and returns filtered data
                         or (filtered_data, changed) tuple
        """
        with self.lock:
            self.filters[filter_key] = filter_func
            self.logger.info(f"Registered filter: {filter_key}")

    def get_filter_metrics(self) -> Dict[str, Any]:
        """Get current filtering performance metrics"""
        with self.lock:
            metrics_snapshot = {}
            for key, values in self.metrics.items():
                if values:
                    metrics_snapshot[key] = {
                        'current': values[-1],
                        'average': sum(values) / len(values),
                        'min': min(values),
                        'max': max(values)
                    }
            return metrics_snapshot

    def get_active_foci(self) -> Dict[str, Dict[str, Any]]:
        """
        Get information about all active attentional foci.

        Returns:
            Dictionary mapping focus IDs to information dictionaries
        """
        active_foci = self._get_active_foci()

        # Convert to serializable format
        result = {}
        for focus_id, focus in active_foci.items():
            result[focus_id] = {
                "mode": focus.mode.name,
                "target_concepts": focus.target_concepts,
                "target_patterns": focus.target_patterns,
                "activation": focus.get_current_activation(),
                "start_time": focus.start_time,
                "duration": focus.duration
            }

        return result

    # -----------------------------------------------------------------------
    # Internal methods
    # -----------------------------------------------------------------------

    def _register_default_filters(self) -> None:
        """Register built-in filter functions"""
        # Bottom-up salience filter
        self.register_filter("salience", self._filter_by_salience)

        # Top-down selective attention
        self.register_filter("selective", self._filter_by_selective_attention)

        # Conceptual relevance filter
        self.register_filter("concept_relevance", self._filter_by_concept_relevance)

        # Pattern matching filter
        self.register_filter("pattern_matching", self._filter_by_pattern_matching)

        # Urgency filter
        self.register_filter("urgency", self._filter_by_urgency)

        # Novelty filter
        self.register_filter("novelty", self._filter_by_novelty)

    def _create_default_focus(self) -> None:
        """Create a default attentional focus"""
        default_focus_id = self.create_focus(
            mode=AttentionMode.HYBRID,
            target_concepts=[],
            target_patterns=[],
            importance_threshold=0.1  # Low threshold to match most things
        )

        self.default_focus = self.attentional_foci[default_focus_id]

    def _register_with_nexus_core(self) -> None:
        """Register with NexusCore for system-wide coordination"""
        if not self.nexus_core:
            self.logger.warning("No NexusCore available for registration")
            return

        try:
            # Register as a module with NexusCore
            dependencies = []
            if self.perception_core:
                dependencies.append("perception_core")
            if self.working_memory:
                dependencies.append("working_memory")
            if self.semantic_memory:
                dependencies.append("semantic_memory")

            self.nexus_core.register_module(
                name="attention_filter",
                module=self,
                dependencies=dependencies
            )

            self.logger.info("Successfully registered with NexusCore")
        except Exception as e:
            self.logger.error(f"Failed to register with NexusCore: {str(e)}")

    def _clone_input(self, input_data: Any) -> Any:
        """Create a working copy of input data"""
        if isinstance(input_data, dict):
            return input_data.copy()
        elif isinstance(input_data, list):
            return input_data.copy()
        elif isinstance(input_data, str):
            return input_data
        else:
            # For other types, we might need more sophisticated cloning
            # This is a simplified approach
            return input_data

    def _get_active_foci(self) -> Dict[str, AttentionalFocus]:
        """Get all currently active attentional foci"""
        current_time = time.time()

        with self.lock:
            active_foci = {}
            for focus_id, focus in self.attentional_foci.items():
                if focus.is_active(current_time):
                    active_foci[focus_id] = focus

            return active_foci

    def _get_filters_for_mode(self, mode: AttentionMode) -> List[str]:
        """Get appropriate filters for a given attention mode"""
        # Define filter sets for each mode
        filter_sets = {
            AttentionMode.BOTTOM_UP: ["salience", "novelty"],
            AttentionMode.TOP_DOWN: ["selective", "concept_relevance"],
            AttentionMode.DIVIDED: ["selective", "salience"],
            AttentionMode.SUSTAINED: ["selective", "concept_relevance", "pattern_matching"],
            AttentionMode.HYBRID: ["salience", "selective", "concept_relevance", "pattern_matching"],
            AttentionMode.EXPLORATORY: ["novelty", "salience"],
            AttentionMode.URGENT: ["urgency", "selective"]
        }

        return filter_sets.get(mode, ["selective"])  # Default to selective as fallback

    def _update_saliency_map(self, content_id: str, content: Dict[str, Any],
                            context: Dict[str, Any]) -> None:
        """Create or update a saliency map for content"""
        # This is a simplified implementation
        # In a real system, this would analyze content more deeply

        # Create base map if it doesn't exist
        if content_id not in self.saliency_maps:
            self.saliency_maps[content_id] = SaliencyMap(content_id=content_id)

        saliency_map = self.saliency_maps[content_id]

        # Update base map
        if "text" in content:
            text = content["text"]
            words = text.split()

            # Simple heuristics for salience:
            # - Words in ALL CAPS are more salient
            # - Words with exclamation marks are more salient
            # - Longer words may be more salient (e.g., technical terms)

            for word in words:
                clean_word = word.strip(".,!?;:()")
                if not clean_word:
                    continue

                salience = 0.5  # Default salience

                # ALL CAPS words
                if clean_word.isupper() and len(clean_word) > 1:
                    salience += 0.3

                # Words with emphasis
                if "!" in word:
                    salience += 0.2

                # Longer words
                if len(clean_word) > 8:
                    salience += 0.1

                saliency_map.base_map[clean_word] = salience

        # Update temporal factors (recency bias)
        current_time = time.time()
        for word in saliency_map.base_map:
            # Newer mentions get higher temporal factors
            saliency_map.temporal_factors[word] = 1.0

        # Update contextual factors based on current context
        if "goal" in context:
            goal = context["goal"]
            goal_words = goal.lower().split()

            for word in saliency_map.base_map:
                if word.lower() in goal_words:
                    saliency_map.contextual_factors[word] = 1.5  # Boost if related to goal
                else:
                    saliency_map.contextual_factors[word] = 1.0

    # -----------------------------------------------------------------------
    # Filter implementations
    # -----------------------------------------------------------------------

    def _filter_by_salience(self, data: Any, focus: AttentionalFocus,
                           context: Dict[str, Any]) -> Tuple[Any, bool]:
        """
        Filter based on bottom-up salience.
        Highlights or enhances salient elements while suppressing less salient ones.

        Args:
            data: Data to filter
            focus: Current attentional focus
            context: Additional context

        Returns:
            Tuple of (filtered_data, changed)
        """
        # This is a simplified implementation
        # In a real system, this would analyze content more deeply

        # For text data, we might highlight salient words
        if isinstance(data, str):
            # For now, we'll just return the original data
            # In a real implementation, we might add markup or structure
            return data, False

        elif isinstance(data, dict) and "text" in data:
            # If we have a content_id, we might have a saliency map
            content_id = data.get("content_id", "")
            if content_id in self.saliency_maps:
                saliency_map = self.saliency_maps[content_id]

                # Get top salient elements
                top_elements = saliency_map.sort_by_salience()[:5]  # Top 5 most salient

                # Add salience information to the data
                filtered_data = data.copy()
                filtered_data["salient_elements"] = top_elements

                return filtered_data, True

        # For other data types or no changes
        return data, False

    def _filter_by_selective_attention(self, data: Any, focus: AttentionalFocus,
                                      context: Dict[str, Any]) -> Tuple[Any, bool]:
        """
        Filter based on top-down selective attention.
        Focuses on elements that match current goals and intentions.

        Args:
            data: Data to filter
            focus: Current attentional focus
            context: Additional context

        Returns:
            Tuple of (filtered_data, changed)
        """
        # For text data, prioritize content related to target concepts
        if isinstance(data, str):
            text = data
            highlighted_text = text

            # Highlight target concepts
            for concept in focus.target_concepts:
                if concept.lower() in text.lower():
                    # In a real implementation, we might add markup or structure
                    # Here we just leave it unchanged
                    pass

            if highlighted_text != text:
                return highlighted_text, True
            return text, False

        elif isinstance(data, dict) and "text" in data:
            text = data["text"]
            filtered_data = data.copy()

            # Extract parts most relevant to focus
            relevant_parts = []

            # Split into sentences (simplistic)
            sentences = text.split(". ")

            for sentence in sentences:
                relevance = 0
                for concept in focus.target_concepts:
                    if concept.lower() in sentence.lower():
                        relevance += 1

                if relevance > 0:
                    relevant_parts.append(sentence)

            if relevant_parts:
                filtered_data["relevant_parts"] = relevant_parts
                return filtered_data, True

        # For other data types or no changes
        return data, False

    def _filter_by_selective_attention(self, data: Any, focus: AttentionalFocus,
                                      context: Dict[str, Any]) -> Tuple[Any, bool]:
        """
        Filter based on top-down selective attention.
        Focuses on elements that match current goals and intentions.

        Args:
            data: Data to filter
            focus: Current attentional focus
            context: Additional context

        Returns:
            Tuple of (filtered_data, changed)
        """
        # For text data, prioritize content related to target concepts
        if isinstance(data, str):
            text = data
            highlighted_text = text

            # Highlight target concepts
            for concept in focus.target_concepts:
                if concept.lower() in text.lower():
                    # In a real implementation, we might add markup or structure
                    # Here we just leave it unchanged
                    pass

            if highlighted_text != text:
                return highlighted_text, True
            return text, False

        elif isinstance(data, dict) and "text" in data:
            text = data["text"]
            filtered_data = data.copy()

            # Extract parts most relevant to focus
            relevant_parts = []

            # Split into sentences (simplistic)
            sentences = text.split(". ")

            for sentence in sentences:
                relevance = 0
                for concept in focus.target_concepts:
                    if concept.lower() in sentence.lower():
                        relevance += 1

                if relevance > 0:
                    relevant_parts.append(sentence)

            if relevant_parts:
                filtered_data["relevant_parts"] = relevant_parts
                return filtered_data, True

        # For other data types or no changes
        return data, False

    def _filter_by_concept_relevance(self, data: Any, focus: AttentionalFocus,
                                    context: Dict[str, Any]) -> Tuple[Any, bool]:
        """
        Filter based on conceptual relevance.
        Enhances content related to targeted concepts in semantic memory.

        Args:
            data: Data to filter
            focus: Current attentional focus
            context: Additional context

        Returns:
            Tuple of (filtered_data, changed)
        """
        # This filter requires semantic memory to work effectively
        if not self.semantic_memory:
            return data, False

        # For text data
        if isinstance(data, str):
            text = data
            # In a real implementation, we could look up related concepts
            # in semantic memory
            return text, False

        elif isinstance(data, dict) and "text" in data:
            text = data["text"]
            filtered_data = data.copy()

            # Extract concepts from text
            # This is a simplistic approach; a real implementation would use NLP
            words = set(text.lower().split())

            # Try to find related concepts in semantic memory
            related_concepts = []
            for concept in focus.target_concepts:
                # Placeholder for semantic memory query
                # In a real implementation, this would query the semantic memory
                # for concepts related to the target concept

                # For now, we'll just do simple string matching
                for word in words:
                    if concept.lower() in word or word in concept.lower():
                        related_concepts.append((word, concept))

            if related_concepts:
                filtered_data["related_concepts"] = related_concepts
                return filtered_data, True

        # For other data types or no changes
        return data, False

    def _filter_by_pattern_matching(self, data: Any, focus: AttentionalFocus,
                                  context: Dict[str, Any]) -> Tuple[Any, bool]:
        """
        Filter based on pattern matching.
        Identifies content that matches specific patterns.

        Args:
            data: Data to filter
            focus: Current attentional focus
            context: Additional context

        Returns:
            Tuple of (filtered_data, changed)
        """
        # For text data
        if isinstance(data, str):
            text = data
            matches = []

            for pattern in focus.target_patterns:
                try:
                    found = re.findall(pattern, text, re.IGNORECASE)
                    matches.extend(found)
                except re.error:
                    # If pattern is not a valid regex, treat it as a literal string
                    if pattern.lower() in text.lower():
                        matches.append(pattern)

            if matches:
                # In a real implementation, we might add markup or structure
                # Here we just return the original text
                return text, False
            return text, False

        elif isinstance(data, dict) and "text" in data:
            text = data["text"]
            filtered_data = data.copy()

            pattern_matches = []

            for pattern in focus.target_patterns:
                try:
                    found = re.findall(pattern, text, re.IGNORECASE)
                    if found:
                        pattern_matches.append((pattern, found))
                except re.error:
                    # If pattern is not a valid regex, treat it as a literal string
                    if pattern.lower() in text.lower():
                        pattern_matches.append((pattern, [pattern]))

            if pattern_matches:
                filtered_data["pattern_matches"] = pattern_matches
                return filtered_data, True

        # For other data types or no changes
        return data, False

    def _filter_by_urgency(self, data: Any, focus: AttentionalFocus,
                         context: Dict[str, Any]) -> Tuple[Any, bool]:
        """
        Filter based on urgency.
        Prioritizes content that requires immediate attention.

        Args:
            data: Data to filter
            focus: Current attentional focus
            context: Additional context

        Returns:
            Tuple of (filtered_data, changed)
        """
        # Check for urgency indicators in context
        is_urgent = (context.get("urgency", 0) > 0.7 or
                     context.get("emergency", False))

        # For text data
        if isinstance(data, str):
            text = data

            # Check for urgency markers in the text
            urgency_markers = ["urgent", "immediately", "emergency", "critical", "asap"]
            text_urgent = any(marker in text.lower() for marker in urgency_markers)

            if is_urgent or text_urgent:
                # In a real implementation, we might add markup or priority flags
                # Here we just return the original text
                return text, False
            return text, False

        elif isinstance(data, dict):
            filtered_data = data.copy()

            # Check for urgency in the data
            text_urgent = False
            if "text" in data:
                text = data["text"]
                urgency_markers = ["urgent", "immediately", "emergency", "critical", "asap"]
                text_urgent = any(marker in text.lower() for marker in urgency_markers)

            if is_urgent or text_urgent:
                filtered_data["is_urgent"] = True
                # Increase priority if that field exists
                if "priority" in filtered_data:
                    filtered_data["priority"] = max(filtered_data["priority"], 0.9)
                else:
                    filtered_data["priority"] = 0.9

                return filtered_data, True

        # For other data types or no changes
        return data, False

    def _filter_by_novelty(self, data: Any, focus: AttentionalFocus,
                         context: Dict[str, Any]) -> Tuple[Any, bool]:
        """
        Filter based on novelty.
        Highlights content that contains new or unexpected information.

        Args:
            data: Data to filter
            focus: Current attentional focus
            context: Additional context

        Returns:
            Tuple of (filtered_data, changed)
        """
        # This filter ideally requires working memory to track what's been seen
        if not self.working_memory:
            return data, False

        # For text data
        if isinstance(data, str):
            text = data
            # In a real implementation, we would check against working memory
            # to determine what's novel
            return text, False

        elif isinstance(data, dict) and "text" in data:
            text = data["text"]
            filtered_data = data.copy()

            # Simplified novelty detection
            # In a real implementation, this would query working memory
            # to check what's been seen recently

            # For demonstration, just randomly mark some content as novel
            if random.random() < 0.3:  # 30% chance
                filtered_data["is_novel"] = True
                return filtered_data, True

        # For other data types or no changes
        return data, False


# Create a submodule for specific attention filters

class SelectiveAttentionFilter:
    """
    Specialized filter for selective attention based on specific criteria.
    This can be used directly or registered with the main AttentionFilter.
    """

    def __init__(self, criteria: Dict[str, Any],
                working_memory: Optional[WorkingMemory] = None):
        """
        Initialize a selective attention filter.

        Args:
            criteria: Dictionary of criteria to filter by
            working_memory: Optional reference to working memory
        """
        self.criteria = criteria
        self.working_memory = working_memory
        self.logger = logging.getLogger("GrandNexus.Perception.SelectiveFilter")

    def filter(self, data: Any, focus: AttentionalFocus,
              context: Dict[str, Any]) -> Tuple[Any, bool]:
        """
        Apply selective filtering based on criteria.

        Args:
            data: Data to filter
            focus: Current attentional focus
            context: Additional context

        Returns:
            Tuple of (filtered_data, changed)
        """
        # Check if data matches criteria
        if isinstance(data, dict):
            filtered_data = data.copy()
            matches = []

            for key, value in self.criteria.items():
                if key in data and data[key] == value:
                    matches.append(key)

            if matches:
                filtered_data["matching_criteria"] = matches
                return filtered_data, True

        # No matches or unsupported data type
        return data, False


class TemporalAttentionFilter:
    """
    Specialized filter for attention based on temporal patterns.
    This can be used to detect sequences, trends, or changes over time.
    """

    def __init__(self, window_size: int = 5,
                working_memory: Optional[WorkingMemory] = None):
        """
        Initialize a temporal attention filter.

        Args:
            window_size: Number of recent items to consider
            working_memory: Optional reference to working memory
        """
        self.window_size = window_size
        self.working_memory = working_memory
        self.history = deque(maxlen=window_size)
        self.logger = logging.getLogger("GrandNexus.Perception.TemporalFilter")

    def filter(self, data: Any, focus: AttentionalFocus,
              context: Dict[str, Any]) -> Tuple[Any, bool]:
        """
        Apply temporal filtering.

        Args:
            data: Data to filter
            focus: Current attentional focus
            context: Additional context

        Returns:
            Tuple of (filtered_data, changed)
        """
        # Add current data to history
        self.history.append(data)

        # Not enough history yet
        if len(self.history) < 2:
            return data, False

        # Check for temporal patterns
        if isinstance(data, dict):
            filtered_data = data.copy()

            # Check for changes
            previous = self.history[-2]
            if isinstance(previous, dict):
                changes = []

                for key in set(data.keys()) | set(previous.keys()):
                    if key in data and key in previous:
                        if data[key] != previous[key]:
                            changes.append((key, previous[key], data[key]))
                    elif key in data:
                        changes.append((key, None, data[key]))
                    elif key in previous:
                        changes.append((key, previous[key], None))

                if changes:
                    filtered_data["temporal_changes"] = changes
                    return filtered_data, True

        # No changes or unsupported data type
        return data, False


# Function to create a specialized attention filter for detecting anomalies

    def create_anomaly_filter(baseline: Dict[str, Any],
                         threshold: float = 0.2,
                         working_memory: Optional[WorkingMemory] = None) -> Callable:
    """
    Create an anomaly detection filter.

    Args:
        baseline: Baseline normal values
        threshold: Threshold for anomaly detection
        working_memory: Optional reference to working memory

    Returns:
        Filter function that can be registered with AttentionFilter
    """
    logger = logging.getLogger("GrandNexus.Perception.AnomalyFilter")

    def anomaly_filter(data: Any, focus: AttentionalFocus,
                      context: Dict[str, Any]) -> Tuple[Any, bool]:
        """
        Detect anomalies compared to baseline.

        Args:
            data: Data to filter
            focus: Current attentional focus
            context: Additional context

        Returns:
            Tuple of (filtered_data, changed)
        """
        if not isinstance(data, dict):
            return data, False

        filtered_data = data.copy()
        anomalies = []

        for key, base_value in baseline.items():
            if key in data:
                current_value = data[key]

                # Check for numerical anomalies
                if isinstance(base_value, (int, float)) and isinstance(current_value, (int, float)):
                    if abs(current_value - base_value) / max(1, abs(base_value)) > threshold:
                        anomalies.append((key, base_value, current_value))
                # Check for categorical anomalies
                elif current_value != base_value:
                    anomalies.append((key, base_value, current_value))

        if anomalies:
            filtered_data["anomalies"] = anomalies
            logger.info(f"Detected {len(anomalies)} anomalies")
            return filtered_data, True

        return data, False

    return anomaly_filter

class ContextualFilter:
    """
    Filter factory for creating context-specific attention filters.
    This class is designed to be used by AttentionFilter to dynamically
    generate filters based on specific contexts and needs.
    """

    @staticmethod
    def create_keyword_filter(keywords: List[str], importance_factor: float = 1.0) -> Callable:
        """
        Create a filter that highlights content containing specific keywords.

        Args:
            keywords: List of keywords to focus on
            importance_factor: Multiplier for matched content importance

        Returns:
            Filter function
        """
        def keyword_filter(data: Any, focus: AttentionalFocus, context: Dict[str, Any]) -> Tuple[Any, bool]:
            if isinstance(data, dict) and "text" in data:
                text = data["text"]
                filtered_data = data.copy()

                keyword_matches = []
                for keyword in keywords:
                    if keyword.lower() in text.lower():
                        keyword_matches.append(keyword)

                if keyword_matches:
                    filtered_data["keyword_matches"] = keyword_matches
                    filtered_data["keyword_importance"] = len(keyword_matches) * importance_factor
                    return filtered_data, True

            return data, False

        return keyword_filter

    @staticmethod
    def create_emotional_filter(emotions: List[str]) -> Callable:
        """
        Create a filter that detects emotional content.

        Args:
            emotions: List of emotions to detect (e.g., "joy", "anger")

        Returns:
            Filter function
        """
        # Simplified emotion keyword mapping
        emotion_keywords = {
            "joy": ["happy", "joy", "excited", "delighted", "pleased"],
            "anger": ["angry", "mad", "furious", "outraged", "irritated"],
            "sadness": ["sad", "unhappy", "depressed", "miserable", "gloomy"],
            "fear": ["afraid", "scared", "fearful", "terrified", "anxious"],
            "surprise": ["surprised", "shocked", "amazed", "astonished", "stunned"]
        }

        # Expand emotions list to keywords
        keywords = []
        for emotion in emotions:
            if emotion in emotion_keywords:
                keywords.extend(emotion_keywords[emotion])

        def emotion_filter(data: Any, focus: AttentionalFocus, context: Dict[str, Any]) -> Tuple[Any, bool]:
            if isinstance(data, dict) and "text" in data:
                text = data["text"]
                filtered_data = data.copy()

                emotion_matches = {}
                for emotion in emotions:
                    if emotion not in emotion_keywords:
                        continue

                    matches = 0
                    for keyword in emotion_keywords[emotion]:
                        matches += text.lower().count(keyword.lower())

                    if matches > 0:
                        emotion_matches[emotion] = matches

                if emotion_matches:
                    filtered_data["emotion_analysis"] = emotion_matches
                    return filtered_data, True

            return data, False

        return emotion_filter
