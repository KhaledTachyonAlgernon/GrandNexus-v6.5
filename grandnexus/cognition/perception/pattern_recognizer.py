"""
pattern_recognizer.py - Pattern Recognition Module for GrandNexus

This module provides capabilities for detecting and extracting patterns from various
data sources including sequences, text, symbolic structures, and graph data. It serves
as a key perceptual component that helps the cognitive system identify regularities
and recurring motifs in observed information.

Key features:
- Sequential pattern detection (temporal patterns, repeating sequences)
- Structural pattern recognition (recurring structures in data)
- Approximate pattern matching with flexible similarity metrics
- Integration with semantic memory for contextual pattern understanding
- Emergent pattern discovery through unsupervised analysis
- Adaptive pattern prioritization based on relevance and attention
"""

import logging
import time
import threading
import uuid
import re
import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set, Tuple, Union, Callable
from enum import Enum, auto
from collections import defaultdict, deque, Counter
import random

# Assuming these exist or will exist
from grandnexus.core.nexus_core import NexusCore
from grandnexus.memory.semantic import SemanticMemory
from grandnexus.memory.working import WorkingMemory
from grandnexus.memory.episodic import EpisodicMemory
from grandnexus.graph.graph_engine import GraphEngine
from grandnexus.graph.graph_engine import create_graph_engine, AbstractGraphEngine

# Try importing optional dependencies
try:
    import numpy as np
    HAS_NUMPY = True
except ImportError:
    HAS_NUMPY = False

try:
    from sklearn.cluster import DBSCAN
    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False

# Pattern types
class PatternType(Enum):
    """Types of patterns the recognizer can detect"""
    SEQUENTIAL = auto()  # Ordered sequences of items
    TEMPORAL = auto()    # Time-dependent patterns
    STRUCTURAL = auto()  # Structure-based patterns (trees, graphs)
    LINGUISTIC = auto()  # Language patterns
    SYMBOLIC = auto()    # Symbolic or mathematical patterns
    BEHAVIORAL = auto()  # Patterns in behavior or action sequences
    SPATIAL = auto()     # Spatial arrangements
    EMERGENT = auto()    # Emergent patterns not fitting other categories

@dataclass
class Pattern:
    """Represents a detected pattern"""
    pattern_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    pattern_type: PatternType = PatternType.SEQUENTIAL
    pattern_data: Any = None  # The actual pattern content
    confidence: float = 0.5
    frequency: int = 1
    first_seen: float = field(default_factory=time.time)
    last_seen: float = field(default_factory=time.time)
    metadata: Dict[str, Any] = field(default_factory=dict)

    # For structural patterns
    structure: Optional[Dict[str, Any]] = None

    # For pattern variations
    variations: List[Dict[str, Any]] = field(default_factory=list)

    # For extended context
    context_before: Optional[Any] = None
    context_after: Optional[Any] = None

    def update_frequency(self) -> None:
        """Update the frequency and last_seen timestamp"""
        self.frequency += 1
        self.last_seen = time.time()

    def add_variation(self, variation_data: Any, similarity: float) -> None:
        """Add a variation of this pattern"""
        self.variations.append({
            "data": variation_data,
            "similarity": similarity,
            "timestamp": time.time()
        })

    def to_dict(self) -> Dict[str, Any]:
        """Convert pattern to a dictionary for serialization"""
        result = {
            "pattern_id": self.pattern_id,
            "pattern_type": self.pattern_type.name,
            "confidence": self.confidence,
            "frequency": self.frequency,
            "first_seen": self.first_seen,
            "last_seen": self.last_seen,
            "metadata": self.metadata,
        }

        # Handle pattern_data based on type for serialization
        if isinstance(self.pattern_data, (str, int, float, bool, type(None))):
            result["pattern_data"] = self.pattern_data
        else:
            # For complex structures, serialize as string or use a specific serializer
            try:
                result["pattern_data"] = str(self.pattern_data)
            except:
                result["pattern_data"] = f"<{type(self.pattern_data).__name__}>"

        if self.structure:
            result["structure"] = self.structure

        if self.variations:
            result["variation_count"] = len(self.variations)

        return result

@dataclass
class PatternMatch:
    """Result of a pattern matching operation"""
    matched: bool
    pattern: Optional[Pattern] = None
    confidence: float = 0.0
    match_start: Optional[int] = None  # For sequential data
    match_end: Optional[int] = None    # For sequential data
    match_elements: List[Any] = field(default_factory=list)
    similarity: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class PatternSearchConfig:
    """Configuration for pattern search operations"""
    pattern_types: List[PatternType] = field(default_factory=lambda: list(PatternType))
    min_pattern_length: int = 2
    max_pattern_length: int = 10
    min_frequency: int = 2
    min_confidence: float = 0.5
    max_patterns: int = 20
    similarity_threshold: float = 0.7
    consider_variations: bool = True
    use_semantic_context: bool = False
    detect_hierarchical: bool = False
    max_search_time: float = 5.0  # seconds

class PatternRecognizer:
    """
    Core pattern recognition engine for GrandNexus, capable of identifying
    various pattern types across different data sources and modalities.
    """

    def __init__(self, nexus_core: Optional[NexusCore] = None,
                 semantic_memory: Optional[SemanticMemory] = None,
                 working_memory: Optional[WorkingMemory] = None,
                 episodic_memory: Optional[EpisodicMemory] = None,
                 graph_engine: Optional[GraphEngine] = None):
        """
        Initialize the pattern recognizer.

        Args:
            nexus_core: Reference to the NexusCore for system integration
            semantic_memory: Reference to semantic memory for context
            working_memory: Reference to working memory for active patterns
            episodic_memory: Reference to episodic memory for historical patterns
            graph_engine: Reference to graph engine for structural patterns
        """
        self.logger = logging.getLogger("GrandNexus.Perception.PatternRecognizer")
        self.nexus_core = nexus_core
        self.semantic_memory = semantic_memory
        self.working_memory = working_memory
        self.episodic_memory = episodic_memory
        self.graph_engine = graph_engine

        # Internal state
        self.instance_id = str(uuid.uuid4())[:8]
        self.initialized = False
        self.lock = threading.RLock()

        # Pattern storage
        self.patterns = {}  # pattern_id -> Pattern
        self.pattern_index = defaultdict(list)  # Various indices for faster lookup

        # Pattern detectors for different types
        self.detectors = {}

        # History of recently seen data
        self.recent_sequences = deque(maxlen=100)
        self.recent_items = deque(maxlen=1000)

        # Performance metrics
        self.metrics = defaultdict(lambda: deque(maxlen=100))

        self.logger.info(f"PatternRecognizer initialized with ID={self.instance_id}")

    def initialize(self) -> bool:
        """Initialize the pattern recognizer and register pattern detectors"""
        if self.initialized:
            return True

        with self.lock:
            try:
                # Register built-in pattern detectors
                self._register_default_detectors()

                # Load any existing patterns from memory
                self._load_patterns_from_memory()

                # Connect to other modules
                if self.nexus_core:
                    self._register_with_nexus_core()

                self.initialized = True
                self.logger.info("PatternRecognizer initialization complete")
                return True
            except Exception as e:
                self.logger.error(f"PatternRecognizer initialization failed: {str(e)}")
                return False

    def process(self, data: Any, data_type: str = "sequential",
               context: Optional[Dict[str, Any]] = None,
               detect_new_patterns: bool = True) -> List[PatternMatch]:
        """
        Process new data, matching against known patterns and optionally
        detecting new patterns.

        Args:
            data: The data to process (sequence, text, structure, etc.)
            data_type: Type of data (sequential, text, structural, etc.)
            context: Additional contextual information
            detect_new_patterns: Whether to look for new patterns

        Returns:
            List of pattern matches found in the data
        """
        if not self.initialized:
            self.initialize()

        # Prepare context if not provided
        if context is None:
            context = {}

        # Track the data for future pattern detection
        self._track_data(data, data_type, context)

        # Prepare result list
        matches = []

        # Match against known patterns
        pattern_matches = self._match_known_patterns(data, data_type, context)
        matches.extend(pattern_matches)

        # Detect new patterns if requested
        if detect_new_patterns:
            start_time = time.time()
            new_patterns = self._detect_new_patterns(data, data_type, context)
            detection_time = time.time() - start_time

            self.metrics["pattern_detection_time"].append(detection_time)

            # Log new pattern detection
            if new_patterns:
                self.logger.info(f"Detected {len(new_patterns)} new patterns in {data_type} data")

                # Register the new patterns
                for pattern in new_patterns:
                    self.register_pattern(pattern)

                # Match the new patterns against the current data
                new_matches = self._match_patterns(new_patterns, data, data_type, context)
                matches.extend(new_matches)

        # Update metrics
        self.metrics["matches_per_process"].append(len(matches))

        return matches

    def register_pattern(self, pattern: Pattern) -> bool:
        """
        Register a new pattern or update an existing similar pattern.

        Args:
            pattern: The pattern to register

        Returns:
            True if registration succeeded, False otherwise
        """
        with self.lock:
            try:
                # Check if this pattern is similar to an existing one
                similar_pattern_id = self._find_similar_pattern(pattern)

                if similar_pattern_id:
                    # Update the existing pattern
                    existing_pattern = self.patterns[similar_pattern_id]
                    existing_pattern.update_frequency()
                    existing_pattern.last_seen = time.time()

                    # Add this as a variation if it's not identical
                    if existing_pattern.pattern_data != pattern.pattern_data:
                        similarity = self._calculate_pattern_similarity(
                            existing_pattern.pattern_data,
                            pattern.pattern_data,
                            pattern.pattern_type
                        )
                        existing_pattern.add_variation(pattern.pattern_data, similarity)

                    # Update confidence if needed
                    if pattern.confidence > existing_pattern.confidence:
                        existing_pattern.confidence = pattern.confidence

                    # Update metadata with any new information
                    for key, value in pattern.metadata.items():
                        if key not in existing_pattern.metadata:
                            existing_pattern.metadata[key] = value

                    return True
                else:
                    # Register as a new pattern
                    self.patterns[pattern.pattern_id] = pattern

                    # Update indices for faster lookup
                    self._update_pattern_indices(pattern)

                    # Add to working memory if available
                    if self.working_memory:
                        pattern_dict = pattern.to_dict()
                        self.working_memory.store_item(
                            content=pattern_dict,
                            source="pattern_recognizer",
                            metadata={
                                "pattern_type": pattern.pattern_type.name,
                                "confidence": pattern.confidence
                            }
                        )

                    self.logger.info(f"Registered new pattern: {pattern.pattern_id} ({pattern.pattern_type.name})")
                    return True
            except Exception as e:
                self.logger.error(f"Error registering pattern: {str(e)}")
                return False

    def find_patterns(self, config: PatternSearchConfig,
                     data: Optional[Any] = None) -> List[Pattern]:
        """
        Find patterns matching the given configuration.

        Args:
            config: Configuration for pattern search
            data: Optional specific data to search in instead of history

        Returns:
            List of patterns matching the criteria
        """
        if not self.initialized:
            self.initialize()

        matches = []

        with self.lock:
            # Filter patterns by type
            candidates = []
            for pattern_id, pattern in self.patterns.items():
                if pattern.pattern_type in config.pattern_types and \
                   pattern.confidence >= config.min_confidence and \
                   pattern.frequency >= config.min_frequency:
                    candidates.append(pattern)

            # Sort by relevance (combination of confidence and frequency)
            candidates.sort(key=lambda p: p.confidence * p.frequency, reverse=True)

            # Limit to max_patterns
            candidates = candidates[:config.max_patterns]

            if data is not None:
                # If specific data is provided, match patterns against it
                for pattern in candidates:
                    detector = self._get_detector_for_type(pattern.pattern_type)
                    if detector:
                        matches = detector.match(pattern, data, {})
                        if matches:
                            matches.append(pattern)
            else:
                # Otherwise return the filtered patterns
                matches = candidates

        return matches

    def get_pattern(self, pattern_id: str) -> Optional[Pattern]:
        """Get a specific pattern by ID"""
        with self.lock:
            return self.patterns.get(pattern_id)

    def remove_pattern(self, pattern_id: str) -> bool:
        """Remove a pattern from the system"""
        with self.lock:
            if pattern_id in self.patterns:
                pattern = self.patterns[pattern_id]

                # Remove from indices
                self._remove_from_indices(pattern)

                # Remove from patterns dictionary
                del self.patterns[pattern_id]

                self.logger.info(f"Removed pattern: {pattern_id}")
                return True
            return False

    def get_pattern_metrics(self) -> Dict[str, Any]:
        """Get metrics about pattern recognition performance"""
        with self.lock:
            metrics_snapshot = {}

            # Overall pattern metrics
            metrics_snapshot["total_patterns"] = len(self.patterns)

            # Pattern types distribution
            type_counts = defaultdict(int)
            for pattern in self.patterns.values():
                type_counts[pattern.pattern_type.name] += 1
            metrics_snapshot["pattern_type_distribution"] = dict(type_counts)

            # Performance metrics
            for key, values in self.metrics.items():
                if values:
                    metrics_snapshot[key] = {
                        'current': values[-1],
                        'average': sum(values) / len(values),
                        'min': min(values),
                        'max': max(values)
                    }

            # Length metrics
            if self.patterns:
                pattern_lengths = []
                for pattern in self.patterns.values():
                    # Estimate length based on pattern_type
                    if pattern.pattern_type == PatternType.SEQUENTIAL and hasattr(pattern.pattern_data, "__len__"):
                        pattern_lengths.append(len(pattern.pattern_data))

                if pattern_lengths:
                    metrics_snapshot["avg_pattern_length"] = sum(pattern_lengths) / len(pattern_lengths)
                    metrics_snapshot["min_pattern_length"] = min(pattern_lengths)
                    metrics_snapshot["max_pattern_length"] = max(pattern_lengths)

            return metrics_snapshot

    # -----------------------------------------------------------------------
    # Internal methods
    # -----------------------------------------------------------------------

    def _register_default_detectors(self) -> None:
        """Register built-in pattern detectors for different pattern types"""
        # Sequential pattern detector
        self.detectors[PatternType.SEQUENTIAL] = SequentialPatternDetector()

        # Temporal pattern detector
        self.detectors[PatternType.TEMPORAL] = TemporalPatternDetector()

        # Structural pattern detector (if graph_engine is available)
        if self.graph_engine:
            self.detectors[PatternType.STRUCTURAL] = StructuralPatternDetector()

        # Linguistic pattern detector
        self.detectors[PatternType.LINGUISTIC] = LinguisticPatternDetector()

        # Symbolic pattern detector
        self.detectors[PatternType.SYMBOLIC] = SymbolicPatternDetector()

    def _load_patterns_from_memory(self) -> None:
        """Load existing patterns from memory if available"""
        # This is a placeholder - in a real implementation,
        # patterns would be loaded from semantic_memory or episodic_memory
        pass

    def _register_with_nexus_core(self) -> None:
        """Register with NexusCore for system-wide coordination"""
        if not self.nexus_core:
            self.logger.warning("No NexusCore available for registration")
            return

        try:
            # Register as a module with NexusCore
            dependencies = []
            if self.semantic_memory:
                dependencies.append("semantic_memory")
            if self.working_memory:
                dependencies.append("working_memory")
            if self.episodic_memory:
                dependencies.append("episodic_memory")
            if self.graph_engine:
                dependencies.append("graph_engine")

            self.nexus_core.register_module(
                name="pattern_recognizer",
                module=self,
                dependencies=dependencies
            )

            self.logger.info("Successfully registered with NexusCore")
        except Exception as e:
            self.logger.error(f"Failed to register with NexusCore: {str(e)}")

    def _track_data(self, data: Any, data_type: str, context: Dict[str, Any]) -> None:
        """
        Track the data for future pattern detection by adding to history.

        Args:
            data: The data to track
            data_type: Type of data (sequential, text, structural, etc.)
            context: Additional contextual information
        """
        # Add to recent items
        self.recent_items.append((data, data_type, time.time(), context))

        # For sequential data, add to sequences
        if data_type == "sequential" or data_type == "temporal":
            self.recent_sequences.append((data, data_type, time.time(), context))

    def _match_known_patterns(self, data: Any, data_type: str,
                             context: Dict[str, Any]) -> List[PatternMatch]:
        """
        Match the given data against known patterns.

        Args:
            data: The data to match against patterns
            data_type: Type of data
            context: Additional contextual information

        Returns:
            List of pattern matches
        """
        matches = []

        # Get relevant patterns based on data_type
        relevant_patterns = []

        # Map data_type to PatternType
        pattern_type_map = {
            "sequential": PatternType.SEQUENTIAL,
            "temporal": PatternType.TEMPORAL,
            "structural": PatternType.STRUCTURAL,
            "text": PatternType.LINGUISTIC,
            "symbolic": PatternType.SYMBOLIC,
            "behavior": PatternType.BEHAVIORAL,
            "spatial": PatternType.SPATIAL
        }

        # Get pattern type from map, default to SEQUENTIAL
        pattern_type = pattern_type_map.get(data_type, PatternType.SEQUENTIAL)

        # Get patterns of matching type
        with self.lock:
            for pattern_id, pattern in self.patterns.items():
                if pattern.pattern_type == pattern_type:
                    relevant_patterns.append(pattern)

        # Match patterns using appropriate detector
        detector = self._get_detector_for_type(pattern_type)
        if detector:
            for pattern in relevant_patterns:
                pattern_matches = detector.match(pattern, data, context)
                matches.extend(pattern_matches)

        return matches

    def _detect_new_patterns(self, data: Any, data_type: str,
                            context: Dict[str, Any]) -> List[Pattern]:
        """
        Detect new patterns in the data and history.

        Args:
            data: The current data
            data_type: Type of data
            context: Additional contextual information

        Returns:
            List of newly detected patterns
        """
        new_patterns = []

        # Map data_type to PatternType
        pattern_type_map = {
            "sequential": PatternType.SEQUENTIAL,
            "temporal": PatternType.TEMPORAL,
            "structural": PatternType.STRUCTURAL,
            "text": PatternType.LINGUISTIC,
            "symbolic": PatternType.SYMBOLIC,
            "behavior": PatternType.BEHAVIORAL,
            "spatial": PatternType.SPATIAL
        }

        # Get pattern type from map, default to SEQUENTIAL
        pattern_type = pattern_type_map.get(data_type, PatternType.SEQUENTIAL)

        # Use appropriate detector
        detector = self._get_detector_for_type(pattern_type)
        if detector:
            history = self._get_relevant_history(data_type)
            detected_patterns = detector.detect(data, history, context)
            new_patterns.extend(detected_patterns)

        return new_patterns

    def _match_patterns(self, patterns: List[Pattern], data: Any,
                       data_type: str, context: Dict[str, Any]) -> List[PatternMatch]:
        """
        Match specific patterns against the given data.

        Args:
            patterns: List of patterns to match
            data: The data to match against
            data_type: Type of data
            context: Additional contextual information

        Returns:
            List of pattern matches
        """
        matches = []

        # Map data_type to PatternType
        pattern_type_map = {
            "sequential": PatternType.SEQUENTIAL,
            "temporal": PatternType.TEMPORAL,
            "structural": PatternType.STRUCTURAL,
            "text": PatternType.LINGUISTIC,
            "symbolic": PatternType.SYMBOLIC,
            "behavior": PatternType.BEHAVIORAL,
            "spatial": PatternType.SPATIAL
        }

        # Get pattern type from map, default to SEQUENTIAL
        pattern_type = pattern_type_map.get(data_type, PatternType.SEQUENTIAL)

        # Match patterns using appropriate detector
        for pattern in patterns:
            detector = self._get_detector_for_type(pattern.pattern_type)
            if detector:
                pattern_matches = detector.match(pattern, data, context)
                matches.extend(pattern_matches)

        return matches

    def _get_detector_for_type(self, pattern_type: PatternType) -> Optional[Any]:
        """
        Get the appropriate detector for a pattern type.

        Args:
            pattern_type: The type of pattern

        Returns:
            Pattern detector instance or None if not available
        """
        return self.detectors.get(pattern_type)

    def _get_relevant_history(self, data_type: str) -> List[Tuple[Any, str, float, Dict[str, Any]]]:
        """
        Get relevant historical data for pattern detection.

        Args:
            data_type: Type of data

        Returns:
            List of (data, data_type, timestamp, context) tuples
        """
        if data_type in ["sequential", "temporal"]:
            return list(self.recent_sequences)
        else:
            # Filter by data_type
            return [(d, dt, ts, ctx) for d, dt, ts, ctx in self.recent_items if dt == data_type]

    def _find_similar_pattern(self, pattern: Pattern) -> Optional[str]:
        """
        Find an existing pattern similar to the given one.

        Args:
            pattern: The pattern to find a similar one for

        Returns:
            ID of similar pattern or None if not found
        """
        # First check for exact matches
        for pattern_id, existing in self.patterns.items():
            if existing.pattern_type == pattern.pattern_type and existing.pattern_data == pattern.pattern_data:
                return pattern_id

        # Then check for similar patterns
        for pattern_id, existing in self.patterns.items():
            if existing.pattern_type == pattern.pattern_type:
                similarity = self._calculate_pattern_similarity(
                    existing.pattern_data,
                    pattern.pattern_data,
                    pattern.pattern_type
                )

                # Consider similar if above threshold
                if similarity >= 0.8:  # High threshold for similarity
                    return pattern_id

        return None

    def _calculate_pattern_similarity(self, pattern1: Any, pattern2: Any,
                                    pattern_type: PatternType) -> float:
        """
        Calculate similarity between two patterns.

        Args:
            pattern1: First pattern data
            pattern2: Second pattern data
            pattern_type: Type of patterns

        Returns:
            Similarity score between 0.0 and 1.0
        """
        if pattern1 == pattern2:
            return 1.0

        # Handle different pattern types
        if pattern_type == PatternType.SEQUENTIAL:
            return self._sequence_similarity(pattern1, pattern2)
        elif pattern_type == PatternType.LINGUISTIC:
            return self._text_similarity(pattern1, pattern2)
        elif pattern_type == PatternType.STRUCTURAL:
            return self._structural_similarity(pattern1, pattern2)
        elif pattern_type == PatternType.SYMBOLIC:
            return self._symbolic_similarity(pattern1, pattern2)
        else:
            # Default to basic comparison
            try:
                # For basic types that support equality
                return 1.0 if pattern1 == pattern2 else 0.0
            except:
                # For complex types, use string representation
                return self._text_similarity(str(pattern1), str(pattern2))

    def _sequence_similarity(self, seq1: Any, seq2: Any) -> float:
        """
        Calculate similarity between two sequences.

        Args:
            seq1: First sequence
            seq2: Second sequence

        Returns:
            Similarity score between 0.0 and 1.0
        """
        # Handle non-sequence types
        if not hasattr(seq1, "__len__") or not hasattr(seq2, "__len__"):
            return 0.0

        # Empty sequences
        if len(seq1) == 0 and len(seq2) == 0:
            return 1.0
        elif len(seq1) == 0 or len(seq2) == 0:
            return 0.0

        # Simple longest common subsequence approach
        # Real implementation would use a more sophisticated algorithm
        lcs_length = self._longest_common_subsequence_length(seq1, seq2)

        return lcs_length / max(len(seq1), len(seq2))

    def _text_similarity(self, text1: str, text2: str) -> float:
        """
        Calculate similarity between two text strings.

        Args:
            text1: First text
            text2: Second text

        Returns:
            Similarity score between 0.0 and 1.0
        """
        # Check if both are strings
        if not isinstance(text1, str) or not isinstance(text2, str):
            return 0.0

        # Simple Jaccard similarity of words
        words1 = set(text1.lower().split())
        words2 = set(text2.lower().split())

        if not words1 and not words2:
            return 1.0
        elif not words1 or not words2:
            return 0.0

        intersection = len(words1.intersection(words2))
        union = len(words1.union(words2))

        return intersection / union

    def _structural_similarity(self, struct1: Any, struct2: Any) -> float:
        """
        Calculate similarity between two structures.

        Args:
            struct1: First structure
            struct2: Second structure

        Returns:
            Similarity score between 0.0 and 1.0
        """
        # This is a placeholder - a real implementation would depend
        # on the specific structure types
        return 0.5

    def _symbolic_similarity(self, symbol1: Any, symbol2: Any) -> float:
        """
        Calculate similarity between two symbolic expressions.

        Args:
            symbol1: First symbolic expression
            symbol2: Second symbolic expression

        Returns:
            Similarity score between 0.0 and 1.0
        """
        # This is a placeholder - a real implementation would use
        # tree edit distance, canonical forms, etc.
        return 0.5

    def _longest_common_subsequence_length(self, seq1: Any, seq2: Any) -> int:
        """
        Calculate the length of the longest common subsequence.

        Args:
            seq1: First sequence
            seq2: Second sequence

        Returns:
            Length of longest common subsequence
        """
        # Simple dynamic programming implementation
        m, n = len(seq1), len(seq2)
        dp = [[0] * (n + 1) for _ in range(m + 1)]

        for i in range(1, m + 1):
            for j in range(1, n + 1):
                if seq1[i-1] == seq2[j-1]:
                    dp[i][j] = dp[i-1][j-1] + 1
                else:
                    dp[i][j] = max(dp[i-1][j], dp[i][j-1])

        return dp[m][n]

    def _update_pattern_indices(self, pattern: Pattern) -> None:
        """
        Update indices for efficient pattern lookup.

        Args:
            pattern: The pattern to index
        """
        # Index by pattern type
        self.pattern_index[f"type:{pattern.pattern_type.name}"].append(pattern.pattern_id)

        # Index by confidence range
        confidence_range = int(pattern.confidence * 10) / 10
        self.pattern_index[f"confidence:{confidence_range}"].append(pattern.pattern_id)

        # Index by frequency bucket (logarithmic scale)
        if pattern.frequency <= 1:
            freq_bucket = "1"
        elif pattern.frequency <= 5:
            freq_bucket = "2-5"
        elif pattern.frequency <= 10:
            freq_bucket = "6-10"
        elif pattern.frequency <= 50:
            freq_bucket = "11-50"
        else:
            freq_bucket = "50+"

        self.pattern_index[f"frequency:{freq_bucket}"].append(pattern.pattern_id)

        # Add metadata indices if available
        if pattern.metadata:
            for key, value in pattern.metadata.items():
                if isinstance(value, (str, int, float, bool)):
                    self.pattern_index[f"metadata:{key}:{value}"].append(pattern.pattern_id)

    def _remove_from_indices(self, pattern: Pattern) -> None:
        """
        Remove a pattern from all indices.

        Args:
            pattern: The pattern to remove from indices
        """
        # Remove from type index
        type_key = f"type:{pattern.pattern_type.name}"
        if type_key in self.pattern_index and pattern.pattern_id in self.pattern_index[type_key]:
            self.pattern_index[type_key].remove(pattern.pattern_id)

        # Remove from confidence index
        confidence_range = int(pattern.confidence * 10) / 10
        conf_key = f"confidence:{confidence_range}"
        if conf_key in self.pattern_index and pattern.pattern_id in self.pattern_index[conf_key]:
            self.pattern_index[conf_key].remove(pattern.pattern_id)

        # Remove from frequency index
        # Determine frequency bucket as in _update_pattern_indices
        if pattern.frequency <= 1:
            freq_bucket = "1"
        elif pattern.frequency <= 5:
            freq_bucket = "2-5"
        elif pattern.frequency <= 10:
            freq_bucket = "6-10"
        elif pattern.frequency <= 50:
            freq_bucket = "11-50"
        else:
            freq_bucket = "50+"

        freq_key = f"frequency:{freq_bucket}"
        if freq_key in self.pattern_index and pattern.pattern_id in self.pattern_index[freq_key]:
            self.pattern_index[freq_key].remove(pattern.pattern_id)

        # Remove from metadata indices
        if pattern.metadata:
            for key, value in pattern.metadata.items():
                if isinstance(value, (str, int, float, bool)):
                    meta_key = f"metadata:{key}:{value}"
                    if meta_key in self.pattern_index and pattern.pattern_id in self.pattern_index[meta_key]:
                        self.pattern_index[meta_key].remove(pattern.pattern_id)


# -----------------------------------------------------------------------------
# Pattern Detector implementations
# -----------------------------------------------------------------------------

class PatternDetectorBase:
    """Base class for pattern detectors"""

    def detect(self, data: Any, history: List[Tuple[Any, str, float, Dict[str, Any]]],
              context: Dict[str, Any]) -> List[Pattern]:
        """
        Detect patterns in data and history.

        Args:
            data: Current data
            history: List of (data, data_type, timestamp, context) tuples
            context: Additional contextual information

        Returns:
            List of detected patterns
        """
        raise NotImplementedError("Subclasses must implement detect method")

    def match(self, pattern: Pattern, data: Any, context: Dict[str, Any]) -> List[PatternMatch]:
        """
        Match a pattern against data.

        Args:
            pattern: Pattern to match
            data: Data to match against
            context: Additional contextual information

        Returns:
            List of pattern matches
        """
        raise NotImplementedError("Subclasses must implement match method")


class SequentialPatternDetector(PatternDetectorBase):
    """Detector for sequential patterns"""

    def detect(self, data: Any, history: List[Tuple[Any, str, float, Dict[str, Any]]],
              context: Dict[str, Any]) -> List[Pattern]:
        """
        Detect sequential patterns in data and history.

        Args:
            data: Current sequential data
            history: List of (data, data_type, timestamp, context) tuples
            context: Additional contextual information

        Returns:
            List of detected sequential patterns
        """
        patterns = []

        # Collect all sequential data from history
        all_sequences = []
        for hist_data, data_type, timestamp, hist_context in history:
            if data_type == "sequential":
                all_sequences.append(hist_data)

        # Add current data
        all_sequences.append(data)

        # Only perform detection if we have enough sequences
        if len(all_sequences) < 2:
            return patterns

        # Simple frequent subsequence detection
        # This is a placeholder for a more sophisticated algorithm
        # (e.g., PrefixSpan, SPADE, etc.)

        # First, find common subsequences across sequences
        min_length = context.get("min_pattern_length", 2)
        max_length = context.get("max_pattern_length", 10)

        # Extract potential subsequences from the current data
        if not hasattr(data, "__len__") or len(data) < min_length:
            return patterns

        # Extract all subsequences of length between min_length and max_length
        subsequences = []
        for length in range(min_length, min(max_length + 1, len(data) + 1)):
            for i in range(len(data) - length + 1):
                subsequence = data[i:i+length]
                subsequences.append(subsequence)

        # Count occurrences of each subsequence in all data
        subsequence_counts = Counter()
        for seq in all_sequences:
            if not hasattr(seq, "__len__"):
                continue

            for subseq in subsequences:
                if self._contains_subsequence(seq, subseq):
                    subsequence_counts[str(subseq)] += 1

        # Create patterns for frequent subsequences
        min_frequency = context.get("min_frequency", 2)
        for subseq_str, count in subsequence_counts.items():
            if count >= min_frequency:
                # Try to parse the string representation back to the original type
                # This is a simplification - real implementation would need to handle various types
                subseq = self._parse_sequence(subseq_str)

                # Create a new pattern
                pattern = Pattern(
                    pattern_type=PatternType.SEQUENTIAL,
                    pattern_data=subseq,
                    confidence=min(0.5 + 0.1 * count, 0.95),  # Higher confidence with more occurrences
                    frequency=count,
                    metadata={"source": "sequential_detector"}
                )

                patterns.append(pattern)

        return patterns

    def match(self, pattern: Pattern, data: Any, context: Dict[str, Any]) -> List[PatternMatch]:
        """
        Match a sequential pattern against data.

        Args:
            pattern: Pattern to match
            data: Data to match against
            context: Additional contextual information

        Returns:
            List of pattern matches
        """
        matches = []

        if not hasattr(data, "__len__") or not hasattr(pattern.pattern_data, "__len__"):
            return matches

        # Find all occurrences of the pattern in the data
        pattern_data = pattern.pattern_data
        pattern_len = len(pattern_data)

        if pattern_len > len(data):
            return matches

        # Exact matching
        for i in range(len(data) - pattern_len + 1):
            if data[i:i+pattern_len] == pattern_data:
                match = PatternMatch(
                    matched=True,
                    pattern=pattern,
                    confidence=pattern.confidence,
                    match_start=i,
                    match_end=i + pattern_len,
                    match_elements=data[i:i+pattern_len],
                    similarity=1.0
                )
                matches.append(match)

        # If no exact matches and we allow approximate matching
        if not matches and context.get("allow_approximate", True):
            # Approximate matching with similarity threshold
            similarity_threshold = context.get("similarity_threshold", 0.7)

            for i in range(len(data) - pattern_len + 1):
                # Calculate similarity between pattern and current window
                current_window = data[i:i+pattern_len]

                # Convert sequences to strings for comparison if needed
                if isinstance(pattern_data, str) and not isinstance(current_window, str):
                    current_window_str = str(current_window)
                    similarity = self._calculate_similarity(pattern_data, current_window_str)
                elif not isinstance(pattern_data, str) and isinstance(current_window, str):
                    pattern_data_str = str(pattern_data)
                    similarity = self._calculate_similarity(pattern_data_str, current_window)
                else:
                    similarity = self._calculate_similarity(pattern_data, current_window)

                if similarity >= similarity_threshold:
                    match = PatternMatch(
                        matched=True,
                        pattern=pattern,
                        confidence=pattern.confidence * similarity,
                        match_start=i,
                        match_end=i + pattern_len,
                        match_elements=current_window,
                        similarity=similarity
                    )
                    matches.append(match)

        return matches

    def _contains_subsequence(self, sequence: Any, subsequence: Any) -> bool:
        """
        Check if a sequence contains a subsequence.

        Args:
            sequence: The sequence to check in
            subsequence: The subsequence to look for

        Returns:
            True if subsequence is in sequence, False otherwise
        """
        if not hasattr(sequence, "__len__") or not hasattr(subsequence, "__len__"):
            return False

        if len(subsequence) > len(sequence):
            return False

        for i in range(len(sequence) - len(subsequence) + 1):
            if sequence[i:i+len(subsequence)] == subsequence:
                return True

        return False

    def _parse_sequence(self, sequence_str: str) -> Any:
        """
        Parse a string representation back to a sequence.

        Args:
            sequence_str: String representation of sequence

        Returns:
            The parsed sequence
        """
        # This is a simplified parsing logic
        # Real implementation would need to handle various sequence types
        try:
            # Try to eval if it looks like a list/tuple
            if (sequence_str.startswith('[') and sequence_str.endswith(']')) or \
               (sequence_str.startswith('(') and sequence_str.endswith(')')):
                return eval(sequence_str)
            else:
                # Return as string otherwise
                return sequence_str
        except:
            # If parsing fails, return the string as is
            return sequence_str

    def _calculate_similarity(self, seq1: Any, seq2: Any) -> float:
        """
        Calculate similarity between two sequences.

        Args:
            seq1: First sequence
            seq2: Second sequence

        Returns:
            Similarity score between 0.0 and 1.0
        """
        # String-based comparison for heterogeneous types
        if isinstance(seq1, str) and isinstance(seq2, str):
            # For strings, use Levenshtein distance or similar
            # This is a simplified version using Jaccard similarity
            words1 = set(seq1.split())
            words2 = set(seq2.split())

            if not words1 and not words2:
                return 1.0

            intersection = len(words1.intersection(words2))
            union = len(words1.union(words2))

            return intersection / union

        # Sequence-based comparison for homogeneous types
        if hasattr(seq1, "__len__") and hasattr(seq2, "__len__"):
            # If lengths are too different, limit similarity
            len_ratio = min(len(seq1), len(seq2)) / max(len(seq1), len(seq2)) if max(len(seq1), len(seq2)) > 0 else 1.0

            # Count matching elements
            matches = 0
            for i in range(min(len(seq1), len(seq2))):
                if seq1[i] == seq2[i]:
                    matches += 1

            element_similarity = matches / min(len(seq1), len(seq2)) if min(len(seq1), len(seq2)) > 0 else 0.0

            # Combine length ratio and element similarity
            return (len_ratio * 0.4 + element_similarity * 0.6)

        # Fallback for incomparable types
        return 0.0


class TemporalPatternDetector(PatternDetectorBase):
    """Detector for time-dependent patterns"""

    def detect(self, data: Any, history: List[Tuple[Any, str, float, Dict[str, Any]]],
              context: Dict[str, Any]) -> List[Pattern]:
        """
        Detect temporal patterns in data and history.

        Args:
            data: Current data
            history: List of (data, data_type, timestamp, context) tuples
            context: Additional contextual information

        Returns:
            List of detected temporal patterns
        """
        patterns = []

        # Need enough history for temporal patterns
        if len(history) < 3:
            return patterns

        # Sort by timestamp
        sorted_history = sorted(history, key=lambda x: x[2])

        # Extract temporal sequence of values
        values = [hist_data for hist_data, _, _, _ in sorted_history]
        timestamps = [ts for _, _, ts, _ in sorted_history]

        # Add current data and timestamp
        current_time = time.time()
        values.append(data)
        timestamps.append(current_time)

        # Check for periodic patterns
        periodic_patterns = self._detect_periodicity(values, timestamps, context)
        patterns.extend(periodic_patterns)

        # Check for trend patterns
        trend_patterns = self._detect_trends(values, timestamps, context)
        patterns.extend(trend_patterns)

        return patterns

    def match(self, pattern: Pattern, data: Any, context: Dict[str, Any]) -> List[PatternMatch]:
        """
        Match a temporal pattern against data.

        Args:
            pattern: Pattern to match
            data: Data to match against
            context: Additional contextual information

        Returns:
            List of pattern matches
        """
        matches = []

        # Temporal patterns often require historical context to match
        # This is a simplified implementation

        if "temporal_type" not in pattern.metadata:
            return matches

        temporal_type = pattern.metadata["temporal_type"]

        if temporal_type == "periodic":
            # For periodic patterns, we check if this value fits the pattern
            expected_value = pattern.metadata.get("expected_value")
            if expected_value is not None:
                # Calculate similarity between expected and actual
                if isinstance(expected_value, (int, float)) and isinstance(data, (int, float)):
                    similarity = 1.0 - min(abs(expected_value - data) / max(abs(expected_value), 1.0), 1.0)
                else:
                    # For non-numeric types, use string comparison
                    similarity = 1.0 if str(expected_value) == str(data) else 0.0

                # If similar enough, we have a match
                similarity_threshold = context.get("similarity_threshold", 0.7)
                if similarity >= similarity_threshold:
                    match = PatternMatch(
                        matched=True,
                        pattern=pattern,
                        confidence=pattern.confidence * similarity,
                        similarity=similarity,
                        metadata={"expected_value": expected_value}
                    )
                    matches.append(match)

        elif temporal_type == "trend":
            # For trend patterns, we check if this value continues the trend
            trend_direction = pattern.metadata.get("trend_direction")
            last_value = pattern.metadata.get("last_value")

            if trend_direction is not None and last_value is not None:
                # Convert to numeric if needed
                if not isinstance(data, (int, float)):
                    try:
                        numeric_data = float(data)
                    except (ValueError, TypeError):
                        return matches
                else:
                    numeric_data = data

                if not isinstance(last_value, (int, float)):
                    try:
                        last_value = float(last_value)
                    except (ValueError, TypeError):
                        return matches

                # Check if value continues the trend
                continues_trend = False
                if trend_direction == "increasing" and numeric_data > last_value:
                    continues_trend = True
                elif trend_direction == "decreasing" and numeric_data < last_value:
                    continues_trend = True
                elif trend_direction == "stable" and abs(numeric_data - last_value) / max(abs(last_value), 1.0) < 0.1:
                    continues_trend = True

                if continues_trend:
                    match = PatternMatch(
                        matched=True,
                        pattern=pattern,
                        confidence=pattern.confidence,
                        similarity=1.0,
                        metadata={
                            "trend_direction": trend_direction,
                            "last_value": last_value,
                            "new_value": numeric_data
                        }
                    )
                    matches.append(match)

        return matches

    def _detect_periodicity(self, values: List[Any], timestamps: List[float],
                           context: Dict[str, Any]) -> List[Pattern]:
        """
        Detect periodic patterns in temporal data.

        Args:
            values: Sequence of values
            timestamps: Corresponding timestamps
            context: Additional contextual information

        Returns:
            List of periodic patterns
        """
        patterns = []

        # Need at least 3 points for periodicity
        if len(values) < 3:
            return patterns

        # This is a placeholder - a real implementation would use
        # Fourier analysis, autocorrelation, or other techniques

        # For demonstration, we'll do a simple check for alternating values
        # if the last 4+ values alternate between two values
        if len(values) >= 4:
            recent_values = values[-4:]

            # Check if odd and even indices have the same values
            if all(recent_values[i] == recent_values[0] for i in range(0, len(recent_values), 2)) and \
               all(recent_values[i] == recent_values[1] for i in range(1, len(recent_values), 2)) and \
               recent_values[0] != recent_values[1]:

                # We have an alternating pattern
                pattern = Pattern(
                    pattern_type=PatternType.TEMPORAL,
                    pattern_data=[recent_values[0], recent_values[1]],
                    confidence=0.7,
                    frequency=2,  # Number of complete cycles
                    metadata={
                        "temporal_type": "periodic",
                        "period": 2,
                        "expected_value": recent_values[0] if len(recent_values) % 2 == 0 else recent_values[1]
                    }
                )

                patterns.append(pattern)

        return patterns

    def _detect_trends(self, values: List[Any], timestamps: List[float],
                      context: Dict[str, Any]) -> List[Pattern]:
        """
        Detect trends in temporal data.

        Args:
            values: Sequence of values
            timestamps: Corresponding timestamps
            context: Additional contextual information

        Returns:
            List of trend patterns
        """
        patterns = []

        # Need at least 3 points for a trend
        if len(values) < 3:
            return patterns

        # Convert to numeric if possible
        numeric_values = []
        for val in values:
            if isinstance(val, (int, float)):
                numeric_values.append(val)
            else:
                try:
                    numeric_values.append(float(val))
                except (ValueError, TypeError):
                    # If we can't convert all values to numeric, we can't detect trends
                    return patterns

        # Simple trend detection - check if values are consistently increasing or decreasing
        increases = 0
        decreases = 0

        for i in range(1, len(numeric_values)):
            if numeric_values[i] > numeric_values[i-1]:
                increases += 1
            elif numeric_values[i] < numeric_values[i-1]:
                decreases += 1

        # Determine if there's a consistent trend
        if increases >= len(numeric_values) - 2 and increases > 0:
            # Increasing trend
            pattern = Pattern(
                pattern_type=PatternType.TEMPORAL,
                pattern_data=numeric_values,
                confidence=min(0.5 + 0.1 * increases, 0.9),
                frequency=1,
                metadata={
                    "temporal_type": "trend",
                    "trend_direction": "increasing",
                    "trend_strength": increases / (len(numeric_values) - 1),
                    "last_value": numeric_values[-1]
                }
            )
            patterns.append(pattern)

        elif decreases >= len(numeric_values) - 2 and decreases > 0:
            # Decreasing trend
            pattern = Pattern(
                pattern_type=PatternType.TEMPORAL,
                pattern_data=numeric_values,
                confidence=min(0.5 + 0.1 * decreases, 0.9),
                frequency=1,
                metadata={
                    "temporal_type": "trend",
                    "trend_direction": "decreasing",
                    "trend_strength": decreases / (len(numeric_values) - 1),
                    "last_value": numeric_values[-1]
                }
            )
            patterns.append(pattern)

        elif (increases + decreases) < len(numeric_values) / 3:
            # Stable trend (not much change)
            pattern = Pattern(
                pattern_type=PatternType.TEMPORAL,
                pattern_data=numeric_values,
                confidence=0.7,
                frequency=1,
                metadata={
                    "temporal_type": "trend",
                    "trend_direction": "stable",
                    "trend_strength": 1.0 - (increases + decreases) / (len(numeric_values) - 1),
                    "last_value": numeric_values[-1]
                }
            )
            patterns.append(pattern)

        return patterns


class StructuralPatternDetector(PatternDetectorBase):
    """Detector for structural patterns in graph-like data"""

    def detect(self, data: Any, history: List[Tuple[Any, str, float, Dict[str, Any]]],
              context: Dict[str, Any]) -> List[Pattern]:
        """
        Detect structural patterns in data and history.

        Args:
            data: Current structured data
            history: List of (data, data_type, timestamp, context) tuples
            context: Additional contextual information

        Returns:
            List of detected structural patterns
        """
        # This is a placeholder - real implementation would depend on
        # specific graph data structure and available graph algorithms

        # Return empty list for now
        return []

    def match(self, pattern: Pattern, data: Any, context: Dict[str, Any]) -> List[PatternMatch]:
        """
        Match a structural pattern against data.

        Args:
            pattern: Pattern to match
            data: Data to match against
            context: Additional contextual information

        Returns:
            List of pattern matches
        """
        # This is a placeholder - real implementation would depend on
        # specific graph data structure and subgraph isomorphism algorithms

        # Return empty list for now
        return []


class LinguisticPatternDetector(PatternDetectorBase):
    """Detector for linguistic patterns in text data"""

    def detect(self, data: Any, history: List[Tuple[Any, str, float, Dict[str, Any]]],
              context: Dict[str, Any]) -> List[Pattern]:
        """
        Detect linguistic patterns in data and history.

        Args:
            data: Current text data
            history: List of (data, data_type, timestamp, context) tuples
            context: Additional contextual information

        Returns:
            List of detected linguistic patterns
        """
        patterns = []

        # Ensure we have text data
        if not isinstance(data, str):
            return patterns

        # Collect text from history
        all_texts = []
        for hist_data, data_type, timestamp, hist_context in history:
            if data_type == "text" and isinstance(hist_data, str):
                all_texts.append(hist_data)

        # Add current data
        all_texts.append(data)

        # Only perform detection if we have enough text
        if len(all_texts) < 2:
            return patterns

        # Extract potential patterns
        text_patterns = []

        # Common phrases (simple n-gram extraction)
        text_patterns.extend(self._extract_common_phrases(all_texts))

        # Regular expressions (simple matching patterns)
        text_patterns.extend(self._extract_regex_patterns(all_texts))

        # Create Pattern objects for the detected patterns
        for pattern_text, pattern_info in text_patterns:
            pattern = Pattern(
                pattern_type=PatternType.LINGUISTIC,
                pattern_data=pattern_text,
                confidence=pattern_info.get("confidence", 0.5),
                frequency=pattern_info.get("frequency", 1),
                metadata={
                    "linguistic_type": pattern_info.get("type", "phrase"),
                    "source": "linguistic_detector"
                }
            )
            patterns.append(pattern)

        return patterns

    def match(self, pattern: Pattern, data: Any, context: Dict[str, Any]) -> List[PatternMatch]:
        """
        Match a linguistic pattern against text data.

        Args:
            pattern: Pattern to match
            data: Data to match against
            context: Additional contextual information

        Returns:
            List of pattern matches
        """
        matches = []

        # Ensure we have text data
        if not isinstance(data, str) or not isinstance(pattern.pattern_data, str):
            return matches

        # Get linguistic type from metadata
        ling_type = pattern.metadata.get("linguistic_type", "phrase")

        if ling_type == "phrase":
            # Simple substring matching
            pattern_text = pattern.pattern_data

            # Find all occurrences
            start_index = 0
            while True:
                start_index = data.find(pattern_text, start_index)
                if start_index == -1:
                    break

                end_index = start_index + len(pattern_text)

                match = PatternMatch(
                    matched=True,
                    pattern=pattern,
                    confidence=pattern.confidence,
                    match_start=start_index,
                    match_end=end_index,
                    match_elements=[pattern_text],
                    similarity=1.0
                )
                matches.append(match)

                # Move past this match
                start_index = end_index

        elif ling_type == "regex":
            # Regular expression matching
            try:
                pattern_regex = pattern.pattern_data
                for match in re.finditer(pattern_regex, data):
                    pattern_match = PatternMatch(
                        matched=True,
                        pattern=pattern,
                        confidence=pattern.confidence,
                        match_start=match.start(),
                        match_end=match.end(),
                        match_elements=[match.group(0)],
                        similarity=1.0,
                        metadata={"groups": match.groups()}
                    )
                    matches.append(pattern_match)
            except re.error:
                # Invalid regex
                pass

        return matches

    def _extract_common_phrases(self, texts: List[str]) -> List[Tuple[str, Dict[str, Any]]]:
        """
        Extract common phrases from a list of texts.

        Args:
            texts: List of text strings

        Returns:
            List of (phrase, info_dict) tuples
        """
        common_phrases = []

        # Simplified n-gram extraction - real implementation would be more sophisticated
        n_values = [2, 3, 4]  # bigrams, trigrams, 4-grams

        for n in n_values:
            # Extract n-grams from all texts
            all_ngrams = []

            for text in texts:
                words = text.split()
                if len(words) < n:
                    continue

                text_ngrams = [' '.join(words[i:i+n]) for i in range(len(words) - n + 1)]
                all_ngrams.extend(text_ngrams)

            # Count occurrences
            ngram_counts = Counter(all_ngrams)

            # Filter for common n-grams (appearing at least twice)
            for ngram, count in ngram_counts.items():
                if count >= 2:
                    common_phrases.append((ngram, {
                        "type": "phrase",
                        "frequency": count,
                        "confidence": min(0.5 + 0.1 * count, 0.9),
                        "n_value": n
                    }))

        return common_phrases

    def _extract_regex_patterns(self, texts: List[str]) -> List[Tuple[str, Dict[str, Any]]]:
        """
        Extract regex patterns from texts.

        Args:
            texts: List of text strings

        Returns:
            List of (regex_pattern, info_dict) tuples
        """
        regex_patterns = []

        # Predefined regex patterns to check for
        common_regexes = [
            (r'\b\d{3}-\d{3}-\d{4}\b', "phone_number"),
            (r'\b\w+@\w+\.\w+\b', "email"),
            (r'\b\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}\b', "ip_address"),
            (r'\b\d{5}(-\d{4})?\b', "zip_code"),
            (r'\b(https?://)\S+\b', "url")
        ]

        for pattern, pattern_name in common_regexes:
            # Count how many texts contain this pattern
            match_count = 0
            for text in texts:
                if re.search(pattern, text):
                    match_count += 1

            # If pattern appears in multiple texts, add it
            if match_count >= 2:
                regex_patterns.append((pattern, {
                    "type": "regex",
                    "frequency": match_count,
                    "confidence": min(0.6 + 0.1 * match_count, 0.9),
                    "pattern_name": pattern_name
                }))

        return regex_patterns


class SymbolicPatternDetector(PatternDetectorBase):
    """Detector for symbolic patterns in mathematical or logical expressions"""

    def detect(self, data: Any, history: List[Tuple[Any, str, float, Dict[str, Any]]],
              context: Dict[str, Any]) -> List[Pattern]:
        """
        Detect symbolic patterns in data and history.

        Args:
            data: Current symbolic data
            history: List of (data, data_type, timestamp, context) tuples
            context: Additional contextual information

        Returns:
            List of detected symbolic patterns
        """
        # This is a placeholder - real implementation would depend on
        # symbolic expression parsing and pattern matching algorithms

        # Return empty list for now
        return []

    def match(self, pattern: Pattern, data: Any, context: Dict[str, Any]) -> List[PatternMatch]:
        """
        Match a symbolic pattern against data.

        Args:
            pattern: Pattern to match
            data: Data to match against
            context: Additional contextual information

        Returns:
            List of pattern matches
        """
        # This is a placeholder - real implementation would depend on
        # symbolic expression matching algorithms

        # Return empty list for now
        return []