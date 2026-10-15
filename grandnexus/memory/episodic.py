"""
episodic.py - Enhanced Episodic Memory Module for GrandNexus

This enhanced implementation combines concepts from:
- BioNeMo: For efficient retrieval mechanisms and embedding structures
- LIDA (Learning Intelligent Distribution Agent): For consciousness-inspired episodic structure
- ACT-R: For activation/decay mechanisms and attention-based memory processes
- HTM (Hierarchical Temporal Memory): For temporal encoding and sequence detection

Features:
- Event-Episode-Epoch hierarchy to organize memory temporally
- Sparse distributed representations for efficient storage and matching
- Activation-based retrieval with context sensitivity
- Salience-guided memory consolidation for long-term storage
- Temporal and causal associative links between episodes
"""

import logging
import time
import uuid
import heapq
import math
import threading
import numpy as np
from collections import defaultdict, deque
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import (
    Any, Dict, List, Optional, Set, Tuple, Callable, Union, Deque
)

# -----------------------------------------------------------------------------
# Enumerations and Types
# -----------------------------------------------------------------------------

class MemoryStrength(Enum):
    """Levels of memory strength, inspired by LIDA's activation levels"""
    UNCONSCIOUS = 1      # Very weak activation, not readily accessible
    PERIPHERAL = 2       # Weak activation, accessible with cues
    PRECONSCIOUS = 3     # Moderate activation, easily accessible with cues
    CONSCIOUS = 4        # Strong activation, part of current context
    FOCAL = 5            # Strongest activation, central to current focus


class EpisodeType(Enum):
    """Types of episodes for better organization and retrieval"""
    PERCEPTION = auto()   # Direct sensory experiences
    INTERACTION = auto()  # Agent interactions with environment
    COMMUNICATION = auto() # Communication with users or other agents
    DECISION = auto()     # Decision-making processes
    REFLECTION = auto()   # Self-reflection and metacognition
    SYSTEM = auto()       # System operations or administrative tasks
    PROCEDURAL = auto()   # Step-by-step procedures or workflows
    GENERAL = auto()      # General/unspecified episode type


class RetrievalMode(Enum):
    """Different modes for memory retrieval, affecting how matches are scored"""
    EXACT = auto()         # Exact match on specified criteria
    SIMILARITY = auto()    # Similarity-based matching using embeddings
    TEMPORAL = auto()      # Retrieval based on temporal relationships
    CAUSAL = auto()        # Retrieval that emphasizes cause-and-effect
    ASSOCIATIVE = auto()   # Retrieval based on associative links
    CONTEXT = auto()       # Context-sensitive retrieval including current state
    MIXED = auto()         # Balanced approach using multiple factors

# -----------------------------------------------------------------------------
# Data Structures
# -----------------------------------------------------------------------------

@dataclass
class EpisodeEvent:
    """
    Represents an atomic event within an episode.
    Inspired by ACT-R's declarative memory chunks and HTM's sparse representations.

    Attributes:
        event_id: Unique identifier for the event
        timestamp: When the event occurred
        content: The actual data (text, sensory data, etc.)
        metadata: Additional context or attributes
        importance: How salient/important this event is (0-1)
        embedding: Vector representation for similarity
        activation: Current activation level (ACT-R concept)
        sparse_rep: Sparse distributed representation (HTM concept)
        causal_links: IDs of events this may have caused/influenced
        predecessor_events: IDs of events that directly preceded this
    """
    event_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    timestamp: float = field(default_factory=time.time)
    content: Any = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    importance: float = 0.5
    embedding: Optional[np.ndarray] = None
    activation: float = 1.0  # Initial full activation
    sparse_rep: Optional[np.ndarray] = None  # Sparse distributed representation
    causal_links: List[str] = field(default_factory=list)
    predecessor_events: List[str] = field(default_factory=list)

    def __post_init__(self):
        # Set memory strength to CONSCIOUS initially
        self.memory_strength = MemoryStrength.CONSCIOUS
        self.access_count: int = 0
        self.last_access_time: float = self.timestamp

    def access(self) -> None:
        """Mark this event as accessed, updating activation and metadata."""
        self.access_count += 1
        self.last_access_time = time.time()
        self.activation = 1.0  # Full activation upon access

        # Update memory strength based on access
        if self.access_count > 5:
            self.memory_strength = MemoryStrength.FOCAL
        elif self.access_count > 2:
            self.memory_strength = MemoryStrength.CONSCIOUS

    def decay_activation(self, decay_rate: float, current_time: float) -> None:
        """
        Apply ACT-R-inspired decay to activation based on time since last access.

        Args:
            decay_rate: Rate at which activation decays
            current_time: Current timestamp for decay calculation
        """
        time_since_access = max(0.001, current_time - self.last_access_time)

        # ACT-R-style power law decay: A = B * (T^-d)
        # Where B is base activation, T is time since access, d is decay parameter
        base_level = 1.0
        decay_factor = math.pow(time_since_access, -decay_rate)

        # Incorporate access count (more accesses = slower decay)
        access_boost = math.log(1 + self.access_count) * 0.1

        self.activation = base_level * decay_factor + access_boost

        # Cap activation within range [0, 1]
        self.activation = max(0.0, min(1.0, self.activation))

        # Update memory strength based on activation
        if self.activation > 0.8:
            self.memory_strength = MemoryStrength.CONSCIOUS
        elif self.activation > 0.5:
            self.memory_strength = MemoryStrength.PRECONSCIOUS
        elif self.activation > 0.2:
            self.memory_strength = MemoryStrength.PERIPHERAL
        else:
            self.memory_strength = MemoryStrength.UNCONSCIOUS

    def __repr__(self) -> str:
        return (f"<EpisodeEvent id={self.event_id[-8:]}, time={self.timestamp:.1f}, "
                f"imp={self.importance:.2f}, act={self.activation:.2f}, "
                f"strength={self.memory_strength.name}>")


@dataclass
class Episode:
    """
    Represents a coherent collection of events, inspired by LIDA's episodic memory
    and HTM's sequence representations.

    Attributes:
        episode_id: Unique identifier for this episode
        episode_type: Type of episode (perception, interaction, etc.)
        start_time: When the episode began
        end_time: When the episode ended (None if ongoing)
        events: Sequence of events in this episode
        context: Contextual information about the environment, agent state, etc.
        tags: Semantic labels for easier retrieval
        embedding: Vector representation of the entire episode
        importance: Overall salience/importance of the episode
        activation: Current activation level (decays over time)
        location: Optional spatial reference
        associations: Links to related episodes
        summary: Brief description of episode content
    """
    episode_id: str
    episode_type: EpisodeType = EpisodeType.GENERAL
    start_time: float = field(default_factory=time.time)
    end_time: Optional[float] = None
    events: List[EpisodeEvent] = field(default_factory=list)
    context: Dict[str, Any] = field(default_factory=dict)
    tags: Set[str] = field(default_factory=set)
    embedding: Optional[np.ndarray] = None
    importance: float = 0.5
    activation: float = 1.0
    location: Optional[str] = None
    associations: Dict[str, float] = field(default_factory=dict)  # episode_id -> strength
    summary: Optional[str] = None

    def __post_init__(self):
        # Set memory strength to CONSCIOUS initially
        self.memory_strength = MemoryStrength.CONSCIOUS
        self.access_count: int = 0
        self.last_access_time: float = self.start_time
        self.temporal_context: List[str] = []  # Episode IDs that occurred before/after

        # HTM-inspired fields for sequence learning
        self.sequence_predictors: Dict[str, Dict[str, float]] = defaultdict(dict)  # event -> (next_event -> probability)

    def add_event(self, event: EpisodeEvent) -> None:
        """
        Adds an event to the episode, maintaining temporal order and
        updating HTM-style sequence predictors.
        """
        if not self.events:
            self.events.append(event)
            return

        # If the new event is out of order, insert it at the right position
        if event.timestamp < self.events[-1].timestamp:
            # Find the correct position
            for i, existing_event in enumerate(self.events):
                if event.timestamp < existing_event.timestamp:
                    # Update predecessor relationships
                    if i > 0:
                        event.predecessor_events.append(self.events[i-1].event_id)
                    if i < len(self.events):
                        self.events[i].predecessor_events.append(event.event_id)

                    # Insert at correct position
                    self.events.insert(i, event)
                    break
        else:
            # Add at the end (normal case)
            if self.events:
                # Update predecessor relationship
                event.predecessor_events.append(self.events[-1].event_id)

                # Update HTM-style sequence predictors
                last_event_id = self.events[-1].event_id
                event_type = event.metadata.get("type", "generic")

                # Increment count for this sequence
                if last_event_id in self.sequence_predictors:
                    self.sequence_predictors[last_event_id][event.event_id] = self.sequence_predictors[last_event_id].get(event.event_id, 0) + 1
                else:
                    self.sequence_predictors[last_event_id] = {event.event_id: 1}

            self.events.append(event)

        # Update episode importance based on event importance
        self.importance = max(self.importance, event.importance)

        # Update episode's last access time
        self.last_access_time = time.time()

    def close(self) -> None:
        """Marks this episode as finished by setting end_time."""
        if self.end_time is None:
            self.end_time = time.time()

        # Normalize sequence predictor probabilities
        for source_event, targets in self.sequence_predictors.items():
            total = sum(targets.values())
            if total > 0:
                for target_event in targets:
                    targets[target_event] /= total

    def is_active(self) -> bool:
        """Returns True if the episode has not been closed yet."""
        return self.end_time is None

    def duration(self) -> float:
        """Return the length of time from start_time to end_time (or now if ongoing)."""
        if self.end_time is None:
            return time.time() - self.start_time
        else:
            return self.end_time - self.start_time

    def access(self) -> None:
        """Mark this episode as accessed, updating activation and metadata."""
        self.access_count += 1
        self.last_access_time = time.time()
        self.activation = 1.0  # Full activation upon access

        # Update memory strength based on access
        if self.access_count > 5:
            self.memory_strength = MemoryStrength.FOCAL
        elif self.access_count > 2:
            self.memory_strength = MemoryStrength.CONSCIOUS

    def decay_activation(self, decay_rate: float, current_time: float) -> None:
        """
        Apply ACT-R-inspired decay to activation based on time since last access.
        """
        time_since_access = max(0.001, current_time - self.last_access_time)

        # Decay formula similar to event decay but slower for episodes
        base_level = 1.0
        decay_factor = math.pow(time_since_access, -decay_rate * 0.5)  # Slower decay for episodes

        # Incorporate access count, importance, and recency
        access_boost = math.log(1 + self.access_count) * 0.1
        importance_boost = self.importance * 0.2

        self.activation = base_level * decay_factor + access_boost + importance_boost

        # Cap activation within range [0, 1]
        self.activation = max(0.0, min(1.0, self.activation))

        # Update memory strength based on activation
        if self.activation > 0.8:
            self.memory_strength = MemoryStrength.CONSCIOUS
        elif self.activation > 0.5:
            self.memory_strength = MemoryStrength.PRECONSCIOUS
        elif self.activation > 0.2:
            self.memory_strength = MemoryStrength.PERIPHERAL
        else:
            self.memory_strength = MemoryStrength.UNCONSCIOUS

    def predict_next_event_type(self, current_event_id: str) -> Optional[str]:
        """
        HTM-inspired prediction of what might come next in a sequence.

        Args:
            current_event_id: ID of the current event

        Returns:
            ID of the most likely next event, or None if no prediction
        """
        if current_event_id not in self.sequence_predictors:
            return None

        # Get probabilities for next events
        next_event_probs = self.sequence_predictors[current_event_id]
        if not next_event_probs:
            return None

        # Return the most probable next event
        return max(next_event_probs.items(), key=lambda x: x[1])[0]

    def generate_summary(self) -> str:
        """
        Generate a brief summary of this episode if not already set.
        In a real implementation, this might use LLM integration.
        """
        if self.summary:
            return self.summary

        # Simple summary generation based on events and context
        event_count = len(self.events)
        duration = self.duration()

        if event_count == 0:
            summary = f"Empty {self.episode_type.name.lower()} episode"
        else:
            # Get the first and last event contents
            first_event = self.events[0].content
            last_event = self.events[-1].content

            # Try to extract content strings
            if isinstance(first_event, str):
                first_content = first_event[:30]
            else:
                first_content = str(type(first_event).__name__)

            if isinstance(last_event, str):
                last_content = last_event[:30]
            else:
                last_content = str(type(last_event).__name__)

            summary = (f"{self.episode_type.name.capitalize()} episode with {event_count} events "
                      f"over {duration:.1f}s, from '{first_content}...' to '{last_content}...'")

        self.summary = summary
        return summary

    def __repr__(self) -> str:
        status = "active" if self.is_active() else "closed"
        return (f"<Episode id={self.episode_id}, type={self.episode_type.name}, "
                f"events={len(self.events)}, {status}, imp={self.importance:.2f}, "
                f"act={self.activation:.2f}, strength={self.memory_strength.name}>")


@dataclass
class Epoch:
    """
    Represents a higher-level time period containing multiple episodes.
    This creates a 3-level hierarchy: Event -> Episode -> Epoch.
    Inspired by LIDA's hierarchical structure and HTM's temporal pooling.

    Attributes:
        epoch_id: Unique identifier for this epoch
        name: Human-readable name for this epoch
        start_time: When the epoch began
        end_time: When the epoch ended (None if ongoing)
        episodes: Episodes contained in this epoch
        themes: Detected themes or patterns across episodes
        embedding: Vector representation of the entire epoch
        importance: Overall salience/importance of the epoch
    """
    epoch_id: str
    name: str
    start_time: float
    end_time: Optional[float] = None
    episodes: List[str] = field(default_factory=list)  # List of episode_ids
    themes: Dict[str, float] = field(default_factory=dict)  # theme -> strength
    embedding: Optional[np.ndarray] = None
    importance: float = 0.5

    def is_active(self) -> bool:
        """Returns True if the epoch has not been closed yet."""
        return self.end_time is None

    def add_episode(self, episode_id: str) -> None:
        """Adds an episode to this epoch."""
        if episode_id not in self.episodes:
            self.episodes.append(episode_id)

    def close(self) -> None:
        """Marks this epoch as finished by setting end_time."""
        if self.end_time is None:
            self.end_time = time.time()

    def duration(self) -> float:
        """Return the length of time from start_time to end_time (or now if ongoing)."""
        if self.end_time is None:
            return time.time() - self.start_time
        else:
            return self.end_time - self.start_time

    def __repr__(self) -> str:
        status = "active" if self.is_active() else "closed"
        return (f"<Epoch id={self.epoch_id}, name='{self.name}', "
                f"episodes={len(self.episodes)}, {status}, imp={self.importance:.2f}>")


@dataclass
class EpisodicQuery:
    """
    Enhanced query structure for retrieving memories from episodic memory.
    Supports multiple retrieval modes and filters.

    Attributes:
        text: Optional text to match against episode/event content
        embedding: Optional embedding vector for similarity search
        tags: Optional set of tags to filter by
        time_range: Optional (start_time, end_time) to filter by
        episode_types: Optional set of episode types to filter by
        importance_threshold: Minimum importance score for retrieved episodes
        activation_threshold: Minimum activation for retrieved episodes
        memory_strength: Minimum memory strength for retrieved episodes
        retrieval_mode: Strategy for ranking and scoring matches
        max_results: Maximum number of results to return
        include_events: Whether to include individual events in results
        recursive: Whether to include associated episodes
        context_factors: Additional context to influence retrieval
    """
    text: Optional[str] = None
    embedding: Optional[np.ndarray] = None
    tags: Optional[Set[str]] = None
    time_range: Optional[Tuple[float, float]] = None
    episode_types: Optional[Set[EpisodeType]] = None
    importance_threshold: float = 0.0
    activation_threshold: float = 0.0
    memory_strength: Optional[MemoryStrength] = None
    retrieval_mode: RetrievalMode = RetrievalMode.MIXED
    max_results: int = 10
    include_events: bool = False
    recursive: bool = False
    context_factors: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        # Convert string episode types to enum values if needed
        if self.episode_types and any(isinstance(t, str) for t in self.episode_types):
            types = set()
            for t in self.episode_types:
                if isinstance(t, str):
                    try:
                        types.add(EpisodeType[t.upper()])
                    except KeyError:
                        pass  # Ignore invalid types
                else:
                    types.add(t)
            self.episode_types = types

        # Convert string memory strength to enum value if needed
        if isinstance(self.memory_strength, str):
            try:
                self.memory_strength = MemoryStrength[self.memory_strength.upper()]
            except KeyError:
                self.memory_strength = None

        # Convert string retrieval mode to enum value if needed
        if isinstance(self.retrieval_mode, str):
            try:
                self.retrieval_mode = RetrievalMode[self.retrieval_mode.upper()]
            except KeyError:
                self.retrieval_mode = RetrievalMode.MIXED


@dataclass
class EpisodicMemoryConfig:
    """
    Configuration for the EpisodicMemory system.

    Attributes:
        short_term_capacity: Max episodes in short-term memory
        long_term_capacity: Max episodes in long-term memory
        event_decay_rate: How quickly event activations decay (ACT-R concept)
        episode_decay_rate: How quickly episode activations decay (ACT-R concept)
        consolidation_interval: How often to run consolidation (seconds)
        embedding_dimension: Size of embedding vectors
        enable_memory_optimization: Whether to use memory optimization techniques
        similarity_threshold: Threshold for considering items similar
        embedding_function: Function to generate embeddings from content
        enable_predictive_coding: Whether to use HTM-style predictive coding
        enable_temporal_pooling: Whether to use HTM-style temporal pooling
        auto_generate_summaries: Whether to auto-generate episode summaries
        auto_tagging: Whether to automatically generate tags for episodes
        max_associative_distance: Max distance for associative retrieval
        attention_span: Number of episodes to keep in attentional focus
        consciousness_threshold: Activation threshold for "conscious" access
        temporal_context_size: Size of temporal context window
        max_epochs: Maximum number of epochs to maintain
        consolidation_scheduler: How to schedule consolidation (periodic/adaptive)
    """
    short_term_capacity: int = 100
    long_term_capacity: int = 1000
    event_decay_rate: float = 0.5
    episode_decay_rate: float = 0.3
    consolidation_interval: float = 300.0  # 5 minutes
    embedding_dimension: int = 256
    enable_memory_optimization: bool = True
    similarity_threshold: float = 0.7
    embedding_function: Optional[Callable[[Any], np.ndarray]] = None
    enable_predictive_coding: bool = True
    enable_temporal_pooling: bool = True
    auto_generate_summaries: bool = True
    auto_tagging: bool = True
    max_associative_distance: int = 3
    attention_span: int = 5
    consciousness_threshold: float = 0.7
    temporal_context_size: int = 10
    max_epochs: int = 100
    consolidation_scheduler: str = "adaptive"  # "periodic" or "adaptive"


@dataclass
class EpisodicMemoryState:
    """
    Internal state for episodic memory, used to track current context
    and attentional focus.
    """
    current_episode_id: Optional[str] = None
    attentional_focus: List[str] = field(default_factory=list)  # episode_ids in focus
    temporal_context: List[str] = field(default_factory=list)  # recent episode_ids
    active_concepts: Dict[str, float] = field(default_factory=dict)  # concept_id -> activation
    last_consolidation_time: float = field(default_factory=time.time)
    operation_count: int = 0
    current_epoch_id: Optional[str] = None


class EpisodicMemory:
    """
    Enhanced episodic memory system incorporating concepts from:
    - LIDA: Consciousness-inspired memory structure
    - ACT-R: Activation-based retrieval and decay
    - HTM: Hierarchical temporal memory patterns

    Features:
    - Three-level memory hierarchy: Events -> Episodes -> Epochs
    - Activation-based memory with ACT-R-inspired decay
    - Consciousness-like attentional mechanisms
    - HTM-inspired sequence learning and prediction
    - Associative linking between related episodes
    - Content-addressable memory with embedding similarity
    """

    def __init__(self, config: EpisodicMemoryConfig):
        """Initialize EpisodicMemory with the provided configuration."""
        self.logger = logging.getLogger("EpisodicMemory")
        self.config = config
        self.lock = threading.RLock()

        # Primary memory stores
        self.episodes: Dict[str, Episode] = {}
        self.events: Dict[str, EpisodeEvent] = {}
        self.epochs: Dict[str, Epoch] = {}

        # Organizational structures
        self.short_term_episodes: List[str] = []  # episode_ids in short-term memory
        self.long_term_episodes: List[str] = []   # episode_ids in long-term memory

        # Indexing structures for efficient retrieval
        self.episodes_by_type: Dict[EpisodeType, List[str]] = {
            etype: [] for etype in EpisodeType
        }
        self.episodes_by_tag: Dict[str, List[str]] = defaultdict(list)
        self.episode_timeline: List[Tuple[float, str]] = []  # (timestamp, episode_id) sorted by time

        # ACT-R inspired activation tracking
        self.episode_activations: Dict[str, float] = {}
        self.event_activations: Dict[str, float] = {}

        # HTM inspired structures
        self.temporal_sequences: Dict[str, Dict[str, float]] = defaultdict(dict)  # ep_id -> (next_ep_id -> probability)
        self.active_predictions: Dict[str, float] = {}  # predicted_episode_id -> confidence

        # LIDA inspired attention and consciousness structures
        self.state = EpisodicMemoryState()

        # Time tracking for adaptive consolidation
        self.last_decay_time = time.time()
        self.last_consolidation_time = time.time()

        self.logger.info(f"EpisodicMemory initialized (short={config.short_term_capacity}, "
                        f"long={config.long_term_capacity})")

    # -------------------------------------------------------------------------
    # Episode Creation and Management
    # -------------------------------------------------------------------------

    def create_episode(self,
                      episode_type: Union[EpisodeType, str] = EpisodeType.GENERAL,
                      context: Dict[str, Any] = None,
                      tags: Set[str] = None,
                      episode_id: Optional[str] = None,
                      importance: float = 0.5,
                      location: Optional[str] = None) -> Episode:
        """
        Creates a new episode and makes it the current active episode.

        Args:
            episode_type: Type of episode being created
            context: Contextual information for this episode
            tags: Semantic tags for categorization
            episode_id: Optional custom ID (generated if None)
            importance: Initial importance score (0-1)
            location: Optional location identifier

        Returns:
            The newly created Episode object
        """
        with self.lock:
            # Convert string episode type to enum if needed
            if isinstance(episode_type, str):
                try:
                    episode_type = EpisodeType[episode_type.upper()]
                except KeyError:
                    episode_type = EpisodeType.GENERAL

            # Generate ID if not provided
            if episode_id is None:
                episode_id = str(uuid.uuid4())

            # Check for duplicates
            if episode_id in self.episodes:
                self.logger.warning(f"Episode {episode_id} already exists. Returning existing episode.")
                return self.episodes[episode_id]

            # Create the episode
            episode = Episode(
                episode_id=episode_id,
                episode_type=episode_type,
                context=context or {},
                tags=tags or set(),
                importance=importance,
                location=location
            )

            # Store the episode
            self.episodes[episode_id] = episode
            self.short_term_episodes.append(episode_id)
            self.episode_activations[episode_id] = 1.0  # Full initial activation

            # Update indices
            self.episodes_by_type[episode_type].append(episode_id)
            for tag in episode.tags:
                self.episodes_by_tag[tag].append(episode_id)

            # Add to timeline
            self.episode_timeline.append((episode.start_time, episode_id))
            self._sort_timeline()

            # Update attention state
            self._update_attention(episode_id)

            # Make this the current episode
            self.state.current_episode_id = episode_id

            # Assign to current epoch if one is active
            if self.state.current_epoch_id and self.state.current_epoch_id in self.epochs:
                current_epoch = self.epochs[self.state.current_epoch_id]
                if current_epoch.is_active():
                    current_epoch.add_episode(episode_id)

            # Auto-tag if enabled
            if self.config.auto_tagging:
                self._auto_tag_episode(episode)

            # Increment operation count
            self.state.operation_count += 1

            # Check if we should run adaptive consolidation
            if (self.config.consolidation_scheduler == "adaptive" and
                self.state.operation_count % 10 == 0):  # Every 10 operations
                self._consolidate_memory()

            self.logger.info(f"Created new episode {episode_id} of type {episode_type.name}")
            return episode

    def add_event(self,
                 episode_id: str,
                 content: Any,
                 metadata: Dict[str, Any] = None,
                 importance: float = 0.5,
                 event_id: Optional[str] = None) -> Optional[str]:
        """
        Add a new event to the specified episode.

        Args:
            episode_id: ID of the episode to add this event to
            content: Event content (text, sensor data, etc.)
            metadata: Additional context or attributes
            importance: Event importance (0-1)
            event_id: Optional custom ID (generated if None)

        Returns:
            The event_id if successful, None otherwise
        """
        with self.lock:
            # Check if episode exists and is active
            if episode_id not in self.episodes:
                self.logger.warning(f"Cannot add event: Episode {episode_id} not found.")
                return None

            episode = self.episodes[episode_id]
            if not episode.is_active():
                self.logger.warning(f"Cannot add event: Episode {episode_id} is closed.")
                return None

            # Generate ID if not provided
            if event_id is None:
                event_id = str(uuid.uuid4())

            # Check for duplicates
            if event_id in self.events:
                self.logger.warning(f"Event {event_id} already exists. Using existing event.")
                return event_id

            # Create event
            event = EpisodeEvent(
                event_id=event_id,
                timestamp=time.time(),
                content=content,
                metadata=metadata or {},
                importance=importance
            )

            # Generate embedding if function provided
            if self.config.embedding_function:
                try:
                    event.embedding = self.config.embedding_function(content)
                except Exception as e:
                    self.logger.error(f"Error generating embedding for event {event_id}: {e}")

            # Generate sparse representation (HTM-inspired)
            if self.config.enable_predictive_coding:
                event.sparse_rep = self._generate_sparse_representation(content)

            # Store event
            self.events[event_id] = event
            self.event_activations[event_id] = 1.0  # Full initial activation

            # Add to episode
            episode.add_event(event)

            # Update episode importance if this event is more important
            if importance > episode.importance:
                episode.importance = importance

            # Update episode's activation
            episode.activation = 1.0
            self.episode_activations[episode_id] = 1.0

            # Check for HTM-style predictions and update accuracy
            self._update_predictions(episode, event)

            # Increment operation count
            self.state.operation_count += 1

            # Check if we should run adaptive consolidation
            if (self.config.consolidation_scheduler == "adaptive" and
                self.state.operation_count % 20 == 0):  # Every 20 operations
                self._consolidate_memory()

            self.logger.debug(f"Added event {event_id} to episode {episode_id}")
            return event_id

    def close_episode(self, episode_id: str) -> bool:
        """
        Mark an episode as complete by setting its end_time.
        If embedding_function is provided, computes an embedding for the episode.

        Args:
            episode_id: ID of the episode to close

        Returns:
            True if successful, False otherwise
        """
        with self.lock:
            if episode_id not in self.episodes:
                self.logger.warning(f"Cannot close episode {episode_id}: Not found.")
                return False

            episode = self.episodes[episode_id]
            if not episode.is_active():
                self.logger.debug(f"Episode {episode_id} is already closed.")
                return True

            # Close the episode
            episode.close()

            # Generate embedding if function provided
            if self.config.embedding_function and episode.events:
                try:
                    # Create a combined representation of all events
                    all_content = []
                    for event in episode.events:
                        if isinstance(event.content, str):
                            all_content.append(event.content)
                        else:
                            # Try to get a string representation
                            all_content.append(str(event.content))

                    combined_content = " ".join(all_content)
                    episode.embedding = self.config.embedding_function(combined_content)
                except Exception as e:
                    self.logger.error(f"Error generating embedding for episode {episode_id}: {e}")

            # Generate summary if auto-generation is enabled
            if self.config.auto_generate_summaries:
                episode.generate_summary()

            # Update temporal links in context
            self._update_temporal_links(episode_id)

            # Make this episode no longer the current one
            if self.state.current_episode_id == episode_id:
                self.state.current_episode_id = None

            # If this was in the attentional focus, update attention
            if episode_id in self.state.attentional_focus:
                self.state.attentional_focus.remove(episode_id)

            # HTM-inspired sequence learning
            if self.config.enable_predictive_coding:
                self._learn_episode_sequence(episode_id)

            # Check if we should consolidate memory after closing an episode
            if self.config.consolidation_scheduler == "periodic":
                current_time = time.time()
                if current_time - self.last_consolidation_time >= self.config.consolidation_interval:
                    self._consolidate_memory()

            self.logger.info(f"Closed episode {episode_id} with {len(episode.events)} events.")
            return True

    def create_epoch(self,
                    name: str,
                    episodes: List[str] = None,
                    epoch_id: Optional[str] = None,
                    importance: float = 0.5) -> Epoch:
        """
        Creates a new epoch to group episodes together.

        Args:
            name: Human-readable name for this epoch
            episodes: Optional list of episode IDs to include initially
            epoch_id: Optional custom ID (generated if None)
            importance: Initial importance score (0-1)

        Returns:
            The newly created Epoch object
        """
        with self.lock:
            # Generate ID if not provided
            if epoch_id is None:
                epoch_id = str(uuid.uuid4())

            # Check for duplicates
            if epoch_id in self.epochs:
                self.logger.warning(f"Epoch {epoch_id} already exists. Returning existing epoch.")
                return self.epochs[epoch_id]

            # Create the epoch
            epoch = Epoch(
                epoch_id=epoch_id,
                name=name,
                start_time=time.time(),
                episodes=[],
                importance=importance
            )

            # Add episodes if provided
            if episodes:
                valid_episodes = [ep_id for ep_id in episodes if ep_id in self.episodes]
                epoch.episodes = valid_episodes

            # Store the epoch
            self.epochs[epoch_id] = epoch

            # Make this the current epoch
            self.state.current_epoch_id = epoch_id

            self.logger.info(f"Created new epoch {epoch_id} named '{name}' with {len(epoch.episodes)} episodes")
            return epoch

    def close_epoch(self, epoch_id: str) -> bool:
        """
        Mark an epoch as complete by setting its end_time.

        Args:
            epoch_id: ID of the epoch to close

        Returns:
            True if successful, False otherwise
        """
        with self.lock:
            if epoch_id not in self.epochs:
                self.logger.warning(f"Cannot close epoch {epoch_id}: Not found.")
                return False

            epoch = self.epochs[epoch_id]
            if not epoch.is_active():
                self.logger.debug(f"Epoch {epoch_id} is already closed.")
                return True

            # Close the epoch
            epoch.close()

            # No longer the current epoch
            if self.state.current_epoch_id == epoch_id:
                self.state.current_epoch_id = None

            # If we have episodic memory integration, we could analyze themes here
            if self.config.auto_generate_summaries:
                self._analyze_epoch_themes(epoch)

            self.logger.info(f"Closed epoch {epoch_id} named '{epoch.name}'")
            return True

    # -------------------------------------------------------------------------
    # Memory Retrieval
    # -------------------------------------------------------------------------

    def get_episode(self, episode_id: str, update_activation: bool = True) -> Optional[Episode]:
        """
        Retrieve an episode by ID, optionally updating its activation.

        Args:
            episode_id: ID of the episode to retrieve
            update_activation: Whether to update activation on access

        Returns:
            The Episode object or None if not found
        """
        with self.lock:
            episode = self.episodes.get(episode_id)
            if not episode:
                return None

            if update_activation:
                # Update episode activation
                episode.access()
                self.episode_activations[episode_id] = episode.activation

                # Update attention
                self._update_attention(episode_id)

            return episode

    def get_event(self, event_id: str, update_activation: bool = True) -> Optional[EpisodeEvent]:
        """
        Retrieve an event by ID, optionally updating its activation.

        Args:
            event_id: ID of the event to retrieve
            update_activation: Whether to update activation on access

        Returns:
            The EpisodeEvent object or None if not found
        """
        with self.lock:
            event = self.events.get(event_id)
            if not event:
                return None

            if update_activation:
                # Update event activation
                event.access()
                self.event_activations[event_id] = event.activation

            return event

    def get_epoch(self, epoch_id: str) -> Optional[Epoch]:
        """
        Retrieve an epoch by ID.

        Args:
            epoch_id: ID of the epoch to retrieve

        Returns:
            The Epoch object or None if not found
        """
        with self.lock:
            return self.epochs.get(epoch_id)

    def replay_episode(self,
                      episode_id: str,
                      callback: Callable[[EpisodeEvent], None],
                      update_activation: bool = True) -> bool:
        """
        Replay each event in the specified episode by calling the callback function.

        Args:
            episode_id: ID of the episode to replay
            callback: Function to call for each event
            update_activation: Whether to update activations during replay

        Returns:
            True if replay successful, False otherwise
        """
        with self.lock:
            episode = self.episodes.get(episode_id)
            if not episode:
                self.logger.warning(f"Cannot replay episode {episode_id}: Not found.")
                return False

            if update_activation:
                # Update episode activation
                episode.access()
                self.episode_activations[episode_id] = episode.activation

                # Update attention
                self._update_attention(episode_id)

            # Replay each event in order
            for event in episode.events:
                if update_activation:
                    # Update event activation
                    event.access()
                    self.event_activations[event.event_id] = event.activation

                # Call callback with the event
                try:
                    callback(event)
                except Exception as e:
                    self.logger.error(f"Error in replay callback for event {event.event_id}: {e}")
                    return False

            return True

    def retrieve_episodes(self, query: EpisodicQuery) -> List[Episode]:
        """
        Enhanced retrieval of episodes based on various criteria.

        Args:
            query: EpisodicQuery object with search parameters

        Returns:
            List of matching episodes, sorted by relevance
        """
        with self.lock:
            # If using consciousness-based retrieval, prime the activation landscape
            if query.retrieval_mode == RetrievalMode.CONTEXT:
                self._prime_activations_for_context(query)

            # Get all candidate episodes
            candidates = self._get_candidate_episodes(query)

            # Score candidates based on retrieval mode
            scored_episodes = []
            for episode_id in candidates:
                episode = self.episodes[episode_id]
                score = self._score_episode(episode, query)
                scored_episodes.append((score, episode))

            # Sort by score (descending)
            scored_episodes.sort(reverse=True, key=lambda x: x[0])

            # Get top results
            top_episodes = [episode for _, episode in scored_episodes[:query.max_results]]

            # If recursive retrieval is enabled, include associated episodes
            if query.recursive and top_episodes:
                associated_episodes = self._get_associated_episodes(top_episodes)

                # Add associated episodes not already in results
                added_ids = {ep.episode_id for ep in top_episodes}
                for score, episode in associated_episodes:
                    if episode.episode_id not in added_ids and len(top_episodes) < query.max_results:
                        top_episodes.append(episode)
                        added_ids.add(episode.episode_id)

            # Update activations and attention for retrieved episodes
            for episode in top_episodes:
                episode.access()
                self.episode_activations[episode.episode_id] = episode.activation

                # Only update attention for the top few episodes
                if len(self.state.attentional_focus) < self.config.attention_span:
                    self._update_attention(episode.episode_id)

            # If include_events is True, also retrieve and activate the events
            if query.include_events:
                for episode in top_episodes:
                    for event in episode.events:
                        event.access()
                        self.event_activations[event.event_id] = event.activation

            self.logger.debug(f"Retrieved {len(top_episodes)} episodes matching query")
            return top_episodes

    def retrieve_events(self,
                       text: Optional[str] = None,
                       embedding: Optional[np.ndarray] = None,
                       time_range: Optional[Tuple[float, float]] = None,
                       importance_threshold: float = 0.0,
                       max_results: int = 20) -> List[EpisodeEvent]:
        """
        Retrieve individual events matching the criteria.

        Args:
            text: Optional text to match against event content
            embedding: Optional embedding for similarity search
            time_range: Optional time range to filter by
            importance_threshold: Minimum importance for events
            max_results: Maximum number of events to return

        Returns:
            List of matching events, sorted by relevance
        """
        with self.lock:
            candidates = []

            # Check all events
            for event_id, event in self.events.items():
                # Apply filters
                if importance_threshold > 0 and event.importance < importance_threshold:
                    continue

                if time_range:
                    start_time, end_time = time_range
                    if event.timestamp < start_time or event.timestamp > end_time:
                        continue

                candidates.append(event)

            # Score candidates
            scored_events = []
            for event in candidates:
                score = 0.0

                # Text matching
                if text and isinstance(event.content, str):
                    if text.lower() in event.content.lower():
                        score += 0.5

                # Embedding similarity
                if embedding is not None and event.embedding is not None:
                    similarity = self._vector_similarity(embedding, event.embedding)
                    score += similarity * 0.7

                # Importance and activation boost
                score += event.importance * 0.2
                score += event.activation * 0.1

                scored_events.append((score, event))

            # Sort by score
            scored_events.sort(reverse=True, key=lambda x: x[0])

            # Get top results
            top_events = [event for _, event in scored_events[:max_results]]

            # Update activations for retrieved events
            for event in top_events:
                event.access()
                self.event_activations[event.event_id] = event.activation

            return top_events

    def get_temporal_context(self) -> List[Episode]:
        """
        Get the current temporal context - episodes that are temporally adjacent
        to the current focus of attention.

        Returns:
            List of episodes in the temporal context
        """
        with self.lock:
            # Get episodes in the temporal context
            context_episodes = []
            for episode_id in self.state.temporal_context:
                if episode_id in self.episodes:
                    context_episodes.append(self.episodes[episode_id])

            return context_episodes

    def get_attentional_focus(self) -> List[Episode]:
        """
        Get episodes currently in the attentional focus.

        Returns:
            List of episodes in the attentional focus
        """
        with self.lock:
            focus_episodes = []
            for episode_id in self.state.attentional_focus:
                if episode_id in self.episodes:
                    focus_episodes.append(self.episodes[episode_id])

            return focus_episodes

    def query_by_time_range(self, start_time: float, end_time: float) -> List[Episode]:
        """
        Find episodes that overlap with the given time range.

        Args:
            start_time: Start of the time range
            end_time: End of the time range

        Returns:
            List of episodes that overlap with the time range
        """
        with self.lock:
            results = []

            # Linear search through episodes
            # (A more efficient implementation would use a time-indexed data structure)
            for episode_id, episode in self.episodes.items():
                episode_end = episode.end_time if episode.end_time is not None else time.time()

                # Check for overlap
                if not (episode_end < start_time or episode.start_time > end_time):
                    results.append(episode)

            # Sort by start time
            results.sort(key=lambda ep: ep.start_time)
            return results

    def query_by_tag(self, tag: str) -> List[Episode]:
        """
        Find episodes that have the specified tag.

        Args:
            tag: Tag to search for

        Returns:
            List of episodes with the tag
        """
        with self.lock:
            results = []

            # Use the tag index
            for episode_id in self.episodes_by_tag.get(tag, []):
                if episode_id in self.episodes:
                    results.append(self.episodes[episode_id])

            return results

    def predict_next_episodes(self,
                             current_episode_id: Optional[str] = None,
                             top_n: int = 3) -> List[Tuple[Episode, float]]:
        """
        Predict the most likely next episodes based on learned temporal patterns.
        Uses HTM-inspired sequence memory.

        Args:
            current_episode_id: ID of the current episode, or None to use the last one
            top_n: Number of predictions to return

        Returns:
            List of (episode, probability) tuples
        """
        with self.lock:
            # If no current episode specified, use the current or last episode
            if current_episode_id is None:
                if self.state.current_episode_id:
                    current_episode_id = self.state.current_episode_id
                elif self.state.temporal_context:
                    current_episode_id = self.state.temporal_context[-1]
                else:
                    return []

            # Check if episode exists
            if current_episode_id not in self.episodes:
                return []

            # Get predictions
            if current_episode_id in self.temporal_sequences:
                predictions = self.temporal_sequences[current_episode_id]

                # Sort by probability
                sorted_predictions = sorted(predictions.items(), key=lambda x: x[1], reverse=True)

                # Get top N
                results = []
                for pred_id, prob in sorted_predictions[:top_n]:
                    if pred_id in self.episodes:
                        results.append((self.episodes[pred_id], prob))

                return results

            return []

    # -------------------------------------------------------------------------
    # Memory Maintenance and Optimization
    # -------------------------------------------------------------------------

    def clear_all(self) -> None:
        """
        Clear all episodes, events, and epochs from memory.
        """
        with self.lock:
            self.episodes.clear()
            self.events.clear()
            self.epochs.clear()

            self.short_term_episodes.clear()
            self.long_term_episodes.clear()

            for etype in EpisodeType:
                self.episodes_by_type[etype].clear()

            self.episodes_by_tag.clear()
            self.episode_timeline.clear()

            self.episode_activations.clear()
            self.event_activations.clear()

            self.temporal_sequences.clear()
            self.active_predictions.clear()

            self.state = EpisodicMemoryState()

            self.last_decay_time = time.time()
            self.last_consolidation_time = time.time()

            self.logger.info("Episodic memory cleared")

    def _consolidate_memory(self) -> None:
        """
        Perform memory maintenance operations:
        1. Decay event and episode activations
        2. Move episodes between short-term and long-term memory
        3. Prune episodes if above capacity
        4. Update associations between episodes

        This implements concepts from ACT-R (activation decay) and
        LIDA (memory consolidation).
        """
        with self.lock:
            current_time = time.time()

            # Update decay time
            self.last_consolidation_time = current_time

            # Apply activation decay
            self._apply_activation_decay()

            # Check short-term capacity
            if len(self.short_term_episodes) > self.config.short_term_capacity:
                # Select episodes to move to long-term memory
                to_consolidate = len(self.short_term_episodes) - self.config.short_term_capacity
                self._move_episodes_to_long_term(to_consolidate)

            # Check long-term capacity
            if len(self.long_term_episodes) > self.config.long_term_capacity:
                # Remove excess episodes
                to_remove = len(self.long_term_episodes) - self.config.long_term_capacity
                self._prune_episodes(to_remove)

            # Update associations between episodes
            self._update_episode_associations()

            # Update temporal sequence models
            if self.config.enable_predictive_coding:
                self._update_sequence_models()

            # Update epochs if needed
            self._update_epochs()

            self.logger.debug(f"Memory consolidation complete. Short-term: {len(self.short_term_episodes)}, "
                             f"Long-term: {len(self.long_term_episodes)}")

    def _apply_activation_decay(self) -> None:
        """
        Apply ACT-R-inspired activation decay to events and episodes.
        """
        current_time = time.time()
        time_since_decay = current_time - self.last_decay_time
        self.last_decay_time = current_time

        # Only decay if enough time has passed (efficiency optimization)
        if time_since_decay < 1.0:
            return

        # Apply decay to events
        for event_id, event in self.events.items():
            event.decay_activation(self.config.event_decay_rate, current_time)
            self.event_activations[event_id] = event.activation

        # Apply decay to episodes
        for episode_id, episode in self.episodes.items():
            episode.decay_activation(self.config.episode_decay_rate, current_time)
            self.episode_activations[episode_id] = episode.activation

    def _move_episodes_to_long_term(self, count: int) -> None:
        """
        Move episodes from short-term to long-term memory based on
        consolidation criteria (importance, activation, etc.).

        Args:
            count: Number of episodes to move
        """
        if count <= 0:
            return

        # Score episodes for consolidation
        scored_episodes = []
        for episode_id in self.short_term_episodes:
            if episode_id in self.episodes:
                episode = self.episodes[episode_id]

                # Compute consolidation score
                # Episodes with high importance but low activation are prime candidates
                consolidation_score = (
                    episode.importance * 0.4 +    # Higher importance -> more likely to consolidate
                    (1.0 - episode.activation) * 0.3 +  # Lower activation -> more likely to consolidate
                    (episode.is_active() == False) * 0.3  # Closed episodes are prioritized
                )

                scored_episodes.append((consolidation_score, episode_id))

        # Sort by score (descending)
        scored_episodes.sort(reverse=True)

        # Select episodes to move
        to_move = [ep_id for _, ep_id in scored_episodes[:count]]

        # Move to long-term memory
        for episode_id in to_move:
            if episode_id in self.short_term_episodes:
                self.short_term_episodes.remove(episode_id)
                self.long_term_episodes.append(episode_id)

                # Generate summary if not already done
                episode = self.episodes[episode_id]
                if self.config.auto_generate_summaries and not episode.summary:
                    episode.generate_summary()

        self.logger.debug(f"Moved {len(to_move)} episodes to long-term memory")

    def _prune_episodes(self, count: int) -> None:
        """
        Remove the least important/relevant episodes from long-term memory.

        Args:
            count: Number of episodes to remove
        """
        if count <= 0:
            return

        # Score episodes for removal
        scored_episodes = []
        for episode_id in self.long_term_episodes:
            if episode_id in self.episodes:
                episode = self.episodes[episode_id]

                # Compute removal score
                # Lower score = higher priority for removal
                removal_score = (
                    episode.importance * 0.4 +          # Higher importance -> less likely to remove
                    episode.activation * 0.3 +          # Higher activation -> less likely to remove
                    (len(episode.events) > 0) * 0.2 +   # Episodes with events are less likely to be removed
                    (len(episode.associations) > 0) * 0.1  # Episodes with associations are less likely to be removed
                )

                scored_episodes.append((removal_score, episode_id))

        # Sort by score (ascending - lowest scores removed first)
        scored_episodes.sort()

        # Select episodes to remove
        to_remove = [ep_id for _, ep_id in scored_episodes[:count]]

        # Remove episodes
        for episode_id in to_remove:
            self._remove_episode(episode_id)

        self.logger.debug(f"Pruned {len(to_remove)} episodes from long-term memory")

    def _remove_episode(self, episode_id: str) -> None:
        """
        Internal method to remove an episode and its events.

        Args:
            episode_id: ID of the episode to remove
        """
        if episode_id not in self.episodes:
            return

        episode = self.episodes[episode_id]

        # Remove from both memory stores
        if episode_id in self.short_term_episodes:
            self.short_term_episodes.remove(episode_id)
        if episode_id in self.long_term_episodes:
            self.long_term_episodes.remove(episode_id)

        # Remove from type index
        if episode.episode_type in self.episodes_by_type and episode_id in self.episodes_by_type[episode.episode_type]:
            self.episodes_by_type[episode.episode_type].remove(episode_id)

        # Remove from tag index
        for tag in episode.tags:
            if tag in self.episodes_by_tag and episode_id in self.episodes_by_tag[tag]:
                self.episodes_by_tag[tag].remove(episode_id)

        # Remove from timeline
        self.episode_timeline = [(t, eid) for t, eid in self.episode_timeline if eid != episode_id]

        # Remove from attention and temporal context
        if episode_id in self.state.attentional_focus:
            self.state.attentional_focus.remove(episode_id)
        if episode_id in self.state.temporal_context:
            self.state.temporal_context.remove(episode_id)

        # Remove from activations
        if episode_id in self.episode_activations:
            del self.episode_activations[episode_id]

        # Remove from temporal sequences
        if episode_id in self.temporal_sequences:
            del self.temporal_sequences[episode_id]
        for seq in self.temporal_sequences.values():
            if episode_id in seq:
                del seq[episode_id]

        # Remove from epochs
        for epoch_id, epoch in self.epochs.items():
            if episode_id in epoch.episodes:
                epoch.episodes.remove(episode_id)

        # Remove associated events
        for event in episode.events:
            if event.event_id in self.events:
                del self.events[event.event_id]
            if event.event_id in self.event_activations:
                del self.event_activations[event.event_id]

        # Remove episode itself
        del self.episodes[episode_id]

    def _update_episode_associations(self) -> None:
        """
        Update associations between episodes based on:
        - Shared tags
        - Temporal proximity
        - Content similarity
        - Causal relationships

        This implements concepts from LIDA (associative networks).
        """
        # This can be computationally expensive, so we'll limit updates
        # to recently accessed episodes
        current_time = time.time()
        recent_threshold = current_time - 3600  # Last hour

        recent_episodes = [
            episode_id for episode_id, episode in self.episodes.items()
            if episode.last_access_time >= recent_threshold or episode.activation > 0.5
        ]

        # Limit to a reasonable number
        if len(recent_episodes) > 50:
            # Select the most recently accessed ones
            recent_episodes = sorted(
                recent_episodes,
                key=lambda ep_id: self.episodes[ep_id].last_access_time,
                reverse=True
            )[:50]

        # Update associations between these episodes
        for i, ep1_id in enumerate(recent_episodes):
            if ep1_id not in self.episodes:
                continue

            ep1 = self.episodes[ep1_id]

            for j in range(i+1, len(recent_episodes)):
                ep2_id = recent_episodes[j]
                if ep2_id not in self.episodes:
                    continue

                ep2 = self.episodes[ep2_id]

                # Calculate association strength
                assoc_strength = 0.0

                # Tag similarity
                tag_intersection = ep1.tags.intersection(ep2.tags)
                tag_union = ep1.tags.union(ep2.tags)
                if tag_union:
                    tag_similarity = len(tag_intersection) / len(tag_union)
                    assoc_strength += tag_similarity * 0.3

                # Temporal proximity
                if ep1.end_time and ep2.start_time:
                    # Episodes that occur close together in time
                    time_diff = abs(ep1.end_time - ep2.start_time)
                    time_factor = math.exp(-time_diff / 3600)  # Decay factor, 1 hour half-life
                    assoc_strength += time_factor * 0.2

                # Embedding similarity
                if ep1.embedding is not None and ep2.embedding is not None:
                    embedding_sim = self._vector_similarity(ep1.embedding, ep2.embedding)
                    assoc_strength += embedding_sim * 0.4

                # Context similarity
                context_match = sum(1 for k, v in ep1.context.items()
                                  if k in ep2.context and ep2.context[k] == v)
                context_total = len(set(ep1.context.keys()).union(ep2.context.keys()))
                if context_total > 0:
                    context_sim = context_match / context_total
                    assoc_strength += context_sim * 0.1

                # Only store significant associations
                if assoc_strength > 0.3:
                    ep1.associations[ep2_id] = assoc_strength
                    ep2.associations[ep1_id] = assoc_strength

    def _update_sequence_models(self) -> None:
        """
        Update HTM-inspired sequence models for episode prediction.
        """
        # Get timeline of recently closed episodes
        closed_episodes = [(t, eid) for t, eid in self.episode_timeline
                        if eid in self.episodes and not self.episodes[eid].is_active()]

        if len(closed_episodes) < 2:
            return

        # Update sequence model based on observed transitions
        for i in range(len(closed_episodes) - 1):
            current_ep_id = closed_episodes[i][1]
            next_ep_id = closed_episodes[i+1][1]

            # Add or update transition probability
            if current_ep_id not in self.temporal_sequences:
                self.temporal_sequences[current_ep_id] = {next_ep_id: 1.0}
            else:
                self.temporal_sequences[current_ep_id][next_ep_id] = self.temporal_sequences[current_ep_id].get(next_ep_id, 0) + 1.0

        # Normalize probabilities
        for source_id, transitions in self.temporal_sequences.items():
            total = sum(transitions.values())
            if total > 0:
                for target_id in transitions:
                    transitions[target_id] /= total

    def _update_epochs(self) -> None:
        """
        Update epoch information, such as theme detection.
        """
        # Find active epochs that have been updated
        for epoch_id, epoch in self.epochs.items():
            if not epoch.is_active():
                continue

            # Check if we have new episodes in this epoch
            episodes_updated = False
            for episode_id in epoch.episodes:
                if episode_id in self.episodes:
                    episode = self.episodes[episode_id]
                    if episode.last_access_time > self.last_consolidation_time:
                        episodes_updated = True
                        break

            if not episodes_updated:
                continue

            # Update epoch themes
            self._analyze_epoch_themes(epoch)

    def _analyze_epoch_themes(self, epoch: Epoch) -> None:
        """
        Analyze themes across episodes in an epoch.
        """
        # Skip epochs with too few episodes
        if len(epoch.episodes) < 2:
            return

        # Collect tags from all episodes
        all_tags = defaultdict(int)
        for episode_id in epoch.episodes:
            if episode_id in self.episodes:
                episode = self.episodes[episode_id]
                for tag in episode.tags:
                    all_tags[tag] += 1

        # Identify common themes
        total_episodes = len(epoch.episodes)
        themes = {}
        for tag, count in all_tags.items():
            if count >= 2:  # At least 2 episodes have this tag
                theme_strength = count / total_episodes
                themes[tag] = theme_strength

        # Update epoch themes
        epoch.themes = themes

    def _update_temporal_links(self, episode_id: str) -> None:
        """
        Update temporal links when an episode is closed.
        """
        # Add to temporal context
        if episode_id not in self.state.temporal_context:
            self.state.temporal_context.append(episode_id)
            # Keep only the most recent episodes in the context
            if len(self.state.temporal_context) > self.config.temporal_context_size:
                self.state.temporal_context.pop(0)

    def _update_attention(self, episode_id: str) -> None:
        """
        Update attentional focus with a newly accessed episode.
        Based on LIDA's attentional mechanisms.
        """
        # If already in focus, move to the end (most recent)
        if episode_id in self.state.attentional_focus:
            self.state.attentional_focus.remove(episode_id)
            self.state.attentional_focus.append(episode_id)
        else:
            # Add to focus
            self.state.attentional_focus.append(episode_id)

            # Limit focus size
            if len(self.state.attentional_focus) > self.config.attention_span:
                self.state.attentional_focus.pop(0)

    def _update_predictions(self, episode: Episode, event: EpisodeEvent) -> None:
        """
        Update HTM-style predictions based on a new event.
        """
        if not self.config.enable_predictive_coding or len(episode.events) < 2:
            return

        # Get previous event(s)
        prev_events = [e for e in episode.events if e.event_id != event.event_id and e.timestamp < event.timestamp]
        if not prev_events:
            return

        prev_event = max(prev_events, key=lambda e: e.timestamp)

        # Update sequence predictions
        if prev_event.event_id in episode.sequence_predictors:
            # Check if our prediction was correct
            predicted_events = episode.sequence_predictors[prev_event.event_id]
            for predicted_id, prob in list(predicted_events.items()):
                # Adjust probability based on match/miss
                if predicted_id == event.event_id:
                    # Correct prediction - increase probability
                    predicted_events[predicted_id] = min(1.0, prob * 1.1)
                else:
                    # Incorrect - decrease probability
                    predicted_events[predicted_id] = max(0.01, prob * 0.9)

        # Make new predictions based on this event
        episode.sequence_predictors.setdefault(event.event_id, {})

    def _learn_episode_sequence(self, episode_id: str) -> None:
        """
        Learn sequence patterns when an episode is closed.
        """
        if not self.config.enable_predictive_coding:
            return

        # Find episodes that preceded and followed this one
        timeline_idx = None
        for i, (_, eid) in enumerate(self.episode_timeline):
            if eid == episode_id:
                timeline_idx = i
                break

        if timeline_idx is None:
            return

        # Get preceding episode
        if timeline_idx > 0:
            prev_id = self.episode_timeline[timeline_idx-1][1]
            # Update sequence model
            if prev_id not in self.temporal_sequences:
                self.temporal_sequences[prev_id] = {episode_id: 1.0}
            else:
                self.temporal_sequences[prev_id][episode_id] = self.temporal_sequences[prev_id].get(episode_id, 0) + 1.0
                # Normalize
                total = sum(self.temporal_sequences[prev_id].values())
                for target_id in self.temporal_sequences[prev_id]:
                    self.temporal_sequences[prev_id][target_id] /= total

    def _prime_activations_for_context(self, query: EpisodicQuery) -> None:
        """
        Prime the activation landscape to enhance context-sensitive retrieval.
        """
        # Boost activation of episodes in the current temporal context
        for episode_id in self.state.temporal_context:
            if episode_id in self.episodes:
                episode = self.episodes[episode_id]
                episode.activation = min(1.0, episode.activation * 1.2)
                self.episode_activations[episode_id] = episode.activation

        # Boost activation of episodes in the attentional focus
        for episode_id in self.state.attentional_focus:
            if episode_id in self.episodes:
                episode = self.episodes[episode_id]
                episode.activation = min(1.0, episode.activation * 1.5)
                self.episode_activations[episode_id] = episode.activation

        # If context factors are provided, boost matching episodes
        if query.context_factors:
            for episode_id, episode in self.episodes.items():
                match_score = 0
                for k, v in query.context_factors.items():
                    if k in episode.context and episode.context[k] == v:
                        match_score += 1

                if match_score > 0:
                    boost_factor = 1.0 + (match_score / len(query.context_factors)) * 0.5
                    episode.activation = min(1.0, episode.activation * boost_factor)
                    self.episode_activations[episode_id] = episode.activation

    def _get_candidate_episodes(self, query: EpisodicQuery) -> List[str]:
        """
        Get the initial set of candidate episodes based on query filters.
        """
        candidates = set()

        # Apply hard filters first

        # Filter by episode type
        if query.episode_types:
            for etype in query.episode_types:
                candidates.update(self.episodes_by_type[etype])
        else:
            # Use all episodes
            candidates.update(self.episodes.keys())

        # Filter by tags (if specified)
        if query.tags:
            tag_matches = set()
            for tag in query.tags:
                tag_matches.update(self.episodes_by_tag.get(tag, []))
            if tag_matches:
                candidates = candidates.intersection(tag_matches)

        # Filter by time range (if specified)
        if query.time_range:
            start_time, end_time = query.time_range
            time_matches = set()
            for episode_id in candidates:
                episode = self.episodes[episode_id]
                episode_end = episode.end_time if episode.end_time is not None else time.time()
                if not (episode_end < start_time or episode.start_time > end_time):
                    time_matches.add(episode_id)
            candidates = time_matches

        # Filter by memory strength (if specified)
        if query.memory_strength:
            strength_matches = set()
            for episode_id in candidates:
                episode = self.episodes[episode_id]
                if episode.memory_strength.value >= query.memory_strength.value:
                    strength_matches.add(episode_id)
            candidates = strength_matches

        # Filter by activation threshold
        activation_matches = set()
        for episode_id in candidates:
            if episode_id in self.episode_activations:
                activation = self.episode_activations[episode_id]
                if activation >= query.activation_threshold:
                    activation_matches.add(episode_id)
        candidates = activation_matches

        # Filter by importance threshold
        importance_matches = set()
        for episode_id in candidates:
            episode = self.episodes[episode_id]
            if episode.importance >= query.importance_threshold:
                importance_matches.add(episode_id)
        candidates = importance_matches

        return list(candidates)

    def _score_episode(self, episode: Episode, query: EpisodicQuery) -> float:
        """
        Score an episode based on how well it matches the query.
        The scoring method depends on the retrieval mode.
        """
        score = 0.0

        # Base score components
        if query.retrieval_mode == RetrievalMode.EXACT:
            # Exact mode prioritizes precise matches
            score += self._exact_match_score(episode, query)

        elif query.retrieval_mode == RetrievalMode.SIMILARITY:
            # Similarity mode prioritizes embedding similarity
            score += self._similarity_score(episode, query)

        elif query.retrieval_mode == RetrievalMode.TEMPORAL:
            # Temporal mode prioritizes recency and sequence patterns
            score += self._temporal_score(episode, query)

        elif query.retrieval_mode == RetrievalMode.CAUSAL:
            # Causal mode prioritizes cause-effect relationships
            score += self._causal_score(episode, query)

        elif query.retrieval_mode == RetrievalMode.ASSOCIATIVE:
            # Associative mode prioritizes linked episodes
            score += self._associative_score(episode, query)

        elif query.retrieval_mode == RetrievalMode.CONTEXT:
            # Context mode prioritizes contextual relevance
            score += self._context_score(episode, query)

        else:  # RetrievalMode.MIXED or fallback
            # Mixed mode uses a balanced approach
            score += self._exact_match_score(episode, query) * 0.2
            score += self._similarity_score(episode, query) * 0.3
            score += self._temporal_score(episode, query) * 0.15
            score += self._causal_score(episode, query) * 0.1
            score += self._associative_score(episode, query) * 0.1
            score += self._context_score(episode, query) * 0.15

        # Additional factors that apply to all modes

        # Boost for episodes in attentional focus
        if episode.episode_id in self.state.attentional_focus:
            score += 0.3

        # Boost for episodes in temporal context
        if episode.episode_id in self.state.temporal_context:
            score += 0.2

        # Boost for high activation (already in consciousness)
        score += episode.activation * 0.2

        # Boost for high importance
        score += episode.importance * 0.1

        return score

    def _exact_match_score(self, episode: Episode, query: EpisodicQuery) -> float:
        """
        Score based on exact matches to query criteria.
        """
        score = 0.0

        # Text matching
        if query.text:
            # Check episode summary
            if episode.summary and query.text.lower() in episode.summary.lower():
                score += 0.5

            # Check event content
            content_matches = 0
            for event in episode.events:
                if isinstance(event.content, str) and query.text.lower() in event.content.lower():
                    content_matches += 1

            if content_matches > 0:
                # Logarithmic scoring to prevent domination by episodes with many events
                score += min(0.5, 0.1 * math.log(1 + content_matches))

        # Tag matching
        if query.tags:
            matching_tags = len(query.tags.intersection(episode.tags))
            if matching_tags > 0:
                score += 0.5 * (matching_tags / len(query.tags))

        return score

    def _similarity_score(self, episode: Episode, query: EpisodicQuery) -> float:
        """
        Score based on embedding similarity.
        """
        score = 0.0

        # Embedding similarity
        if query.embedding is not None and episode.embedding is not None:
            similarity = self._vector_similarity(query.embedding, episode.embedding)
            score += similarity

        return score

    def _temporal_score(self, episode: Episode, query: EpisodicQuery) -> float:
        """
        Score based on temporal factors (recency, sequence patterns).
        """
        score = 0.0

        # Recency factor
        current_time = time.time()
        episode_time = episode.end_time if episode.end_time else episode.start_time
        time_diff = current_time - episode_time
        recency = math.exp(-time_diff / 86400)  # 1-day half-life
        score += recency * 0.5

        # Sequence factor
        if self.state.current_episode_id in self.temporal_sequences:
            predictions = self.temporal_sequences[self.state.current_episode_id]
            if episode.episode_id in predictions:
                sequence_prob = predictions[episode.episode_id]
                score += sequence_prob * 0.5

        return score

    def _causal_score(self, episode: Episode, query: EpisodicQuery) -> float:
        """
        Score based on causal relationships.
        """
        score = 0.0

        # Causal links through events
        causal_count = 0
        for event in episode.events:
            causal_count += len(event.causal_links)

        if causal_count > 0:
            # Logarithmic scaling to prevent domination
            score += min(0.5, 0.1 * math.log(1 + causal_count))

        # If current episode is active, check for causal links
        if self.state.current_episode_id and self.state.current_episode_id in self.episodes:
            current_ep = self.episodes[self.state.current_episode_id]
            for event in current_ep.events:
                if episode.episode_id in event.causal_links:
                    score += 0.5
                    break

        return score

    def _associative_score(self, episode: Episode, query: EpisodicQuery) -> float:
        """
        Score based on associative links to other episodes.
        """
        score = 0.0

        # Associations with episodes in attentional focus
        for focus_id in self.state.attentional_focus:
            if focus_id in episode.associations:
                assoc_strength = episode.associations[focus_id]
                score += assoc_strength * 0.5

        # Associations with episodes matching the query
        if query.text and self.config.embedding_function:
            try:
                query_embedding = self.config.embedding_function(query.text)
                for assoc_id, strength in episode.associations.items():
                    if assoc_id in self.episodes:
                        assoc_ep = self.episodes[assoc_id]
                        if assoc_ep.embedding is not None:
                            sim = self._vector_similarity(query_embedding, assoc_ep.embedding)
                            if sim > 0.7:  # High similarity threshold
                                score += strength * sim * 0.5
            except Exception as e:
                self.logger.error(f"Error computing associative score: {e}")

        return score

    def _context_score(self, episode: Episode, query: EpisodicQuery) -> float:
        """
        Score based on contextual relevance.
        """
        score = 0.0

        # Context factors
        if query.context_factors:
            matches = 0
            for k, v in query.context_factors.items():
                if k in episode.context and episode.context[k] == v:
                    matches += 1

            if matches > 0:
                score += 0.5 * (matches / len(query.context_factors))

        # Location match
        if "location" in query.context_factors and episode.location:
            if query.context_factors["location"] == episode.location:
                score += 0.3

        return score

    def _get_associated_episodes(self, episodes: List[Episode]) -> List[Tuple[float, Episode]]:
        """
        Find episodes associated with the given list of episodes.

        Args:
            episodes: List of episodes to find associations for

        Returns:
            List of (score, episode) tuples
        """
        associated = {}

        # Collect all associations
        for episode in episodes:
            for assoc_id, strength in episode.associations.items():
                if assoc_id in self.episodes and assoc_id != episode.episode_id:
                    if assoc_id in associated:
                        associated[assoc_id] = max(associated[assoc_id], strength)
                    else:
                        associated[assoc_id] = strength

        # Convert to list of (score, episode) tuples
        result = []
        for assoc_id, strength in associated.items():
            result.append((strength, self.episodes[assoc_id]))

        # Sort by strength
        result.sort(reverse=True, key=lambda x: x[0])

        return result

    def _sort_timeline(self) -> None:
        """Sort the episode timeline by timestamp."""
        self.episode_timeline.sort(key=lambda x: x[0])

    def _generate_sparse_representation(self, content: Any) -> Optional[np.ndarray]:
        """
        Generate a sparse distributed representation (SDR) of content.
        Inspired by HTM's sparse encodings.

        This is a simplified implementation that:
        1. Creates a sparse binary vector
        2. Sets a small percentage of bits to 1
        3. Positions of 1s represent semantic features

        Args:
            content: Content to generate SDR for

        Returns:
            Sparse binary vector or None if cannot be generated
        """
        if not self.config.enable_predictive_coding:
            return None

        # SDR parameters
        n = 1024  # Total bits
        w = 40    # Active bits (sparsity ~4%)

        # Initialize empty SDR
        sdr = np.zeros(n, dtype=np.int8)

        try:
            # Simple hash-based approach for deterministic SDRs
            if isinstance(content, str):
                # For strings, hash character sequences
                chunks = [content[i:i+3] for i in range(0, len(content), 3)]
                for chunk in chunks:
                    # Get a hash value for this chunk
                    hash_val = hash(chunk) % n
                    # Set that bit to 1
                    sdr[hash_val] = 1
            elif isinstance(content, (int, float)):
                # For numbers, use value-based encoding
                val = float(content)
                # Scale to 0-1 range (assuming reasonable bounds)
                normalized = (val + 1000) / 2000
                # Set w bits based on this value
                center = int(normalized * n)
                half_width = w // 2
                start = max(0, center - half_width)
                end = min(n, center + half_width)
                sdr[start:end] = 1
            else:
                # For other types, use hash of string representation
                s = str(content)
                hash_val = hash(s)
                for i in range(w):
                    bit_pos = (hash_val + i * 16777619) % n  # FNV-like hash spreading
                    sdr[bit_pos] = 1
        except Exception as e:
            self.logger.error(f"Error generating SDR: {e}")
            return None

        # Ensure we have exactly w bits set to 1
        active_bits = np.sum(sdr)
        if active_bits < w:
            # Need to add more active bits
            zero_positions = np.where(sdr == 0)[0]
            to_activate = np.random.choice(zero_positions, size=w-active_bits, replace=False)
            sdr[to_activate] = 1
        elif active_bits > w:
            # Need to remove some active bits
            one_positions = np.where(sdr == 1)[0]
            to_deactivate = np.random.choice(one_positions, size=active_bits-w, replace=False)
            sdr[to_deactivate] = 0

        return sdr

    def _vector_similarity(self, vec1: np.ndarray, vec2: np.ndarray) -> float:
        """
        Calculate similarity between two vectors.

        Args:
            vec1: First vector
            vec2: Second vector

        Returns:
            Similarity score between 0 and 1
        """
        # Check if SDRs
        if vec1.dtype == np.int8 and vec2.dtype == np.int8 and np.max(vec1) <= 1 and np.max(vec2) <= 1:
            # For SDRs, use overlap measure
            intersection = np.sum(np.logical_and(vec1, vec2))
            union = np.sum(np.logical_or(vec1, vec2))
            if union == 0:
                return 0.0
            return intersection / union

        # Otherwise use cosine similarity
        norm1 = np.linalg.norm(vec1)
        norm2 = np.linalg.norm(vec2)

        if norm1 == 0 or norm2 == 0:
            return 0.0

        dot_product = np.dot(vec1, vec2)
        similarity = dot_product / (norm1 * norm2)

        # Ensure result is in [0, 1] range
        return max(0.0, min(1.0, similarity))

    def _auto_tag_episode(self, episode: Episode) -> None:
        """
        Automatically generate tags for an episode based on its content.
        """
        if not self.config.auto_tagging:
            return

        # This is a simplified implementation
        # A real implementation might use more sophisticated NLP

        # Collect words from content and context
        words = set()

        # Add words from summary if available
        if episode.summary:
            words.update(episode.summary.lower().split())

        # Add words from context
        for k, v in episode.context.items():
            if isinstance(v, str):
                words.update(v.lower().split())

        # Add words from event content
        for event in episode.events:
            if isinstance(event.content, str):
                words.update(event.content.lower().split())

        # Filter out common stop words
        stop_words = {"the", "a", "an", "and", "or", "but", "in", "on", "at", "to", "for", "with", "by", "of", "is", "are", "was", "were"}
        words = words - stop_words

        # Keep only words of reasonable length
        words = {w for w in words if len(w) >= 4 and len(w) <= 20}

        # Add as tags (limit to top 5)
        for word in list(words)[:5]:
            episode.tags.add(word)
            # Update tag index
            self.episodes_by_tag[word].append(episode.episode_id)