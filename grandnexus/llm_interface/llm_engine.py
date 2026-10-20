"""
llm_engine.py - Multi-provider LLM Interface for GrandNexus

This module provides a unified interface for interacting with various
Large Language Models (LLMs) and embedding models. It supports multiple
providers and offers caching, fallback mechanisms, and integration with
GrandNexus's memory and cognitive systems.

Key features:
- Support for multiple LLM providers (local models, OpenAI, Anthropic, etc.)
- Unified embedding generation interface
- Caching of results to reduce redundant API calls
- Integration with memory systems for context enhancement
- Fallback mechanisms for resilience
- Structured response parsing
"""

import logging
import threading
import time
import uuid
import json
import os
import re
import hashlib
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Callable, Union, Tuple, Set
from collections import defaultdict, deque
import numpy as np



# Assuming these exist or will exist
from grandnexus.core.nexus_core import NexusCore
from grandnexus.memory.working import WorkingMemory
from grandnexus.memory.semantic import SemanticMemory
from grandnexus.memory.episodic import EpisodicMemory
from grandnexus.graph.graph_engine import GraphEngine
from grandnexus.graph.graph_engine import GraphQuery  # réparation 6.5

class LLMProvider(str, Enum):
    """Types of supported LLM providers"""
    LOCAL = "local"          # Local models (via transformers, etc.)
    OPENAI = "openai"        # OpenAI API
    ANTHROPIC = "anthropic"  # Anthropic/Claude API
    DEEPSEEK = "deepseek"    # DeepSeek API
    CUSTOM = "custom"        # Custom integration


@dataclass
class LLMConfig:
    """Configuration for the LLMEngine"""
    provider: LLMProvider = LLMProvider.LOCAL
    model_name: str = "text-embedding-3-small"  # Default model name
    api_key: Optional[str] = None
    temperature: float = 0.7
    max_tokens: int = 1024
    embedding_dimension: int = 1536
    context_window: int = 8192
    cache_enabled: bool = True
    cache_size: int = 1000
    request_timeout: float = 60.0
    fallback_providers: List[LLMProvider] = field(default_factory=list)
    synergy_with_memory: bool = True
    synergy_with_graph: bool = True


@dataclass
class LLMRequest:
    """Represents a request to an LLM"""
    prompt: str
    temperature: Optional[float] = None
    max_tokens: Optional[int] = None
    request_type: str = "completion"  # "completion", "embedding", "chat"
    context: Dict[str, Any] = field(default_factory=dict)
    streaming: bool = False
    stream_callback: Optional[Callable] = None
    request_id: str = field(default_factory=lambda: str(uuid.uuid4()))


@dataclass
class LLMResponse:
    """Result of an LLM request"""
    text: str
    embedding: Optional[np.ndarray] = None
    tokens_used: int = 0
    model_used: str = ""
    request_time: float = 0.0
    request_id: str = ""
    parsed_data: Optional[Dict[str, Any]] = None
    confidence: float = 1.0


class LLMEngine:
    """
    Core LLM Engine for GrandNexus, integrating multiple providers and models.

    This engine manages access to language models and embeddings, providing
    a unified interface regardless of the underlying provider. It also integrates
    with GrandNexus's memory systems for context enhancement and result storage.
    """

    def __init__(self, config: LLMConfig,
                 working_memory: Optional[WorkingMemory] = None,
                 semantic_memory: Optional[SemanticMemory] = None,
                 episodic_memory: Optional[EpisodicMemory] = None,
                 graph_engine: Optional[GraphEngine] = None,
                 nexus_core: Optional[NexusCore] = None):
        """
        Initialize the LLM engine.

        Args:
            config: Configuration for the LLM engine
            working_memory: Reference to working memory for context
            semantic_memory: Reference to semantic memory for knowledge retrieval
            episodic_memory: Reference to episodic memory for experience access
            graph_engine: Reference to graph engine for knowledge graph integration
            nexus_core: Reference to the NexusCore for system-wide coordination
        """
        self.logger = logging.getLogger("GrandNexus.LLMInterface.Engine")
        self.config = config
        self.nexus_core = nexus_core
        self.working_memory = working_memory
        self.semantic_memory = semantic_memory
        self.episodic_memory = episodic_memory
        self.graph_engine = graph_engine

        # Internal state
        self.instance_id = str(uuid.uuid4())[:8]
        self.running = False
        self.initialized = False
        self.lock = threading.RLock()

        # Provider clients
        self.clients = {}

        # Caching
        self.cache = {}
        self.cache_order = deque(maxlen=self.config.cache_size)

        # Request history and metrics
        self.request_history = deque(maxlen=100)
        self.metrics = defaultdict(lambda: deque(maxlen=100))

        # Parser registry
        self.parsers = {}

        self.logger.info(f"LLMEngine initialized with ID={self.instance_id}, provider={config.provider}")

    def initialize(self) -> bool:
        """Initialize the LLM engine and connect to providers"""
        if self.initialized:
            return True

        with self.lock:
            try:
                # Initialize provider clients
                self._init_provider_clients()

                # Initialize parsers
                self._init_response_parsers()

                # Connect to memory systems
                if self.config.synergy_with_memory:
                    self._init_memory_integration()

                # Connect to graph system
                if self.config.synergy_with_graph:
                    self._init_graph_integration()

                # Register with NexusCore
                if self.nexus_core:
                    self._register_with_nexus_core()

                self.initialized = True
                self.logger.info("LLMEngine initialization complete")
                return True
            except Exception as e:
                self.logger.error(f"LLMEngine initialization failed: {str(e)}")
                return False

    def start(self) -> bool:
        """Start the LLM engine"""
        if not self.initialized:
            if not self.initialize():
                return False

        with self.lock:
            if self.running:
                return True

            try:
                self.running = True
                self.logger.info("LLMEngine started successfully")
                return True
            except Exception as e:
                self.logger.error(f"Failed to start LLMEngine: {str(e)}")
                return False

    def stop(self) -> bool:
        """Stop the LLM engine"""
        with self.lock:
            if not self.running:
                return True

            try:
                self.running = False
                self.logger.info("LLMEngine stopped successfully")
                return True
            except Exception as e:
                self.logger.error(f"Error stopping LLMEngine: {str(e)}")
                return False

    def complete(self, request: LLMRequest) -> LLMResponse:
        """
        Generate a completion from a prompt.

        Args:
            request: The LLMRequest with prompt and parameters

        Returns:
            LLMResponse containing the generated text
        """
        if not self.initialized:
            self.initialize()

        # Check cache if enabled
        if self.config.cache_enabled:
            cache_key = self._create_cache_key(request)
            if cache_key in self.cache:
                self.logger.debug(f"Cache hit for request {request.request_id}")
                self.metrics["cache_hit_rate"].append(1.0)
                return self.cache[cache_key]
            else:
                self.metrics["cache_hit_rate"].append(0.0)

        # Prepare context
        context = request.context.copy()

        # Enrich prompt with memory if enabled
        enriched_prompt = request.prompt
        if self.config.synergy_with_memory:
            enriched_prompt = self._enrich_prompt_with_memory(request.prompt, context)

        # Set request parameters
        temperature = request.temperature if request.temperature is not None else self.config.temperature
        max_tokens = request.max_tokens if request.max_tokens is not None else self.config.max_tokens

        # Record start time
        start_time = time.time()

        try:
            # Call the provider
            response = self._call_provider({
                "prompt": enriched_prompt,
                "temperature": temperature,
                "max_tokens": max_tokens,
                "request_type": request.request_type,
                "streaming": request.streaming,
                "stream_callback": request.stream_callback,
                "request_id": request.request_id,
                "model": context.get("model") or self.config.model_name,  # réparation 6.5
                "system": context.get("system"),                          # réparation 6.5
            })

            # Calculate request time
            request_time = time.time() - start_time
            response.request_time = request_time

            # Update metrics
            self.metrics["request_time"].append(request_time)
            self.metrics["tokens_used"].append(response.tokens_used)
            self.metrics["completion_success_rate"].append(1.0)

            # Store in cache if enabled
            if self.config.cache_enabled:
                self._add_to_cache(cache_key, response)

            # Record in history
            self.request_history.append({
                "request_id": request.request_id,
                "request_type": request.request_type,
                "timestamp": start_time,
                "success": True
            })

            return response

        except Exception as e:
            self.logger.error(f"Error in completion request: {str(e)}")

            # Update metrics
            self.metrics["completion_success_rate"].append(0.0)

            # Record in history
            self.request_history.append({
                "request_id": request.request_id,
                "request_type": request.request_type,
                "timestamp": start_time,
                "success": False,
                "error": str(e)
            })

            # Try fallback providers if available
            if self.config.fallback_providers:
                return self._call_fallback_provider(
                    request, e, self.config.fallback_providers[0]
                )

            # Return error response
            return LLMResponse(
                text=f"Error: {str(e)}",
                request_id=request.request_id,
                confidence=0.0
            )

    def create_embedding(self, text: str, model: Optional[str] = None) -> np.ndarray:
        """
        Create an embedding vector for a text.

        Args:
            text: The text to embed
            model: Optional specific embedding model to use

        Returns:
            Numpy array with the embedding vector
        """
        if not self.initialized:
            self.initialize()

        # Check cache if enabled
        if self.config.cache_enabled:
            cache_key = f"embed:{model or self.config.model_name}:{hashlib.md5(text.encode()).hexdigest()}"
            if cache_key in self.cache:
                self.logger.debug(f"Cache hit for embedding")
                self.metrics["cache_hit_rate"].append(1.0)
                return self.cache[cache_key].embedding
            else:
                self.metrics["cache_hit_rate"].append(0.0)

        # Record start time
        start_time = time.time()

        try:
            # Create embedding request
            request = LLMRequest(
                prompt=text,
                request_type="embedding",
                context={"model": model} if model else {}
            )

            # Call the provider
            response = self._call_provider({
                "prompt": text,
                "request_type": "embedding",
                "request_id": request.request_id,
                "model": model
            })

            # Calculate request time
            request_time = time.time() - start_time
            response.request_time = request_time

            # Update metrics
            self.metrics["request_time"].append(request_time)
            self.metrics["embedding_success_rate"].append(1.0)

            # Store in cache if enabled
            if self.config.cache_enabled:
                self._add_to_cache(cache_key, response)

            # Record in history
            self.request_history.append({
                "request_id": request.request_id,
                "request_type": "embedding",
                "timestamp": start_time,
                "success": True
            })

            return response.embedding

        except Exception as e:
            self.logger.error(f"Error in embedding request: {str(e)}")

            # Update metrics
            self.metrics["embedding_success_rate"].append(0.0)

            # Record in history
            self.request_history.append({
                "request_id": str(uuid.uuid4()),
                "request_type": "embedding",
                "timestamp": start_time,
                "success": False,
                "error": str(e)
            })

            # Try fallback providers if available
            if self.config.fallback_providers:
                fallback_response = self._call_fallback_provider(
                    LLMRequest(prompt=text, request_type="embedding"),
                    e, self.config.fallback_providers[0]
                )
                if fallback_response.embedding is not None:
                    return fallback_response.embedding

            # Return zero embedding as fallback
            return np.zeros(self.config.embedding_dimension)

    def parse_response(self, response: LLMResponse, parser_type: str) -> Dict[str, Any]:
        """
        Parse an LLM response using a specific parser.

        Args:
            response: The LLMResponse to parse
            parser_type: The type of parser to use

        Returns:
            Dictionary with parsed data
        """
        if not parser_type in self.parsers:
            self.logger.warning(f"Unknown parser type: {parser_type}")
            return {"error": f"Unknown parser type: {parser_type}"}

        try:
            parser_func = self.parsers[parser_type]
            parsed_data = parser_func(response.text)

            # Update the response with parsed data
            response.parsed_data = parsed_data

            return parsed_data
        except Exception as e:
            self.logger.error(f"Error parsing response with {parser_type}: {str(e)}")
            return {"error": f"Parsing error: {str(e)}"}

    def register_parser(self, parser_type: str, parser_func: Callable[[str], Dict[str, Any]]) -> None:
        """
        Register a custom response parser.

        Args:
            parser_type: Unique identifier for the parser
            parser_func: Function that takes response text and returns parsed data
        """
        with self.lock:
            self.parsers[parser_type] = parser_func
            self.logger.info(f"Registered parser: {parser_type}")

    def get_metrics(self) -> Dict[str, Any]:
        """Get current performance metrics"""
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

    def clear_cache(self) -> None:
        """Clear the response cache"""
        with self.lock:
            self.cache.clear()
            self.cache_order.clear()
            self.logger.info("Cache cleared")

    # -----------------------------------------------------------------------
    # Internal methods
    # -----------------------------------------------------------------------

    def _init_provider_clients(self) -> None:
        """Initialize clients for the configured provider and fallbacks"""
        # Initialize main provider
        self._init_provider(self.config.provider)

        # Initialize fallback providers
        for provider in self.config.fallback_providers:
            self._init_provider(provider)

    def _init_provider(self, provider: LLMProvider) -> None:
        """
        Initialize a specific provider client.

        Args:
            provider: The provider to initialize
        """
        if provider == LLMProvider.LOCAL:
            self._init_local_model()
        elif provider == LLMProvider.OPENAI:
            self._init_openai_client()
        elif provider == LLMProvider.ANTHROPIC:
            self._init_anthropic_client()
        elif provider == LLMProvider.DEEPSEEK:
            self._init_deepseek_client()
        elif provider == LLMProvider.CUSTOM:
            self._init_custom_client()
        else:
            self.logger.warning(f"Unknown provider: {provider}")

    def _init_local_model(self) -> None:
        """Initialize a local model using Hugging Face Transformers"""
        try:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

            # Check if we should use GPU
            device = "cuda" if torch.cuda.is_available() else "cpu"

            # Create a basic transformers pipeline
            model_name = self.config.model_name
            tokenizer = AutoTokenizer.from_pretrained(model_name)
            model = AutoModelForCausalLM.from_pretrained(model_name)

            # Move model to appropriate device
            model = model.to(device)

            # Create text generation pipeline
            text_generator = pipeline(
                "text-generation",
                model=model,
                tokenizer=tokenizer,
                device=0 if device == "cuda" else -1
            )

            # Create embedding pipeline if needed
            try:
                from sentence_transformers import SentenceTransformer
                embedding_model = SentenceTransformer(model_name)
                embedding_model = embedding_model.to(device)
            except (ImportError, ValueError):
                self.logger.warning(f"No suitable embedding model found for {model_name}")
                embedding_model = None

            self.clients[LLMProvider.LOCAL] = {
                "text_generator": text_generator,
                "embedding_model": embedding_model,
                "tokenizer": tokenizer,
                "device": device
            }

            self.logger.info(f"Initialized local model: {model_name} on {device}")
        except ImportError:
            self.logger.error("Failed to import transformers. Please install with: pip install transformers torch sentence-transformers")
            raise
        except Exception as e:
            self.logger.error(f"Error initializing local model: {str(e)}")
            raise

    def _init_openai_client(self) -> None:
        """Initialize the OpenAI client"""
        try:
            import openai

            # Set up API key
            api_key = self.config.api_key
            if api_key is None:
                api_key = os.environ.get("OPENAI_API_KEY")

            if not api_key:
                raise ValueError("No OpenAI API key provided in config or environment")

            # Create client
            client = openai.OpenAI(api_key=api_key)

            self.clients[LLMProvider.OPENAI] = {
                "client": client
            }

            self.logger.info("Initialized OpenAI client")
        except ImportError:
            self.logger.error("Failed to import openai. Please install with: pip install openai")
            raise
        except Exception as e:
            self.logger.error(f"Error initializing OpenAI client: {str(e)}")
            raise

    def _init_anthropic_client(self) -> None:
        """Initialize the Anthropic/Claude client"""
        try:
            import anthropic

            # Set up API key
            api_key = self.config.api_key
            if api_key is None:
                api_key = os.environ.get("ANTHROPIC_API_KEY")

            if not api_key:
                raise ValueError("No Anthropic API key provided in config or environment")

            # Create client
            client = anthropic.Anthropic(api_key=api_key)

            self.clients[LLMProvider.ANTHROPIC] = {
                "client": client
            }

            self.logger.info("Initialized Anthropic client")
        except ImportError:
            self.logger.error("Failed to import anthropic. Please install with: pip install anthropic")
            raise
        except Exception as e:
            self.logger.error(f"Error initializing Anthropic client: {str(e)}")
            raise

    def _init_deepseek_client(self) -> None:
        """Initialize the DeepSeek client"""
        # This is a placeholder - actual implementation would depend on DeepSeek's API
        self.logger.warning("DeepSeek client initialization not implemented yet")

    def _init_custom_client(self) -> None:
        """Initialize a custom client"""
        # This is a placeholder for custom LLM integrations
        self.logger.warning("Custom client initialization not implemented yet")

    def _init_response_parsers(self) -> None:
        """Initialize standard response parsers"""
        # Register some basic parsers
        self.register_parser("json", self._parse_json_response)
        self.register_parser("list", self._parse_list_response)
        self.register_parser("key_value", self._parse_key_value_response)
        self.register_parser("structured_text", self._parse_structured_text)

    def _init_memory_integration(self) -> None:
        """Initialize integration with memory systems"""
        if not (self.working_memory or self.semantic_memory or self.episodic_memory):
            self.logger.warning("No memory modules available for integration")
            return

        self.logger.info("Initialized memory integration")

    def _init_graph_integration(self) -> None:
        """Initialize integration with graph engine"""
        if not self.graph_engine:
            self.logger.warning("No graph engine available for integration")
            return

        self.logger.info("Initialized graph integration")

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
            if self.graph_engine:
                dependencies.append("graph_engine")

            self.nexus_core.register_module(
                name="llm_engine",
                module=self,
                dependencies=dependencies
            )

            self.logger.info("Successfully registered with NexusCore")
        except Exception as e:
            self.logger.error(f"Failed to register with NexusCore: {str(e)}")

    def _enrich_prompt_with_memory(self, prompt: str, context: Dict[str, Any]) -> str:
        """
        Enhance a prompt with relevant information from memory.

        Args:
            prompt: The original prompt
            context: Context information

        Returns:
            Enriched prompt with memory information
        """
        enriched_prompt = prompt

        try:
            # Extract potential memory cues from the prompt
            # This is a simplified approach - a real implementation would be more sophisticated
            memory_cues = self._extract_memory_cues(prompt)

            # Check working memory for relevant information
            if self.working_memory:
                working_mem_items = self._retrieve_from_working_memory(memory_cues, context)
                if working_mem_items:
                    working_context = "\nContext from working memory:\n"
                    for item in working_mem_items:
                        working_context += f"- {item}\n"
                    enriched_prompt = working_context + "\n" + enriched_prompt

            # Check semantic memory for relevant concepts
            if self.semantic_memory:
                semantic_concepts = self._retrieve_from_semantic_memory(memory_cues, context)
                if semantic_concepts:
                    semantic_context = "\nRelevant concepts:\n"
                    for concept, description in semantic_concepts:
                        semantic_context += f"- {concept}: {description}\n"
                    enriched_prompt = semantic_context + "\n" + enriched_prompt

            # Check episodic memory for relevant experiences
            if self.episodic_memory:
                episodic_items = self._retrieve_from_episodic_memory(memory_cues, context)
                if episodic_items:
                    episodic_context = "\nRelevant past interactions:\n"
                    for item in episodic_items:
                        episodic_context += f"- {item}\n"
                    enriched_prompt = episodic_context + "\n" + enriched_prompt

            return enriched_prompt
        except Exception as e:
            self.logger.error(f"Error enriching prompt with memory: {str(e)}")
            return prompt

    def _extract_memory_cues(self, prompt: str) -> List[str]:
        """
        Extract potential memory cues from a prompt.

        Args:
            prompt: The prompt to analyze

        Returns:
            List of extracted cues
        """
        # This is a simplified implementation
        # In a real system, this would use NLP techniques to extract key concepts

        words = prompt.split()
        # Filter out common stop words and punctuation
        stop_words = {"the", "a", "an", "and", "or", "but", "in", "on", "at", "to", "for", "with", "by"}
        cues = [word.strip(".,!?;:()[]{}\"'").lower() for word in words
                if len(word) > 3 and word.lower() not in stop_words]

        # Remove duplicates while preserving order
        unique_cues = []
        for cue in cues:
            if cue not in unique_cues:
                unique_cues.append(cue)

        return unique_cues[:10]  # Limit to top 10 cues

    def _retrieve_from_working_memory(self, cues: List[str],
                                     context: Dict[str, Any]) -> List[str]:
        """
        Retrieve relevant items from working memory.

        Args:
            cues: Memory cues to search for
            context: Additional context

        Returns:
            List of relevant items as strings
        """
        if not self.working_memory:
            return []

        # This is a simplified implementation
        # In a real system, this would use more sophisticated retrieval

        try:
            # Assume working_memory has a method to query items
            items = self.working_memory.get_all_items()

            relevant_items = []
            for item in items:
                # Check if any cue is in the item's content
                if hasattr(item, 'content'):
                    content = str(item.content)
                    if any(cue in content.lower() for cue in cues):
                        relevant_items.append(content)

            return relevant_items[:3]  # Limit to 3 items
        except Exception as e:
            self.logger.error(f"Error retrieving from working memory: {str(e)}")
            return []

    def _retrieve_from_semantic_memory(self, cues: List[str],
                                     context: Dict[str, Any]) -> List[Tuple[str, str]]:
        """
        Retrieve relevant concepts from semantic memory.

        Args:
            cues: Memory cues to search for
            context: Additional context

        Returns:
            List of (concept, description) tuples
        """
        if not self.semantic_memory:
            return []

        # This is a simplified implementation
        # In a real system, this would use more sophisticated retrieval

        try:
            relevant_concepts = []

            for cue in cues:
                # Assume semantic_memory has a method to query concepts
                query = {"text": cue}
                concepts = self.semantic_memory.query_concepts(query)

                for concept in concepts:
                    if hasattr(concept, 'name') and hasattr(concept, 'description'):
                        relevant_concepts.append((concept.name, concept.description))

            return relevant_concepts[:3]  # Limit to 3 concepts
        except Exception as e:
            self.logger.error(f"Error retrieving from semantic memory: {str(e)}")
            return []

    def _retrieve_from_episodic_memory(self, cues: List[str],
                                     context: Dict[str, Any]) -> List[str]:
        """
        Retrieve relevant episodes from episodic memory.

        Args:
            cues: Memory cues to search for
            context: Additional context

        Returns:
            List of relevant episodes as strings
        """
        if not self.episodic_memory:
            return []

        # This is a simplified implementation
        # In a real system, this would use more sophisticated retrieval

        try:
            relevant_episodes = []

            # Assume episodic_memory has a method to query episodes
            # This would depend on the actual implementation of EpisodicMemory

            return relevant_episodes[:2]  # Limit to 2 episodes
        except Exception as e:
            self.logger.error(f"Error retrieving from episodic memory: {str(e)}")
            return []

    def _enrich_response_with_graph(self, response: LLMResponse) -> LLMResponse:
        """
        Enhance an LLM response with information from the knowledge graph.

        Args:
            response: The LLM response to enhance

        Returns:
            Enhanced response
        """
        if not self.graph_engine or not self.config.synergy_with_graph:
            return response

        try:
            # Extract entities from the response
            entities = self._extract_entities(response.text)

            if not entities:
                return response

            # Query the graph for information about these entities
            graph_info = self._query_graph_for_entities(entities)

            if not graph_info:
                return response

            # Create enhanced response
            enhanced_response = LLMResponse(
                text=response.text,
                embedding=response.embedding,
                tokens_used=response.tokens_used,
                model_used=response.model_used,
                request_time=response.request_time,
                request_id=response.request_id,
                confidence=response.confidence,
                parsed_data=response.parsed_data
            )

            # Add graph information to parsed data
            if enhanced_response.parsed_data is None:
                enhanced_response.parsed_data = {}

            enhanced_response.parsed_data["graph_context"] = graph_info

            return enhanced_response
        except Exception as e:
            self.logger.error(f"Error enriching response with graph: {str(e)}")
            return response

    def _extract_entities(self, text: str) -> List[str]:
        """
        Extract entities from text for graph queries.

        Args:
            text: The text to analyze

        Returns:
            List of extracted entities
        """
        # This is a simplified implementation
        # In a real system, this would use NER or more sophisticated techniques

        # Split into sentences and words
        sentences = text.split(". ")
        entities = []

        for sentence in sentences:
            words = sentence.split()

            # Look for capitalized words that might be entities
            for i, word in enumerate(words):
                clean_word = word.strip(".,!?;:()[]{}\"'")

                # Simple heuristic: capitalized words not at the start of a sentence
                if (i > 0 and clean_word and clean_word[0].isupper() and
                    len(clean_word) > 1 and clean_word.lower() not in
                    {"i", "i'm", "i've", "i'll", "i'd"}):
                    entities.append(clean_word)

                # Also look for quoted phrases that might be entities
                if '"' in sentence:
                    quoted = re.findall(r'"([^"]*)"', sentence)
                    entities.extend(quoted)

        # Remove duplicates
        unique_entities = []
        for entity in entities:
            if entity not in unique_entities:
                unique_entities.append(entity)

        return unique_entities[:5]  # Limit to 5 entities

    def _query_graph_for_entities(self, entities: List[str]) -> Dict[str, Any]:
        """
        Query the knowledge graph for information about entities.

        Args:
            entities: List of entities to query

        Returns:
            Dictionary with graph information
        """
        if not self.graph_engine:
            return {}

        result = {}

        try:
            for entity in entities:
                # Create a query for the graph engine
                query = GraphQuery(
                    node_criteria={"label": entity},
                    max_depth=1,
                    max_results=5
                )

                # Execute the query
                query_result = self.graph_engine.execute_query(query)

                if query_result.get("nodes"):
                    result[entity] = query_result

            return result
        except Exception as e:
            self.logger.error(f"Error querying graph for entities: {str(e)}")
            return {}

    def _call_provider(self, request_params: Dict[str, Any]) -> LLMResponse:
        """
        Call the configured LLM provider.

        Args:
            request_params: Parameters for the request

        Returns:
            LLMResponse from the provider
        """
        provider = self.config.provider

        if provider not in self.clients:
            raise ValueError(f"Provider {provider} not initialized")

        if provider == LLMProvider.LOCAL:
            return self._call_local_model(request_params)
        elif provider == LLMProvider.OPENAI:
            return self._call_openai(request_params)
        elif provider == LLMProvider.ANTHROPIC:
            return self._call_anthropic(request_params)
        elif provider == LLMProvider.DEEPSEEK:
            return self._call_deepseek(request_params)
        elif provider == LLMProvider.CUSTOM:
            return self._call_custom_provider(request_params)
        else:
            raise ValueError(f"Unsupported provider: {provider}")

    def _call_local_model(self, request_params: Dict[str, Any]) -> LLMResponse:
        """
        Call a local model using Transformers.

        Args:
            request_params: Parameters for the request

        Returns:
            LLMResponse from the local model
        """
        client = self.clients[LLMProvider.LOCAL]

        if "request_type" not in request_params:
            raise ValueError("Missing request_type parameter")

        request_type = request_params["request_type"]

        if request_type == "completion" or request_type == "chat":
            # Use text generation pipeline
            if "text_generator" not in client:
                raise ValueError("Text generator not initialized")

            text_generator = client["text_generator"]
            prompt = request_params["prompt"]
            max_tokens = request_params.get("max_tokens", 100)
            temperature = request_params.get("temperature", 0.7)

            # Generate text
            result = text_generator(
                prompt,
                max_length=len(prompt.split()) + max_tokens,
                num_return_sequences=1,
                temperature=temperature
            )

            # Extract generated text (removing the prompt)
            generated_text = result[0]["generated_text"]
            if generated_text.startswith(prompt):
                generated_text = generated_text[len(prompt):].strip()

            # Count tokens
            tokenizer = client["tokenizer"]
            tokens = len(tokenizer.encode(generated_text))

            return LLMResponse(
                text=generated_text,
                tokens_used=tokens,
                model_used=self.config.model_name,
                request_id=request_params.get("request_id", "")
            )

        elif request_type == "embedding":
            # Use embedding model
            if "embedding_model" not in client or client["embedding_model"] is None:
                raise ValueError("Embedding model not initialized")

            embedding_model = client["embedding_model"]
            text = request_params["prompt"]

            # Generate embedding
            embedding = embedding_model.encode(text)

            # Ensure it's a numpy array
            if not isinstance(embedding, np.ndarray):
                embedding = np.array(embedding)

            return LLMResponse(
                text="",
                embedding=embedding,
                tokens_used=0,
                model_used=self.config.model_name,
                request_id=request_params.get("request_id", "")
            )

        else:
            raise ValueError(f"Unsupported request type: {request_type}")

    def _call_openai(self, request_params: Dict[str, Any]) -> LLMResponse:
        """
        Call the OpenAI API.

        Args:
            request_params: Parameters for the request

        Returns:
            LLMResponse from OpenAI
        """
        client = self.clients[LLMProvider.OPENAI]["client"]

        if "request_type" not in request_params:
            raise ValueError("Missing request_type parameter")

        request_type = request_params["request_type"]
        model = request_params.get("model", "gpt-3.5-turbo")

        if request_type == "completion" or request_type == "chat":
            # Use chat completion API
            prompt = request_params["prompt"]
            max_tokens = request_params.get("max_tokens", 1000)
            temperature = request_params.get("temperature", 0.7)
            streaming = request_params.get("streaming", False)

            messages = [{"role": "user", "content": prompt}]

            if streaming:
                stream_callback = request_params.get("stream_callback")
                if not stream_callback:
                    raise ValueError("Stream callback required for streaming")

                response_text = ""

                # Stream the response
                stream = client.chat.completions.create(
                    model=model,
                    messages=messages,
                    max_tokens=max_tokens,
                    temperature=temperature,
                    stream=True
                )

                for chunk in stream:
                    if chunk.choices and chunk.choices[0].delta.content:
                        content = chunk.choices[0].delta.content
                        response_text += content
                        stream_callback(content)

                # Return the full response
                return LLMResponse(
                    text=response_text,
                    tokens_used=len(response_text.split()),
                    model_used=model,
                    request_id=request_params.get("request_id", "")
                )
            else:
                # Non-streaming request
                response = client.chat.completions.create(
                    model=model,
                    messages=messages,
                    max_tokens=max_tokens,
                    temperature=temperature
                )

                response_text = response.choices[0].message.content

                return LLMResponse(
                    text=response_text,
                    tokens_used=response.usage.total_tokens,
                    model_used=model,
                    request_id=request_params.get("request_id", "")
                )

        elif request_type == "embedding":
            # Use embedding API
            text = request_params["prompt"]
            embedding_model = request_params.get("model", "text-embedding-3-small")

            response = client.embeddings.create(
                model=embedding_model,
                input=text
            )

            embedding = np.array(response.data[0].embedding)

            return LLMResponse(
                text="",
                embedding=embedding,
                tokens_used=response.usage.total_tokens,
                model_used=embedding_model,
                request_id=request_params.get("request_id", "")
            )

        else:
            raise ValueError(f"Unsupported request type: {request_type}")

    def _call_anthropic(self, request_params: Dict[str, Any]) -> LLMResponse:
        """
        Call the Anthropic/Claude API.

        Args:
            request_params: Parameters for the request

        Returns:
            LLMResponse from Anthropic
        """
        client = self.clients[LLMProvider.ANTHROPIC]["client"]

        if "request_type" not in request_params:
            raise ValueError("Missing request_type parameter")

        request_type = request_params["request_type"]
        model = request_params.get("model") or "claude-sonnet-5-5"  # réparation 6.5 : ancien défaut retiré
        if model.startswith(("text-embedding", "gpt-")):
            model = "claude-sonnet-5-5"
        _system = {"system": request_params["system"]} if request_params.get("system") else {}

        if request_type == "completion" or request_type == "chat":
            # Use completion API
            prompt = request_params["prompt"]
            max_tokens = request_params.get("max_tokens", 1000)
            temperature = request_params.get("temperature", 0.7)
            streaming = request_params.get("streaming", False)

            if streaming:
                stream_callback = request_params.get("stream_callback")
                if not stream_callback:
                    raise ValueError("Stream callback required for streaming")

                response_text = ""

                # Stream the response
                with client.messages.stream(
                    model=model,
                    max_tokens=max_tokens,
                    temperature=temperature,
                    messages=[
                        {"role": "user", "content": prompt}
                    ]
                ) as stream:
                    for text in stream.text_stream:
                        response_text += text
                        stream_callback(text)

                # Return the full response
                return LLMResponse(
                    text=response_text,
                    tokens_used=len(response_text.split()),
                    model_used=model,
                    request_id=request_params.get("request_id", "")
                )
            else:
                # Non-streaming request
                response = client.messages.create(
                    model=model,
                    max_tokens=max_tokens,
                    temperature=temperature,
                    **_system,
                    messages=[
                        {"role": "user", "content": prompt}
                    ]
                )

                response_text = response.content[0].text

                return LLMResponse(
                    text=response_text,
                    tokens_used=response.usage.output_tokens,
                    model_used=model,
                    request_id=request_params.get("request_id", "")
                )

        elif request_type == "embedding":
            # Anthropic doesn't provide embeddings, fallback to OpenAI if available
            if LLMProvider.OPENAI in self.clients:
                self.logger.warning("Anthropic doesn't provide embeddings, falling back to OpenAI")
                return self._call_openai(request_params)
            else:
                raise ValueError("Anthropic doesn't provide embeddings and no fallback available")

        else:
            raise ValueError(f"Unsupported request type: {request_type}")

    def _call_deepseek(self, request_params: Dict[str, Any]) -> LLMResponse:
        """
        Call the DeepSeek API.

        Args:
            request_params: Parameters for the request

        Returns:
            LLMResponse from DeepSeek
        """
        # This is a placeholder - implementation would depend on DeepSeek's API
        raise NotImplementedError("DeepSeek API not implemented yet")

    def _call_custom_provider(self, request_params: Dict[str, Any]) -> LLMResponse:
        """
        Call a custom provider.

        Args:
            request_params: Parameters for the request

        Returns:
            LLMResponse from the custom provider
        """
        # This is a placeholder for custom LLM integrations
        raise NotImplementedError("Custom provider not implemented yet")

    def _call_fallback_provider(self, request: LLMRequest, original_error: Exception,
                              fallback_provider: LLMProvider) -> LLMResponse:
        """
        Call a fallback provider after a failure.

        Args:
            request: The original request
            original_error: The error from the primary provider
            fallback_provider: The fallback provider to use

        Returns:
            LLMResponse from the fallback provider
        """
        self.logger.info(f"Falling back to {fallback_provider} after error: {str(original_error)}")

        # Create a backup of the current provider
        original_provider = self.config.provider

        try:
            # Temporarily switch to fallback provider
            self.config.provider = fallback_provider

            # Retry the request
            return self.complete(request)
        except Exception as e:
            self.logger.error(f"Fallback to {fallback_provider} also failed: {str(e)}")

            # Try next fallback if available
            fallback_index = self.config.fallback_providers.index(fallback_provider)
            if fallback_index + 1 < len(self.config.fallback_providers):
                next_fallback = self.config.fallback_providers[fallback_index + 1]
                return self._call_fallback_provider(request, e, next_fallback)

            # All fallbacks failed, return error response
            return LLMResponse(
                text=f"Error: {str(original_error)}. Fallbacks also failed.",
                request_id=request.request_id,
                confidence=0.0
            )
        finally:
            # Restore original provider
            self.config.provider = original_provider

    def _create_cache_key(self, request: LLMRequest) -> str:
        """
        Create a cache key for a request.

        Args:
            request: The request to create a key for

        Returns:
            Cache key string
        """
        # Create a deterministic key based on request parameters
        components = [
            request.request_type,
            request.prompt,
            str(request.temperature),
            str(request.max_tokens)
        ]

        # Add sorted context items if present
        if request.context:
            context_str = json.dumps(request.context, sort_keys=True)
            components.append(context_str)

        # Join and hash
        key_string = "|".join(components)
        return hashlib.md5(key_string.encode()).hexdigest()

    def _add_to_cache(self, cache_key: str, response: LLMResponse) -> None:
        """
        Add a response to the cache.

        Args:
            cache_key: Cache key for the response
            response: Response to cache
        """
        # Check if key already exists
        if cache_key in self.cache:
            return

        # If cache is full, remove oldest item
        if len(self.cache) >= self.config.cache_size:
            if self.cache_order:
                oldest_key = self.cache_order.popleft()
                if oldest_key in self.cache:
                    del self.cache[oldest_key]

        # Add to cache and order tracker
        self.cache[cache_key] = response
        self.cache_order.append(cache_key)

    def _parse_json_response(self, text: str) -> Dict[str, Any]:
        """
        Parse a JSON response from LLM.

        Args:
            text: Response text to parse

        Returns:
            Parsed JSON data
        """
        # Try to extract JSON from the response
        try:
            # Look for JSON pattern
            json_pattern = r'```(?:json)?\s*([\s\S]*?)```'
            json_match = re.search(json_pattern, text)

            if json_match:
                json_str = json_match.group(1)
                return json.loads(json_str)

            # If no match with code blocks, try to parse the whole text
            return json.loads(text)
        except json.JSONDecodeError:
            # Try to be more lenient by fixing common issues
            fixed_text = text.strip()

            # Remove trailing commas in lists/objects
            fixed_text = re.sub(r',\s*}', '}', fixed_text)
            fixed_text = re.sub(r',\s*]', ']', fixed_text)

            # Add quotes around unquoted keys
            fixed_text = re.sub(r'([{,])\s*(\w+)\s*:', r'\1"\2":', fixed_text)

            try:
                return json.loads(fixed_text)
            except json.JSONDecodeError as e:
                raise ValueError(f"Failed to parse JSON response: {str(e)}")

    def _parse_list_response(self, text: str) -> Dict[str, Any]:
        """
        Parse a list response from LLM.

        Args:
            text: Response text to parse

        Returns:
            Dictionary with parsed list
        """
        items = []

        # Look for numbered or bulleted list items
        lines = text.split('\n')

        for line in lines:
            line = line.strip()

            # Check for numbered items (1. Item)
            numbered_match = re.match(r'^\d+\.\s+(.+)$', line)
            if numbered_match:
                items.append(numbered_match.group(1))
                continue

            # Check for bulleted items (- Item or * Item)
            bulleted_match = re.match(r'^[-*•]\s+(.+)$', line)
            if bulleted_match:
                items.append(bulleted_match.group(1))

        return {"items": items}

    def _parse_key_value_response(self, text: str) -> Dict[str, Any]:
        """
        Parse a key-value response from LLM.

        Args:
            text: Response text to parse

        Returns:
            Dictionary with parsed key-value pairs
        """
        result = {}

        # Look for key-value pairs (Key: Value)
        lines = text.split('\n')

        for line in lines:
            line = line.strip()

            # Skip empty lines
            if not line:
                continue

            # Check for key-value pattern
            kv_match = re.match(r'^([^:]+):\s*(.+)$', line)
            if kv_match:
                key = kv_match.group(1).strip()
                value = kv_match.group(2).strip()
                result[key] = value

        return result

    def _parse_structured_text(self, text: str) -> Dict[str, Any]:
        """
        Parse structured text response from LLM.

        Args:
            text: Response text to parse

        Returns:
            Dictionary with parsed structure
        """
        result = {"sections": []}
        current_section = None

        # Look for headers and content
        lines = text.split('\n')

        for line in lines:
            line = line.strip()

            # Skip empty lines
            if not line:
                continue

            # Check for headers (# Header)
            header_match = re.match(r'^#+\s+(.+)$', line)
            if header_match:
                # Save previous section if exists
                if current_section:
                    result["sections"].append(current_section)

                # Start new section
                current_section = {
                    "title": header_match.group(1),
                    "content": ""
                }
            elif current_section:
                # Add line to current section content
                if current_section["content"]:
                    current_section["content"] += "\n"
                current_section["content"] += line

        # Save last section if exists
        if current_section:
            result["sections"].append(current_section)

        return result