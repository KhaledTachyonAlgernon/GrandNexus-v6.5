# llm_interface/llm_engine.py

import uuid  # réparation 6.5 : fourni avant par l'espace de noms partagé du notebook
import logging
import threading
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Callable, Union, Tuple
import numpy as np
from enum import Enum

class LLMProvider(str, Enum):
    """Types de providers LLM supportés"""
    LOCAL = "local"          # Modèle local (ex: via transformers)
    OPENAI = "openai"        # API OpenAI
    ANTHROPIC = "anthropic"  # API Anthropic/Claude
    DEEPSEEK = "deepseek"    # API DeepSeek
    CUSTOM = "custom"        # Intégration personnalisée

@dataclass
class LLMConfig:
    """Configuration pour le LLMEngine"""
    provider: LLMProvider = LLMProvider.LOCAL
    model_name: str = "text-embedding-3-small"  # Nom du modèle par défaut
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
    """Représente une requête au LLM"""
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
    """Résultat d'une requête LLM"""
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
    Moteur LLM pour GrandNexus intégrant plusieurs providers et modèles.
    Gère l'accès aux LLM, les embeddings, et le caching.
    S'intègre avec les autres modules de GrandNexus.
    """

    def __init__(self,
                 config: LLMConfig,
                 working_memory: Optional[Any] = None,
                 semantic_memory: Optional[Any] = None,
                 episodic_memory: Optional[Any] = None,
                 graph_engine: Optional[Any] = None):
        """
        Initialise le moteur LLM avec la configuration et les références aux autres modules.
        """
        self.logger = logging.getLogger("LLMEngine")
        self.config = config
        self.working_memory = working_memory
        self.semantic_memory = semantic_memory
        self.episodic_memory = episodic_memory
        self.graph_engine = graph_engine

        self.lock = threading.RLock()
        self.response_cache = {}
        self.embedding_cache = {}
        self.active_clients = {}

        # Initialize provider clients based on config
        self._init_provider_clients()

        self.logger.info(f"LLMEngine initialized with provider: {config.provider}, model: {config.model_name}")

    def _init_provider_clients(self) -> None:
        """Initialise les clients d'API selon le provider configuré"""
        try:
            if self.config.provider == LLMProvider.LOCAL:
                self._init_local_model()
            elif self.config.provider == LLMProvider.OPENAI:
                self._init_openai_client()
            elif self.config.provider == LLMProvider.ANTHROPIC:
                self._init_anthropic_client()
            elif self.config.provider == LLMProvider.DEEPSEEK:
                self._init_deepseek_client()
            elif self.config.provider == LLMProvider.CUSTOM:
                self._init_custom_client()

            # Initialize fallback providers if specified
            for provider in self.config.fallback_providers:
                self._init_fallback_provider(provider)

        except Exception as e:
            self.logger.error(f"Error initializing LLM provider: {e}", exc_info=True)
            raise

    def _init_local_model(self):
        """Initialise un modèle local via HuggingFace Transformers"""
        try:
            from transformers import AutoModelForCausalLM, AutoTokenizer

            self.logger.info(f"Loading local model: {self.config.model_name}")
            self.tokenizer = AutoTokenizer.from_pretrained(self.config.model_name)
            self.model = AutoModelForCausalLM.from_pretrained(
                self.config.model_name,
                torch_dtype="auto",
                device_map="auto"
            )
            self.active_clients["local"] = True
        except ImportError:
            self.logger.error("Transformers library not installed. Cannot use LOCAL provider.")
            self.active_clients["local"] = False
        except Exception as e:
            self.logger.error(f"Error loading local model: {e}", exc_info=True)
            self.active_clients["local"] = False

    def _init_openai_client(self):
        """Initialise le client OpenAI"""
        try:
            import openai
            openai.api_key = self.config.api_key
            self.active_clients["openai"] = openai
        except ImportError:
            self.logger.error("OpenAI library not installed. Cannot use OPENAI provider.")
            self.active_clients["openai"] = False

    def _init_anthropic_client(self):
        """Initialise le client Anthropic/Claude"""
        try:
            import anthropic
            self.anthropic_client = anthropic.Anthropic(api_key=self.config.api_key)
            self.active_clients["anthropic"] = self.anthropic_client
        except ImportError:
            self.logger.error("Anthropic library not installed. Cannot use ANTHROPIC provider.")
            self.active_clients["anthropic"] = False

    def _init_deepseek_client(self):
        """Initialise le client DeepSeek"""
        # Implémentation spécifique à DeepSeek basée sur leur API/SDK
        try:
            # Exemple fictif - à adapter selon l'API réelle de DeepSeek
            import deepseek
            self.deepseek_client = deepseek.Client(api_key=self.config.api_key)
            self.active_clients["deepseek"] = self.deepseek_client
        except ImportError:
            self.logger.error("DeepSeek library not installed. Cannot use DEEPSEEK provider.")
            self.active_clients["deepseek"] = False

    def _init_custom_client(self):
        """Initialise un client personnalisé"""
        # Implémentation pour intégration personnalisée
        self.active_clients["custom"] = True

    def _init_fallback_provider(self, provider: LLMProvider):
        """Initialise un provider de fallback"""
        if provider == LLMProvider.LOCAL:
            if "local" not in self.active_clients:
                self._init_local_model()
        elif provider == LLMProvider.OPENAI:
            if "openai" not in self.active_clients:
                self._init_openai_client()
        # etc.

    def complete(self, request: LLMRequest) -> LLMResponse:
        """
        Génère une complétion de texte à partir d'un prompt.
        Intègre caching et gestion des fallbacks.
        """
        with self.lock:
            # Check cache first if enabled
            if self.config.cache_enabled:
                cache_key = self._create_cache_key(request)
                cached_response = self.response_cache.get(cache_key)
                if cached_response:
                    self.logger.debug(f"Cache hit for request: {request.request_id}")
                    return cached_response

            start_time = time.time()

            # Enrichir le prompt avec le context si synergy activée
            if self.config.synergy_with_memory:
                enriched_prompt = self._enrich_prompt_with_memory(request.prompt, request.context)
                request.prompt = enriched_prompt

            # Tenter l'appel au provider principal
            try:
                response = self._call_provider(request)

                # Cache result if enabled
                if self.config.cache_enabled:
                    self._add_to_cache(cache_key, response)

                return response

            except Exception as e:
                self.logger.error(f"Error completing request with primary provider: {e}", exc_info=True)

                # Try fallback providers
                for provider in self.config.fallback_providers:
                    try:
                        self.logger.info(f"Trying fallback provider: {provider}")
                        fallback_response = self._call_fallback_provider(provider, request)
                        return fallback_response
                    except Exception as fallback_e:
                        self.logger.error(f"Fallback provider {provider} also failed: {fallback_e}")

                # All providers failed
                raise Exception(f"All LLM providers failed to complete request: {request.request_id}")

    def _call_provider(self, request: LLMRequest) -> LLMResponse:
        """Appelle le provider configuré"""
        if self.config.provider == LLMProvider.LOCAL:
            return self._call_local_model(request)
        elif self.config.provider == LLMProvider.OPENAI:
            return self._call_openai(request)
        elif self.config.provider == LLMProvider.ANTHROPIC:
            return self._call_anthropic(request)
        elif self.config.provider == LLMProvider.DEEPSEEK:
            return self._call_deepseek(request)
        elif self.config.provider == LLMProvider.CUSTOM:
            return self._call_custom_provider(request)
        else:
            raise ValueError(f"Unsupported provider: {self.config.provider}")

    def _call_local_model(self, request: LLMRequest) -> LLMResponse:
        """Appelle un modèle local via Transformers"""
        try:
            import torch

            start_time = time.time()

            inputs = self.tokenizer(request.prompt, return_tensors="pt").to(self.model.device)

            temp = request.temperature if request.temperature is not None else self.config.temperature
            max_new_tokens = request.max_tokens if request.max_tokens is not None else self.config.max_tokens

            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    temperature=temp,
                    do_sample=temp > 0,
                    pad_token_id=self.tokenizer.eos_token_id
                )

            response_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            # Remove prompt from response if needed
            if response_text.startswith(request.prompt):
                response_text = response_text[len(request.prompt):].strip()

            tokens_used = outputs.shape[1]  # Total tokens including prompt

            end_time = time.time()

            return LLMResponse(
                text=response_text,
                tokens_used=tokens_used,
                model_used=self.config.model_name,
                request_time=end_time - start_time,
                request_id=request.request_id
            )
        except Exception as e:
            self.logger.error(f"Error in local model call: {e}", exc_info=True)
            raise

    def _call_openai(self, request: LLMRequest) -> LLMResponse:
        """Appelle l'API OpenAI"""
        client = self.active_clients.get("openai")
        if not client:
            raise Exception("OpenAI client not initialized")

        start_time = time.time()

        temp = request.temperature if request.temperature is not None else self.config.temperature
        max_tokens = request.max_tokens if request.max_tokens is not None else self.config.max_tokens

        response = client.chat.completions.create(
            model=self.config.model_name,
            messages=[{"role": "user", "content": request.prompt}],
            temperature=temp,
            max_tokens=max_tokens,
            timeout=self.config.request_timeout
        )

        end_time = time.time()

        return LLMResponse(
            text=response.choices[0].message.content,
            tokens_used=response.usage.total_tokens,
            model_used=self.config.model_name,
            request_time=end_time - start_time,
            request_id=request.request_id
        )

    def _call_anthropic(self, request: LLMRequest) -> LLMResponse:
        """Appelle l'API Anthropic/Claude"""
        client = self.active_clients.get("anthropic")
        if not client:
            raise Exception("Anthropic client not initialized")

        start_time = time.time()

        temp = request.temperature if request.temperature is not None else self.config.temperature
        max_tokens = request.max_tokens if request.max_tokens is not None else self.config.max_tokens

        response = client.messages.create(
            model=self.config.model_name,
            messages=[{"role": "user", "content": request.prompt}],
            temperature=temp,
            max_tokens=max_tokens
        )

        end_time = time.time()

        return LLMResponse(
            text=response.content[0].text,
            tokens_used=response.usage.input_tokens + response.usage.output_tokens,
            model_used=self.config.model_name,
            request_time=end_time - start_time,
            request_id=request.request_id
        )

    def _call_deepseek(self, request: LLMRequest) -> LLMResponse:
        """Appelle l'API DeepSeek"""
        client = self.active_clients.get("deepseek")
        if not client:
            raise Exception("DeepSeek client not initialized")

        # Implémentation spécifique à l'API DeepSeek
        # À adapter selon la documentation de leur API
        start_time = time.time()

        response = client.generate(
            prompt=request.prompt,
            temperature=request.temperature or self.config.temperature,
            max_tokens=request.max_tokens or self.config.max_tokens
        )

        end_time = time.time()

        return LLMResponse(
            text=response.get("text", ""),
            tokens_used=response.get("usage", {}).get("total_tokens", 0),
            model_used=self.config.model_name,
            request_time=end_time - start_time,
            request_id=request.request_id
        )

    def _call_custom_provider(self, request: LLMRequest) -> LLMResponse:
        """Appelle un provider personnalisé"""
        # Implémentation spécifique pour intégration personnalisée
        start_time = time.time()

        # Logique personnalisée d'appel LLM ici
        response_text = f"Custom provider response for: {request.prompt[:30]}..."

        end_time = time.time()

        return LLMResponse(
            text=response_text,
            tokens_used=len(request.prompt.split()),  # Estimation basique
            model_used="custom_model",
            request_time=end_time - start_time,
            request_id=request.request_id
        )

    def _call_fallback_provider(self, provider: LLMProvider, request: LLMRequest) -> LLMResponse:
        """Gère les appels aux providers de fallback"""
        # Temporairement changer de provider et appeler
        original_provider = self.config.provider
        self.config.provider = provider

        try:
            response = self._call_provider(request)
            return response
        finally:
            # Restaurer le provider original
            self.config.provider = original_provider

    def create_embedding(self, text: str) -> np.ndarray:
        """
        Crée un embedding vectoriel pour le texte d'entrée.
        Utilise le caching pour éviter des calculs redondants.
        """
        with self.lock:
            # Check cache
            if self.config.cache_enabled and text in self.embedding_cache:
                return self.embedding_cache[text]

            start_time = time.time()

            try:
                if self.config.provider == LLMProvider.OPENAI:
                    embedding = self._create_openai_embedding(text)
                elif self.config.provider == LLMProvider.LOCAL:
                    embedding = self._create_local_embedding(text)
                # Autres providers...
                else:
                    raise ValueError(f"Embedding not supported for provider: {self.config.provider}")

                # Cache result
                if self.config.cache_enabled:
                    self.embedding_cache[text] = embedding

                self.logger.debug(f"Created embedding in {time.time() - start_time:.3f}s")
                return embedding

            except Exception as e:
                self.logger.error(f"Error creating embedding: {e}", exc_info=True)
                raise

    def _create_openai_embedding(self, text: str) -> np.ndarray:
        """Crée un embedding via l'API OpenAI"""
        client = self.active_clients.get("openai")
        if not client:
            raise Exception("OpenAI client not initialized")

        response = client.embeddings.create(
            model="text-embedding-3-small",  # ou autre modèle d'embedding
            input=text
        )

        return np.array(response.data[0].embedding)

    def _create_local_embedding(self, text: str) -> np.ndarray:
        """Crée un embedding avec un modèle local"""
        try:
            from sentence_transformers import SentenceTransformer

            # Lazy initialize embedding model if needed
            if not hasattr(self, "embedding_model"):
                model_name = "all-MiniLM-L6-v2"  # Modèle léger par défaut
                self.embedding_model = SentenceTransformer(model_name)

            embedding = self.embedding_model.encode(text)
            return embedding

        except ImportError:
            self.logger.error("SentenceTransformers not installed. Cannot create local embeddings.")
            raise

    def _enrich_prompt_with_memory(self, prompt: str, context: Dict[str, Any]) -> str:
        """
        Enrichit le prompt avec des informations de la mémoire.
        Intègre des éléments pertinents des mémoires working, semantic et episodic.
        """
        if not self.config.synergy_with_memory:
            return prompt

        enriched_parts = []

        # Working Memory
        if self.working_memory:
            # On pourrait obtenir les éléments récents ou correspondant à des critères
            memory_items = []  # Placeholder pour la vraie implémentation
            if memory_items:
                enriched_parts.append("\nRelevant working memory:\n" + "\n".join(str(item) for item in memory_items))

        # Semantic Memory - concepts liés au contexte
        if self.semantic_memory and context.get("query_concepts"):
            # On pourrait chercher des concepts reliés au prompt
            concepts = []  # Placeholder pour la vraie implémentation
            if concepts:
                enriched_parts.append("\nRelevant concepts:\n" + "\n".join(str(concept) for concept in concepts))

        # Episodic Memory - épisodes pertinents
        if self.episodic_memory and context.get("query_episodes"):
            # On pourrait chercher des épisodes similaires ou pertinents
            episodes = []  # Placeholder pour la vraie implémentation
            if episodes:
                enriched_parts.append("\nRelevant episodes:\n" + "\n".join(str(episode) for episode in episodes))

        # S'il y a des enrichissements, les ajouter au prompt
        if enriched_parts:
            enriched_prompt = prompt + "\n\nContext Information:" + "".join(enriched_parts)
            return enriched_prompt

        return prompt

    def _enrich_response_with_graph(self, response: LLMResponse) -> LLMResponse:
        """
        Enrichit la réponse avec des informations du graphe de connaissances
        si la synergy avec le graphe est activée.
        """
        if not self.config.synergy_with_graph or not self.graph_engine:
            return response

        # Exemple d'enrichissement (à adapter selon vos besoins réels)
        try:
            # Extraire des entités clés de la réponse
            entities = self._extract_entities(response.text)

            if entities:
                # Rechercher ces entités dans le graphe
                graph_data = self._query_graph_for_entities(entities)

                if graph_data:
                    # Stocker les informations du graphe dans parsed_data
                    if response.parsed_data is None:
                        response.parsed_data = {}
                    response.parsed_data["graph_enrichment"] = graph_data

        except Exception as e:
            self.logger.error(f"Error enriching response with graph data: {e}", exc_info=True)

        return response

    def _extract_entities(self, text: str) -> List[str]:
        """
        Extrait des entités du texte.
        Placeholder pour une implémentation NER plus sophistiquée.
        """
        # Implémentation simpliste pour l'exemple
        import re
        # Cherche les mots commençant par une majuscule qui ne sont pas au début d'une phrase
        entities = re.findall(r'(?<![\.\?\!]\s)(?<!\A)([A-Z][a-z]+)', text)
        return list(set(entities))  # Éliminer les doublons

    def _query_graph_for_entities(self, entities: List[str]) -> Dict[str, Any]:
        """
        Interroge le graphe pour obtenir des informations sur les entités.
        """
        if not self.graph_engine:
            return {}

        result = {}

        for entity in entities:
            # Exemple simplifié - à adapter selon votre API graph_engine
            try:
                # Rechercher l'entité dans le graphe
                nodes = self.graph_engine.find_nodes({"name": entity})

                if nodes:
                    node_id = nodes[0]
                    # Obtenir les relations pour ce nœud
                    relations = self.graph_engine.get_relations(node_id)

                    # Stocker dans le résultat
                    result[entity] = {
                        "node_id": node_id,
                        "relations": relations
                    }
            except Exception as e:
                self.logger.error(f"Error querying graph for entity '{entity}': {e}")

        return result

    def _create_cache_key(self, request: LLMRequest) -> str:
        """Crée une clé de cache unique pour une requête"""
        # Générer une clé basée sur les paramètres essentiels
        key_parts = [
            request.prompt,
            str(request.temperature or self.config.temperature),
            str(request.max_tokens or self.config.max_tokens),
            request.request_type,
            self.config.model_name
        ]
        return "|".join(key_parts)

    def _add_to_cache(self, cache_key: str, response: LLMResponse) -> None:
        """Ajoute une réponse au cache"""
        # Limiter la taille du cache si nécessaire
        if self.config.cache_size > 0 and len(self.response_cache) >= self.config.cache_size:
            # Stratégie simple: supprimer une entrée aléatoire
            # Une stratégie LRU serait plus efficace en pratique
            if self.response_cache:
                self.response_cache.pop(next(iter(self.response_cache)))

        self.response_cache[cache_key] = response

    def parse_response(self, response: LLMResponse, parser_type: str) -> Dict[str, Any]:
        """
        Parse la réponse du LLM selon un type de parser spécifique.
        Utilise les fonctions de llm_interface.response_parser
        """
        if not response or not response.text:
            return {}

        try:
            # Importer dynamiquement le parser approprié
            parser_module = __import__("llm_interface.response_parser", fromlist=["parse_" + parser_type])
            parser_func = getattr(parser_module, "parse_" + parser_type, None)

            if parser_func:
                parsed_data = parser_func(response.text)
                return parsed_data
            else:
                self.logger.warning(f"Parser function 'parse_{parser_type}' not found")
                return {}

        except (ImportError, AttributeError) as e:
            self.logger.error(f"Error importing parser: {e}")
            return {}
        except Exception as e:
            self.logger.error(f"Error parsing response: {e}", exc_info=True)
            return {}