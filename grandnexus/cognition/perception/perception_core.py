"""
perception_core.py - Core Perception Module for GrandNexus

This module provides the foundation for perception capabilities in GrandNexus.
It processes incoming data, interpreting and structuring different types of inputs
including text, mathematical expressions, code, structured data, and conceptual relationships.

The perception module serves as the initial layer of cognitive processing,
transforming raw inputs into structured representations that can be utilized
by reasoning, learning, and memory systems.

Key features:
- Multi-modal input processing
- Context-aware perception filters
- Pattern recognition in various data types
- Symbolic parsing capabilities
- Integration with memory for context enrichment
"""

import logging
import time
import uuid
import re
import json
from typing import Dict, List, Any, Optional, Union, Tuple, Set, Type, Callable
from dataclasses import dataclass, field
from enum import Enum, auto
import threading

# Will be imported from other modules
from grandnexus.cognition.cognition_manager import (
    CognitionManager, CognitiveModuleBase, CognitiveProcessType,
    CognitiveMode, CognitiveContext, CognitiveRequest
)
from grandnexus.memory.working import WorkingMemory
from grandnexus.memory.semantic import SemanticMemory
from grandnexus.llm_interface.llm_engine import LLMEngine


class InputType(Enum):
    """Types of inputs that the perception module can process"""
    TEXT = auto()             # Natural language text
    MATHEMATICAL = auto()      # Mathematical expressions, formulas, equations
    CODE = auto()             # Programming code in various languages
    STRUCTURED_DATA = auto()  # JSON, XML, CSV, etc.
    GRAPH = auto()            # Graph structures, relationships, networks
    CONCEPT = auto()          # Conceptual or abstract notions
    MIXED = auto()            # Mixed or ambiguous input types
    UNKNOWN = auto()          # Unrecognized input


class PerceptionFilter(Enum):
    """Filters that can be applied during perception"""
    NONE = auto()             # No filtering
    RELEVANCE = auto()        # Filter by relevance to current context
    NOVELTY = auto()          # Filter by novelty (emphasis on new information)
    SALIENCE = auto()         # Filter by saliency (emphasis on important features)
    ABSTRACTION = auto()       # Filter by level of abstraction
    TEMPORAL = auto()          # Filter by recency or temporal relevance
    SEMANTIC = auto()          # Filter by semantic meaning or category


@dataclass
class PerceptionResult:
    """Result of a perception process"""
    result_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    input_type: InputType = InputType.UNKNOWN
    parsed_data: Any = None
    structured_representation: Any = None
    detected_patterns: List[Dict[str, Any]] = field(default_factory=list)
    confidence: float = 1.0
    metadata: Dict[str, Any] = field(default_factory=dict)
    processing_time: float = 0.0
    errors: List[str] = field(default_factory=list)


class PerceptionCore(CognitiveModuleBase):
    """
    Core class for the perception module in GrandNexus.

    Provides capabilities for processing inputs of different types,
    extracting structured information, and preparing it for higher
    cognitive functions like reasoning and learning.
    """

    def __init__(self,
                 cognition_manager: Optional[CognitionManager] = None,
                 working_memory: Optional[WorkingMemory] = None,
                 semantic_memory: Optional[SemanticMemory] = None,
                 llm_engine: Optional[LLMEngine] = None):
        """
        Initialize the Perception Core.

        Args:
            cognition_manager: Reference to the CognitionManager
            working_memory: Reference to the WorkingMemory module
            semantic_memory: Reference to the SemanticMemory module
            llm_engine: Reference to the LLM interface
        """
        self.cognition_manager = cognition_manager
        self.working_memory = working_memory
        self.semantic_memory = semantic_memory
        self.llm_engine = llm_engine

        # Logger setup
        self.logger = logging.getLogger("PerceptionCore")

        # Internal state
        self.lock = threading.RLock()
        self.input_type_detectors: Dict[InputType, Callable] = {}
        self.input_type_parsers: Dict[InputType, Dict[str, Callable]] = {
            input_type: {} for input_type in InputType
        }
        self.perception_filters: Dict[PerceptionFilter, Callable] = {}

        # Initialize default detectors and parsers
        self._initialize_default_components()

        self.logger.info("PerceptionCore initialized")

    def _initialize_default_components(self) -> None:
        """Initialize default input type detectors and parsers"""
        # Register input type detectors
        self.register_input_type_detector(InputType.TEXT, self._detect_text_input)
        self.register_input_type_detector(InputType.MATHEMATICAL, self._detect_mathematical_input)
        self.register_input_type_detector(InputType.CODE, self._detect_code_input)
        self.register_input_type_detector(InputType.STRUCTURED_DATA, self._detect_structured_data)

        # Register input parsers
        self.register_input_parser(InputType.TEXT, "default", self._parse_text_input)
        self.register_input_parser(InputType.MATHEMATICAL, "default", self._parse_mathematical_input)
        self.register_input_parser(InputType.CODE, "default", self._parse_code_input)
        self.register_input_parser(InputType.STRUCTURED_DATA, "default", self._parse_structured_data)

        # Register perception filters
        self.register_perception_filter(PerceptionFilter.RELEVANCE, self._filter_by_relevance)
        self.register_perception_filter(PerceptionFilter.NOVELTY, self._filter_by_novelty)

    def register_input_type_detector(self, input_type: InputType, detector_func: Callable) -> None:
        """
        Register a function to detect a specific input type.

        Args:
            input_type: The input type to detect
            detector_func: Function that returns confidence (0-1) of input being the specified type
        """
        with self.lock:
            self.input_type_detectors[input_type] = detector_func
            self.logger.debug(f"Registered detector for {input_type.name}")

    def register_input_parser(self, input_type: InputType, parser_name: str, parser_func: Callable) -> None:
        """
        Register a parser for a specific input type.

        Args:
            input_type: The input type to parse
            parser_name: Name of the parser (allows multiple parsers per input type)
            parser_func: Function that parses input of the specified type
        """
        with self.lock:
            self.input_type_parsers[input_type][parser_name] = parser_func
            self.logger.debug(f"Registered parser '{parser_name}' for {input_type.name}")

    def register_perception_filter(self, filter_type: PerceptionFilter, filter_func: Callable) -> None:
        """
        Register a perception filter function.

        Args:
            filter_type: The type of filter
            filter_func: Function that applies the filter
        """
        with self.lock:
            self.perception_filters[filter_type] = filter_func
            self.logger.debug(f"Registered filter for {filter_type.name}")

    def unregister_input_type_detector(self, input_type: InputType) -> bool:
        """
        Unregister an input type detector.

        Args:
            input_type: The input type detector to remove

        Returns:
            bool: True if successful, False otherwise
        """
        with self.lock:
            if input_type in self.input_type_detectors:
                del self.input_type_detectors[input_type]
                self.logger.debug(f"Unregistered detector for {input_type.name}")
                return True
            return False

    def unregister_input_parser(self, input_type: InputType, parser_name: str) -> bool:
        """
        Unregister an input parser.

        Args:
            input_type: The input type
            parser_name: Name of the parser to remove

        Returns:
            bool: True if successful, False otherwise
        """
        with self.lock:
            if input_type in self.input_type_parsers and parser_name in self.input_type_parsers[input_type]:
                del self.input_type_parsers[input_type][parser_name]
                self.logger.debug(f"Unregistered parser '{parser_name}' for {input_type.name}")
                return True
            return False

    def unregister_perception_filter(self, filter_type: PerceptionFilter) -> bool:
        """
        Unregister a perception filter.

        Args:
            filter_type: The filter type to remove

        Returns:
            bool: True if successful, False otherwise
        """
        with self.lock:
            if filter_type in self.perception_filters:
                del self.perception_filters[filter_type]
                self.logger.debug(f"Unregistered filter for {filter_type.name}")
                return True
            return False

    @CognitionManager.cognitive_capability(
        capability_type=CognitiveProcessType.PERCEPTION,
        description="Detect input type and structure",
        priority=50,
        compatible_modes={CognitiveMode.REACTIVE, CognitiveMode.DELIBERATIVE}
    )
    def perceive_input(self,
                     request: CognitiveRequest,
                     context: CognitiveContext,
                     query: Any,
                     **kwargs) -> Dict[str, Any]:
        """
        Main perception capability to process input data.

        This function performs the following steps:
        1. Determine the type of input
        2. Parse the input according to its type
        3. Apply any filters based on context
        4. Return structured results

        Args:
            request: The cognitive request
            context: Cognitive context
            query: The input data to process
            **kwargs: Additional arguments

        Returns:
            Dict[str, Any]: Perception results
        """
        start_time = time.time()

        try:
            self.logger.info(f"Processing perception request for query type: {type(query)}")

            # Create a perception result object
            result = PerceptionResult()

            # If input is not a string, convert to string or handle specially
            if not isinstance(query, str):
                if isinstance(query, dict):
                    # Handle dictionary input
                    result.input_type = InputType.STRUCTURED_DATA
                    result.parsed_data = query
                    result.structured_representation = query
                elif isinstance(query, (list, tuple)):
                    # Handle list/tuple input
                    result.input_type = InputType.STRUCTURED_DATA
                    result.parsed_data = query
                    result.structured_representation = {"items": query}
                else:
                    # Convert to string
                    query = str(query)
                    # Then determine type
                    result.input_type = self._determine_input_type(query)
            else:
                # Determine the type of input
                result.input_type = self._determine_input_type(query)

            # Parse the input based on its type
            result = self._parse_input(query, result)

            # Apply filters if specified in context
            if "perception_filters" in context.parameters:
                filters = context.parameters["perception_filters"]
                result = self._apply_filters(result, filters, context)

            # Detect patterns in the input
            result.detected_patterns = self._detect_patterns(query, result)

            # Update processing time
            end_time = time.time()
            result.processing_time = end_time - start_time

            # Convert result to dictionary for return
            return self._perception_result_to_dict(result)

        except Exception as e:
            self.logger.error(f"Error in perceive_input: {e}", exc_info=True)
            return {
                "error": str(e),
                "input_type": InputType.UNKNOWN.name,
                "processing_time": time.time() - start_time
            }

    def _determine_input_type(self, input_data: Any) -> InputType:
        """
        Determine the type of input data.

        Args:
            input_data: The input data to analyze

        Returns:
            InputType: The detected input type
        """
        type_confidences = {}

        for input_type, detector in self.input_type_detectors.items():
            try:
                confidence = detector(input_data)
                type_confidences[input_type] = confidence
            except Exception as e:
                self.logger.debug(f"Error in detector for {input_type.name}: {e}")
                type_confidences[input_type] = 0.0

        # If no detector returned a positive confidence, default to TEXT
        if not type_confidences or max(type_confidences.values()) <= 0.2:
            return InputType.TEXT

        # Return the type with highest confidence
        return max(type_confidences.items(), key=lambda x: x[1])[0]

    def _parse_input(self, input_data: Any, result: PerceptionResult) -> PerceptionResult:
        """
        Parse the input data based on its detected type.

        Args:
            input_data: The input data to parse
            result: The current perception result

        Returns:
            PerceptionResult: Updated perception result
        """
        input_type = result.input_type

        # If we don't have any parsers for this type, leave as is
        if input_type not in self.input_type_parsers or not self.input_type_parsers[input_type]:
            result.parsed_data = input_data
            return result

        # Get default parser for this input type
        parser = self.input_type_parsers[input_type].get("default")

        if not parser:
            # If no default parser, use the first available
            parser = next(iter(self.input_type_parsers[input_type].values()))

        try:
            # Parse the input
            parsed_data = parser(input_data)
            result.parsed_data = parsed_data

            # Create structured representation
            if input_type == InputType.TEXT:
                result.structured_representation = {"text": parsed_data}
            elif input_type == InputType.MATHEMATICAL:
                result.structured_representation = {"expression": parsed_data}
            elif input_type == InputType.CODE:
                result.structured_representation = parsed_data
            elif input_type == InputType.STRUCTURED_DATA:
                result.structured_representation = parsed_data
            else:
                result.structured_representation = {"data": parsed_data}

        except Exception as e:
            self.logger.error(f"Error parsing input of type {input_type.name}: {e}", exc_info=True)
            result.errors.append(f"Parsing error: {str(e)}")
            result.confidence = 0.5  # Reduce confidence due to parsing error

        return result

    def _apply_filters(self,
                     result: PerceptionResult,
                     filters: List[Union[PerceptionFilter, str]],
                     context: CognitiveContext) -> PerceptionResult:
        """
        Apply perception filters to the result.

        Args:
            result: The perception result to filter
            filters: List of filters to apply
            context: The cognitive context

        Returns:
            PerceptionResult: Filtered perception result
        """
        for filter_item in filters:
            # Convert string to enum if needed
            if isinstance(filter_item, str):
                try:
                    filter_item = PerceptionFilter[filter_item.upper()]
                except KeyError:
                    self.logger.warning(f"Unknown filter type: {filter_item}")
                    continue

            # Get filter function
            filter_func = self.perception_filters.get(filter_item)

            if filter_func:
                try:
                    result = filter_func(result, context)
                except Exception as e:
                    self.logger.error(f"Error applying filter {filter_item.name}: {e}", exc_info=True)

        return result

    def _detect_patterns(self, input_data: Any, result: PerceptionResult) -> List[Dict[str, Any]]:
        """
        Detect patterns in the input data.

        Args:
            input_data: The original input data
            result: The current perception result

        Returns:
            List[Dict[str, Any]]: Detected patterns
        """
        patterns = []

        # Simple pattern detection based on input type
        if result.input_type == InputType.TEXT:
            # Detect URLs
            url_pattern = re.compile(r'https?://\S+')
            urls = url_pattern.findall(input_data) if isinstance(input_data, str) else []
            if urls:
                patterns.append({
                    "pattern_type": "url",
                    "instances": urls,
                    "confidence": 1.0
                })

            # Detect email addresses
            email_pattern = re.compile(r'\S+@\S+\.\S+')
            emails = email_pattern.findall(input_data) if isinstance(input_data, str) else []
            if emails:
                patterns.append({
                    "pattern_type": "email",
                    "instances": emails,
                    "confidence": 1.0
                })

        elif result.input_type == InputType.CODE:
            # Detect function definitions (simplified)
            if isinstance(input_data, str):
                function_pattern = re.compile(r'def\s+(\w+)\s*\(')
                functions = function_pattern.findall(input_data)
                if functions:
                    patterns.append({
                        "pattern_type": "function_definition",
                        "instances": functions,
                        "confidence": 0.9
                    })

        # Add more specialized pattern detection in future implementations

        return patterns

    def _perception_result_to_dict(self, result: PerceptionResult) -> Dict[str, Any]:
        """
        Convert a PerceptionResult to a dictionary.

        Args:
            result: The perception result to convert

        Returns:
            Dict[str, Any]: Dictionary representation of the result
        """
        return {
            "result_id": result.result_id,
            "input_type": result.input_type.name,
            "parsed_data": result.parsed_data,
            "structured_representation": result.structured_representation,
            "detected_patterns": result.detected_patterns,
            "confidence": result.confidence,
            "metadata": result.metadata,
            "processing_time": result.processing_time,
            "errors": result.errors
        }

    #------------------------------------------------------------------
    # Input type detectors
    #------------------------------------------------------------------

    def _detect_text_input(self, input_data: Any) -> float:
        """
        Detect if input is natural language text.

        Args:
            input_data: The input data to analyze

        Returns:
            float: Confidence that input is text (0-1)
        """
        if not isinstance(input_data, str):
            return 0.0

        # Text is our default, but we can still apply some heuristics

        # Check for structured data markers
        if input_data.strip().startswith(("{", "[", "<")) and input_data.strip().endswith(("}", "]", ">")):
            # Might be JSON, XML, etc.
            return 0.3

        # Check for code markers
        code_indicators = ["def ", "class ", "function", "import ", "from ", "#include", "public class"]
        if any(indicator in input_data for indicator in code_indicators):
            return 0.4

        # Check for mathematical expressions
        math_symbols = ["=", "+", "-", "*", "/", "^", "\\frac", "\\sum", "\\int"]
        math_symbol_count = sum(1 for symbol in math_symbols if symbol in input_data)
        if math_symbol_count > 3:
            return 0.4

        # Default confidence for text
        return 0.8

    def _detect_mathematical_input(self, input_data: Any) -> float:
        """
        Detect if input contains mathematical expressions.

        Args:
            input_data: The input data to analyze

        Returns:
            float: Confidence that input is mathematical (0-1)
        """
        if not isinstance(input_data, str):
            return 0.0

        # Check for LaTeX math delimiters
        if "\\begin{equation}" in input_data or "\\begin{align}" in input_data:
            return 0.9

        if "$" in input_data and input_data.count("$") >= 2:
            return 0.8

        # Check for common math operators and patterns
        math_operators = ["+", "-", "*", "/", "=", "<", ">", "±", "≤", "≥", "≠", "≈", "∫", "∑"]
        math_functions = ["sin", "cos", "tan", "log", "ln", "exp", "sqrt", "max", "min"]

        # Count math operators and functions
        operator_count = sum(input_data.count(op) for op in math_operators)
        function_count = sum(1 for func in math_functions if func in input_data)

        # Check for equation patterns
        equation_patterns = ["=", "equation", "formula", "solve", "compute"]
        has_equation_pattern = any(pattern in input_data.lower() for pattern in equation_patterns)

        # Calculate confidence score
        score = 0.0
        if has_equation_pattern:
            score += 0.3

        # Add score based on operator and function density
        if len(input_data) > 0:
            op_density = operator_count / len(input_data)
            score += min(0.5, op_density * 10)  # Cap at 0.5 for operator density
            score += min(0.3, function_count * 0.1)  # 0.1 per function, up to 0.3

        return min(0.95, score)  # Cap at 0.95

    def _detect_code_input(self, input_data: Any) -> float:
        """
        Detect if input is programming code.

        Args:
            input_data: The input data to analyze

        Returns:
            float: Confidence that input is code (0-1)
        """
        if not isinstance(input_data, str):
            return 0.0

        # Common code indicators
        code_keywords = [
            "function", "def ", "class ", "if ", "else", "for ", "while ", "return", "import ",
            "from ", "var ", "let ", "const ", "public", "private", "protected", "#include",
            "package", "namespace", "using "
        ]

        code_symbols = ["{}", "[]", "()", ";", "=>", "->", "<<", ">>", "===", "!==", "&&", "||"]

        # Count occurrences
        keyword_count = sum(input_data.count(keyword) for keyword in code_keywords)
        symbol_count = sum(input_data.count(symbol[0]) for symbol in code_symbols if len(symbol) == 1)

        # Check for indentation patterns
        indentation_pattern = re.compile(r'^( {2,}|\t+)', re.MULTILINE)
        indentation_matches = indentation_pattern.findall(input_data)

        # Check for common code structures
        function_def = re.search(r'(function|def|public|private)\s+\w+\s*\(', input_data)
        class_def = re.search(r'(class|interface|struct|enum)\s+\w+', input_data)
        import_statement = re.search(r'(import|from|#include|using)\s+[\w\.]+', input_data)

        # Calculate confidence score
        score = 0.0

        if function_def or class_def:
            score += 0.4

        if import_statement:
            score += 0.3

        if len(indentation_matches) > 2:
            score += 0.2

        # Add score based on keyword and symbol density
        if len(input_data) > 0:
            keyword_factor = min(0.3, keyword_count * 0.05)  # 0.05 per keyword, up to 0.3
            symbol_factor = min(0.2, symbol_count * 0.02)   # 0.02 per symbol, up to 0.2
            score += keyword_factor + symbol_factor

        return min(0.95, score)  # Cap at 0.95

    def _detect_structured_data(self, input_data: Any) -> float:
        """
        Detect if input is structured data (JSON, XML, etc.).

        Args:
            input_data: The input data to analyze

        Returns:
            float: Confidence that input is structured data (0-1)
        """
        # If it's already a dict or list, it's definitely structured data
        if isinstance(input_data, (dict, list)):
            return 1.0

        if not isinstance(input_data, str):
            return 0.0

        input_data = input_data.strip()

        # Check for JSON format
        if (input_data.startswith("{") and input_data.endswith("}")) or (input_data.startswith("[") and input_data.endswith("]")):
            try:
                json.loads(input_data)
                return 0.95  # High confidence for valid JSON
            except json.JSONDecodeError:
                # Looks like JSON but isn't valid
                return 0.5

        # Check for XML format
        if input_data.startswith("<") and input_data.endswith(">"):
            if input_data.startswith("<?xml") or ("</" in input_data and ">" in input_data):
                return 0.9

        # Check for CSV-like format
        if "," in input_data:
            lines = input_data.split("\n")
            if len(lines) > 1:
                comma_counts = [line.count(",") for line in lines if line.strip()]
                if len(comma_counts) > 1 and len(set(comma_counts[:min(5, len(comma_counts))])) <= 1:
                    # Consistent number of commas in first few lines suggests CSV
                    return 0.8

        # Check for key-value pairs
        kv_pattern = re.compile(r'^\s*[\w-]+\s*[:=]\s*.+$', re.MULTILINE)
        kv_matches = kv_pattern.findall(input_data)
        if len(kv_matches) > 2 and len(kv_matches) / input_data.count("\n") > 0.5:
            return 0.7  # Looks like key-value configuration format

        return 0.1  # Low confidence

    #------------------------------------------------------------------
    # Input parsers
    #------------------------------------------------------------------

    def _parse_text_input(self, input_data: Any) -> Any:
        """
        Parse natural language text input.

        Args:
            input_data: The text input to parse

        Returns:
            Any: Parsed text data
        """
        if not isinstance(input_data, str):
            return str(input_data)

        # For text, we might extract entities, keywords, etc.
        # For now, we'll just return the text itself with basic structure

        # Split into paragraphs
        paragraphs = [p.strip() for p in input_data.split("\n\n") if p.strip()]

        # Extract potential keywords (simple implementation)
        words = input_data.lower().split()
        # Filter common stopwords (very simplified)
        stopwords = {"the", "a", "an", "in", "on", "at", "to", "for", "of", "with", "by"}
        keywords = [word for word in words if word not in stopwords and len(word) > 3]
        # Count word frequencies
        word_counts = {}
        for word in keywords:
            word_counts[word] = word_counts.get(word, 0) + 1
        # Sort by frequency
        potential_keywords = sorted(word_counts.items(), key=lambda x: x[1], reverse=True)[:10]

        return {
            "text": input_data,
            "paragraphs": paragraphs,
            "potential_keywords": [k[0] for k in potential_keywords],
            "word_count": len(words),
            "paragraph_count": len(paragraphs)
        }

    def _parse_mathematical_input(self, input_data: Any) -> Any:
        """
        Parse mathematical expressions.

        Args:
            input_data: The mathematical input to parse

        Returns:
            Any: Parsed mathematical data
        """
        if not isinstance(input_data, str):
            return {"expression": str(input_data), "parsed": False}

        # Basic identification of expression type
        expression_type = "unknown"
        properties = {}

        # Check for equation
        if "=" in input_data:
            expression_type = "equation"
            sides = input_data.split("=", 1)
            properties["left_side"] = sides[0].strip()
            properties["right_side"] = sides[1].strip()

        # Check for function definition
        elif ":" in input_data or "->" in input_data:
            expression_type = "function_definition"
            # Try to identify function name and mapping
            if "->" in input_data:
                parts = input_data.split("->", 1)
                properties["domain"] = parts[0].strip()
                properties["codomain"] = parts[1].strip()
            elif ":" in input_data:
                parts = input_data.split(":", 1)
                properties["name"] = parts[0].strip()
                properties["definition"] = parts[1].strip()

        # Check for inequality
        elif any(op in input_data for op in ["<", ">", "≤", "≥", "≠"]):
            expression_type = "inequality"
            # Identify the operator
            for op in ["≤", "≥", "≠", "<", ">"]:
                if op in input_data:
                    sides = input_data.split(op, 1)
                    properties["left_side"] = sides[0].strip()
                    properties["right_side"] = sides[1].strip()
                    properties["operator"] = op
                    break

        # Check for expression (no equality/inequality)
        else:
            expression_type = "expression"
            properties["expression"] = input_data.strip()

        return {
            "expression": input_data,
            "type": expression_type,
            "properties": properties,
            "parsed": True
        }

    def _parse_code_input(self, input_data: Any) -> Any:
        """
        Parse programming code input.

        Args:
            input_data: The code input to parse

        Returns:
            Any: Parsed code data with structure information
        """
        if not isinstance(input_data, str):
            return {"code": str(input_data), "parsed": False}

        # Try to determine language
        language = self._detect_code_language(input_data)

        # Extract functions and classes (basic regex-based approach)
        functions = []
        classes = []
        imports = []

        # Extract function definitions
        if language in ["python", "unknown"]:
            # Python-style function definitions
            function_pattern = re.compile(r'def\s+([a-zA-Z_][a-zA-Z0-9_]*)\s*\(([^)]*)\)(?:\s*->?[^:]*)?:', re.MULTILINE)
            func_matches = function_pattern.findall(input_data)
            for name, params in func_matches:
                functions.append({
                    "name": name,
                    "parameters": [p.strip() for p in params.split(",") if p.strip()],
                    "language": language
                })

            # Python-style class definitions
            class_pattern = re.compile(r'class\s+([a-zA-Z_][a-zA-Z0-9_]*)(?:\s*\(([^)]*)\))?:', re.MULTILINE)
            class_matches = class_pattern.findall(input_data)
            for name, parents in class_matches:
                classes.append({
                    "name": name,
                    "parents": [p.strip() for p in parents.split(",") if p.strip()],
                    "language": language
                })

            # Python-style imports
            import_pattern = re.compile(r'(from\s+[a-zA-Z_][a-zA-Z0-9_.]+\s+import|import)\s+([^#\n]+)', re.MULTILINE)
            import_matches = import_pattern.findall(input_data)
            for import_type, import_names in import_matches:
                imports.append({
                    "statement": (import_type + " " + import_names).strip(),
                    "language": language
                })

        elif language == "javascript":
            # JavaScript-style function definitions (including ES6 arrow functions)
            function_pattern = re.compile(r'(?:function\s+([a-zA-Z_][a-zA-Z0-9_]*)\s*\(([^)]*)\)|(?:const|let|var)\s+([a-zA-Z_][a-zA-Z0-9_]*)\s*=\s*(?:function\s*\(([^)]*)\)|(?:\(([^)]*)\)\s*=>)))', re.MULTILINE)
            func_matches = function_pattern.findall(input_data)
            for match in func_matches:
                if match[0]:  # Standard function
                    functions.append({
                        "name": match[0],
                        "parameters": [p.strip() for p in match[1].split(",") if p.strip()],
                        "language": language
                    })
                elif match[2]:  # Arrow function or function expression
                    params = match[3] if match[3] else match[4]
                    functions.append({
                        "name": match[2],
                        "parameters": [p.strip() for p in params.split(",") if p.strip()],
                        "language": language
                    })

            # JavaScript-style class definitions
            class_pattern = re.compile(r'class\s+([a-zA-Z_][a-zA-Z0-9_]*)(?:\s+extends\s+([a-zA-Z_][a-zA-Z0-9_.]*))?', re.MULTILINE)
            class_matches = class_pattern.findall(input_data)
            for name, parent in class_matches:
                parents = [parent] if parent else []
                classes.append({
                    "name": name,
                    "parents": parents,
                    "language": language
                })

            # JavaScript-style imports
            import_pattern = re.compile(r'import\s+(?:{[^}]+}|[a-zA-Z_][a-zA-Z0-9_]*)\s+from\s+[\'"][^\'")]+[\'"]', re.MULTILINE)
            import_matches = import_pattern.findall(input_data)
            for import_stmt in import_matches:
                imports.append({
                    "statement": import_stmt.strip(),
                    "language": language
                })

        # Count lines of code
        lines = input_data.split("\n")
        non_empty_lines = sum(1 for line in lines if line.strip())

        # Look for comments
        comment_patterns = {
            "python": r'#.*$',
            "javascript": r'(?://.*$|/\*[\s\S]*?\*/)',
            "unknown": r'(?://.*$|/\*[\s\S]*?\*/|#.*$)'
        }

        comment_pattern = re.compile(comment_patterns.get(language, comment_patterns["unknown"]), re.MULTILINE)
        comments = comment_pattern.findall(input_data)

        return {
            "code": input_data,
            "language": language,
            "functions": functions,
            "classes": classes,
            "imports": imports,
            "line_count": len(lines),
            "non_empty_line_count": non_empty_lines,
            "comment_count": len(comments),
            "parsed": True
        }

    def _detect_code_language(self, code: str) -> str:
        """
        Attempt to detect the programming language of a code snippet.

        Args:
            code: The code snippet

        Returns:
            str: Detected language or "unknown"
        """
        # Simple heuristics for language detection
        if re.search(r'def\s+\w+\s*\(|import\s+\w+|from\s+\w+\s+import', code):
            return "python"
        elif re.search(r'(?:function\s+\w+|var\s+|let\s+|const\s+|=>|{|}|;$)', code):
            return "javascript"
        elif re.search(r'#include\s+<\w+\.h>|int\s+main\s*\(\s*\)', code):
            return "c"
        elif re.search(r'public\s+class|public\s+static\s+void\s+main|import\s+java\.\w+', code):
            return "java"
        else:
            return "unknown"

    def _parse_structured_data(self, input_data: Any) -> Any:
        """
        Parse structured data input (JSON, XML, etc.).

        Args:
            input_data: The structured data input to parse

        Returns:
            Any: Parsed structured data
        """
        # If already a dict or list, return as is
        if isinstance(input_data, (dict, list)):
            return input_data

        if not isinstance(input_data, str):
            return {"data": str(input_data), "parsed": False}

        input_data = input_data.strip()
        data_format = "unknown"
        parsed_data = None

        # Try to parse as JSON
        if (input_data.startswith("{") and input_data.endswith("}")) or (input_data.startswith("[") and input_data.endswith("]")):
            try:
                parsed_data = json.loads(input_data)
                data_format = "json"
            except json.JSONDecodeError:
                # Not valid JSON
                pass

        # Try to parse as CSV (simplified)
        if data_format == "unknown" and "," in input_data:
            lines = input_data.split("\n")
            if len(lines) > 1:
                # Check if the first line could be a header
                header = [h.strip() for h in lines[0].split(",")]
                if all(h for h in header):  # Non-empty header values
                    try:
                        parsed_data = []
                        for i, line in enumerate(lines[1:], 1):
                            if not line.strip():
                                continue
                            values = [v.strip() for v in line.split(",")]
                            if len(values) == len(header):
                                row = {header[j]: values[j] for j in range(len(header))}
                                parsed_data.append(row)
                            else:
                                # Inconsistent column count, not a valid CSV
                                parsed_data = None
                                break
                        if parsed_data:
                            data_format = "csv"
                    except Exception:
                        # Error during CSV parsing
                        parsed_data = None

        # Try to parse as key-value pairs
        if data_format == "unknown":
            lines = input_data.split("\n")
            kv_data = {}
            valid_kv = True
            for line in lines:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                if ":" in line:
                    key, value = line.split(":", 1)
                    kv_data[key.strip()] = value.strip()
                elif "=" in line:
                    key, value = line.split("=", 1)
                    kv_data[key.strip()] = value.strip()
                else:
                    valid_kv = False
                    break
            if valid_kv and kv_data:
                parsed_data = kv_data
                data_format = "key-value"

        # If we couldn't parse it, just return the raw string
        if parsed_data is None:
            return {
                "data": input_data,
                "format": data_format,
                "parsed": False
            }

        return {
            "data": parsed_data,
            "format": data_format,
            "raw": input_data,
            "parsed": True
        }

    #------------------------------------------------------------------
    # Perception filters
    #------------------------------------------------------------------

    def _filter_by_relevance(self, result: PerceptionResult, context: CognitiveContext) -> PerceptionResult:
        """
        Filter perception result based on relevance to current context.

        Args:
            result: The perception result to filter
            context: The cognitive context

        Returns:
            PerceptionResult: Filtered perception result
        """
        # Simple relevance filtering, could be expanded significantly

        # If no context parameters or no structured representation, return as is
        if not context.parameters or not result.structured_representation:
            return result

        # If there's a current topic or focus in the context
        if "topic" in context.parameters:
            current_topic = context.parameters["topic"]

            # If dealing with text
            if result.input_type == InputType.TEXT and isinstance(result.parsed_data, dict):
                # Check if we have keywords and filter based on topic relevance
                if "potential_keywords" in result.parsed_data:
                    # Keep keywords potentially relevant to the topic
                    original_keywords = result.parsed_data["potential_keywords"]
                    topic_keywords = [k for k in original_keywords if self._is_related(k, current_topic)]

                    # Update the parsed data
                    if topic_keywords:
                        result.parsed_data["potential_keywords"] = topic_keywords
                        result.parsed_data["topic_filtered"] = True

                        # Set a relevance metadata score
                        result.metadata["relevance_score"] = len(topic_keywords) / len(original_keywords)

        return result

    def _is_related(self, term: str, topic: str) -> bool:
        """
        Simple helper to check if a term is related to a topic.
        Could be expanded with semantic similarity, knowledge graph, etc.

        Args:
            term: The term to check
            topic: The topic to check against

        Returns:
            bool: True if related, False otherwise
        """
        # Basic implementation - check if the term is contained in the topic or vice versa
        return term.lower() in topic.lower() or topic.lower() in term.lower()

    def _filter_by_novelty(self, result: PerceptionResult, context: CognitiveContext) -> PerceptionResult:
        """
        Filter perception result based on novelty (emphasize new information).

        Args:
            result: The perception result to filter
            context: The cognitive context

        Returns:
            PerceptionResult: Filtered perception result with novelty information
        """
        # If working memory is not available, we can't determine novelty
        if not self.working_memory:
            return result

        # Initialize novelty score as neutral
        novelty_score = 0.5

        # If we have text input with keywords
        if result.input_type == InputType.TEXT and isinstance(result.parsed_data, dict) and "potential_keywords" in result.parsed_data:
            keywords = result.parsed_data["potential_keywords"]

            # Check which keywords are new relative to working memory
            new_keywords = []
            for keyword in keywords:
                # Simplified check - a more sophisticated approach would use embeddings
                query_result = self.working_memory.retrieve(
                    query_text=keyword,
                    max_results=1
                )

                # If no results, it's potentially new
                if not query_result:
                    new_keywords.append(keyword)

            # Calculate novelty score based on proportion of new keywords
            if keywords:
                novelty_score = len(new_keywords) / len(keywords)

            # Add novelty information to result
            result.metadata["novelty_score"] = novelty_score
            result.metadata["new_keywords"] = new_keywords

        return result

    #------------------------------------------------------------------
    # Helper methods
    #------------------------------------------------------------------

    @CognitionManager.cognitive_capability(
        capability_type=CognitiveProcessType.PERCEPTION,
        description="Extract structured information from text",
        priority=60,
        compatible_modes={CognitiveMode.REACTIVE, CognitiveMode.DELIBERATIVE}
    )
    def extract_structured_info(self,
                              request: CognitiveRequest,
                              context: CognitiveContext,
                              query: Any,
                              **kwargs) -> Dict[str, Any]:
        """
        Extract structured information from text.
        This is a higher-level capability that builds on basic perception.

        Args:
            request: The cognitive request
            context: Cognitive context
            query: The input data to process
            **kwargs: Additional arguments

        Returns:
            Dict[str, Any]: Extracted structured information
        """
        # First perform basic perception
        perception_result = self.perceive_input(request, context, query, **kwargs)

        # If there was an error in basic perception, return it
        if "error" in perception_result:
            return perception_result

        # If input is not text, return the basic perception result
        if perception_result["input_type"] != "TEXT":
            return {
                "message": "Extraction of structured information is only supported for text input",
                "basic_perception": perception_result
            }

        # Extract structured information using more advanced techniques
        # In a full implementation, this could use NLP techniques or call a LLM
        structured_info = self._extract_structured_info_from_text(query)

        return {
            "structured_info": structured_info,
            "basic_perception": perception_result
        }

    def _extract_structured_info_from_text(self, text: str) -> Dict[str, Any]:
        """
        Extract structured information from text.
        This is a simplified implementation.

        Args:
            text: The text to analyze

        Returns:
            Dict[str, Any]: Extracted structured information
        """
        # Simple extraction of entities (people, locations, dates)
        entities = {
            "people": [],
            "locations": [],
            "dates": [],
            "organizations": []
        }

        # Very simple name detection (looking for capitalized words)
        name_pattern = re.compile(r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b')
        potential_names = name_pattern.findall(text)

        # Simple date detection
        date_patterns = [
            r'\b\d{1,2}/\d{1,2}/\d{2,4}\b',  # MM/DD/YYYY or DD/MM/YYYY
            r'\b\d{1,2}-\d{1,2}-\d{2,4}\b',  # MM-DD-YYYY or DD-MM-YYYY
            r'\b(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\s+\d{1,2}(?:st|nd|rd|th)?,?\s+\d{2,4}\b'  # Month DD, YYYY
        ]

        dates = []
        for pattern in date_patterns:
            dates.extend(re.findall(pattern, text))

        # Attempt to categorize entities
        for name in potential_names:
            # This is a very simplistic approach; a real implementation would use NER
            if len(name.split()) >= 2:
                # Assume people usually have first and last names
                entities["people"].append(name)
            elif name in ["America", "USA", "Canada", "Mexico", "China", "Russia", "London", "Paris", "Tokyo"]:
                # Just a few example locations
                entities["locations"].append(name)
            elif "Inc" in name or "Corp" in name or "Company" in name:
                entities["organizations"].append(name)

        entities["dates"] = dates

        # Extract key phrases (very simplified)
        sentences = re.split(r'[.!?]+', text)
        key_phrases = []

        for sentence in sentences:
            sentence = sentence.strip()
            if not sentence:
                continue

            # Look for sentences that might contain key information
            if any(indicator in sentence.lower() for indicator in [
                "important", "significant", "key", "critical", "essential",
                "main", "primary", "major", "crucial", "vital"
            ]):
                key_phrases.append(sentence)

        # Extract potential action items (very simplified)
        action_items = []
        action_indicators = ["need to", "should", "must", "have to", "required to"]

        for sentence in sentences:
            sentence = sentence.strip()
            if not sentence:
                continue

            if any(indicator in sentence.lower() for indicator in action_indicators):
                action_items.append(sentence)

        return {
            "entities": entities,
            "key_phrases": key_phrases,
            "action_items": action_items
        }

    @CognitionManager.cognitive_capability(
        capability_type=CognitiveProcessType.PERCEPTION,
        description="Classify input content",
        priority=70,
        compatible_modes={CognitiveMode.DELIBERATIVE}
    )
    def classify_content(self,
                       request: CognitiveRequest,
                       context: CognitiveContext,
                       query: Any,
                       **kwargs) -> Dict[str, Any]:
        """
        Classify the content of input data.

        Args:
            request: The cognitive request
            context: Cognitive context
            query: The input data to classify
            **kwargs: Additional arguments

        Returns:
            Dict[str, Any]: Classification results
        """
        # First perform basic perception
        perception_result = self.perceive_input(request, context, query, **kwargs)

        # If there was an error in basic perception, return it
        if "error" in perception_result:
            return perception_result

        # Build classification based on input type
        classification = {
            "categories": [],
            "confidence": 0.0,
            "topic_areas": []
        }

        # Use LLM for classification if available
        if self.llm_engine:
            try:
                # Create a classification prompt
                if isinstance(query, str):
                    prompt = f"Classify the following content into categories and topics:\n\n{query[:1000]}\n\nProvide the response as JSON with 'categories' (list of category names) and 'topic_areas' (list of relevant topics)."

                    # Call LLM for classification
                    llm_response = self.llm_engine.complete({
                        "prompt": prompt,
                        "max_tokens": 200,
                        "temperature": 0.3
                    })

                    # Try to parse JSON response
                    try:
                        if hasattr(llm_response, 'text'):
                            llm_text = llm_response.text
                        else:
                            llm_text = str(llm_response)

                        # Extract JSON part if exists
                        json_match = re.search(r'```json\s*(.*?)\s*```', llm_text, re.DOTALL)
                        if json_match:
                            llm_result = json.loads(json_match.group(1))
                        else:
                            # Try to find JSON-like structure
                            json_match = re.search(r'({[^{]*"categories"[^}]*})', llm_text, re.DOTALL)
                            if json_match:
                                llm_result = json.loads(json_match.group(1))
                            else:
                                # Fallback to rule-based classification
                                raise ValueError("Could not parse LLM response as JSON")

                        # Update classification with LLM results
                        if "categories" in llm_result:
                            classification["categories"] = llm_result["categories"]
                        if "topic_areas" in llm_result:
                            classification["topic_areas"] = llm_result["topic_areas"]

                        classification["confidence"] = 0.8  # Higher confidence with LLM
                        classification["method"] = "llm"

                    except (json.JSONDecodeError, ValueError, AttributeError) as e:
                        self.logger.warning(f"Error parsing LLM classification response: {e}")
                        # Fall back to rule-based classification
                        classification.update(self._rule_based_classification(query, perception_result))
                else:
                    # Non-string input, use rule-based
                    classification.update(self._rule_based_classification(query, perception_result))
            except Exception as e:
                self.logger.error(f"Error in LLM classification: {e}")
                # Fall back to rule-based classification
                classification.update(self._rule_based_classification(query, perception_result))
        else:
            # No LLM available, use rule-based classification
            classification.update(self._rule_based_classification(query, perception_result))

        return {
            "classification": classification,
            "basic_perception": perception_result
        }

    def _rule_based_classification(self, query: Any, perception_result: Dict[str, Any]) -> Dict[str, Any]:
        """
        Perform rule-based classification of content.

        Args:
            query: The original query
            perception_result: Result of basic perception

        Returns:
            Dict[str, Any]: Classification results
        """
        classification = {
            "categories": [],
            "confidence": 0.5,  # Lower confidence for rule-based
            "topic_areas": [],
            "method": "rule-based"
        }

        input_type = perception_result.get("input_type", "UNKNOWN")

        # Classify based on input type
        if input_type == "TEXT":
            text = query if isinstance(query, str) else str(query)

            # Simple topic detection based on keywords
            topic_indicators = {
                "technology": ["computer", "software", "hardware", "tech", "digital", "internet", "web", "online", "code", "program"],
                "science": ["science", "scientific", "research", "experiment", "theory", "hypothesis", "lab", "study"],
                "business": ["business", "company", "market", "finance", "economic", "investment", "stock", "profit", "revenue"],
                "health": ["health", "medical", "doctor", "patient", "disease", "treatment", "medicine", "hospital", "clinic"],
                "education": ["education", "school", "student", "learn", "teach", "course", "class", "professor", "academic"],
                "politics": ["politic", "government", "election", "vote", "party", "policy", "law", "regulation", "president"]
            }

            # Check for topics
            text_lower = text.lower()
            for topic, keywords in topic_indicators.items():
                if any(keyword in text_lower for keyword in keywords):
                    classification["topic_areas"].append(topic)

            # Classify content type
            if len(text.split()) > 100:
                classification["categories"].append("long-form")
            else:
                classification["categories"].append("short-form")

            if "?" in text:
                classification["categories"].append("question")

            if len(text.split("\n")) > 5:
                classification["categories"].append("multi-paragraph")

        elif input_type == "CODE":
            classification["categories"].append("code")
            language = perception_result.get("parsed_data", {}).get("language", "unknown")
            if language != "unknown":
                classification["categories"].append(f"{language}-code")

        elif input_type == "MATHEMATICAL":
            classification["categories"].append("mathematical")
            math_type = perception_result.get("parsed_data", {}).get("type", "expression")
            classification["categories"].append(f"math-{math_type}")

        elif input_type == "STRUCTURED_DATA":
            classification["categories"].append("structured-data")
            data_format = perception_result.get("parsed_data", {}).get("format", "unknown")
            if data_format != "unknown":
                classification["categories"].append(f"{data_format}-format")

        return classification


# Standalone execution for testing
if __name__ == "__main__":
    # Set up logging
    logging.basicConfig(level=logging.INFO)

    # Create a PerceptionCore instance
    perception = PerceptionCore()

    # Create a simple context for testing
    context = CognitiveContext(
        parameters={"perception_filters": [PerceptionFilter.RELEVANCE]}
    )

    # Test with different input types
    test_inputs = [
        "This is a simple text input for testing perception capabilities.",
        "x^2 + 3x + 2 = 0",
        "def hello_world():\n    print('Hello, world!')",
        '{"name": "John", "age": 30, "city": "New York"}'
    ]

    for input_data in test_inputs:
        request = CognitiveRequest(query=input_data, context=context)
        result = perception.perceive_input(request, context, input_data)
        print(f"\nPerception result for: {input_data[:30]}...")
        print(f"Input type: {result['input_type']}")
        print(f"Confidence: {result['confidence']}")
        if result['detected_patterns']:
            print(f"Detected patterns: {result['detected_patterns']}")
        print("-" * 40)