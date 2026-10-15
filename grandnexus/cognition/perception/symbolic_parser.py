"""
symbolic_parser.py - Symbolic Expression Parsing Module for GrandNexus

This module provides functionality for parsing and interpreting symbolic expressions
such as mathematical formulas, logical statements, and structured patterns. It serves
as a bridge between textual representations and formal computational structures.

The module supports:
- Mathematical expression parsing and evaluation
- Logical formula parsing and interpretation
- Custom DSL (Domain Specific Language) processing
- Structural pattern recognition in symbolic forms

Key design principles:
- Extensibility: Easy to add new symbolic languages and parsing strategies
- Progressive capacity: Start with simple expressions, evolve to handle complex ones
- Integration: Connect with semantic memory for concept recognition and lookup
- Emergence: Support discovery of patterns and structures in symbolic content
"""

import logging
import re
import uuid
import math
from typing import Any, Dict, List, Optional, Set, Tuple, Union, Callable
from enum import Enum, auto
from dataclasses import dataclass, field
import ast
import operator
import threading
from collections import defaultdict, deque

# Assuming these exist or will exist
from grandnexus.core.nexus_core import NexusCore
from grandnexus.memory.semantic import SemanticMemory
from grandnexus.memory.working import WorkingMemory

# Import sympy with try/except to make it optional
try:
    import sympy
    from sympy.parsing.sympy_parser import parse_expr, standard_transformations
    from sympy import Symbol, symbols, solve, Eq, simplify
    HAS_SYMPY = True
except ImportError:
    HAS_SYMPY = False

# Define symbolic expression types
class SymbolicType(Enum):
    """Types of symbolic expressions the parser can handle"""
    MATHEMATICAL = auto()    # Mathematical formulas, equations
    LOGICAL = auto()         # Logical propositions, boolean algebra
    CODE = auto()            # Code snippets or DSL expressions
    PATTERN = auto()         # Pattern expressions or templates
    CONSTRAINT = auto()      # Constraint expressions
    UNKNOWN = auto()         # Unable to determine type

@dataclass
class SymbolicAST:
    """Abstract Syntax Tree for parsed symbolic expressions"""
    node_type: str
    value: Any = None
    children: List["SymbolicAST"] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def traverse(self, callback: Callable[["SymbolicAST"], None]) -> None:
        """
        Traverse the AST, calling the callback for each node.

        Args:
            callback: Function to call on each node
        """
        callback(self)
        for child in self.children:
            child.traverse(callback)

    def to_dict(self) -> Dict[str, Any]:
        """Convert AST to dictionary representation"""
        result = {
            "node_type": self.node_type,
            "value": self.value,
            "metadata": self.metadata,
            "children": [child.to_dict() for child in self.children]
        }
        return result

    def __str__(self) -> str:
        """String representation of the AST node"""
        if not self.children:
            return f"{self.node_type}({self.value})"
        return f"{self.node_type}({self.value}, {len(self.children)} children)"

@dataclass
class ParseResult:
    """Result of parsing a symbolic expression"""
    success: bool
    expr_type: SymbolicType
    ast: Optional[SymbolicAST] = None
    sympy_expr: Optional[Any] = None  # SymPy expression if applicable
    python_ast: Optional[ast.AST] = None  # Python AST if applicable
    error_message: Optional[str] = None
    confidence: float = 1.0
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        """Initialize id and timestamp after creation"""
        self.id = str(uuid.uuid4())
        self.timestamp = field(default_factory=lambda: __import__('time').time())

    def is_valid(self) -> bool:
        """Check if parse result is valid and usable"""
        return self.success and (self.ast is not None or
                                 self.sympy_expr is not None or
                                 self.python_ast is not None)

    def to_dict(self) -> Dict[str, Any]:
        """Convert parse result to dictionary representation"""
        result = {
            "success": self.success,
            "expr_type": self.expr_type.name,
            "confidence": self.confidence,
            "metadata": self.metadata
        }

        if self.error_message:
            result["error_message"] = self.error_message

        if self.ast:
            result["ast"] = self.ast.to_dict()

        # Can't directly serialize sympy expressions or Python AST
        # Just indicate their presence
        if self.sympy_expr is not None:
            result["has_sympy_expr"] = True

        if self.python_ast is not None:
            result["has_python_ast"] = True

        return result

class SymbolicParser:
    """
    Core symbolic parser for GrandNexus, capable of understanding
    and interpreting various forms of symbolic expressions.

    This parser serves as an interface between textual/string representations
    and structured symbolic forms that can be manipulated, reasoned about,
    and integrated into GrandNexus's cognitive processes.
    """

    def __init__(self, nexus_core: Optional[NexusCore] = None,
                 semantic_memory: Optional[SemanticMemory] = None,
                 working_memory: Optional[WorkingMemory] = None):
        """
        Initialize the symbolic parser.

        Args:
            nexus_core: Reference to the NexusCore for system integration
            semantic_memory: Reference to semantic memory for concept lookup
            working_memory: Reference to working memory for context
        """
        self.logger = logging.getLogger("GrandNexus.Perception.SymbolicParser")
        self.nexus_core = nexus_core
        self.semantic_memory = semantic_memory
        self.working_memory = working_memory

        # Internal state
        self.instance_id = str(uuid.uuid4())[:8]
        self.initialized = False
        self.lock = threading.RLock()

        # Parsers registry for different symbolic types
        self.parsers = {}

        # Recognized symbol registry
        self.known_symbols = {}
        self.symbol_patterns = []

        # Cache recently parsed expressions
        self.expression_cache = {}
        self.expression_history = deque(maxlen=100)

        # Performance metrics
        self.metrics = defaultdict(lambda: deque(maxlen=100))

        self.logger.info(f"SymbolicParser initialized with ID={self.instance_id}")

    def initialize(self) -> bool:
        """Initialize the parser and load parsers for different expression types"""
        if self.initialized:
            return True

        with self.lock:
            try:
                # Register built-in parsers
                self._register_default_parsers()

                # Initialize symbol registry
                self._initialize_symbol_registry()

                # Connect to other modules
                if self.nexus_core:
                    self._register_with_nexus_core()

                self.initialized = True
                self.logger.info("SymbolicParser initialization complete")
                return True
            except Exception as e:
                self.logger.error(f"SymbolicParser initialization failed: {str(e)}")
                return False

    def parse(self, expression: str, expected_type: Optional[SymbolicType] = None,
             context: Optional[Dict[str, Any]] = None) -> ParseResult:
        """
        Parse a symbolic expression into a structured form.

        Args:
            expression: The expression string to parse
            expected_type: Optional hint about the expected expression type
            context: Optional contextual information to aid parsing

        Returns:
            ParseResult containing the parsed expression
        """
        if not self.initialized:
            self.initialize()

        expression = expression.strip()

        # Check cache for identical expression
        cache_key = (expression, str(expected_type), str(sorted(context.items())) if context else None)
        if cache_key in self.expression_cache:
            return self.expression_cache[cache_key]

        # Prepare context if not provided
        if context is None:
            context = {}

        # Determine the expression type if not specified
        expr_type = expected_type if expected_type else self._detect_expression_type(expression, context)

        # Select the appropriate parser
        if expr_type in self.parsers:
            parser_func = self.parsers[expr_type]
            try:
                # Parse the expression
                result = parser_func(expression, context)
                # Update metrics
                self.metrics["parse_success_rate"].append(1.0 if result.success else 0.0)

                # Cache successful results
                if result.success:
                    self.expression_cache[cache_key] = result
                    self.expression_history.append(expression)

                return result
            except Exception as e:
                self.logger.error(f"Error parsing expression '{expression}': {str(e)}")
                self.metrics["parse_success_rate"].append(0.0)
                return ParseResult(
                    success=False,
                    expr_type=expr_type,
                    error_message=f"Parser error: {str(e)}",
                    confidence=0.0
                )
        else:
            self.logger.warning(f"No parser available for expression type: {expr_type}")
            self.metrics["parse_success_rate"].append(0.0)
            return ParseResult(
                success=False,
                expr_type=expr_type,
                error_message=f"No parser available for type: {expr_type}",
                confidence=0.0
            )

    def evaluate(self, parse_result: ParseResult,
                 variable_values: Optional[Dict[str, Any]] = None,
                 context: Optional[Dict[str, Any]] = None) -> Any:
        """
        Evaluate a parsed expression with given variable values.

        Args:
            parse_result: The parsed expression to evaluate
            variable_values: Dictionary mapping variable names to values
            context: Additional contextual information for evaluation

        Returns:
            The result of evaluation, or None if evaluation fails
        """
        if not parse_result.success:
            self.logger.warning("Cannot evaluate unsuccessful parse result")
            return None

        # Prepare variable values if not provided
        if variable_values is None:
            variable_values = {}

        # Prepare context if not provided
        if context is None:
            context = {}

        try:
            # Handle different expression types
            if parse_result.expr_type == SymbolicType.MATHEMATICAL:
                return self._evaluate_mathematical(parse_result, variable_values, context)
            elif parse_result.expr_type == SymbolicType.LOGICAL:
                return self._evaluate_logical(parse_result, variable_values, context)
            elif parse_result.expr_type == SymbolicType.CODE:
                return self._evaluate_code(parse_result, variable_values, context)
            elif parse_result.expr_type == SymbolicType.PATTERN:
                return self._evaluate_pattern(parse_result, variable_values, context)
            else:
                self.logger.warning(f"Evaluation not supported for type: {parse_result.expr_type}")
                return None
        except Exception as e:
            self.logger.error(f"Error evaluating expression: {str(e)}")
            return None

    def transform(self, parse_result: ParseResult,
                 transformation: str,
                 context: Optional[Dict[str, Any]] = None) -> ParseResult:
        """
        Apply a transformation to a parsed expression.

        Args:
            parse_result: The parsed expression to transform
            transformation: The type of transformation to apply (e.g., "simplify", "factor")
            context: Additional contextual information for transformation

        Returns:
            A new ParseResult with the transformed expression
        """
        if not parse_result.success:
            self.logger.warning("Cannot transform unsuccessful parse result")
            return parse_result

        # Prepare context if not provided
        if context is None:
            context = {}

        try:
            # Handle different expression types
            if parse_result.expr_type == SymbolicType.MATHEMATICAL:
                return self._transform_mathematical(parse_result, transformation, context)
            elif parse_result.expr_type == SymbolicType.LOGICAL:
                return self._transform_logical(parse_result, transformation, context)
            else:
                self.logger.warning(f"Transformation not supported for type: {parse_result.expr_type}")
                return parse_result
        except Exception as e:
            self.logger.error(f"Error transforming expression: {str(e)}")
            return parse_result

    def find_symbols(self, expression: str) -> List[Dict[str, Any]]:
        """
        Extract symbols (variables, constants, functions) from an expression.

        Args:
            expression: The expression string to analyze

        Returns:
            List of dictionaries containing symbol information
        """
        # Parse the expression first
        parse_result = self.parse(expression)
        if not parse_result.success:
            return []

        symbols = []
        try:
            # Extract symbols based on expression type
            if parse_result.expr_type == SymbolicType.MATHEMATICAL and parse_result.sympy_expr is not None:
                # Extract symbols from sympy expression
                if HAS_SYMPY:
                    sympy_symbols = list(parse_result.sympy_expr.free_symbols)
                    symbols = [
                        {
                            "name": str(sym),
                            "type": "variable",
                            "metadata": {
                                "is_known": str(sym) in self.known_symbols
                            }
                        }
                        for sym in sympy_symbols
                    ]
            elif parse_result.ast is not None:
                # Extract symbols from AST by traversing
                found_symbols = set()

                def collect_symbols(node):
                    if node.node_type == "variable":
                        found_symbols.add(node.value)

                parse_result.ast.traverse(collect_symbols)

                symbols = [
                    {
                        "name": sym,
                        "type": "variable",
                        "metadata": {
                            "is_known": sym in self.known_symbols
                        }
                    }
                    for sym in found_symbols
                ]
        except Exception as e:
            self.logger.error(f"Error extracting symbols: {str(e)}")

        return symbols

    def register_symbol(self, name: str, symbol_type: str,
                       value: Optional[Any] = None,
                       metadata: Optional[Dict[str, Any]] = None) -> bool:
        """
        Register a known symbol for future reference.

        Args:
            name: The symbol name
            symbol_type: Type of symbol (variable, constant, function, etc.)
            value: Optional value or definition
            metadata: Additional information about the symbol

        Returns:
            True if registration succeeded, False otherwise
        """
        with self.lock:
            if metadata is None:
                metadata = {}

            self.known_symbols[name] = {
                "type": symbol_type,
                "value": value,
                "metadata": metadata
            }

            self.logger.info(f"Registered symbol: {name} ({symbol_type})")
            return True

    def register_parser(self, expr_type: SymbolicType,
                       parser_func: Callable[[str, Dict[str, Any]], ParseResult]) -> None:
        """
        Register a custom parser for a specific expression type.

        Args:
            expr_type: The type of expression this parser handles
            parser_func: Function that takes (expression, context) and returns ParseResult
        """
        with self.lock:
            self.parsers[expr_type] = parser_func
            self.logger.info(f"Registered parser for expression type: {expr_type}")

    def get_parsing_metrics(self) -> Dict[str, Any]:
        """Get current parsing performance metrics"""
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

    # -----------------------------------------------------------------------
    # Internal methods
    # -----------------------------------------------------------------------

    def _register_default_parsers(self) -> None:
        """Register built-in parsers for different expression types"""
        self.register_parser(SymbolicType.MATHEMATICAL, self._parse_mathematical)
        self.register_parser(SymbolicType.LOGICAL, self._parse_logical)
        self.register_parser(SymbolicType.CODE, self._parse_code)
        self.register_parser(SymbolicType.PATTERN, self._parse_pattern)
        self.register_parser(SymbolicType.CONSTRAINT, self._parse_constraint)

    def _initialize_symbol_registry(self) -> None:
        """Initialize registry with common mathematical and logical symbols"""
        # Mathematical constants
        self.register_symbol("pi", "constant", math.pi, {"description": "Ratio of circumference to diameter"})
        self.register_symbol("e", "constant", math.e, {"description": "Base of natural logarithm"})

        # Mathematical functions
        self.register_symbol("sin", "function", math.sin, {"description": "Sine function"})
        self.register_symbol("cos", "function", math.cos, {"description": "Cosine function"})
        self.register_symbol("tan", "function", math.tan, {"description": "Tangent function"})
        self.register_symbol("log", "function", math.log, {"description": "Natural logarithm"})
        self.register_symbol("exp", "function", math.exp, {"description": "Exponential function"})

        # Logical operators
        self.register_symbol("and", "operator", lambda x, y: x and y, {"description": "Logical AND"})
        self.register_symbol("or", "operator", lambda x, y: x or y, {"description": "Logical OR"})
        self.register_symbol("not", "operator", lambda x: not x, {"description": "Logical NOT"})

        # Common symbol patterns
        self.symbol_patterns = [
            (r'\b[a-zA-Z]\b', "single_letter_variable"),
            (r'\b[a-zA-Z][0-9]\b', "indexed_variable"),
            (r'\b[a-zA-Z][a-zA-Z0-9_]*\b', "identifier")
        ]

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

            self.nexus_core.register_module(
                name="symbolic_parser",
                module=self,
                dependencies=dependencies
            )

            self.logger.info("Successfully registered with NexusCore")
        except Exception as e:
            self.logger.error(f"Failed to register with NexusCore: {str(e)}")

    def _detect_expression_type(self, expression: str, context: Dict[str, Any]) -> SymbolicType:
        """
        Determine the type of symbolic expression.

        Args:
            expression: The expression string to analyze
            context: Contextual information that might help with detection

        Returns:
            SymbolicType indicating the detected type
        """
        # Check context for explicit type hint
        if context and "expr_type" in context:
            hint = context["expr_type"]
            if isinstance(hint, SymbolicType):
                return hint
            elif isinstance(hint, str) and hasattr(SymbolicType, hint.upper()):
                return getattr(SymbolicType, hint.upper())

        # Apply heuristics to detect type

        # Check for mathematical expressions
        if re.search(r'[-+*/^=<>()]|\b(sin|cos|tan|log|exp)\b', expression):
            return SymbolicType.MATHEMATICAL

        # Check for logical expressions
        if re.search(r'\b(and|or|not|if|then|implies|forall|exists)\b|[∧∨¬⟹∀∃]', expression, re.IGNORECASE):
            return SymbolicType.LOGICAL

        # Check for code-like expressions
        if re.search(r'[;{}]|\b(function|def|return|for|while)\b', expression):
            return SymbolicType.CODE

        # Check for pattern expressions
        if re.search(r'[*?+]|\{\d+,\d*\}|\[\w+\]', expression):
            return SymbolicType.PATTERN

        # Default to mathematical if it contains common operators
        if re.search(r'[-+*/]', expression):
            return SymbolicType.MATHEMATICAL

        return SymbolicType.UNKNOWN

    def _parse_mathematical(self, expression: str, context: Dict[str, Any]) -> ParseResult:
        """
        Parse a mathematical expression.

        Args:
            expression: The expression string to parse
            context: Additional context for parsing

        Returns:
            ParseResult containing the parsed mathematical expression
        """
        # Use SymPy for advanced mathematical parsing if available
        if HAS_SYMPY:
            try:
                # Parse with sympy
                sympy_expr = parse_expr(expression, transformations=standard_transformations)

                # Create a basic AST representation
                ast_root = self._sympy_to_ast(sympy_expr)

                return ParseResult(
                    success=True,
                    expr_type=SymbolicType.MATHEMATICAL,
                    ast=ast_root,
                    sympy_expr=sympy_expr,
                    confidence=1.0
                )
            except Exception as e:
                self.logger.warning(f"SymPy parsing failed: {str(e)}, falling back to basic parsing")

        # Fall back to basic parsing if SymPy is not available or fails
        try:
            ast_root = self._basic_math_parse(expression)

            return ParseResult(
                success=True,
                expr_type=SymbolicType.MATHEMATICAL,
                ast=ast_root,
                confidence=0.8,  # Lower confidence for basic parsing
                metadata={"parser": "basic"}
            )
        except Exception as e:
            self.logger.error(f"Basic math parsing failed: {str(e)}")
            return ParseResult(
                success=False,
                expr_type=SymbolicType.MATHEMATICAL,
                error_message=f"Parsing error: {str(e)}",
                confidence=0.0
            )

    def _parse_logical(self, expression: str, context: Dict[str, Any]) -> ParseResult:
        """
        Parse a logical expression.

        Args:
            expression: The expression string to parse
            context: Additional context for parsing

        Returns:
            ParseResult containing the parsed logical expression
        """
        # Normalize logical operators
        normalized = expression.lower()
        normalized = normalized.replace("&&", " and ")
        normalized = normalized.replace("||", " or ")
        normalized = normalized.replace("!", "not ")

        try:
            # Try to parse as Python expression if it's simple enough
            try:
                # Convert to a Python boolean expression
                # This is a simplistic approach - a real implementation would use a proper logical parser
                python_expr = normalized
                python_expr = re.sub(r'\b(implies|⟹)\b', "<=", python_expr)
                python_expr = re.sub(r'\b(iff|⟺)\b', "==", python_expr)

                # Parse with Python's ast
                python_ast = ast.parse(python_expr, mode='eval')

                # Convert to our AST format
                ast_root = self._python_ast_to_symbolic_ast(python_ast)

                return ParseResult(
                    success=True,
                    expr_type=SymbolicType.LOGICAL,
                    ast=ast_root,
                    python_ast=python_ast,
                    confidence=0.9,
                    metadata={"parser": "python_ast"}
                )
            except SyntaxError:
                # Fall back to basic parsing
                raise ValueError("Not parsable as Python expression")

            # A real implementation would have a proper logical expression parser here
            # For now, we'll just create a basic AST for demonstration

        except Exception as e:
            self.logger.error(f"Logical parsing failed: {str(e)}")

            # Create a very basic AST even if parsing fails
            # In a real implementation, this would be more sophisticated
            ast_root = SymbolicAST(node_type="expression", value=expression)

            return ParseResult(
                success=True,  # Let's say it succeeds but with low confidence
                expr_type=SymbolicType.LOGICAL,
                ast=ast_root,
                confidence=0.5,
                metadata={"parser": "basic", "warning": str(e)}
            )

    def _parse_code(self, expression: str, context: Dict[str, Any]) -> ParseResult:
        """
        Parse a code expression or snippet.

        Args:
            expression: The code string to parse
            context: Additional context for parsing

        Returns:
            ParseResult containing the parsed code
        """
        try:
            # Try to parse as Python code
            python_ast = ast.parse(expression)

            # Convert to our AST format
            ast_root = self._python_ast_to_symbolic_ast(python_ast)

            return ParseResult(
                success=True,
                expr_type=SymbolicType.CODE,
                ast=ast_root,
                python_ast=python_ast,
                confidence=0.9,
                metadata={"language": "python"}
            )
        except SyntaxError as e:
            # Not valid Python code
            self.logger.info(f"Code is not valid Python: {str(e)}")

            # For other languages, we would have additional parsers
            # For now, just create a generic code AST
            ast_root = SymbolicAST(node_type="code", value=expression)

            return ParseResult(
                success=True,
                expr_type=SymbolicType.CODE,
                ast=ast_root,
                confidence=0.5,
                metadata={"language": "unknown"}
            )
        except Exception as e:
            self.logger.error(f"Code parsing failed: {str(e)}")
            return ParseResult(
                success=False,
                expr_type=SymbolicType.CODE,
                error_message=f"Parsing error: {str(e)}",
                confidence=0.0
            )

    def _parse_pattern(self, expression: str, context: Dict[str, Any]) -> ParseResult:
        """
        Parse a pattern expression.

        Args:
            expression: The pattern string to parse
            context: Additional context for parsing

        Returns:
            ParseResult containing the parsed pattern
        """
        try:
            # Check if it looks like a regex pattern
            if re.search(r'[*+?{}()\[\]\|\.]', expression):
                # Try to compile as regex to validate
                try:
                    re.compile(expression)
                    pattern_type = "regex"
                    confidence = 0.9
                except re.error:
                    pattern_type = "invalid_regex"
                    confidence = 0.3
            else:
                # Treat as a simple glob-like pattern
                pattern_type = "glob"
                confidence = 0.7

            # Create a basic AST for the pattern
            ast_root = SymbolicAST(
                node_type="pattern",
                value=expression,
                metadata={"pattern_type": pattern_type}
            )

            return ParseResult(
                success=True,
                expr_type=SymbolicType.PATTERN,
                ast=ast_root,
                confidence=confidence,
                metadata={"pattern_type": pattern_type}
            )
        except Exception as e:
            self.logger.error(f"Pattern parsing failed: {str(e)}")
            return ParseResult(
                success=False,
                expr_type=SymbolicType.PATTERN,
                error_message=f"Parsing error: {str(e)}",
                confidence=0.0
            )

    def _parse_constraint(self, expression: str, context: Dict[str, Any]) -> ParseResult:
        """
        Parse a constraint expression.

        Args:
            expression: The constraint string to parse
            context: Additional context for parsing

        Returns:
            ParseResult containing the parsed constraint
        """
        # Constraints are often mathematical equalities or inequalities
        if re.search(r'[<>=]', expression):
            # Try to parse as mathematical expression first
            math_result = self._parse_mathematical(expression, context)
            if math_result.success:
                # It's a mathematical constraint
                math_result.expr_type = SymbolicType.CONSTRAINT
                math_result.metadata["constraint_type"] = "mathematical"
                return math_result

        # For other types of constraints, a more specialized parser would be used
        # For now, create a basic constraint AST
        ast_root = SymbolicAST(node_type="constraint", value=expression)

        return ParseResult(
            success=True,
            expr_type=SymbolicType.CONSTRAINT,
            ast=ast_root,
            confidence=0.6,
            metadata={"constraint_type": "general"}
        )

    def _sympy_to_ast(self, sympy_expr) -> SymbolicAST:
        """
        Convert a SymPy expression to our internal AST format.

        Args:
            sympy_expr: The SymPy expression to convert

        Returns:
            SymbolicAST representation of the expression
        """
        if not HAS_SYMPY:
            raise ImportError("SymPy is not available")

        # Handle different types of expressions
        if sympy_expr.is_number:
            return SymbolicAST(node_type="number", value=float(sympy_expr))

        elif sympy_expr.is_symbol:
            return SymbolicAST(node_type="variable", value=str(sympy_expr))

        elif sympy_expr.is_Add:
            # Addition operation
            ast_node = SymbolicAST(node_type="operation", value="add")
            for arg in sympy_expr.args:
                ast_node.children.append(self._sympy_to_ast(arg))
            return ast_node

        elif sympy_expr.is_Mul:
            # Multiplication operation
            ast_node = SymbolicAST(node_type="operation", value="multiply")
            for arg in sympy_expr.args:
                ast_node.children.append(self._sympy_to_ast(arg))
            return ast_node

        elif sympy_expr.is_Pow:
            # Power operation
            ast_node = SymbolicAST(node_type="operation", value="power")
            # Add base and exponent as children
            ast_node.children.append(self._sympy_to_ast(sympy_expr.args[0]))
            ast_node.children.append(self._sympy_to_ast(sympy_expr.args[1]))
            return ast_node

        elif isinstance(sympy_expr, sympy.Function):
            # Function call
            func_name = type(sympy_expr).__name__.lower()
            ast_node = SymbolicAST(node_type="function", value=func_name)
            # Add arguments as children
            for arg in sympy_expr.args:
                ast_node.children.append(self._sympy_to_ast(arg))
            return ast_node

        # Handle equations
        elif isinstance(sympy_expr, sympy.Eq):
            ast_node = SymbolicAST(node_type="equation", value="equals")
            # Add left and right sides as children
            ast_node.children.append(self._sympy_to_ast(sympy_expr.args[0]))
            ast_node.children.append(self._sympy_to_ast(sympy_expr.args[1]))
            return ast_node

        # Handle inequalities
        elif isinstance(sympy_expr, sympy.Gt):
            ast_node = SymbolicAST(node_type="inequality", value="greater_than")
            ast_node.children.append(self._sympy_to_ast(sympy_expr.args[0]))
            ast_node.children.append(self._sympy_to_ast(sympy_expr.args[1]))
            return ast_node

        elif isinstance(sympy_expr, sympy.Lt):
            ast_node = SymbolicAST(node_type="inequality", value="less_than")
            ast_node.children.append(self._sympy_to_ast(sympy_expr.args[0]))
            ast_node.children.append(self._sympy_to_ast(sympy_expr.args[1]))
            return ast_node

        elif isinstance(sympy_expr, sympy.Ge):
            ast_node = SymbolicAST(node_type="inequality", value="greater_equal")
            ast_node.children.append(self._sympy_to_ast(sympy_expr.args[0]))
            ast_node.children.append(self._sympy_to_ast(sympy_expr.args[1]))
            return ast_node

        elif isinstance(sympy_expr, sympy.Le):
            ast_node = SymbolicAST(node_type="inequality", value="less_equal")
            ast_node.children.append(self._sympy_to_ast(sympy_expr.args[0]))
            ast_node.children.append(self._sympy_to_ast(sympy_expr.args[1]))
            return ast_node

        else:
            # Generic expression, just convert to string
            ast_node = SymbolicAST(node_type="expression", value=str(sympy_expr))
            return ast_node

    def _basic_math_parse(self, expression: str) -> SymbolicAST:
        """
        Simple recursive descent parser for basic mathematical expressions.
        Only handles +, -, *, /, ^, and parentheses.

        Args:
            expression: The expression string to parse

        Returns:
            SymbolicAST representation of the expression
        """
        # Remove whitespace
        expression = expression.strip()

        # Handle empty expression
        if not expression:
            return SymbolicAST(node_type="empty", value="")

        # Handle parenthesized expressions
        if expression.startswith('(') and expression.endswith(')'):
            # Check if the parentheses are balanced and outer-most
            count = 0
            for i, char in enumerate(expression):
                if char == '(':
                    count += 1
                elif char == ')':
                    count -= 1
                if count == 0 and i < len(expression) - 1:
                    # Found a closing parenthesis before the end, not outer-most
                    break
            else:
                # Parentheses are outer-most, parse the inner expression
                return self._basic_math_parse(expression[1:-1])

        # Look for operators: + and - (lowest precedence)
        count = 0
        for i in range(len(expression) - 1, -1, -1):  # Scan right to left
            char = expression[i]
            if char == '(':
                count += 1
            elif char == ')':
                count -= 1
            elif count == 0 and char in '+-' and i > 0 and expression[i-1] not in '+-*/^(':
                # Found an operator outside of parentheses
                # Make sure it's not a unary operator or part of another operator
                left = expression[:i]
                right = expression[i+1:]

                op_node = SymbolicAST(
                    node_type="operation",
                    value="add" if char == '+' else "subtract"
                )
                op_node.children.append(self._basic_math_parse(left))
                op_node.children.append(self._basic_math_parse(right))
                return op_node

        # No + or - found, look for * and / (medium precedence)
        count = 0
        for i in range(len(expression) - 1, -1, -1):  # Scan right to left
            char = expression[i]
            if char == '(':
                count += 1
            elif char == ')':
                count -= 1
            elif count == 0 and char in '*/':
                # Found an operator outside of parentheses
                left = expression[:i]
                right = expression[i+1:]

                op_node = SymbolicAST(
                    node_type="operation",
                    value="multiply" if char == '*' else "divide"
                )
                op_node.children.append(self._basic_math_parse(left))
                op_node.children.append(self._basic_math_parse(right))
                return op_node

        # No +, -, *, / found, look for ^ (highest precedence)
        count = 0
        for i in range(len(expression) - 1, -1, -1):  # Scan right to left
            char = expression[i]
            if char == '(':
                count += 1
            elif char == ')':
                count -= 1
            elif count == 0 and char == '^':
                # Found an operator outside of parentheses
                left = expression[:i]
                right = expression[i+1:]

                op_node = SymbolicAST(
                    node_type="operation",
                    value="power"
                )
                op_node.children.append(self._basic_math_parse(left))
                op_node.children.append(self._basic_math_parse(right))
                return op_node

        # No operators found, check for functions
        match = re.match(r'(\w+)\((.+)\)$', expression)
        if match:
            func_name = match.group(1)
            args_str = match.group(2)

            # Parse arguments (simple comma-separated list)
            # This is naive and doesn't handle nested functions properly
            args = []
            current_arg = ""
            paren_count = 0

            for char in args_str:
                if char == '(':
                    paren_count += 1
                    current_arg += char
                elif char == ')':
                    paren_count -= 1
                    current_arg += char
                elif char == ',' and paren_count == 0:
                    args.append(current_arg.strip())
                    current_arg = ""
                else:
                    current_arg += char

            if current_arg:
                args.append(current_arg.strip())

            # Create function node
            func_node = SymbolicAST(node_type="function", value=func_name)

            # Parse each argument
            for arg in args:
                func_node.children.append(self._basic_math_parse(arg))

            return func_node

        # No operators or functions, check for number
        try:
            value = float(expression)
            return SymbolicAST(node_type="number", value=value)
        except ValueError:
            pass

        # Must be a variable
        return SymbolicAST(node_type="variable", value=expression)

    def _python_ast_to_symbolic_ast(self, python_ast) -> SymbolicAST:
        """
        Convert a Python AST to our internal AST format.

        Args:
            python_ast: The Python AST to convert

        Returns:
            SymbolicAST representation of the expression
        """
        # Handle different types of expressions

        if isinstance(python_ast, ast.Module):
            # Multiple statements
            module_node = SymbolicAST(node_type="module", value="module")
            for statement in python_ast.body:
                module_node.children.append(self._python_ast_to_symbolic_ast(statement))
            return module_node

        elif isinstance(python_ast, ast.Expr):
            # Expression statement
            return self._python_ast_to_symbolic_ast(python_ast.value)

        elif isinstance(python_ast, ast.Expression):
            # Expression wrapper
            return self._python_ast_to_symbolic_ast(python_ast.body)

        elif isinstance(python_ast, ast.Constant):
            # Literal value (Python 3.8+)
            if isinstance(python_ast.value, bool):
                return SymbolicAST(node_type="boolean", value=python_ast.value)
            elif isinstance(python_ast.value, (int, float)):
                return SymbolicAST(node_type="number", value=python_ast.value)
            elif isinstance(python_ast.value, str):
                return SymbolicAST(node_type="string", value=python_ast.value)
            else:
                return SymbolicAST(node_type="constant", value=str(python_ast.value))

        elif isinstance(python_ast, ast.Num):
            # Number literal (older Python versions)
            return SymbolicAST(node_type="number", value=python_ast.n)

        elif isinstance(python_ast, ast.Str):
            # String literal (older Python versions)
            return SymbolicAST(node_type="string", value=python_ast.s)

        elif isinstance(python_ast, ast.NameConstant):
            # Name constant (older Python versions)
            return SymbolicAST(node_type="constant", value=str(python_ast.value))

        elif isinstance(python_ast, ast.Name):
            # Variable name
            return SymbolicAST(node_type="variable", value=python_ast.id)

        elif isinstance(python_ast, ast.BinOp):
            # Binary operation
            op_map = {
                ast.Add: "add",
                ast.Sub: "subtract",
                ast.Mult: "multiply",
                ast.Div: "divide",
                ast.Pow: "power",
                ast.Mod: "modulo",
                ast.FloorDiv: "floor_divide",
            }

            op_type = type(python_ast.op)
            op_name = op_map.get(op_type, op_type.__name__.lower())

            op_node = SymbolicAST(node_type="operation", value=op_name)
            op_node.children.append(self._python_ast_to_symbolic_ast(python_ast.left))
            op_node.children.append(self._python_ast_to_symbolic_ast(python_ast.right))

            return op_node

        elif isinstance(python_ast, ast.UnaryOp):
            # Unary operation
            op_map = {
                ast.USub: "negate",
                ast.UAdd: "positive",
                ast.Not: "not",
                ast.Invert: "invert",
            }

            op_type = type(python_ast.op)
            op_name = op_map.get(op_type, op_type.__name__.lower())

            op_node = SymbolicAST(node_type="operation", value=op_name)
            op_node.children.append(self._python_ast_to_symbolic_ast(python_ast.operand))

            return op_node

        elif isinstance(python_ast, ast.Compare):
            # Comparison operation
            op_map = {
                ast.Eq: "equals",
                ast.NotEq: "not_equals",
                ast.Lt: "less_than",
                ast.LtE: "less_equal",
                ast.Gt: "greater_than",
                ast.GtE: "greater_equal",
                ast.Is: "is",
                ast.IsNot: "is_not",
                ast.In: "in",
                ast.NotIn: "not_in",
            }

            # Handle multiple comparisons (e.g., a < b < c)
            if len(python_ast.ops) == 1:
                # Simple comparison
                op_type = type(python_ast.ops[0])
                op_name = op_map.get(op_type, op_type.__name__.lower())

                if op_name in ["equals", "not_equals"]:
                    node_type = "equation"
                else:
                    node_type = "inequality"

                op_node = SymbolicAST(node_type=node_type, value=op_name)
                op_node.children.append(self._python_ast_to_symbolic_ast(python_ast.left))
                op_node.children.append(self._python_ast_to_symbolic_ast(python_ast.comparators[0]))

                return op_node
            else:
                # Multiple comparisons, create a logical AND of comparisons
                and_node = SymbolicAST(node_type="operation", value="and")

                left = python_ast.left
                for i, op in enumerate(python_ast.ops):
                    op_type = type(op)
                    op_name = op_map.get(op_type, op_type.__name__.lower())

                    if op_name in ["equals", "not_equals"]:
                        node_type = "equation"
                    else:
                        node_type = "inequality"

                    comp_node = SymbolicAST(node_type=node_type, value=op_name)
                    comp_node.children.append(self._python_ast_to_symbolic_ast(left))
                    comp_node.children.append(self._python_ast_to_symbolic_ast(python_ast.comparators[i]))

                    and_node.children.append(comp_node)

                    # For chained comparisons, the right side becomes the left for the next comparison
                    left = python_ast.comparators[i]

                return and_node

        elif isinstance(python_ast, ast.BoolOp):
            # Boolean operation (and, or)
            op_map = {
                ast.And: "and",
                ast.Or: "or",
            }

            op_type = type(python_ast.op)
            op_name = op_map.get(op_type, op_type.__name__.lower())

            op_node = SymbolicAST(node_type="operation", value=op_name)

            # Add all values as children
            for value in python_ast.values:
                op_node.children.append(self._python_ast_to_symbolic_ast(value))

            return op_node

        elif isinstance(python_ast, ast.Call):
            # Function call
            func_node = SymbolicAST(node_type="function", value=self._get_func_name(python_ast.func))

            # Add arguments as children
            for arg in python_ast.args:
                func_node.children.append(self._python_ast_to_symbolic_ast(arg))

            # Add keyword arguments as metadata
            if python_ast.keywords:
                func_node.metadata["keywords"] = {}
                for keyword in python_ast.keywords:
                    key = keyword.arg
                    value_ast = self._python_ast_to_symbolic_ast(keyword.value)
                    func_node.metadata["keywords"][key] = value_ast

            return func_node

        # Add more conversions as needed for other AST node types

        else:
            # Generic node, just convert to string
            return SymbolicAST(node_type="unknown", value=str(python_ast.__class__.__name__))

    def _get_func_name(self, func_node) -> str:
        """
        Extract function name from a func AST node.

        Args:
            func_node: The function node from Python AST

        Returns:
            String representation of the function name
        """
        if isinstance(func_node, ast.Name):
            return func_node.id
        elif isinstance(func_node, ast.Attribute):
            base = self._get_func_name(func_node.value)
            return f"{base}.{func_node.attr}"
        else:
            return str(func_node.__class__.__name__)

    def _evaluate_mathematical(self, parse_result: ParseResult,
                              variable_values: Dict[str, Any],
                              context: Dict[str, Any]) -> Any:
        """
        Evaluate a mathematical expression with given variable values.

        Args:
            parse_result: The parsed mathematical expression
            variable_values: Dictionary mapping variable names to values
            context: Additional contextual information

        Returns:
            The result of evaluation
        """
        if HAS_SYMPY and parse_result.sympy_expr is not None:
            # Evaluate using SymPy
            expr = parse_result.sympy_expr

            # Substitute variables with values
            for var_name, value in variable_values.items():
                if var_name in [str(sym) for sym in expr.free_symbols]:
                    expr = expr.subs(sympy.Symbol(var_name), value)

            # Try to evaluate to a numerical value
            try:
                result = float(expr.evalf())
                return result
            except (TypeError, ValueError):
                # If it can't be converted to float, return the SymPy expression
                return expr

        elif parse_result.ast is not None:
            # Evaluate using our own AST walker
            return self._evaluate_ast(parse_result.ast, variable_values, context)

        else:
            raise ValueError("No evaluable expression found in parse result")

    def _evaluate_logical(self, parse_result: ParseResult,
                         variable_values: Dict[str, Any],
                         context: Dict[str, Any]) -> Any:
        """
        Evaluate a logical expression with given variable values.

        Args:
            parse_result: The parsed logical expression
            variable_values: Dictionary mapping variable names to values
            context: Additional contextual information

        Returns:
            The result of evaluation (typically a boolean)
        """
        if parse_result.ast is not None:
            # Evaluate using our own AST walker
            return self._evaluate_ast(parse_result.ast, variable_values, context)

        elif parse_result.python_ast is not None:
            # Evaluate using Python's eval (use with caution!)
            # This is simplified and not secure for production

            # Create a safe environment with only the variables provided
            safe_env = {k: v for k, v in variable_values.items()}

            # Add basic logical operations
            safe_env.update({
                'and': operator.and_,
                'or': operator.or_,
                'not': operator.not_,
                'True': True,
                'False': False
            })

            # Compile and evaluate the AST
            code = compile(parse_result.python_ast, '<string>', 'eval')
            return eval(code, {"__builtins__": {}}, safe_env)

        else:
            raise ValueError("No evaluable expression found in parse result")

    def _evaluate_code(self, parse_result: ParseResult,
                      variable_values: Dict[str, Any],
                      context: Dict[str, Any]) -> Any:
        """
        Evaluate a code expression with given variable values.

        Args:
            parse_result: The parsed code expression
            variable_values: Dictionary mapping variable names to values
            context: Additional contextual information

        Returns:
            The result of evaluation
        """
        if parse_result.python_ast is not None:
            # This is a simplified and not secure evaluation
            # In a real system, use a sandbox or restricted execution environment

            # Create an environment with only the variables provided
            safe_env = {k: v for k, v in variable_values.items()}

            # Compile and evaluate the AST
            mode = 'exec' if isinstance(parse_result.python_ast, ast.Module) else 'eval'
            code = compile(parse_result.python_ast, '<string>', mode)

            # For 'exec' mode, we need to capture potential return values
            local_env = {}
            exec(code, {"__builtins__": {}}, local_env)

            # Return the updated local environment as the result
            return local_env

        elif parse_result.ast is not None:
            # For non-Python code, we would need a custom interpreter
            # This is just a placeholder
            self.logger.warning("Custom code interpretation not implemented")
            return None

        else:
            raise ValueError("No evaluable code found in parse result")

    def _evaluate_pattern(self, parse_result: ParseResult,
                         variable_values: Dict[str, Any],
                         context: Dict[str, Any]) -> Any:
        """
        Evaluate a pattern expression, typically by matching it against a target.

        Args:
            parse_result: The parsed pattern expression
            variable_values: Dictionary with target string to match against
            context: Additional contextual information

        Returns:
            Match results, typically a boolean or match object
        """
        if not parse_result.ast:
            raise ValueError("No pattern AST found in parse result")

        pattern_type = parse_result.metadata.get("pattern_type", "unknown")
        pattern_str = parse_result.ast.value

        # Get the target string to match against
        target = variable_values.get("target", "")
        if not target:
            raise ValueError("No target string provided for pattern matching")

        if pattern_type == "regex":
            # Regular expression matching
            try:
                match = re.search(pattern_str, target)
                if match:
                    # Return match object details
                    return {
                        "matched": True,
                        "span": match.span(),
                        "groups": match.groups(),
                        "group_dict": match.groupdict()
                    }
                else:
                    return {"matched": False}
            except re.error as e:
                raise ValueError(f"Invalid regex pattern: {str(e)}")

        elif pattern_type == "glob":
            # Simple glob pattern matching
            # Convert glob to regex
            regex = pattern_str.replace("*", ".*").replace("?", ".")
            regex = f"^{regex}$"  # Match whole string

            match = re.match(regex, target)
            return {"matched": bool(match)}

        else:
            raise ValueError(f"Unsupported pattern type: {pattern_type}")

    def _evaluate_ast(self, ast_node: SymbolicAST,
                     variable_values: Dict[str, Any],
                     context: Dict[str, Any]) -> Any:
        """
        Evaluate an AST node with given variable values.

        Args:
            ast_node: The AST node to evaluate
            variable_values: Dictionary mapping variable names to values
            context: Additional contextual information

        Returns:
            The result of evaluation
        """
        # Handle different node types
        if ast_node.node_type == "number":
            return ast_node.value

        elif ast_node.node_type == "boolean":
            return ast_node.value

        elif ast_node.node_type == "string":
            return ast_node.value

        elif ast_node.node_type == "variable":
            var_name = ast_node.value
            if var_name in variable_values:
                return variable_values[var_name]
            elif var_name in self.known_symbols:
                symbol_info = self.known_symbols[var_name]
                if "value" in symbol_info:
                    return symbol_info["value"]

            raise ValueError(f"Unknown variable: {var_name}")

        elif ast_node.node_type == "operation":
            op = ast_node.value

            if op == "add":
                if len(ast_node.children) != 2:
                    raise ValueError("Add operation requires exactly 2 operands")
                left = self._evaluate_ast(ast_node.children[0], variable_values, context)
                right = self._evaluate_ast(ast_node.children[1], variable_values, context)
                return left + right

            elif op == "subtract":
                if len(ast_node.children) != 2:
                    raise ValueError("Subtract operation requires exactly 2 operands")
                left = self._evaluate_ast(ast_node.children[0], variable_values, context)
                right = self._evaluate_ast(ast_node.children[1], variable_values, context)
                return left - right

            elif op == "multiply":
                if len(ast_node.children) != 2:
                    raise ValueError("Multiply operation requires exactly 2 operands")
                left = self._evaluate_ast(ast_node.children[0], variable_values, context)
                right = self._evaluate_ast(ast_node.children[1], variable_values, context)
                return left * right

            elif op == "divide":
                if len(ast_node.children) != 2:
                    raise ValueError("Divide operation requires exactly 2 operands")
                left = self._evaluate_ast(ast_node.children[0], variable_values, context)
                right = self._evaluate_ast(ast_node.children[1], variable_values, context)
                return left / right

            elif op == "power":
                if len(ast_node.children) != 2:
                    raise ValueError("Power operation requires exactly 2 operands")
                base = self._evaluate_ast(ast_node.children[0], variable_values, context)
                exponent = self._evaluate_ast(ast_node.children[1], variable_values, context)
                return base ** exponent

            elif op == "negate":
                if len(ast_node.children) != 1:
                    raise ValueError("Negate operation requires exactly 1 operand")
                operand = self._evaluate_ast(ast_node.children[0], variable_values, context)
                return -operand

            elif op == "and":
                results = [self._evaluate_ast(child, variable_values, context) for child in ast_node.children]
                return all(results)

            elif op == "or":
                results = [self._evaluate_ast(child, variable_values, context) for child in ast_node.children]
                return any(results)

            elif op == "not":
                if len(ast_node.children) != 1:
                    raise ValueError("Not operation requires exactly 1 operand")
                operand = self._evaluate_ast(ast_node.children[0], variable_values, context)
                return not operand

            else:
                raise ValueError(f"Unsupported operation: {op}")

        elif ast_node.node_type == "function":
            func_name = ast_node.value
            args = [self._evaluate_ast(child, variable_values, context) for child in ast_node.children]

            # Check if function is in known symbols
            if func_name in self.known_symbols:
                symbol_info = self.known_symbols[func_name]
                if symbol_info["type"] == "function" and "value" in symbol_info:
                    func = symbol_info["value"]
                    return func(*args)

            # Check built-in functions
            if func_name == "sin":
                return math.sin(args[0])
            elif func_name == "cos":
                return math.cos(args[0])
            elif func_name == "tan":
                return math.tan(args[0])
            elif func_name == "log":
                return math.log(args[0])
            elif func_name == "exp":
                return math.exp(args[0])
            else:
                raise ValueError(f"Unknown function: {func_name}")

        elif ast_node.node_type in ["equation", "inequality"]:
            op = ast_node.value

            if len(ast_node.children) != 2:
                raise ValueError(f"{op} requires exactly 2 operands")

            left = self._evaluate_ast(ast_node.children[0], variable_values, context)
            right = self._evaluate_ast(ast_node.children[1], variable_values, context)

            if op == "equals":
                return left == right
            elif op == "not_equals":
                return left != right
            elif op == "less_than":
                return left < right
            elif op == "less_equal":
                return left <= right
            elif op == "greater_than":
                return left > right
            elif op == "greater_equal":
                return left >= right
            else:
                raise ValueError(f"Unsupported comparison: {op}")

        else:
            raise ValueError(f"Unsupported node type: {ast_node.node_type}")

    def _transform_mathematical(self, parse_result: ParseResult,
                               transformation: str,
                               context: Dict[str, Any]) -> ParseResult:
        """
        Apply a transformation to a mathematical expression.

        Args:
            parse_result: The parsed mathematical expression
            transformation: The type of transformation to apply
            context: Additional contextual information

        Returns:
            A new ParseResult with the transformed expression
        """
        if not HAS_SYMPY or parse_result.sympy_expr is None:
            self.logger.warning("SymPy is required for mathematical transformations")
            return parse_result

        try:
            expr = parse_result.sympy_expr

            # Apply the requested transformation
            if transformation == "simplify":
                result = sympy.simplify(expr)
            elif transformation == "expand":
                result = sympy.expand(expr)
            elif transformation == "factor":
                result = sympy.factor(expr)
            elif transformation == "collect":
                if "var" in context:
                    var_name = context["var"]
                    var = sympy.Symbol(var_name)
                    result = sympy.collect(expr, var)
                else:
                    raise ValueError("Collect transformation requires 'var' in context")
            elif transformation == "solve":
                if isinstance(expr, sympy.Eq):
                    # Solve equation for x by default, or specified variable
                    var_name = context.get("var", "x")
                    var = sympy.Symbol(var_name)
                    result = sympy.solve(expr, var)

                    # Create a new result with the solution
                    # This is a bit special since we're changing the expression type
                    new_ast = SymbolicAST(node_type="solution", value=str(result))
                    return ParseResult(
                        success=True,
                        expr_type=SymbolicType.MATHEMATICAL,
                        ast=new_ast,
                        sympy_expr=result,  # This will be a list for solve, not an expr
                        confidence=parse_result.confidence,
                        metadata={**parse_result.metadata, "transformation": transformation}
                    )
                else:
                    raise ValueError("Solve transformation requires an equation")
            else:
                raise ValueError(f"Unsupported transformation: {transformation}")

            # Create a new AST from the transformed expression
            new_ast = self._sympy_to_ast(result)

            # Create a new ParseResult with the transformed expression
            return ParseResult(
                success=True,
                expr_type=parse_result.expr_type,
                ast=new_ast,
                sympy_expr=result,
                confidence=parse_result.confidence,
                metadata={**parse_result.metadata, "transformation": transformation}
            )

        except Exception as e:
            self.logger.error(f"Error applying transformation '{transformation}': {str(e)}")
            return parse_result

    def _transform_logical(self, parse_result: ParseResult,
                          transformation: str,
                          context: Dict[str, Any]) -> ParseResult:
        """
        Apply a transformation to a logical expression.

        Args:
            parse_result: The parsed logical expression
            transformation: The type of transformation to apply
            context: Additional contextual information

        Returns:
            A new ParseResult with the transformed expression
        """
        # This is a simplified implementation
        # A real implementation would use a logical transformation library

        if not parse_result.ast:
            self.logger.warning("AST required for logical transformations")
            return parse_result

        try:
            # Very basic transformations that don't really transform anything
            # In a real system, this would use proper logical equivalence rules

            if transformation == "negate":
                # Create a NOT wrapper around the expression
                new_ast = SymbolicAST(node_type="operation", value="not")
                new_ast.children.append(parse_result.ast)

                return ParseResult(
                    success=True,
                    expr_type=parse_result.expr_type,
                    ast=new_ast,
                    confidence=parse_result.confidence,
                    metadata={**parse_result.metadata, "transformation": transformation}
                )

            elif transformation == "distribute":
                # This would actually distribute AND over OR or vice versa
                # But we'll just return the original for now
                self.logger.warning("Logical distribution not fully implemented")
                return parse_result

            else:
                raise ValueError(f"Unsupported logical transformation: {transformation}")

        except Exception as e:
            self.logger.error(f"Error applying logical transformation '{transformation}': {str(e)}")
            return parse_result