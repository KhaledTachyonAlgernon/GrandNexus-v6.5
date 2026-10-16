"""
code_analyzer.py - Code Analysis Module for GrandNexus

This module provides functionality for analyzing, understanding, and processing
code structures across various programming languages. It serves as a specialized
component of the perception system, focusing on code as a form of structured input.

The module supports:
- Code structure analysis (AST generation and traversal)
- Semantic understanding of code components
- Pattern recognition in coding structures
- Integration with symbolic and neural reasoning systems
- Cross-language code analysis (Python, JavaScript, etc.)

Key design principles:
- Language-agnostic where possible, specialized where necessary
- Progressive understanding (from syntax to semantics to intent)
- Support for both static and dynamic code analysis
- Extensible plugin system for different programming languages
"""

import math  # fix 6.5: provided earlier by the notebook's shared namespace
import logging
import ast
import re
import uuid
import threading
import time
from enum import Enum, auto
from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional, Set, Tuple, Union, Callable
from collections import defaultdict, deque

# Assuming these exist or will exist in the GrandNexus architecture
from grandnexus.core.nexus_core import NexusCore
from grandnexus.memory.semantic import SemanticMemory
from grandnexus.memory.working import WorkingMemory

# Try to import optional dependencies
try:
    import astroid
    HAS_ASTROID = True
except ImportError:
    HAS_ASTROID = False

try:
    import esprima
    HAS_ESPRIMA = True
except ImportError:
    HAS_ESPRIMA = False

class CodeLanguage(Enum):
    """Supported programming languages for code analysis"""
    PYTHON = auto()
    JAVASCRIPT = auto()
    JAVA = auto()
    CPP = auto()
    RUST = auto()
    GO = auto()
    RUBY = auto()
    PHP = auto()
    CSHARP = auto()
    TYPESCRIPT = auto()
    HTML = auto()
    CSS = auto()
    SQL = auto()
    SHELL = auto()
    UNKNOWN = auto()

class CodeNodeType(Enum):
    """Types of code nodes in the unified AST representation"""
    MODULE = auto()          # Complete module/file
    FUNCTION = auto()        # Function/method definition
    CLASS = auto()           # Class/struct definition
    VARIABLE = auto()        # Variable declaration/reference
    EXPRESSION = auto()      # Expression
    STATEMENT = auto()       # Statement (e.g., assignment)
    CONTROL_FLOW = auto()    # Control flow (if, for, while)
    IMPORT = auto()          # Import/include statement
    LITERAL = auto()         # Literal value (string, number)
    OPERATOR = auto()        # Operator (+, -, etc.)
    COMMENT = auto()         # Comment
    DECORATOR = auto()       # Decorator/annotation
    ARGUMENT = auto()        # Function argument
    ATTRIBUTE = auto()       # Object attribute/property
    CALL = auto()            # Function/method call
    BLOCK = auto()           # Code block
    UNKNOWN = auto()         # Unknown node type

@dataclass
class CodeMetrics:
    """Metrics calculated from code analysis"""
    loc: int = 0                      # Lines of code
    lloc: int = 0                     # Logical lines of code
    comment_lines: int = 0            # Comment lines
    cyclomatic_complexity: int = 0    # Cyclomatic complexity
    function_count: int = 0           # Number of functions
    class_count: int = 0              # Number of classes
    parameter_count: Dict[str, int] = field(default_factory=dict)  # Parameters per function
    nesting_depth: int = 0            # Maximum nesting depth
    dependency_count: int = 0         # Number of dependencies/imports
    maintainability_index: float = 0.0  # Maintainability index
    halstead_metrics: Dict[str, float] = field(default_factory=dict)  # Halstead complexity metrics

@dataclass
class CodeNode:
    """Node in the unified code AST representation"""
    node_id: str
    node_type: CodeNodeType
    language: CodeLanguage
    name: Optional[str] = None
    value: Optional[Any] = None
    code: Optional[str] = None
    line_start: int = 0
    line_end: int = 0
    column_start: int = 0
    column_end: int = 0
    metadata: Dict[str, Any] = field(default_factory=dict)
    parent_id: Optional[str] = None
    children: List[str] = field(default_factory=list)  # List of child node IDs

    def __post_init__(self):
        """Initialize additional fields after creation"""
        self.creation_time = time.time()

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation"""
        return {
            "node_id": self.node_id,
            "node_type": self.node_type.name,
            "language": self.language.name,
            "name": self.name,
            "value": str(self.value) if self.value is not None else None,
            "code": self.code,
            "line_start": self.line_start,
            "line_end": self.line_end,
            "column_start": self.column_start,
            "column_end": self.column_end,
            "metadata": self.metadata,
            "parent_id": self.parent_id,
            "children": self.children
        }

@dataclass
class CodeAnalysisResult:
    """Result of a code analysis operation"""
    success: bool
    language: CodeLanguage
    root_node_id: Optional[str] = None
    metrics: Optional[CodeMetrics] = None
    error_message: Optional[str] = None
    warnings: List[Dict[str, Any]] = field(default_factory=list)
    timing: Dict[str, float] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        """Initialize additional fields after creation"""
        self.id = str(uuid.uuid4())
        self.timestamp = time.time()

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation"""
        result = {
            "id": self.id,
            "timestamp": self.timestamp,
            "success": self.success,
            "language": self.language.name,
            "root_node_id": self.root_node_id,
            "timing": self.timing,
            "metadata": self.metadata
        }

        if self.error_message:
            result["error_message"] = self.error_message

        if self.warnings:
            result["warnings"] = self.warnings

        if self.metrics:
            result["metrics"] = {
                "loc": self.metrics.loc,
                "lloc": self.metrics.lloc,
                "comment_lines": self.metrics.comment_lines,
                "cyclomatic_complexity": self.metrics.cyclomatic_complexity,
                "function_count": self.metrics.function_count,
                "class_count": self.metrics.class_count,
                "parameter_count": self.metrics.parameter_count,
                "nesting_depth": self.metrics.nesting_depth,
                "dependency_count": self.metrics.dependency_count,
                "maintainability_index": self.metrics.maintainability_index,
                "halstead_metrics": self.metrics.halstead_metrics
            }

        return result

class CodeAnalyzer:
    """
    Core code analysis module for GrandNexus.

    This module provides functionality for parsing, analyzing, and understanding
    code across multiple programming languages. It serves as a specialized
    component of the perception subsystem focused on code structures.
    """

    def __init__(self, nexus_core: Optional[NexusCore] = None,
                 semantic_memory: Optional[SemanticMemory] = None,
                 working_memory: Optional[WorkingMemory] = None):
        """
        Initialize the code analyzer.

        Args:
            nexus_core: Reference to the NexusCore for system-wide coordination
            semantic_memory: Reference to semantic memory for concept storage
            working_memory: Reference to working memory for context
        """
        self.logger = logging.getLogger("GrandNexus.Perception.CodeAnalyzer")
        self.nexus_core = nexus_core
        self.semantic_memory = semantic_memory
        self.working_memory = working_memory

        # Internal state
        self.instance_id = str(uuid.uuid4())[:8]
        self.initialized = False
        self.lock = threading.RLock()

        # Node storage
        self.nodes = {}  # Map of node_id -> CodeNode

        # Language parsers registry
        self.language_parsers = {}

        # Analysis plugins registry
        self.analysis_plugins = {}

        # Recent results cache
        self.result_cache = {}
        self.recent_results = deque(maxlen=20)

        # Performance metrics
        self.metrics = defaultdict(lambda: deque(maxlen=100))

        self.logger.info(f"CodeAnalyzer initialized with ID={self.instance_id}")

    def initialize(self) -> bool:
        """Initialize the code analyzer and register language parsers"""
        if self.initialized:
            return True

        with self.lock:
            try:
                # Register built-in language parsers
                self._register_default_parsers()

                # Register built-in analysis plugins
                self._register_default_plugins()

                # Connect to other modules
                if self.nexus_core:
                    self._register_with_nexus_core()

                self.initialized = True
                self.logger.info("CodeAnalyzer initialization complete")
                return True
            except Exception as e:
                self.logger.error(f"CodeAnalyzer initialization failed: {str(e)}")
                return False

    def analyze(self, code: str, language: Optional[CodeLanguage] = None,
               filename: Optional[str] = None,
               analysis_options: Optional[Dict[str, Any]] = None) -> CodeAnalysisResult:
        """
        Analyze a code snippet or file.

        Args:
            code: The source code to analyze
            language: Optional language hint, will auto-detect if not provided
            filename: Optional filename for context
            analysis_options: Additional options for the analysis

        Returns:
            CodeAnalysisResult containing the analysis results
        """
        if not self.initialized:
            self.initialize()

        # Start timing
        start_time = time.time()
        timing = {}

        # Check cache for identical code
        cache_key = hash(code)
        if cache_key in self.result_cache:
            return self.result_cache[cache_key]

        # Prepare options
        if analysis_options is None:
            analysis_options = {}

        # Detect language if not provided
        detected_language = language
        if detected_language is None:
            detect_start = time.time()
            detected_language = self._detect_language(code, filename)
            timing["language_detection"] = time.time() - detect_start

        # Select the appropriate parser
        if detected_language in self.language_parsers:
            parser_func = self.language_parsers[detected_language]
            try:
                # Parse the code
                parse_start = time.time()
                parse_result = parser_func(code, filename, analysis_options)
                timing["parsing"] = time.time() - parse_start

                # Gather code metrics
                metrics_start = time.time()
                metrics = self._calculate_metrics(code, detected_language, parse_result["root_node_id"])
                timing["metrics"] = time.time() - metrics_start

                # Run analysis plugins
                plugins_start = time.time()
                warnings = self._run_analysis_plugins(detected_language, parse_result["root_node_id"], analysis_options)
                timing["plugins"] = time.time() - plugins_start

                # Create result
                result = CodeAnalysisResult(
                    success=True,
                    language=detected_language,
                    root_node_id=parse_result["root_node_id"],
                    metrics=metrics,
                    warnings=warnings,
                    timing=timing,
                    metadata={"filename": filename} if filename else {}
                )

                # Update metrics
                self.metrics["analysis_success_rate"].append(1.0)
                self.metrics["analysis_time"].append(time.time() - start_time)

                # Cache successful results
                self.result_cache[cache_key] = result
                self.recent_results.append(result.id)

                return result

            except Exception as e:
                self.logger.error(f"Error analyzing code: {str(e)}")
                self.metrics["analysis_success_rate"].append(0.0)
                return CodeAnalysisResult(
                    success=False,
                    language=detected_language,
                    error_message=f"Analysis error: {str(e)}",
                    timing=timing
                )
        else:
            self.logger.warning(f"No parser available for language: {detected_language}")
            self.metrics["analysis_success_rate"].append(0.0)
            return CodeAnalysisResult(
                success=False,
                language=detected_language,
                error_message=f"No parser available for language: {detected_language}",
                timing=timing
            )

    def get_node(self, node_id: str) -> Optional[CodeNode]:
        """
        Retrieve a code node by ID.

        Args:
            node_id: The ID of the node to retrieve

        Returns:
            The CodeNode if found, None otherwise
        """
        with self.lock:
            return self.nodes.get(node_id)

    def get_subtree(self, node_id: str) -> Dict[str, CodeNode]:
        """
        Retrieve a node and all its descendants.

        Args:
            node_id: The ID of the root node

        Returns:
            Dictionary mapping node IDs to CodeNodes
        """
        with self.lock:
            subtree = {}
            self._collect_subtree(node_id, subtree)
            return subtree

    def find_nodes(self, criteria: Dict[str, Any]) -> List[CodeNode]:
        """
        Find nodes matching specified criteria.

        Args:
            criteria: Dictionary of criteria to match against node properties

        Returns:
            List of matching CodeNodes
        """
        with self.lock:
            results = []

            for node in self.nodes.values():
                match = True
                for key, value in criteria.items():
                    if key == "node_type":
                        if isinstance(value, CodeNodeType):
                            if node.node_type != value:
                                match = False
                                break
                        elif isinstance(value, str):
                            if node.node_type.name != value:
                                match = False
                                break
                    elif key == "language":
                        if isinstance(value, CodeLanguage):
                            if node.language != value:
                                match = False
                                break
                        elif isinstance(value, str):
                            if node.language.name != value:
                                match = False
                                break
                    elif hasattr(node, key):
                        if getattr(node, key) != value:
                            match = False
                            break
                    elif key in node.metadata:
                        if node.metadata[key] != value:
                            match = False
                            break
                    else:
                        match = False
                        break

                if match:
                    results.append(node)

            return results

    def register_language_parser(self, language: CodeLanguage,
                                parser_func: Callable[[str, Optional[str], Dict[str, Any]], Dict[str, Any]]) -> None:
        """
        Register a parser for a specific language.

        Args:
            language: The language this parser handles
            parser_func: Function that takes (code, filename, options) and returns parse result
        """
        with self.lock:
            self.language_parsers[language] = parser_func
            self.logger.info(f"Registered parser for language: {language}")

    def register_analysis_plugin(self, plugin_id: str,
                                plugin_func: Callable[[CodeLanguage, str, Dict[str, Any]], List[Dict[str, Any]]]) -> None:
        """
        Register an analysis plugin.

        Args:
            plugin_id: Unique identifier for the plugin
            plugin_func: Function that takes (language, root_node_id, options) and returns warnings
        """
        with self.lock:
            self.analysis_plugins[plugin_id] = plugin_func
            self.logger.info(f"Registered analysis plugin: {plugin_id}")

    def get_analysis_metrics(self) -> Dict[str, Any]:
        """Get current analysis performance metrics"""
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

    def _register_default_parsers(self) -> None:
        """Register built-in language parsers"""
        self.register_language_parser(CodeLanguage.PYTHON, self._parse_python)

        if HAS_ESPRIMA:
            self.register_language_parser(CodeLanguage.JAVASCRIPT, self._parse_javascript)

        # Other language parsers would be registered here

    def _register_default_plugins(self) -> None:
        """Register built-in analysis plugins"""
        self.register_analysis_plugin("complexity_checker", self._plugin_complexity_checker)
        self.register_analysis_plugin("variable_naming", self._plugin_variable_naming)
        self.register_analysis_plugin("security_checker", self._plugin_security_checker)

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
                name="code_analyzer",
                module=self,
                dependencies=dependencies
            )

            self.logger.info("Successfully registered with NexusCore")
        except Exception as e:
            self.logger.error(f"Failed to register with NexusCore: {str(e)}")

    def _detect_language(self, code: str, filename: Optional[str] = None) -> CodeLanguage:
        """
        Detect the programming language of the code.

        Args:
            code: The source code to analyze
            filename: Optional filename for context

        Returns:
            Detected CodeLanguage
        """
        # First check file extension if available
        if filename:
            ext = filename.split('.')[-1].lower()
            if ext in ['py', 'pyw']:
                return CodeLanguage.PYTHON
            elif ext in ['js']:
                return CodeLanguage.JAVASCRIPT
            elif ext in ['java']:
                return CodeLanguage.JAVA
            elif ext in ['cpp', 'cc', 'cxx', 'c++']:
                return CodeLanguage.CPP
            elif ext in ['rs']:
                return CodeLanguage.RUST
            elif ext in ['go']:
                return CodeLanguage.GO
            elif ext in ['rb']:
                return CodeLanguage.RUBY
            elif ext in ['php']:
                return CodeLanguage.PHP
            elif ext in ['cs']:
                return CodeLanguage.CSHARP
            elif ext in ['ts']:
                return CodeLanguage.TYPESCRIPT
            elif ext in ['html', 'htm']:
                return CodeLanguage.HTML
            elif ext in ['css']:
                return CodeLanguage.CSS
            elif ext in ['sql']:
                return CodeLanguage.SQL
            elif ext in ['sh', 'bash']:
                return CodeLanguage.SHELL

        # Language detection heuristics based on code content

        # Python indicators
        if re.search(r'\bdef\s+\w+\s*\(', code) or re.search(r'\bimport\s+\w+', code) or re.search(r'\bclass\s+\w+\s*:', code):
            return CodeLanguage.PYTHON

        # JavaScript indicators
        if re.search(r'\bfunction\s+\w+\s*\(', code) or re.search(r'\bconst\s+\w+\s*=', code) or re.search(r'\blet\s+\w+\s*=', code):
            return CodeLanguage.JAVASCRIPT

        # Java indicators
        if re.search(r'\bpublic\s+class\s+\w+', code) or re.search(r'\bprivate\s+\w+\s+\w+\s*\(', code):
            return CodeLanguage.JAVA

        # C++ indicators
        if re.search(r'\#include\s*<\w+>', code) or re.search(r'\bstd::', code):
            return CodeLanguage.CPP

        # Default to UNKNOWN if no patterns match
        return CodeLanguage.UNKNOWN

    def _parse_python(self, code: str, filename: Optional[str],
                     options: Dict[str, Any]) -> Dict[str, Any]:
        """
        Parse Python code using Python's built-in ast module.

        Args:
            code: Python source code
            filename: Optional filename for context
            options: Additional parsing options

        Returns:
            Dictionary with parsing results
        """
        try:
            # Parse the code with Python's ast module
            tree = ast.parse(code, filename or "<string>")

            # Transform to our unified representation
            root_id = self._create_node_id()

            # Create the root module node
            root_node = CodeNode(
                node_id=root_id,
                node_type=CodeNodeType.MODULE,
                language=CodeLanguage.PYTHON,
                name=filename or "<string>",
                code=code,
                line_start=0,
                line_end=len(code.splitlines())
            )

            # Store the node
            self.nodes[root_id] = root_node

            # Process the AST recursively
            self._process_python_ast(tree, root_id, code)

            return {
                "root_node_id": root_id,
                "language": CodeLanguage.PYTHON
            }

        except SyntaxError as e:
            self.logger.error(f"Python syntax error: {str(e)}")
            raise ValueError(f"Python syntax error: {str(e)}")

        except Exception as e:
            self.logger.error(f"Error parsing Python code: {str(e)}")
            raise

    def _parse_javascript(self, code: str, filename: Optional[str],
                         options: Dict[str, Any]) -> Dict[str, Any]:
        """
        Parse JavaScript code using esprima (if available).

        Args:
            code: JavaScript source code
            filename: Optional filename for context
            options: Additional parsing options

        Returns:
            Dictionary with parsing results
        """
        if not HAS_ESPRIMA:
            raise ImportError("esprima is required for JavaScript parsing")

        try:
            # Parse the code with esprima
            tree = esprima.parseScript(code, {'loc': True, 'comment': True})

            # Transform to our unified representation
            root_id = self._create_node_id()

            # Create the root module node
            root_node = CodeNode(
                node_id=root_id,
                node_type=CodeNodeType.MODULE,
                language=CodeLanguage.JAVASCRIPT,
                name=filename or "<string>",
                code=code,
                line_start=0,
                line_end=len(code.splitlines())
            )

            # Store the node
            self.nodes[root_id] = root_node

            # Process the AST recursively
            self._process_javascript_ast(tree, root_id, code)

            return {
                "root_node_id": root_id,
                "language": CodeLanguage.JAVASCRIPT
            }

        except Exception as e:
            self.logger.error(f"Error parsing JavaScript code: {str(e)}")
            raise

    def _process_python_ast(self, node: ast.AST, parent_id: str, code: str) -> None:
        """
        Process a Python AST node and add it to our representation.

        Args:
            node: Python AST node
            parent_id: ID of the parent node
            code: Original source code
        """
        # Skip if node is None
        if node is None:
            return

        # Get the parent node
        parent_node = self.nodes[parent_id]

        # Handle different node types
        if isinstance(node, ast.Module):
            # Module already created, just process children
            for child in node.body:
                self._process_python_ast(child, parent_id, code)

        elif isinstance(node, ast.FunctionDef):
            # Create function node
            node_id = self._create_node_id()

            func_node = CodeNode(
                node_id=node_id,
                node_type=CodeNodeType.FUNCTION,
                language=CodeLanguage.PYTHON,
                name=node.name,
                line_start=node.lineno,
                line_end=getattr(node, 'end_lineno', node.lineno) if hasattr(node, 'end_lineno') else node.lineno,
                column_start=node.col_offset,
                column_end=getattr(node, 'end_col_offset', node.col_offset) if hasattr(node, 'end_col_offset') else node.col_offset,
                parent_id=parent_id,
                metadata={
                    "is_async": isinstance(node, ast.AsyncFunctionDef),
                    "decorators": len(node.decorator_list)
                }
            )

            # Extract function code if we have line numbers
            if hasattr(node, 'end_lineno'):
                lines = code.splitlines()
                if 0 <= node.lineno - 1 < len(lines) and 0 <= node.end_lineno - 1 < len(lines):
                    func_code = '\n'.join(lines[node.lineno - 1:node.end_lineno])
                    func_node.code = func_code

            # Store the node
            self.nodes[node_id] = func_node

            # Add to parent's children
            parent_node.children.append(node_id)

            # Process arguments
            if hasattr(node, 'args'):
                args_id = self._create_node_id()
                args_node = CodeNode(
                    node_id=args_id,
                    node_type=CodeNodeType.ARGUMENT,
                    language=CodeLanguage.PYTHON,
                    name="arguments",
                    parent_id=node_id,
                    metadata={
                        "arg_count": len(node.args.args),
                        "has_vararg": node.args.vararg is not None,
                        "has_kwarg": node.args.kwarg is not None
                    }
                )
                self.nodes[args_id] = args_node
                func_node.children.append(args_id)

                # Process individual arguments
                for arg in node.args.args:
                    arg_id = self._create_node_id()
                    arg_node = CodeNode(
                        node_id=arg_id,
                        node_type=CodeNodeType.ARGUMENT,
                        language=CodeLanguage.PYTHON,
                        name=arg.arg,
                        parent_id=args_id,
                        line_start=getattr(arg, 'lineno', 0),
                        column_start=getattr(arg, 'col_offset', 0)
                    )
                    self.nodes[arg_id] = arg_node
                    args_node.children.append(arg_id)

            # Process function body
            for child in node.body:
                self._process_python_ast(child, node_id, code)

        elif isinstance(node, ast.ClassDef):
            # Create class node
            node_id = self._create_node_id()

            class_node = CodeNode(
                node_id=node_id,
                node_type=CodeNodeType.CLASS,
                language=CodeLanguage.PYTHON,
                name=node.name,
                line_start=node.lineno,
                line_end=getattr(node, 'end_lineno', node.lineno) if hasattr(node, 'end_lineno') else node.lineno,
                column_start=node.col_offset,
                column_end=getattr(node, 'end_col_offset', node.col_offset) if hasattr(node, 'end_col_offset') else node.col_offset,
                parent_id=parent_id,
                metadata={
                    "bases": [self._get_name_from_python_expr(base) for base in node.bases],
                    "decorators": len(node.decorator_list)
                }
            )

            # Extract class code if we have line numbers
            if hasattr(node, 'end_lineno'):
                lines = code.splitlines()
                if 0 <= node.lineno - 1 < len(lines) and 0 <= node.end_lineno - 1 < len(lines):
                    class_code = '\n'.join(lines[node.lineno - 1:node.end_lineno])
                    class_node.code = class_code

            # Store the node
            self.nodes[node_id] = class_node

            # Add to parent's children
            parent_node.children.append(node_id)

            # Process class body
            for child in node.body:
                self._process_python_ast(child, node_id, code)

        elif isinstance(node, ast.Assign):
            # Create assignment node
            node_id = self._create_node_id()

            # Get target names (simplistic, just for display)
            target_names = []
            for target in node.targets:
                if isinstance(target, ast.Name):
                    target_names.append(target.id)
                elif isinstance(target, ast.Attribute):
                    target_names.append(f"{self._get_name_from_python_expr(target.value)}.{target.attr}")
                else:
                    target_names.append("complex_target")

            assign_node = CodeNode(
                node_id=node_id,
                node_type=CodeNodeType.STATEMENT,
                language=CodeLanguage.PYTHON,
                name=",".join(target_names),
                line_start=node.lineno,
                line_end=getattr(node, 'end_lineno', node.lineno) if hasattr(node, 'end_lineno') else node.lineno,
                column_start=node.col_offset,
                column_end=getattr(node, 'end_col_offset', node.col_offset) if hasattr(node, 'end_col_offset') else node.col_offset,
                parent_id=parent_id,
                metadata={
                    "statement_type": "assignment",
                    "targets": target_names
                }
            )

            # Store the node
            self.nodes[node_id] = assign_node

            # Add to parent's children
            parent_node.children.append(node_id)

            # Process value
            value_id = self._create_node_id()
            value_node = CodeNode(
                node_id=value_id,
                node_type=CodeNodeType.EXPRESSION,
                language=CodeLanguage.PYTHON,
                parent_id=node_id,
                line_start=getattr(node.value, 'lineno', node.lineno),
                column_start=getattr(node.value, 'col_offset', 0)
            )
            self.nodes[value_id] = value_node
            assign_node.children.append(value_id)

            # Process the value expression
            self._process_python_ast(node.value, value_id, code)

        elif isinstance(node, ast.If):
            # Create if statement node
            node_id = self._create_node_id()

            if_node = CodeNode(
                node_id=node_id,
                node_type=CodeNodeType.CONTROL_FLOW,
                language=CodeLanguage.PYTHON,
                name="if",
                line_start=node.lineno,
                line_end=getattr(node, 'end_lineno', node.lineno) if hasattr(node, 'end_lineno') else node.lineno,
                column_start=node.col_offset,
                column_end=getattr(node, 'end_col_offset', node.col_offset) if hasattr(node, 'end_col_offset') else node.col_offset,
                parent_id=parent_id,
                metadata={
                    "control_type": "if",
                    "has_else": bool(node.orelse)
                }
            )

            # Store the node
            self.nodes[node_id] = if_node

            # Add to parent's children
            parent_node.children.append(node_id)

            # Process test condition
            test_id = self._create_node_id()
            test_node = CodeNode(
                node_id=test_id,
                node_type=CodeNodeType.EXPRESSION,
                language=CodeLanguage.PYTHON,
                name="condition",
                parent_id=node_id,
                line_start=getattr(node.test, 'lineno', node.lineno),
                column_start=getattr(node.test, 'col_offset', 0)
            )
            self.nodes[test_id] = test_node
            if_node.children.append(test_id)

            # Process test expression
            self._process_python_ast(node.test, test_id, code)

            # Process if body
            body_id = self._create_node_id()
            body_node = CodeNode(
                node_id=body_id,
                node_type=CodeNodeType.BLOCK,
                language=CodeLanguage.PYTHON,
                name="if_body",
                parent_id=node_id
            )
            self.nodes[body_id] = body_node
            if_node.children.append(body_id)

            for child in node.body:
                self._process_python_ast(child, body_id, code)

            # Process else body if present
            if node.orelse:
                orelse_id = self._create_node_id()
                orelse_node = CodeNode(
                    node_id=orelse_id,
                    node_type=CodeNodeType.BLOCK,
                    language=CodeLanguage.PYTHON,
                    name="else_body",
                    parent_id=node_id
                )
                self.nodes[orelse_id] = orelse_node
                if_node.children.append(orelse_id)

                for child in node.orelse:
                    self._process_python_ast(child, orelse_id, code)

        elif isinstance(node, ast.For) or isinstance(node, ast.AsyncFor):
            # Create for loop node
            node_id = self._create_node_id()

            for_node = CodeNode(
                node_id=node_id,
                node_type=CodeNodeType.CONTROL_FLOW,
                language=CodeLanguage.PYTHON,
                name="for",
                line_start=node.lineno,
                line_end=getattr(node, 'end_lineno', node.lineno) if hasattr(node, 'end_lineno') else node.lineno,
                column_start=node.col_offset,
                column_end=getattr(node, 'end_col_offset', node.col_offset) if hasattr(node, 'end_col_offset') else node.col_offset,
                parent_id=parent_id,
                metadata={
                    "control_type": "for",
                    "is_async": isinstance(node, ast.AsyncFor),
                    "has_else": bool(node.orelse)
                }
            )

            # Store the node
            self.nodes[node_id] = for_node

            # Add to parent's children
            parent_node.children.append(node_id)

            # Process target
            target_id = self._create_node_id()
            target_node = CodeNode(
                node_id=target_id,
                node_type=CodeNodeType.EXPRESSION,
                language=CodeLanguage.PYTHON,
                name="target",
                parent_id=node_id,
                line_start=getattr(node.target, 'lineno', node.lineno),
                column_start=getattr(node.target, 'col_offset', 0)
            )
            self.nodes[target_id] = target_node
            for_node.children.append(target_id)

            # Process target expression
            self._process_python_ast(node.target, target_id, code)

            # Process iter
            iter_id = self._create_node_id()
            iter_node = CodeNode(
                node_id=iter_id,
                node_type=CodeNodeType.EXPRESSION,
                language=CodeLanguage.PYTHON,
                name="iter",
                parent_id=node_id,
                line_start=getattr(node.iter, 'lineno', node.lineno),
                column_start=getattr(node.iter, 'col_offset', 0)
            )
            self.nodes[iter_id] = iter_node
            for_node.children.append(iter_id)

            # Process iter expression
            self._process_python_ast(node.iter, iter_id, code)

            # Process for body
            body_id = self._create_node_id()
            body_node = CodeNode(
                node_id=body_id,
                node_type=CodeNodeType.BLOCK,
                language=CodeLanguage.PYTHON,
                name="for_body",
                parent_id=node_id
            )
            self.nodes[body_id] = body_node
            for_node.children.append(body_id)

            for child in node.body:
                self._process_python_ast(child, body_id, code)

            # Process else body if present
            if node.orelse:
                orelse_id = self._create_node_id()
                orelse_node = CodeNode(
                    node_id=orelse_id,
                    node_type=CodeNodeType.BLOCK,
                    language=CodeLanguage.PYTHON,
                    name="else_body",
                    parent_id=node_id
                )
                self.nodes[orelse_id] = orelse_node
                for_node.children.append(orelse_id)

                for child in node.orelse:
                    self._process_python_ast(child, orelse_id, code)

        elif isinstance(node, ast.Call):
            # Create function call node
            node_id = self._create_node_id()

            func_name = self._get_name_from_python_expr(node.func)

            call_node = CodeNode(
                node_id=node_id,
                node_type=CodeNodeType.CALL,
                language=CodeLanguage.PYTHON,
                name=func_name,
                line_start=node.lineno,
                line_end=getattr(node, 'end_lineno', node.lineno) if hasattr(node, 'end_lineno') else node.lineno,
                column_start=node.col_offset,
                column_end=getattr(node, 'end_col_offset', node.col_offset) if hasattr(node, 'end_col_offset') else node.col_offset,
                parent_id=parent_id,
                metadata={
                    "arg_count": len(node.args),
                    "kwarg_count": len(node.keywords)
                }
            )

            # Store the node
            self.nodes[node_id] = call_node

            # Add to parent's children
            parent_node.children.append(node_id)

            # Process function expression
            func_id = self._create_node_id()
            func_node = CodeNode(
                node_id=func_id,
                node_type=CodeNodeType.EXPRESSION,
                language=CodeLanguage.PYTHON,
                name="function",
                parent_id=node_id,
                line_start=getattr(node.func, 'lineno', node.lineno),
                column_start=getattr(node.func, 'col_offset', 0)
            )
            self.nodes[func_id] = func_node
            call_node.children.append(func_id)

            self._process_python_ast(node.func, func_id, code)

            # Process arguments
            if node.args:
                args_id = self._create_node_id()
                args_node = CodeNode(
                    node_id=args_id,
                    node_type=CodeNodeType.ARGUMENT,
                    language=CodeLanguage.PYTHON,
                    name="arguments",
                    parent_id=node_id
                )
                self.nodes[args_id] = args_node
                call_node.children.append(args_id)

                for arg in node.args:
                    self._process_python_ast(arg, args_id, code)

            # Process keyword arguments
            if node.keywords:
                kwargs_id = self._create_node_id()
                kwargs_node = CodeNode(
                    node_id=kwargs_id,
                    node_type=CodeNodeType.ARGUMENT,
                    language=CodeLanguage.PYTHON,
                    name="keyword_arguments",
                    parent_id=node_id
                )
                self.nodes[kwargs_id] = kwargs_node
                call_node.children.append(kwargs_id)

                for kw in node.keywords:
                    kw_id = self._create_node_id()
                    kw_name = kw.arg if kw.arg else "**"
                    kw_node = CodeNode(
                        node_id=kw_id,
                        node_type=CodeNodeType.ARGUMENT,
                        language=CodeLanguage.PYTHON,
                        name=kw_name,
                        parent_id=kwargs_id,
                        line_start=getattr(kw, 'lineno', 0),
                        column_start=getattr(kw, 'col_offset', 0)
                    )
                    self.nodes[kw_id] = kw_node
                    kwargs_node.children.append(kw_id)

                    self._process_python_ast(kw.value, kw_id, code)

        elif isinstance(node, ast.Import) or isinstance(node, ast.ImportFrom):
            # Create import node
            node_id = self._create_node_id()

            if isinstance(node, ast.Import):
                import_names = [name.name for name in node.names]
                module_name = None
            else:  # ImportFrom
                import_names = [name.name for name in node.names]
                module_name = node.module

            import_node = CodeNode(
                node_id=node_id,
                node_type=CodeNodeType.IMPORT,
                language=CodeLanguage.PYTHON,
                name=', '.join(import_names),
                line_start=node.lineno,
                line_end=getattr(node, 'end_lineno', node.lineno) if hasattr(node, 'end_lineno') else node.lineno,
                column_start=node.col_offset,
                column_end=getattr(node, 'end_col_offset', node.col_offset) if hasattr(node, 'end_col_offset') else node.col_offset,
                parent_id=parent_id,
                metadata={
                    "is_from": isinstance(node, ast.ImportFrom),
                    "module": module_name,
                    "level": getattr(node, 'level', 0) if isinstance(node, ast.ImportFrom) else 0,
                    "names": import_names
                }
            )

            # Store the node
            self.nodes[node_id] = import_node

            # Add to parent's children
            parent_node.children.append(node_id)

        elif isinstance(node, ast.Expr):
            # Create expression statement node
            node_id = self._create_node_id()

            expr_node = CodeNode(
                node_id=node_id,
                node_type=CodeNodeType.EXPRESSION,
                language=CodeLanguage.PYTHON,
                line_start=node.lineno,
                line_end=getattr(node, 'end_lineno', node.lineno) if hasattr(node, 'end_lineno') else node.lineno,
                column_start=node.col_offset,
                column_end=getattr(node, 'end_col_offset', node.col_offset) if hasattr(node, 'end_col_offset') else node.col_offset,
                parent_id=parent_id
            )

            # Store the node
            self.nodes[node_id] = expr_node

            # Add to parent's children
            parent_node.children.append(node_id)

            # Process the expression value
            self._process_python_ast(node.value, node_id, code)

        elif isinstance(node, ast.Constant):
            # Create literal node for constant (Python 3.8+)
            node_id = self._create_node_id()

            # Determine the type of constant
            value_type = type(node.value).__name__

            literal_node = CodeNode(
                node_id=node_id,
                node_type=CodeNodeType.LITERAL,
                language=CodeLanguage.PYTHON,
                value=node.value,
                line_start=getattr(node, 'lineno', 0),
                line_end=getattr(node, 'end_lineno', getattr(node, 'lineno', 0)) if hasattr(node, 'end_lineno') else getattr(node, 'lineno', 0),
                column_start=getattr(node, 'col_offset', 0),
                column_end=getattr(node, 'end_col_offset', getattr(node, 'col_offset', 0)) if hasattr(node, 'end_col_offset') else getattr(node, 'col_offset', 0),
                parent_id=parent_id,
                metadata={
                    "value_type": value_type
                }
            )

            # Store the node
            self.nodes[node_id] = literal_node

            # Add to parent's children
            parent_node.children.append(node_id)

        elif isinstance(node, ast.Name):
            # Create variable reference node
            node_id = self._create_node_id()

            name_node = CodeNode(
                node_id=node_id,
                node_type=CodeNodeType.VARIABLE,
                language=CodeLanguage.PYTHON,
                name=node.id,
                line_start=getattr(node, 'lineno', 0),
                line_end=getattr(node, 'end_lineno', getattr(node, 'lineno', 0)) if hasattr(node, 'end_lineno') else getattr(node, 'lineno', 0),
                column_start=getattr(node, 'col_offset', 0),
                column_end=getattr(node, 'end_col_offset', getattr(node, 'col_offset', 0)) if hasattr(node, 'end_col_offset') else getattr(node, 'col_offset', 0),
                parent_id=parent_id,
                metadata={
                    "ctx": node.ctx.__class__.__name__
                }
            )

            # Store the node
            self.nodes[node_id] = name_node

            # Add to parent's children
            parent_node.children.append(node_id)

        # Add handlers for other node types as needed

        # For nodes we don't explicitly handle, create a generic unknown node
        else:
            node_id = self._create_node_id()

            unknown_node = CodeNode(
                node_id=node_id,
                node_type=CodeNodeType.UNKNOWN,
                language=CodeLanguage.PYTHON,
                name=node.__class__.__name__,
                line_start=getattr(node, 'lineno', 0),
                line_end=getattr(node, 'end_lineno', getattr(node, 'lineno', 0)) if hasattr(node, 'end_lineno') else getattr(node, 'lineno', 0),
                column_start=getattr(node, 'col_offset', 0),
                column_end=getattr(node, 'end_col_offset', getattr(node, 'col_offset', 0)) if hasattr(node, 'end_col_offset') else getattr(node, 'col_offset', 0),
                parent_id=parent_id
            )

            # Store the node
            self.nodes[node_id] = unknown_node

            # Add to parent's children
            parent_node.children.append(node_id)

    def _process_javascript_ast(self, node: Any, parent_id: str, code: str) -> None:
        """
        Process a JavaScript AST node (from esprima) and add it to our representation.

        Args:
            node: Esprima AST node
            parent_id: ID of the parent node
            code: Original source code
        """
        # This would be implemented similarly to _process_python_ast
        # but with esprima's node types
        pass

    def _get_name_from_python_expr(self, expr: ast.AST) -> str:
        """
        Extract a displayable name from a Python expression.

        Args:
            expr: Python AST expression node

        Returns:
            String representation of the expression
        """
        if isinstance(expr, ast.Name):
            return expr.id
        elif isinstance(expr, ast.Attribute):
            return f"{self._get_name_from_python_expr(expr.value)}.{expr.attr}"
        elif isinstance(expr, ast.Call):
            return f"{self._get_name_from_python_expr(expr.func)}(...)"
        elif isinstance(expr, ast.Subscript):
            return f"{self._get_name_from_python_expr(expr.value)}[...]"
        else:
            return expr.__class__.__name__

    def _create_node_id(self) -> str:
        """Generate a unique node ID"""
        return str(uuid.uuid4())

    def _collect_subtree(self, node_id: str, result: Dict[str, CodeNode]) -> None:
        """
        Recursively collect a node and all its descendants.

        Args:
            node_id: The ID of the node to start from
            result: Dictionary to populate with nodes
        """
        if node_id not in self.nodes:
            return

        node = self.nodes[node_id]
        result[node_id] = node

        for child_id in node.children:
            self._collect_subtree(child_id, result)

    def _calculate_metrics(self, code: str, language: CodeLanguage, root_node_id: str) -> CodeMetrics:
        """
        Calculate code metrics for the analyzed code.

        Args:
            code: The original source code
            language: The programming language
            root_node_id: ID of the root node

        Returns:
            CodeMetrics with calculated metrics
        """
        # Initialize metrics
        metrics = CodeMetrics()

        # Calculate basic metrics
        lines = code.splitlines()
        metrics.loc = len(lines)

        # Calculate logical lines of code (non-blank, non-comment)
        if language == CodeLanguage.PYTHON:
            # Simple heuristic for Python
            logical_lines = 0
            comment_lines = 0

            for line in lines:
                stripped = line.strip()
                if not stripped:
                    continue  # Skip blank lines

                if stripped.startswith('#'):
                    comment_lines += 1
                else:
                    logical_lines += 1

            metrics.lloc = logical_lines
            metrics.comment_lines = comment_lines

        # Calculate other metrics based on the AST
        if root_node_id in self.nodes:
            # Count functions and classes
            functions = self.find_nodes({"node_type": CodeNodeType.FUNCTION})
            classes = self.find_nodes({"node_type": CodeNodeType.CLASS})

            metrics.function_count = len(functions)
            metrics.class_count = len(classes)

            # Count dependencies
            imports = self.find_nodes({"node_type": CodeNodeType.IMPORT})
            metrics.dependency_count = len(imports)

            # Calculate cyclomatic complexity
            # Simple approximation: count control flow nodes
            control_flow_nodes = self.find_nodes({"node_type": CodeNodeType.CONTROL_FLOW})
            metrics.cyclomatic_complexity = 1 + len(control_flow_nodes)  # Base 1 + number of branches

            # Calculate function parameter counts
            for func in functions:
                arg_nodes = []
                for child_id in func.children:
                    child = self.nodes.get(child_id)
                    if child and child.node_type == CodeNodeType.ARGUMENT:
                        arg_nodes.append(child)

                func_name = func.name or "anonymous"
                if arg_nodes:
                    # Count direct argument children if available
                    params = 0
                    for arg_node in arg_nodes:
                        if arg_node.name == "arguments":
                            # If it's an arguments container, count its children
                            params += len(arg_node.children)
                        else:
                            # Otherwise count the node itself
                            params += 1

                    metrics.parameter_count[func_name] = params

        # Calculate maintainability index
        # Simplified formula: 171 - 5.2 * ln(HV) - 0.23 * CC - 16.2 * ln(SLOC)
        # where HV is Halstead Volume, CC is cyclomatic complexity, SLOC is source lines of code
        try:
            # Placeholder for Halstead volume (would need proper calculation)
            halstead_volume = metrics.lloc * 2.5  # Very rough approximation
            metrics.halstead_metrics["volume"] = halstead_volume

            metrics.maintainability_index = 171 - 5.2 * math.log(halstead_volume) - 0.23 * metrics.cyclomatic_complexity - 16.2 * math.log(metrics.lloc)
            # Normalize to 0-100 range
            metrics.maintainability_index = max(0, min(100, metrics.maintainability_index))
        except:
            metrics.maintainability_index = 50.0  # Default if calculation fails

        return metrics

    def _run_analysis_plugins(self, language: CodeLanguage, root_node_id: str,
                             options: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Run analysis plugins on the parsed code.

        Args:
            language: The code language
            root_node_id: ID of the root node
            options: Analysis options

        Returns:
            List of warnings/issues found by the plugins
        """
        all_warnings = []

        # Determine which plugins to run
        plugins_to_run = options.get("plugins", list(self.analysis_plugins.keys()))

        for plugin_id in plugins_to_run:
            if plugin_id in self.analysis_plugins:
                try:
                    plugin_func = self.analysis_plugins[plugin_id]
                    warnings = plugin_func(language, root_node_id, options)

                    # Add plugin ID to each warning
                    for warning in warnings:
                        warning["plugin"] = plugin_id

                    all_warnings.extend(warnings)
                except Exception as e:
                    self.logger.error(f"Error running plugin {plugin_id}: {str(e)}")

        return all_warnings

    def _plugin_complexity_checker(self, language: CodeLanguage, root_node_id: str,
                                 options: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Plugin to check code complexity.

        Args:
            language: The code language
            root_node_id: ID of the root node
            options: Analysis options

        Returns:
            List of complexity warnings
        """
        warnings = []

        # Get thresholds from options or use defaults
        max_function_complexity = options.get("max_function_complexity", 10)
        max_function_params = options.get("max_function_params", 5)

        # Find function nodes
        functions = self.find_nodes({"node_type": CodeNodeType.FUNCTION})

        for func in functions:
            # Check function complexity
            complexity = 1  # Base complexity

            # Add complexity for control flow statements
            subtree = self.get_subtree(func.node_id)
            for node in subtree.values():
                if node.node_type == CodeNodeType.CONTROL_FLOW:
                    complexity += 1

            if complexity > max_function_complexity:
                warnings.append({
                    "type": "complexity",
                    "message": f"Function '{func.name}' has high cyclomatic complexity ({complexity})",
                    "node_id": func.node_id,
                    "severity": "warning",
                    "line": func.line_start
                })

            # Check parameter count
            arg_nodes = []
            for child_id in func.children:
                child = self.nodes.get(child_id)
                if child and child.node_type == CodeNodeType.ARGUMENT:
                    arg_nodes.append(child)

            param_count = 0
            for arg_node in arg_nodes:
                if arg_node.name == "arguments":
                    # If it's an arguments container, count its children
                    param_count += len(arg_node.children)
                else:
                    # Otherwise count the node itself
                    param_count += 1

            if param_count > max_function_params:
                warnings.append({
                    "type": "parameters",
                    "message": f"Function '{func.name}' has too many parameters ({param_count})",
                    "node_id": func.node_id,
                    "severity": "warning",
                    "line": func.line_start
                })

        return warnings

    def _plugin_variable_naming(self, language: CodeLanguage, root_node_id: str,
                              options: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Plugin to check variable naming conventions.

        Args:
            language: The code language
            root_node_id: ID of the root node
            options: Analysis options

        Returns:
            List of naming convention warnings
        """
        warnings = []

        if language != CodeLanguage.PYTHON:
            return warnings  # Only implemented for Python for now

        # Get naming patterns from options or use defaults
        naming_patterns = options.get("naming_patterns", {
            "function": r"^[a-z][a-z0-9_]*$",  # snake_case
            "class": r"^[A-Z][a-zA-Z0-9]*$",   # PascalCase
            "variable": r"^[a-z][a-z0-9_]*$",  # snake_case
            "constant": r"^[A-Z][A-Z0-9_]*$"   # UPPER_CASE
        })

        # Check function names
        functions = self.find_nodes({"node_type": CodeNodeType.FUNCTION})
        for func in functions:
            if func.name and not re.match(naming_patterns["function"], func.name):
                warnings.append({
                    "type": "naming",
                    "message": f"Function name '{func.name}' does not follow naming convention",
                    "node_id": func.node_id,
                    "severity": "info",
                    "line": func.line_start
                })

        # Check class names
        classes = self.find_nodes({"node_type": CodeNodeType.CLASS})
        for cls in classes:
            if cls.name and not re.match(naming_patterns["class"], cls.name):
                warnings.append({
                    "type": "naming",
                    "message": f"Class name '{cls.name}' does not follow naming convention",
                    "node_id": cls.node_id,
                    "severity": "info",
                    "line": cls.line_start
                })

        # Check variable names
        variables = self.find_nodes({"node_type": CodeNodeType.VARIABLE})
        for var in variables:
            if var.name:
                # Check if it might be a constant (all caps)
                if re.match(r"^[A-Z][A-Z0-9_]*$", var.name):
                    continue  # Skip constants

                if not re.match(naming_patterns["variable"], var.name):
                    warnings.append({
                        "type": "naming",
                        "message": f"Variable name '{var.name}' does not follow naming convention",
                        "node_id": var.node_id,
                        "severity": "info",
                        "line": var.line_start
                    })

        return warnings

    def _plugin_security_checker(self, language: CodeLanguage, root_node_id: str,
                               options: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Plugin to check for common security issues.

        Args:
            language: The code language
            root_node_id: ID of the root node
            options: Analysis options

        Returns:
            List of security warnings
        """
        warnings = []

        if language == CodeLanguage.PYTHON:
            # Check for potentially unsafe function calls
            unsafe_functions = {
                "eval": "Evaluates arbitrary code which may be unsafe",
                "exec": "Executes arbitrary code which may be unsafe",
                "pickle.loads": "Deserializing untrusted data with pickle can lead to code execution",
                "marshal.loads": "Deserializing untrusted data with marshal can lead to code execution",
                "subprocess.call": "Check for shell=True which may allow command injection",
                "subprocess.Popen": "Check for shell=True which may allow command injection",
                "yaml.load": "Use yaml.safe_load instead for untrusted data",
                "os.system": "May allow command injection if input is not sanitized",
                "__import__": "Dynamic imports can lead to code execution"
            }

            # Find all function calls
            calls = self.find_nodes({"node_type": CodeNodeType.CALL})

            for call in calls:
                func_name = call.name

                # Check for direct matches
                if func_name in unsafe_functions:
                    warnings.append({
                        "type": "security",
                        "message": f"Potential security issue: {func_name} - {unsafe_functions[func_name]}",
                        "node_id": call.node_id,
                        "severity": "warning",
                        "line": call.line_start
                    })

                # Check for module.function matches
                for unsafe_func, description in unsafe_functions.items():
                    if '.' in unsafe_func and func_name.endswith('.' + unsafe_func.split('.')[-1]):
                        # This is a heuristic check - would need deeper analysis for accuracy
                        warnings.append({
                            "type": "security",
                            "message": f"Potential security issue: {func_name} - {description}",
                            "node_id": call.node_id,
                            "severity": "warning",
                            "line": call.line_start
                        })

            # Check for hardcoded secrets or credentials
            literals = self.find_nodes({"node_type": CodeNodeType.LITERAL})

            # Patterns that might indicate secrets
            secret_patterns = [
                (r'password', 'Possible hardcoded password'),
                (r'passwd', 'Possible hardcoded password'),
                (r'secret', 'Possible hardcoded secret'),
                (r'key', 'Possible hardcoded key'),
                (r'token', 'Possible hardcoded token'),
                (r'api.+key', 'Possible hardcoded API key')
            ]

            # Check assignments for potential secrets
            assignments = self.find_nodes({"node_type": CodeNodeType.STATEMENT})
            for assign in assignments:
                if not assign.name:
                    continue

                # Check variable name against secret patterns
                for pattern, message in secret_patterns:
                    if re.search(pattern, assign.name, re.IGNORECASE):
                        warnings.append({
                            "type": "security",
                            "message": f"{message} in variable '{assign.name}'",
                            "node_id": assign.node_id,
                            "severity": "warning",
                            "line": assign.line_start
                        })

            # Check for SQL injection vulnerabilities
            # This is a very simple heuristic and would need deeper analysis for accuracy
            string_format_calls = []
            for call in calls:
                if call.name in ['format', 'f-string', '%']:
                    string_format_calls.append(call)

            # Find strings that look like SQL queries
            sql_patterns = [
                r'SELECT\s+.+\s+FROM',
                r'INSERT\s+INTO',
                r'UPDATE\s+.+\s+SET',
                r'DELETE\s+FROM'
            ]

            for literal in literals:
                if not isinstance(literal.value, str):
                    continue

                for pattern in sql_patterns:
                    if re.search(pattern, literal.value, re.IGNORECASE):
                        # Check if this SQL string is used with string formatting
                        # This is a simplified check and not very accurate
                        warnings.append({
                            "type": "security",
                            "message": "Possible SQL query string - check for proper parameterization to avoid SQL injection",
                            "node_id": literal.node_id,
                            "severity": "info",
                            "line": literal.line_start
                        })
                        break

        elif language == CodeLanguage.JAVASCRIPT:
            # Similar checks for JavaScript would go here
            pass

        return warnings