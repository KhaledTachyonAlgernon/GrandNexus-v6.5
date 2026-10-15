"""
symbolic_engine.py - Formal Symbolic Reasoning Engine for GrandNexus

This module provides a robust symbolic reasoning system capable of formal logical operations,
theorem proving, and knowledge representation. It serves as the foundation for GrandNexus's
symbolic reasoning capabilities, complementing the neural reasoning components.

Key features:
- First-order logic representation and manipulation
- Inference rules and deduction mechanisms
- Knowledge base management with belief revision
- Integration with the semantic graph for knowledge retrieval
- Support for different reasoning strategies (forward/backward chaining, resolution)
- Uncertainty handling via probabilistic logic extensions
- Flexible rule systems that can be extended or adapted

The design emphasizes:
- Modularity: Different reasoning strategies can be swapped or combined
- Explainability: All inference steps can be traced and explained
- Integration: Seamless connection with other cognitive modules
- Extensibility: New logical systems or reasoning approaches can be added
"""

import logging
import time
import threading
import uuid
import re
from typing import Dict, List, Any, Optional, Set, Tuple, Union, Callable
from enum import Enum, auto
from dataclasses import dataclass, field
from collections import defaultdict, deque
import copy

# Assuming these exist or will exist in the GrandNexus architecture
from grandnexus.core.nexus_core import NexusCore
from grandnexus.memory.semantic import SemanticMemory
from grandnexus.memory.working import WorkingMemory
from grandnexus.perception.symbolic_parser import SymbolicParser, ParseResult, SymbolicType
from grandnexus.graph.graph_engine import create_graph_engine, AbstractGraphEngine

# Try to import common symbolic reasoning libraries
try:
    import sympy
    from sympy.logic.boolalg import to_cnf, simplify_logic
    HAS_SYMPY = True
except ImportError:
    HAS_SYMPY = False

try:
    from pyke import knowledge_engine
    HAS_PYKE = True
except ImportError:
    HAS_PYKE = False

# -----------------------------------------------------------------------------
# Core Data Structures
# -----------------------------------------------------------------------------

class SymbolicEntityType(Enum):
    """Types of symbolic entities in the reasoning system"""
    CONSTANT = auto()    # Individual constant (e.g., "a", "John")
    VARIABLE = auto()    # Logical variable (e.g., "x", "Y")
    FUNCTION = auto()    # Function symbol (e.g., "father_of")
    PREDICATE = auto()   # Predicate symbol (e.g., "isHuman")
    OPERATOR = auto()    # Logical operator (e.g., "AND", "OR")
    QUANTIFIER = auto()  # Logical quantifier (e.g., "∀", "∃")
    TERM = auto()        # Compound term (function application)
    ATOM = auto()        # Atomic formula (predicate application)
    FORMULA = auto()     # Compound formula (logical expressions)
    RULE = auto()        # Inference rule or axiom

class InferenceStrategy(Enum):
    """Reasoning strategies supported by the engine"""
    FORWARD_CHAINING = auto()  # Data-driven reasoning
    BACKWARD_CHAINING = auto() # Goal-driven reasoning
    RESOLUTION = auto()        # Resolution refutation
    NATURAL_DEDUCTION = auto() # Natural deduction system
    TABLEAUX = auto()          # Semantic tableaux method
    HYBRID = auto()            # Combined strategy

@dataclass
class SymbolicEntity:
    """Base class for all symbolic entities in the reasoning system"""
    entity_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    entity_type: SymbolicEntityType = SymbolicEntityType.CONSTANT
    symbol: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __hash__(self):
        return hash(self.entity_id)

    def __eq__(self, other):
        if not isinstance(other, SymbolicEntity):
            return False
        return self.entity_id == other.entity_id

    def __str__(self):
        return self.symbol

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation"""
        return {
            "entity_id": self.entity_id,
            "entity_type": self.entity_type.name,
            "symbol": self.symbol,
            "metadata": self.metadata
        }

@dataclass
class Term(SymbolicEntity):
    """Represents a term in first-order logic (constant, variable, or function application)"""
    arguments: List[Union["Term", "SymbolicEntity"]] = field(default_factory=list)

    def __post_init__(self):
        if not self.entity_type:
            if not self.arguments:
                # Simple term (constant or variable)
                if self.symbol.isupper() or self.symbol.startswith('?'):
                    self.entity_type = SymbolicEntityType.VARIABLE
                else:
                    self.entity_type = SymbolicEntityType.CONSTANT
            else:
                # Function application
                self.entity_type = SymbolicEntityType.TERM

    def __str__(self):
        if not self.arguments:
            return self.symbol

        args_str = ", ".join(str(arg) for arg in self.arguments)
        return f"{self.symbol}({args_str})"

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation"""
        result = super().to_dict()
        result["arguments"] = [
            arg.to_dict() if hasattr(arg, "to_dict") else str(arg)
            for arg in self.arguments
        ]
        return result

@dataclass
class Formula(SymbolicEntity):
    """Represents a logical formula (atomic or compound)"""
    operator: Optional[str] = None
    arguments: List[Union["Formula", Term]] = field(default_factory=list)
    quantifier: Optional[str] = None
    variables: List[Term] = field(default_factory=list)

    def __post_init__(self):
        if not self.entity_type:
            if not self.operator and not self.quantifier:
                self.entity_type = SymbolicEntityType.ATOM
            else:
                self.entity_type = SymbolicEntityType.FORMULA

    def __str__(self):
        if self.quantifier:
            vars_str = ", ".join(str(var) for var in self.variables)
            arg_str = str(self.arguments[0]) if self.arguments else ""
            return f"{self.quantifier}{vars_str}. {arg_str}"

        if not self.operator:
            # Atomic formula
            args_str = ", ".join(str(arg) for arg in self.arguments)
            return f"{self.symbol}({args_str})"

        # Compound formula
        if len(self.arguments) == 1:
            # Unary operator (e.g., NOT)
            return f"{self.operator}({str(self.arguments[0])})"

        # Binary or n-ary operator
        args_str = f" {self.operator} ".join(str(arg) for arg in self.arguments)
        return f"({args_str})"

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation"""
        result = super().to_dict()
        result["operator"] = self.operator
        result["arguments"] = [
            arg.to_dict() if hasattr(arg, "to_dict") else str(arg)
            for arg in self.arguments
        ]
        result["quantifier"] = self.quantifier
        result["variables"] = [var.to_dict() for var in self.variables]
        return result

@dataclass
class Rule(SymbolicEntity):
    """Represents an inference rule or axiom"""
    premises: List[Formula] = field(default_factory=list)
    conclusion: Optional[Formula] = None
    bidirectional: bool = False
    weight: float = 1.0

    def __post_init__(self):
        self.entity_type = SymbolicEntityType.RULE

    def __str__(self):
        premises_str = ", ".join(str(premise) for premise in self.premises)
        conclusion_str = str(self.conclusion) if self.conclusion else ""
        connector = "⟺" if self.bidirectional else "⟹"
        return f"{premises_str} {connector} {conclusion_str}"

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation"""
        result = super().to_dict()
        result["premises"] = [premise.to_dict() for premise in self.premises]
        result["conclusion"] = self.conclusion.to_dict() if self.conclusion else None
        result["bidirectional"] = self.bidirectional
        result["weight"] = self.weight
        return result

@dataclass
class InferenceStep:
    """Represents a single step in an inference process"""
    step_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    rule_id: Optional[str] = None
    premises: List[str] = field(default_factory=list)  # IDs of formulas used as premises
    conclusion: Optional[str] = None  # ID of concluded formula
    substitutions: Dict[str, str] = field(default_factory=dict)  # Variable substitutions
    confidence: float = 1.0
    timestamp: float = field(default_factory=time.time)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation"""
        return {
            "step_id": self.step_id,
            "rule_id": self.rule_id,
            "premises": self.premises,
            "conclusion": self.conclusion,
            "substitutions": self.substitutions,
            "confidence": self.confidence,
            "timestamp": self.timestamp
        }

@dataclass
class InferenceResult:
    """Result of a symbolic reasoning process"""
    success: bool
    proved_formulas: List[Formula] = field(default_factory=list)
    inference_steps: List[InferenceStep] = field(default_factory=list)
    confidence: float = 1.0
    execution_time: float = 0.0
    message: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation"""
        return {
            "success": self.success,
            "proved_formulas": [f.to_dict() for f in self.proved_formulas],
            "inference_steps": [step.to_dict() for step in self.inference_steps],
            "confidence": self.confidence,
            "execution_time": self.execution_time,
            "message": self.message
        }

class KnowledgeBase:
    """
    Knowledge base for symbolic reasoning, storing facts, rules, and supporting
    belief revision and consistency management.
    """

    def __init__(self):
        """Initialize the knowledge base"""
        self.facts = {}  # Formula ID -> Formula
        self.rules = {}  # Rule ID -> Rule
        self.predicates = {}  # Predicate symbol -> metadata
        self.functions = {}  # Function symbol -> metadata

        # Indexes for efficient querying
        self.fact_index = defaultdict(set)  # Predicate symbol -> set of fact IDs
        self.rule_index = defaultdict(set)  # Predicate in conclusion -> set of rule IDs

        # Belief tracking
        self.confidence = {}  # Formula/Rule ID -> confidence value
        self.provenance = {}  # Formula/Rule ID -> source information
        self.timestamps = {}  # Formula/Rule ID -> timestamp

        # For conflict resolution
        self.dependencies = defaultdict(set)  # Formula ID -> set of IDs it depends on
        self.contradictions = set()  # Pairs of contradictory formula IDs

    def add_fact(self, formula: Formula, confidence: float = 1.0,
                source: Optional[str] = None) -> bool:
        """
        Add a fact to the knowledge base.

        Args:
            formula: The fact to add
            confidence: Confidence value (0.0 to 1.0)
            source: Optional source information

        Returns:
            True if added successfully, False if rejected
        """
        # Check for existing identical fact
        for existing_id, existing_formula in self.facts.items():
            if self._formulas_equivalent(formula, existing_formula):
                # Update confidence if new fact has higher confidence
                if confidence > self.confidence.get(existing_id, 0.0):
                    self.confidence[existing_id] = confidence
                    if source:
                        self.provenance[existing_id] = source
                    self.timestamps[existing_id] = time.time()
                return True

        # Add new fact
        self.facts[formula.entity_id] = formula
        self.confidence[formula.entity_id] = confidence
        self.timestamps[formula.entity_id] = time.time()

        if source:
            self.provenance[formula.entity_id] = source

        # Update indexes
        if formula.entity_type == SymbolicEntityType.ATOM:
            self.fact_index[formula.symbol].add(formula.entity_id)

        # Check for contradictions
        self._check_contradictions(formula)

        return True

    def add_rule(self, rule: Rule, confidence: float = 1.0,
                source: Optional[str] = None) -> bool:
        """
        Add a rule to the knowledge base.

        Args:
            rule: The rule to add
            confidence: Confidence value (0.0 to 1.0)
            source: Optional source information

        Returns:
            True if added successfully, False if rejected
        """
        # Check for existing identical rule
        for existing_id, existing_rule in self.rules.items():
            if self._rules_equivalent(rule, existing_rule):
                # Update confidence if new rule has higher confidence
                if confidence > self.confidence.get(existing_id, 0.0):
                    self.confidence[existing_id] = confidence
                    if source:
                        self.provenance[existing_id] = source
                    self.timestamps[existing_id] = time.time()
                return True

        # Add new rule
        self.rules[rule.entity_id] = rule
        self.confidence[rule.entity_id] = confidence
        self.timestamps[rule.entity_id] = time.time()

        if source:
            self.provenance[rule.entity_id] = source

        # Update indexes
        if rule.conclusion and rule.conclusion.entity_type == SymbolicEntityType.ATOM:
            self.rule_index[rule.conclusion.symbol].add(rule.entity_id)

        return True

    def get_fact(self, fact_id: str) -> Optional[Formula]:
        """Retrieve a fact by ID"""
        return self.facts.get(fact_id)

    def get_rule(self, rule_id: str) -> Optional[Rule]:
        """Retrieve a rule by ID"""
        return self.rules.get(rule_id)

    def query_facts(self, predicate: Optional[str] = None,
                   confidence_threshold: float = 0.0) -> List[Formula]:
        """
        Query facts based on predicate and confidence.

        Args:
            predicate: Optional predicate symbol to filter by
            confidence_threshold: Minimum confidence value

        Returns:
            List of matching facts
        """
        if predicate:
            fact_ids = self.fact_index.get(predicate, set())
        else:
            fact_ids = set(self.facts.keys())

        # Filter by confidence
        facts = []
        for fact_id in fact_ids:
            if self.confidence.get(fact_id, 0.0) >= confidence_threshold:
                facts.append(self.facts[fact_id])

        return facts

    def query_rules(self, conclusion_predicate: Optional[str] = None,
                   confidence_threshold: float = 0.0) -> List[Rule]:
        """
        Query rules based on conclusion predicate and confidence.

        Args:
            conclusion_predicate: Optional predicate symbol in the conclusion
            confidence_threshold: Minimum confidence value

        Returns:
            List of matching rules
        """
        if conclusion_predicate:
            rule_ids = self.rule_index.get(conclusion_predicate, set())
        else:
            rule_ids = set(self.rules.keys())

        # Filter by confidence
        rules = []
        for rule_id in rule_ids:
            if self.confidence.get(rule_id, 0.0) >= confidence_threshold:
                rules.append(self.rules[rule_id])

        return rules

    def remove_fact(self, fact_id: str) -> bool:
        """Remove a fact from the knowledge base"""
        if fact_id not in self.facts:
            return False

        fact = self.facts[fact_id]

        # Remove from indexes
        if fact.entity_type == SymbolicEntityType.ATOM:
            self.fact_index[fact.symbol].discard(fact_id)

        # Remove from contradictions
        to_remove = set()
        for pair in self.contradictions:
            if fact_id in pair:
                to_remove.add(pair)
        self.contradictions -= to_remove

        # Remove from dependencies
        self.dependencies.pop(fact_id, None)
        for deps in self.dependencies.values():
            deps.discard(fact_id)

        # Remove fact
        del self.facts[fact_id]
        self.confidence.pop(fact_id, None)
        self.provenance.pop(fact_id, None)
        self.timestamps.pop(fact_id, None)

        return True

    def remove_rule(self, rule_id: str) -> bool:
        """Remove a rule from the knowledge base"""
        if rule_id not in self.rules:
            return False

        rule = self.rules[rule_id]

        # Remove from indexes
        if rule.conclusion and rule.conclusion.entity_type == SymbolicEntityType.ATOM:
            self.rule_index[rule.conclusion.symbol].discard(rule_id)

        # Remove rule
        del self.rules[rule_id]
        self.confidence.pop(rule_id, None)
        self.provenance.pop(rule_id, None)
        self.timestamps.pop(rule_id, None)

        return True

    def get_contradictions(self) -> List[Tuple[Formula, Formula]]:
        """Get all contradictory facts in the knowledge base"""
        result = []
        for fact1_id, fact2_id in self.contradictions:
            if fact1_id in self.facts and fact2_id in self.facts:
                result.append((self.facts[fact1_id], self.facts[fact2_id]))
        return result

    def resolve_contradiction(self, fact1_id: str, fact2_id: str,
                             keep_id: Optional[str] = None) -> None:
        """
        Resolve a contradiction by removing one of the facts.

        Args:
            fact1_id: ID of the first fact
            fact2_id: ID of the second fact
            keep_id: ID of the fact to keep (if None, keep the highest confidence one)
        """
        if (fact1_id, fact2_id) not in self.contradictions and (fact2_id, fact1_id) not in self.contradictions:
            return

        # Determine which fact to keep
        if keep_id is None:
            conf1 = self.confidence.get(fact1_id, 0.0)
            conf2 = self.confidence.get(fact2_id, 0.0)
            keep_id = fact1_id if conf1 >= conf2 else fact2_id

        # Remove the other fact
        remove_id = fact2_id if keep_id == fact1_id else fact1_id
        self.remove_fact(remove_id)

        # Remove from contradictions
        self.contradictions.discard((fact1_id, fact2_id))
        self.contradictions.discard((fact2_id, fact1_id))

    def clear(self) -> None:
        """Clear the knowledge base"""
        self.facts.clear()
        self.rules.clear()
        self.fact_index.clear()
        self.rule_index.clear()
        self.confidence.clear()
        self.provenance.clear()
        self.timestamps.clear()
        self.dependencies.clear()
        self.contradictions.clear()

    # -----------------------------------------------------------------------
    # Internal methods
    # -----------------------------------------------------------------------

    def _formulas_equivalent(self, formula1: Formula, formula2: Formula) -> bool:
        """
        Check if two formulas are logically equivalent.
        This is a simplified implementation - a real system would use more
        sophisticated formula normalization and equivalence checking.

        Args:
            formula1: First formula
            formula2: Second formula

        Returns:
            True if formulas are equivalent, False otherwise
        """
        # Simple syntactic equality check
        # In a real system, this would use a proper logical equivalence check
        return str(formula1) == str(formula2)

    def _rules_equivalent(self, rule1: Rule, rule2: Rule) -> bool:
        """
        Check if two rules are logically equivalent.

        Args:
            rule1: First rule
            rule2: Second rule

        Returns:
            True if rules are equivalent, False otherwise
        """
        # Check bidirectionality
        if rule1.bidirectional != rule2.bidirectional:
            return False

        # Check conclusion equality
        if not self._formulas_equivalent(rule1.conclusion, rule2.conclusion):
            return False

        # Check premises (order-independent)
        if len(rule1.premises) != len(rule2.premises):
            return False

        # For simplicity, just check that all premises in rule1 have an equivalent in rule2
        # This is a simplification - a real system would handle more complex equivalence
        for premise1 in rule1.premises:
            found_match = False
            for premise2 in rule2.premises:
                if self._formulas_equivalent(premise1, premise2):
                    found_match = True
                    break
            if not found_match:
                return False

        return True

    def _check_contradictions(self, formula: Formula) -> None:
        """
        Check if a formula contradicts existing facts in the knowledge base.

        Args:
            formula: The formula to check
        """
        # This is a simplified implementation
        # A real system would use more sophisticated contradiction detection

        if formula.entity_type != SymbolicEntityType.ATOM:
            return

        # Look for simple negation contradictions
        for existing_id, existing_formula in self.facts.items():
            if existing_formula.entity_type != SymbolicEntityType.ATOM:
                continue

            # Check for negation contradiction
            # This is naive - a real system would check more complex contradictions
            if (existing_formula.symbol == formula.symbol and
                len(existing_formula.arguments) == len(formula.arguments)):

                # Check if one is a negation of the other
                # This is extremely simplified - real systems use unification and logical inference
                if existing_formula.metadata.get("negated", False) != formula.metadata.get("negated", False):

                    # Check if arguments match
                    args_match = True
                    for arg1, arg2 in zip(existing_formula.arguments, formula.arguments):
                        if str(arg1) != str(arg2):
                            args_match = False
                            break

                    if args_match:
                        self.contradictions.add((existing_id, formula.entity_id))

class SymbolicEngine:
    """
    Core symbolic reasoning engine for GrandNexus, supporting formal logical
    operations, theorem proving, and knowledge representation.
    """

    def __init__(self, nexus_core: Optional[NexusCore] = None,
                 semantic_memory: Optional[SemanticMemory] = None,
                 working_memory: Optional[WorkingMemory] = None,
                 symbolic_parser: Optional[SymbolicParser] = None,
                 graph_engine: Optional[AbstractGraphEngine] = None):
        """
        Initialize the symbolic reasoning engine.

        Args:
            nexus_core: Reference to the NexusCore for system integration
            semantic_memory: Reference to semantic memory for knowledge retrieval
            working_memory: Reference to working memory for context
            symbolic_parser: Reference to symbolic parser for parsing logical expressions
            graph_engine: Reference to graph engine for graph-based reasoning
        """
        self.logger = logging.getLogger("GrandNexus.Reasoning.SymbolicEngine")
        self.nexus_core = nexus_core
        self.semantic_memory = semantic_memory
        self.working_memory = working_memory
        self.symbolic_parser = symbolic_parser
        self.graph_engine = graph_engine

        # Internal state
        self.instance_id = str(uuid.uuid4())[:8]
        self.initialized = False
        self.lock = threading.RLock()

        # Knowledge base
        self.knowledge_base = KnowledgeBase()

        # Inference rules registry
        self.inference_rules = {}
        self.inference_strategies = {}

        # Reasoning statistics
        self.stats = defaultdict(lambda: defaultdict(float))

        self.logger.info(f"SymbolicEngine initialized with ID={self.instance_id}")

    def initialize(self) -> bool:
        """Initialize the engine and load standard inference rules"""
        if self.initialized:
            return True

        with self.lock:
            try:
                # Register standard inference rules
                self._register_standard_inference_rules()

                # Register inference strategies
                self._register_inference_strategies()

                # Connect to other modules
                if self.nexus_core:
                    self._register_with_nexus_core()

                self.initialized = True
                self.logger.info("SymbolicEngine initialization complete")
                return True
            except Exception as e:
                self.logger.error(f"SymbolicEngine initialization failed: {str(e)}")
                return False

    def parse_formula(self, formula_str: str, context: Optional[Dict[str, Any]] = None) -> Optional[Formula]:
        """
        Parse a logical formula from string representation.

        Args:
            formula_str: String representation of the formula
            context: Optional context information

        Returns:
            Parsed Formula object, or None if parsing fails
        """
        if not self.symbolic_parser:
            self.logger.error("No symbolic parser available")
            return None

        if not self.initialized:
            self.initialize()

        # Set context for logical expression parsing
        if context is None:
            context = {}
        context["expr_type"] = SymbolicType.LOGICAL

        # Parse the formula using symbolic parser
        parse_result = self.symbolic_parser.parse(formula_str, expected_type=SymbolicType.LOGICAL, context=context)

        if not parse_result.success:
            self.logger.warning(f"Failed to parse formula: {formula_str}")
            return None

        # Convert the parsed result to a Formula object
        try:
            formula = self._parse_result_to_formula(parse_result)
            return formula
        except Exception as e:
            self.logger.error(f"Error converting parse result to formula: {str(e)}")
            return None

    def parse_rule(self, rule_str: str, context: Optional[Dict[str, Any]] = None) -> Optional[Rule]:
        """
        Parse a logical rule from string representation.

        Args:
            rule_str: String representation of the rule
            context: Optional context information

        Returns:
            Parsed Rule object, or None if parsing fails
        """
        if not self.symbolic_parser:
            self.logger.error("No symbolic parser available")
            return None

        if not self.initialized:
            self.initialize()

        # Split into premises and conclusion
        if "=>" in rule_str:
            parts = rule_str.split("=>")
            bidirectional = False
        elif "<=>" in rule_str:
            parts = rule_str.split("<=>")
            bidirectional = True
        else:
            self.logger.warning(f"Rule must contain => or <=> operator: {rule_str}")
            return None

        if len(parts) != 2:
            self.logger.warning(f"Invalid rule format: {rule_str}")
            return None

        premises_str = parts[0].strip()
        conclusion_str = parts[1].strip()

        # Parse premises
        premises = []
        for premise_str in premises_str.split(","):
            premise = self.parse_formula(premise_str.strip(), context)
            if premise:
                premises.append(premise)
            else:
                self.logger.warning(f"Failed to parse premise: {premise_str}")
                return None

        # Parse conclusion
        conclusion = self.parse_formula(conclusion_str, context)
        if not conclusion:
            self.logger.warning(f"Failed to parse conclusion: {conclusion_str}")
            return None

        # Create rule
        rule = Rule(
            symbol=rule_str,
            premises=premises,
            conclusion=conclusion,
            bidirectional=bidirectional
        )

        return rule

    def add_fact(self, fact: Union[Formula, str], confidence: float = 1.0,
                source: Optional[str] = None) -> bool:
        """
        Add a fact to the knowledge base.

        Args:
            fact: The fact to add (either Formula object or string representation)
            confidence: Confidence value (0.0 to 1.0)
            source: Optional source information

        Returns:
            True if added successfully, False otherwise
        """
        if not self.initialized:
            self.initialize()

        # Parse if string
        if isinstance(fact, str):
            formula = self.parse_formula(fact)
            if not formula:
                return False
        else:
            formula = fact

        # Add to knowledge base
        return self.knowledge_base.add_fact(formula, confidence, source)

    def add_rule(self, rule: Union[Rule, str], confidence: float = 1.0,
                source: Optional[str] = None) -> bool:
        """
        Add a rule to the knowledge base.

        Args:
            rule: The rule to add (either Rule object or string representation)
            confidence: Confidence value (0.0 to 1.0)
            source: Optional source information

        Returns:
            True if added successfully, False otherwise
        """
        if not self.initialized:
            self.initialize()

        # Parse if string
        if isinstance(rule, str):
            rule_obj = self.parse_rule(rule)
            if not rule_obj:
                return False
        else:
            rule_obj = rule

        # Add to knowledge base
        return self.knowledge_base.add_rule(rule_obj, confidence, source)

    def query_facts(self, predicate: Optional[str] = None,
                   confidence_threshold: float = 0.0) -> List[Formula]:
        """
        Query facts from the knowledge base.

        Args:
            predicate: Optional predicate symbol to filter by
            confidence_threshold: Minimum confidence value

        Returns:
            List of matching facts
        """
        if not self.initialized:
            self.initialize()

        return self.knowledge_base.query_facts(predicate, confidence_threshold)

    def query_rules(self, conclusion_predicate: Optional[str] = None,
                   confidence_threshold: float = 0.0) -> List[Rule]:
        """
        Query rules from the knowledge base.

        Args:
            conclusion_predicate: Optional predicate symbol in the conclusion
            confidence_threshold: Minimum confidence value

        Returns:
            List of matching rules
        """
        if not self.initialized:
            self.initialize()

        return self.knowledge_base.query_rules(conclusion_predicate, confidence_threshold)

    def prove(self, goal: Union[Formula, str],
             strategy: InferenceStrategy = InferenceStrategy.BACKWARD_CHAINING,
             context: Optional[Dict[str, Any]] = None) -> InferenceResult:
        """
        Attempt to prove a goal using the specified inference strategy.

        Args:
            goal: The goal formula to prove
            strategy: The inference strategy to use
            context: Optional contextual information

        Returns:
            InferenceResult object with the proof result
        """
        if not self.initialized:
            self.initialize()

        # Parse goal if string
        if isinstance(goal, str):
            goal_formula = self.parse_formula(goal)
            if not goal_formula:
                return InferenceResult(
                    success=False,
                    message="Failed to parse goal formula"
                )
        else:
            goal_formula = goal

        # Prepare context
        if context is None:
            context = {}

        # Select strategy
        strategy_func = self.inference_strategies.get(strategy)
        if not strategy_func:
            return InferenceResult(
                success=False,
                message=f"Unsupported inference strategy: {strategy}"
            )

        # Record execution time
        start_time = time.time()

        # Execute strategy
        try:
            result = strategy_func(goal_formula, context)
            result.execution_time = time.time() - start_time

            # Update stats
            self.stats["proof_attempts"][strategy.name] += 1
            if result.success:
                self.stats["proof_successes"][strategy.name] += 1

            return result
        except Exception as e:
            self.logger.error(f"Error during proof: {str(e)}")
            return InferenceResult(
                success=False,
                message=f"Error during proof: {str(e)}",
                execution_time=time.time() - start_time
            )

    def check_consistency(self) -> Dict[str, Any]:
        """
        Check the consistency of the knowledge base.

        Returns:
            Dictionary with consistency information
        """
        if not self.initialized:
            self.initialize()

        contradictions = self.knowledge_base.get_contradictions()

        return {
            "consistent": len(contradictions) == 0,
            "contradictions": [{
                "fact1": str(fact1),
                "fact2": str(fact2),
                "fact1_id": fact1.entity_id,
                "fact2_id": fact2.entity_id
            } for fact1, fact2 in contradictions]
        }

    def resolve_contradictions(self, strategy: str = "confidence") -> int:
        """
        Resolve contradictions in the knowledge base.

        Args:
            strategy: Strategy for resolving contradictions
                     ("confidence", "recency", "custom")

        Returns:
            Number of contradictions resolved
        """
        if not self.initialized:
            self.initialize()

        contradictions = self.knowledge_base.get_contradictions()
        resolved = 0

        for fact1, fact2 in contradictions:
            fact1_id = fact1.entity_id
            fact2_id = fact2.entity_id

            if strategy == "confidence":
                # Keep the fact with higher confidence
                conf1 = self.knowledge_base.confidence.get(fact1_id, 0.0)
                conf2 = self.knowledge_base.confidence.get(fact2_id, 0.0)
                keep_id = fact1_id if conf1 >= conf2 else fact2_id

            elif strategy == "recency":
                # Keep the most recent fact
                time1 = self.knowledge_base.timestamps.get(fact1_id, 0.0)
                time2 = self.knowledge_base.timestamps.get(fact2_id, 0.0)
                keep_id = fact1_id if time1 >= time2 else fact2_id

            else:
                # Default to confidence
                conf1 = self.knowledge_base.confidence.get(fact1_id, 0.0)
                conf2 = self.knowledge_base.confidence.get(fact2_id, 0.0)
                keep_id = fact1_id if conf1 >= conf2 else fact2_id

            self.knowledge_base.resolve_contradiction(fact1_id, fact2_id, keep_id)
            resolved += 1

        return resolved

    def clear_knowledge_base(self) -> None:
        """Clear the knowledge base"""
        if not self.initialized:
            self.initialize()

        self.knowledge_base.clear()

    def get_stats(self) -> Dict[str, Any]:
        """Get reasoning statistics"""
        return {k: dict(v) for k, v in self.stats.items()}

    def register_inference_rule(self, rule_name: str, rule_func: Callable) -> bool:
        """
        Register a custom inference rule.

        Args:
            rule_name: Name of the inference rule
            rule_func: Function implementing the rule

        Returns:
            True if registered successfully, False otherwise
        """
        with self.lock:
            self.inference_rules[rule_name] = rule_func
            self.logger.info(f"Registered inference rule: {rule_name}")
            return True

    def register_inference_strategy(self, strategy: InferenceStrategy,
                                  strategy_func: Callable) -> bool:
        """
        Register a custom inference strategy.

        Args:
            strategy: The inference strategy type
            strategy_func: Function implementing the strategy

        Returns:
            True if registered successfully, False otherwise
        """
        with self.lock:
            self.inference_strategies[strategy] = strategy_func
            self.logger.info(f"Registered inference strategy: {strategy.name}")
            return True

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
            if self.semantic_memory:
                dependencies.append("semantic_memory")
            if self.working_memory:
                dependencies.append("working_memory")
            if self.symbolic_parser:
                dependencies.append("symbolic_parser")
            if self.graph_engine:
                dependencies.append("graph_engine")

            self.nexus_core.register_module(
                name="symbolic_engine",
                module=self,
                dependencies=dependencies
            )

            self.logger.info("Successfully registered with NexusCore")
        except Exception as e:
            self.logger.error(f"Failed to register with NexusCore: {str(e)}")

    def _register_standard_inference_rules(self) -> None:
        """Register standard logical inference rules"""
        # Modus Ponens: A, A => B ⊢ B
        self.register_inference_rule("modus_ponens", self._apply_modus_ponens)

        # Modus Tollens: ¬B, A => B ⊢ ¬A
        self.register_inference_rule("modus_tollens", self._apply_modus_tollens)

        # Conjunction Introduction: A, B ⊢ A ∧ B
        self.register_inference_rule("conjunction_introduction", self._apply_conjunction_introduction)

        # Conjunction Elimination: A ∧ B ⊢ A, B
        self.register_inference_rule("conjunction_elimination", self._apply_conjunction_elimination)

        # Disjunction Introduction: A ⊢ A ∨ B
        self.register_inference_rule("disjunction_introduction", self._apply_disjunction_introduction)

        # Disjunction Elimination: A ∨ B, ¬A ⊢ B
        self.register_inference_rule("disjunction_elimination", self._apply_disjunction_elimination)

        # Double Negation Elimination: ¬¬A ⊢ A
        self.register_inference_rule("double_negation_elimination", self._apply_double_negation_elimination)

    def _register_inference_strategies(self) -> None:
        """Register inference strategies"""
        self.register_inference_strategy(InferenceStrategy.FORWARD_CHAINING, self._forward_chaining)
        self.register_inference_strategy(InferenceStrategy.BACKWARD_CHAINING, self._backward_chaining)
        self.register_inference_strategy(InferenceStrategy.RESOLUTION, self._resolution)
        self.register_inference_strategy(InferenceStrategy.HYBRID, self._hybrid_inference)

    def _parse_result_to_formula(self, parse_result: ParseResult) -> Formula:
        """
        Convert a parse result to a Formula object.

        Args:
            parse_result: The parse result from symbolic parser

        Returns:
            Formula object representing the parsed expression
        """
        # This is a simplified implementation
        # A real implementation would carefully convert the AST

        if not parse_result.ast:
            raise ValueError("Parse result has no AST")

        ast_node = parse_result.ast

        if ast_node.node_type == "equation":
            # Handle equation (A = B)
            if len(ast_node.children) != 2:
                raise ValueError("Equation must have exactly 2 children")

            left = self._ast_node_to_term(ast_node.children[0])
            right = self._ast_node_to_term(ast_node.children[1])

            return Formula(
                entity_type=SymbolicEntityType.ATOM,
                symbol="equals",
                arguments=[left, right]
            )

        elif ast_node.node_type == "operation":
            # Handle logical operations (AND, OR, NOT, etc.)
            op = ast_node.value.upper()

            children = []
            for child in ast_node.children:
                children.append(self._parse_result_to_formula(ParseResult(
                    success=True,
                    expr_type=SymbolicType.LOGICAL,
                    ast=child
                )))

            return Formula(
                entity_type=SymbolicEntityType.FORMULA,
                symbol=op,
                operator=op,
                arguments=children
            )

        elif ast_node.node_type == "function" or ast_node.node_type == "atom":
            # Handle predicate application (P(x, y))
            func_name = ast_node.value

            args = []
            for child in ast_node.children:
                args.append(self._ast_node_to_term(child))

            return Formula(
                entity_type=SymbolicEntityType.ATOM,
                symbol=func_name,
                arguments=args
            )

        # Default case
        return Formula(
            entity_type=SymbolicEntityType.FORMULA,
            symbol=str(ast_node)
        )

    def _ast_node_to_term(self, ast_node) -> Term:
        """
        Convert an AST node to a Term object.

        Args:
            ast_node: The AST node to convert

        Returns:
            Term object representing the AST node
        """
        if ast_node.node_type == "variable":
            return Term(
                entity_type=SymbolicEntityType.VARIABLE,
                symbol=ast_node.value
            )

        elif ast_node.node_type == "constant" or ast_node.node_type == "number":
            return Term(
                entity_type=SymbolicEntityType.CONSTANT,
                symbol=str(ast_node.value)
            )

        elif ast_node.node_type == "function":
            func_name = ast_node.value

            args = []
            for child in ast_node.children:
                args.append(self._ast_node_to_term(child))

            return Term(
                entity_type=SymbolicEntityType.TERM,
                symbol=func_name,
                arguments=args
            )

        # Default case
        return Term(
            entity_type=SymbolicEntityType.CONSTANT,
            symbol=str(ast_node)
        )

    # -----------------------------------------------------------------------
    # Inference rules implementation
    # -----------------------------------------------------------------------

    def _apply_modus_ponens(self, premises: List[Formula]) -> Optional[Formula]:
        """
        Apply modus ponens: A, A => B ⊢ B

        Args:
            premises: List of premises to apply the rule to

        Returns:
            The derived formula, or None if the rule is not applicable
        """
        # This is a simplified implementation
        # A real implementation would use unification and proper logical inference

        if len(premises) < 2:
            return None

        # For all pairs of premises
        for i, premise1 in enumerate(premises):
            for j, premise2 in enumerate(premises):
                if i == j:
                    continue

                # Check if premise2 is an implication
                if (premise2.entity_type == SymbolicEntityType.FORMULA and
                    premise2.operator == "IMPLIES" and
                    len(premise2.arguments) == 2):

                    # Check if premise1 matches the antecedent
                    antecedent = premise2.arguments[0]
                    if self._formulas_match(premise1, antecedent):
                        # Return the consequent
                        return premise2.arguments[1]

        return None

    def _apply_modus_tollens(self, premises: List[Formula]) -> Optional[Formula]:
        """
        Apply modus tollens: ¬B, A => B ⊢ ¬A

        Args:
            premises: List of premises to apply the rule to

        Returns:
            The derived formula, or None if the rule is not applicable
        """
        # This is a simplified implementation
        # A real implementation would use unification and proper logical inference

        if len(premises) < 2:
            return None

        # For all pairs of premises
        for i, premise1 in enumerate(premises):
            for j, premise2 in enumerate(premises):
                if i == j:
                    continue

                # Check if premise1 is a negation
                if (premise1.entity_type == SymbolicEntityType.FORMULA and
                    premise1.operator == "NOT" and
                    len(premise1.arguments) == 1):

                    negated_formula = premise1.arguments[0]

                    # Check if premise2 is an implication
                    if (premise2.entity_type == SymbolicEntityType.FORMULA and
                        premise2.operator == "IMPLIES" and
                        len(premise2.arguments) == 2):

                        # Check if negated_formula matches the consequent
                        consequent = premise2.arguments[1]
                        if self._formulas_match(negated_formula, consequent):
                            # Return the negation of the antecedent
                            antecedent = premise2.arguments[0]
                            return Formula(
                                entity_type=SymbolicEntityType.FORMULA,
                                operator="NOT",
                                arguments=[antecedent]
                            )

        return None

    def _apply_conjunction_introduction(self, premises: List[Formula]) -> Optional[Formula]:
        """
        Apply conjunction introduction: A, B ⊢ A ∧ B

        Args:
            premises: List of premises to apply the rule to

        Returns:
            The derived formula, or None if the rule is not applicable
        """
        if len(premises) < 2:
            return None

        # Take the first two premises and create a conjunction
        return Formula(
            entity_type=SymbolicEntityType.FORMULA,
            operator="AND",
            arguments=premises[:2]
        )

    def _apply_conjunction_elimination(self, premises: List[Formula]) -> Optional[List[Formula]]:
        """
        Apply conjunction elimination: A ∧ B ⊢ A, B

        Args:
            premises: List of premises to apply the rule to

        Returns:
            List of derived formulas, or None if the rule is not applicable
        """
        if not premises:
            return None

        # Look for a conjunction
        for premise in premises:
            if (premise.entity_type == SymbolicEntityType.FORMULA and
                premise.operator == "AND" and
                len(premise.arguments) >= 2):

                # Return the conjuncts
                return premise.arguments

        return None

    def _apply_disjunction_introduction(self, premises: List[Formula]) -> Optional[Formula]:
        """
        Apply disjunction introduction: A ⊢ A ∨ B

        Args:
            premises: List of premises to apply the rule to

        Returns:
            The derived formula, or None if the rule is not applicable
        """
        if not premises:
            return None

        # Take the first premise and create a disjunction with a dummy formula
        dummy = Formula(
            entity_type=SymbolicEntityType.ATOM,
            symbol="dummy_formula"
        )

        return Formula(
            entity_type=SymbolicEntityType.FORMULA,
            operator="OR",
            arguments=[premises[0], dummy]
        )

    def _apply_disjunction_elimination(self, premises: List[Formula]) -> Optional[Formula]:
        """
        Apply disjunction elimination: A ∨ B, ¬A ⊢ B

        Args:
            premises: List of premises to apply the rule to

        Returns:
            The derived formula, or None if the rule is not applicable
        """
        if len(premises) < 2:
            return None

        # For all pairs of premises
        for i, premise1 in enumerate(premises):
            for j, premise2 in enumerate(premises):
                if i == j:
                    continue

                # Check if premise1 is a disjunction
                if (premise1.entity_type == SymbolicEntityType.FORMULA and
                    premise1.operator == "OR" and
                    len(premise1.arguments) == 2):

                    disjunct1 = premise1.arguments[0]
                    disjunct2 = premise1.arguments[1]

                    # Check if premise2 is a negation of disjunct1
                    if (premise2.entity_type == SymbolicEntityType.FORMULA and
                        premise2.operator == "NOT" and
                        len(premise2.arguments) == 1 and
                        self._formulas_match(premise2.arguments[0], disjunct1)):

                        # Return disjunct2
                        return disjunct2

                    # Check if premise2 is a negation of disjunct2
                    if (premise2.entity_type == SymbolicEntityType.FORMULA and
                        premise2.operator == "NOT" and
                        len(premise2.arguments) == 1 and
                        self._formulas_match(premise2.arguments[0], disjunct2)):

                        # Return disjunct1
                        return disjunct1

        return None

    def _apply_double_negation_elimination(self, premises: List[Formula]) -> Optional[Formula]:
        """
        Apply double negation elimination: ¬¬A ⊢ A

        Args:
            premises: List of premises to apply the rule to

        Returns:
            The derived formula, or None if the rule is not applicable
        """
        if not premises:
            return None

        # Look for a double negation
        for premise in premises:
            if (premise.entity_type == SymbolicEntityType.FORMULA and
                premise.operator == "NOT" and
                len(premise.arguments) == 1 and
                premise.arguments[0].entity_type == SymbolicEntityType.FORMULA and
                premise.arguments[0].operator == "NOT" and
                len(premise.arguments[0].arguments) == 1):

                # Return the doubly negated formula
                return premise.arguments[0].arguments[0]

        return None

    def _formulas_match(self, formula1: Formula, formula2: Formula) -> bool:
        """
        Check if two formulas match (for rule application).
        This is a simplified implementation that just checks string equality.
        A real implementation would use unification.

        Args:
            formula1: First formula
            formula2: Second formula

        Returns:
            True if formulas match, False otherwise
        """
        return str(formula1) == str(formula2)

    # -----------------------------------------------------------------------
    # Inference strategies implementation
    # -----------------------------------------------------------------------

    def _forward_chaining(self, goal: Formula, context: Dict[str, Any]) -> InferenceResult:
        """
        Apply forward chaining inference to prove a goal.

        Args:
            goal: The goal formula to prove
            context: Contextual information for inference

        Returns:
            InferenceResult object with the proof result
        """
        # This is a simplified implementation
        # A real implementation would be more sophisticated

        # Initialize result
        result = InferenceResult(success=False)

        # Get all facts and rules
        facts = list(self.knowledge_base.facts.values())
        rules = list(self.knowledge_base.rules.values())

        # Keep track of derived formulas
        derived = set(str(fact) for fact in facts)
        fact_id_map = {str(fact): fact.entity_id for fact in facts}

        # Add facts to proved formulas
        result.proved_formulas.extend(facts)

        # Set a limit on inference steps to avoid infinite loops
        max_steps = context.get("max_steps", 100)
        steps = 0

        # Forward chaining loop
        while steps < max_steps:
            steps += 1
            new_fact_derived = False

            # Try to apply each rule
            for rule in rules:
                if not rule.premises or not rule.conclusion:
                    continue

                # Check if all premises are satisfied
                premises_satisfied = True
                premise_ids = []

                for premise in rule.premises:
                    if str(premise) not in derived:
                        premises_satisfied = False
                        break
                    premise_ids.append(fact_id_map.get(str(premise), "unknown"))

                if not premises_satisfied:
                    continue

                # Apply the rule to derive a new fact
                conclusion = rule.conclusion

                # Check if the conclusion is already derived
                if str(conclusion) in derived:
                    continue

                # Derive the conclusion
                derived.add(str(conclusion))
                fact_id_map[str(conclusion)] = conclusion.entity_id
                result.proved_formulas.append(conclusion)

                # Record inference step
                result.inference_steps.append(InferenceStep(
                    rule_id=rule.entity_id,
                    premises=premise_ids,
                    conclusion=conclusion.entity_id
                ))

                new_fact_derived = True

                # Check if the goal is reached
                if str(conclusion) == str(goal):
                    result.success = True
                    return result

            # If no new facts were derived, we've reached a fixed point
            if not new_fact_derived:
                break

        # Check if the goal is already in the derived facts
        if str(goal) in derived:
            result.success = True

        return result

    def _backward_chaining(self, goal: Formula, context: Dict[str, Any]) -> InferenceResult:
        """
        Apply backward chaining inference to prove a goal.

        Args:
            goal: The goal formula to prove
            context: Contextual information for inference

        Returns:
            InferenceResult object with the proof result
        """
        # This is a simplified implementation
        # A real implementation would be more sophisticated

        # Initialize result
        result = InferenceResult(success=False)

        # Get all facts
        facts = list(self.knowledge_base.facts.values())

        # Initialize agenda with the goal
        agenda = [goal]

        # Keep track of proved formulas and already processed goals
        proved = {str(fact): fact for fact in facts}
        processed_goals = set()

        # Keep track of inference steps
        inference_steps = []

        # Set a limit on inference steps to avoid infinite loops
        max_steps = context.get("max_steps", 100)
        steps = 0

        # Backward chaining loop
        while agenda and steps < max_steps:
            steps += 1
            current_goal = agenda.pop()

            # Skip if already processed
            if str(current_goal) in processed_goals:
                continue

            processed_goals.add(str(current_goal))

            # Check if the goal is already proved
            if str(current_goal) in proved:
                continue

            # Get relevant rules (rules whose conclusion matches the goal)
            relevant_rules = []
            for rule in self.knowledge_base.rules.values():
                if rule.conclusion and self._formulas_match(rule.conclusion, current_goal):
                    relevant_rules.append(rule)

            # Try each relevant rule
            goal_proved = False

            for rule in relevant_rules:
                # Check if all premises can be proved
                all_premises_proved = True
                rule_premises = []

                for premise in rule.premises:
                    if str(premise) in proved:
                        # Premise already proved
                        rule_premises.append(proved[str(premise)])
                    else:
                        # Add premise to agenda
                        agenda.append(premise)
                        all_premises_proved = False

                if all_premises_proved:
                    # All premises are proved, so the current goal is proved
                    proved[str(current_goal)] = current_goal

                    # Record inference step
                    inference_steps.append(InferenceStep(
                        rule_id=rule.entity_id,
                        premises=[premise.entity_id for premise in rule_premises],
                        conclusion=current_goal.entity_id
                    ))

                    goal_proved = True
                    break

            # If no rule could prove the goal, it remains unproved
            if not goal_proved:
                # Try to find the goal in the facts
                for fact in facts:
                    if self._formulas_match(fact, current_goal):
                        proved[str(current_goal)] = current_goal
                        goal_proved = True
                        break

            # If still not proved, try to apply inference rules directly
            if not goal_proved:
                for rule_name, rule_func in self.inference_rules.items():
                    derived = rule_func([proved[str(p)] for p in proved])
                    if derived and self._formulas_match(derived, current_goal):
                        proved[str(current_goal)] = current_goal
                        goal_proved = True
                        break

        # Check if the original goal was proved
        result.success = str(goal) in proved

        # Add all proved formulas to the result
        result.proved_formulas = list(proved.values())

        # Add inference steps to the result
        result.inference_steps = inference_steps

        return result

    def _resolution(self, goal: Formula, context: Dict[str, Any]) -> InferenceResult:
        """
        Apply resolution inference to prove a goal.

        Args:
            goal: The goal formula to prove
            context: Contextual information for inference

        Returns:
            InferenceResult object with the proof result
        """
        # Resolution is more complex and typically used for refutation proofs
        # This is a very simplified placeholder implementation

        # For now, fall back to backward chaining
        self.logger.warning("Resolution strategy not fully implemented, falling back to backward chaining")
        return self._backward_chaining(goal, context)

    def _hybrid_inference(self, goal: Formula, context: Dict[str, Any]) -> InferenceResult:
        """
        Apply a hybrid inference strategy to prove a goal.

        Args:
            goal: The goal formula to prove
            context: Contextual information for inference

        Returns:
            InferenceResult object with the proof result
        """
        # Try backward chaining first
        result = self._backward_chaining(goal, context)

        # If successful, return the result
        if result.success:
            return result

        # If not, try forward chaining
        forward_result = self._forward_chaining(goal, context)

        # If forward chaining is successful, return that result
        if forward_result.success:
            return forward_result

        # If neither succeeded, return the backward chaining result
        # (could also choose to return a combined result)
        return result