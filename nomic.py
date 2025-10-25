#!/usr/bin/env python3
"""
Nomic - Semantic Rule Enforcement for C

High-level goals:
- Parse C (via Clang) into a rich IR (AST, CFG, symbols, macros, etc.)
- Load declarative rules from YAML
- Evaluate rules against IR + project-wide context
- Emit rich structured JSON for CI / IDEs

This file is intentionally single-module so you can iterate quickly in VS Code / Codex.
Later, you can split it into packages.
"""

from __future__ import annotations
from dataclasses import dataclass, field, asdict
from typing import List, Dict, Optional, Tuple, Union, Literal, Any, Set
import argparse
import ast
from collections import deque
from contextlib import contextmanager
import functools
import json
import operator
import os
import re
import shlex
import sys
try:  # Optional dependency; tool still runs without PyYAML.
    import yaml  # type: ignore
except ImportError:  # pragma: no cover - environment without PyYAML
    yaml = None  # type: ignore
try:  # Optional libclang integration.
    from clang import cindex as clang_cindex  # type: ignore
except ImportError:  # pragma: no cover - environment without libclang
    clang_cindex = None  # type: ignore


# ============================================================
# =============== SOURCE LOCATION & CONTEXT ==================
# ============================================================

@dataclass
class SourceLocation:
    file: str
    line: int
    column: int


@dataclass
class SourceRange:
    file: str
    line_start: int
    col_start: int
    line_end: int
    col_end: int


@dataclass
class PreprocessorContext:
    """
    Records what #defines / conditional compilation context was active
    for a given node.
    """
    active_defines: Set[str] = field(default_factory=set)
    guard_condition: Optional[str] = None  # e.g. "HW_TIMER1 && DEBUG"


@dataclass
class Annotation:
    """
    Inline annotations / pragmas / special comments relevant to rules.
    Example: // @ALLOW_BLOCKING_IN_ISR
    """
    text: str
    location: SourceLocation


# ============================================================
# ==================== TYPE DECLARATIONS =====================
# ============================================================

@dataclass
class FieldDecl:
    name: str
    ctype: str  # resolved type string
    is_bitfield: bool = False
    bit_width: Optional[int] = None
    source_range: Optional[SourceRange] = None


@dataclass
class TypeDecl:
    """
    typedefs, struct/union/enum.
    """
    name: str
    kind: Literal["typedef", "struct", "union", "enum"]
    fields: List[FieldDecl] = field(default_factory=list)  # for struct/union/enum
    underlying_type: Optional[str] = None  # e.g. "unsigned long *" for typedef
    attributes: List[str] = field(default_factory=list)    # e.g. ['packed', 'aligned(4)']
    visibility: Literal["file_local", "project_visible"] = "project_visible"
    source_range: Optional[SourceRange] = None
    preprocessor: PreprocessorContext = field(default_factory=PreprocessorContext)
    annotations: List[Annotation] = field(default_factory=list)


# ============================================================
# ===================== SYMBOL USAGE =========================
# ============================================================

@dataclass
class WriteSite:
    location: SourceRange
    in_branch: Optional[str] = None     # "then", "else", "case", etc.
    in_loop: Optional[str] = None       # "for", "while", "do_while"
    in_macro_expansion: bool = False
    guarded_condition: Optional[str] = None  # "if (err != 0)"
    preprocessor: PreprocessorContext = field(default_factory=PreprocessorContext)


@dataclass
class ReadSite:
    location: SourceRange
    in_branch: Optional[str] = None
    in_loop: Optional[str] = None
    in_macro_expansion: bool = False
    guarded_condition: Optional[str] = None
    preprocessor: PreprocessorContext = field(default_factory=PreprocessorContext)


# ============================================================
# ======================= VARIABLES ==========================
# ============================================================

@dataclass
class Variable:
    """
    Represents globals, statics, locals, params, etc.
    """
    name: str
    ctype: str  # fully resolved C type
    storage: Literal["auto", "static", "extern", "register", "typedef_name", "unknown"] = "unknown"
    linkage: Literal["internal", "external", "none"] = "none"  # internal=static, external=linker-visible
    is_const: bool = False
    is_volatile: bool = False
    is_atomic: bool = False

    scope: Literal["file", "function", "block", "param"] = "file"
    decl_function: Optional[str] = None  # function name if local or param

    # Naming helpers
    prefix: Optional[str] = None
    suffix: Optional[str] = None

    source_range: Optional[SourceRange] = None
    preprocessor: PreprocessorContext = field(default_factory=PreprocessorContext)
    annotations: List[Annotation] = field(default_factory=list)

    writes: List[WriteSite] = field(default_factory=list)
    reads: List[ReadSite] = field(default_factory=list)


# ============================================================
# ===================== CALL SITES ===========================
# ============================================================

@dataclass
class CallSite:
    callee_name: str                           # text in code
    callee_symbol: Optional[str] = None        # resolved canonical name, if known

    args: List[str] = field(default_factory=list)  # simplified arg reprs
    location: Optional[SourceRange] = None

    in_branch: Optional[str] = None            # "then", "else", "case"
    in_loop: Optional[str] = None              # "for", "while", "do_while"
    in_switch: bool = False
    in_macro_expansion: bool = False

    # Semantic classification for rule logic
    is_blocking_api: bool = False
    is_allocator: bool = False
    is_lock: bool = False
    is_unlock: bool = False

    preprocessor: PreprocessorContext = field(default_factory=PreprocessorContext)


# ============================================================
# ================= CONTROL FLOW GRAPH =======================
# ============================================================

@dataclass
class BasicBlock:
    block_id: int

    statements: List[str] = field(default_factory=list)    # human-readable summaries/snippets
    calls: List[CallSite] = field(default_factory=list)
    writes: List[WriteSite] = field(default_factory=list)

    enclosing_construct: Optional[str] = None  # "if_then", "if_else", "loop", "switch_case", etc.
    branch_condition: Optional[str] = None     # textual condition for this arm

    successors: List[int] = field(default_factory=list)
    predecessors: List[int] = field(default_factory=list)
    dominators: Set[int] = field(default_factory=set)
    postdominators: Set[int] = field(default_factory=set)

    is_exit_block: bool = False

    source_range: Optional[SourceRange] = None
    preprocessor: PreprocessorContext = field(default_factory=PreprocessorContext)


@dataclass
class ControlFlowGraph:
    blocks: Dict[int, BasicBlock] = field(default_factory=dict)
    entry_block: Optional[int] = None
    exit_blocks: List[int] = field(default_factory=list)

    def all_exit_paths_postdominated_by(self, matcher: Any) -> bool:
        """
        Semantic contract:
        Returns True if for every exit path from entry_block to any exit block,
        there exists a node satisfying `matcher` that postdominates the point
        of interest (e.g. an unlock after lock).

        The actual dataflow/CFG logic will get implemented later.
        """
        # TODO: real CFG/postdom analysis
        # TODO: real CFG/postdom analysis
        return True


# ============================================================
# ================= FLOW CONSTRUCT NODES =====================
# ============================================================

@dataclass
class BlockStmt:
    """
    Generic block { ... } of code:
    - then-block of an if
    - else-block of an if
    - body of a loop
    - body of a switch case
    """
    statements: List[str] = field(default_factory=list)
    writes: List[WriteSite] = field(default_factory=list)
    calls: List[CallSite] = field(default_factory=list)
    locals_declared_here: List[Variable] = field(default_factory=list)

    source_range: Optional[SourceRange] = None
    preprocessor: PreprocessorContext = field(default_factory=PreprocessorContext)


@dataclass
class IfStmt:
    condition: str
    then_block: BlockStmt
    else_block: Optional[BlockStmt] = None

    parent_function: Optional[str] = None  # function name
    source_range: Optional[SourceRange] = None
    preprocessor: PreprocessorContext = field(default_factory=PreprocessorContext)
    annotations: List[Annotation] = field(default_factory=list)


@dataclass
class LoopStmt:
    kind: Literal["for", "while", "do_while"]
    condition: Optional[str]  # do/while has postcondition
    body: BlockStmt

    parent_function: Optional[str] = None
    source_range: Optional[SourceRange] = None
    preprocessor: PreprocessorContext = field(default_factory=PreprocessorContext)
    annotations: List[Annotation] = field(default_factory=list)


@dataclass
class SwitchCaseBlock:
    labels: List[str]                     # e.g. ["STATE_ERR", "3"] or ["default"]
    body: BlockStmt
    source_range: Optional[SourceRange] = None


@dataclass
class SwitchStmt:
    control_expr: str
    cases: List[SwitchCaseBlock] = field(default_factory=list)
    has_default: bool = False

    parent_function: Optional[str] = None
    source_range: Optional[SourceRange] = None
    preprocessor: PreprocessorContext = field(default_factory=PreprocessorContext)
    annotations: List[Annotation] = field(default_factory=list)


# ============================================================
# ======================== FUNCTION ==========================
# ============================================================

@dataclass
class Function:
    name: str
    return_type: str
    parameters: List[Variable] = field(default_factory=list)
    local_vars: List[Variable] = field(default_factory=list)

    linkage: Literal["internal", "external"] = "external"  # static => internal
    storage: Optional[str] = None  # "static", "extern", etc.

    attributes: List[str] = field(default_factory=list)  # compiler attrs like interrupt, naked
    annotations: List[Annotation] = field(default_factory=list)

    is_isr: bool = False
    is_inline: bool = False

    calls: List[CallSite] = field(default_factory=list)
    cfg: ControlFlowGraph = field(default_factory=ControlFlowGraph)
    exit_points: List[int] = field(default_factory=list)  # block_ids of exit blocks

    if_stmts: List[IfStmt] = field(default_factory=list)
    loops: List[LoopStmt] = field(default_factory=list)
    switches: List[SwitchStmt] = field(default_factory=list)

    globals_written: List[str] = field(default_factory=list)
    globals_read: List[str] = field(default_factory=list)

    source_range: Optional[SourceRange] = None
    preprocessor: PreprocessorContext = field(default_factory=PreprocessorContext)

    # For naming rules
    prefix: Optional[str] = None
    suffix: Optional[str] = None


# ============================================================
# ========================= MACROS ===========================
# ============================================================

@dataclass
class MacroDefinition:
    name: str
    kind: Literal["object_like", "function_like"]
    params: List[str] = field(default_factory=list)

    body_tokens: List[str] = field(default_factory=list)
    body_statements: List[str] = field(default_factory=list)

    is_wrapped_single_stmt: bool = False  # e.g. do { ... } while(0)

    source_range: Optional[SourceRange] = None
    preprocessor: PreprocessorContext = field(default_factory=PreprocessorContext)
    annotations: List[Annotation] = field(default_factory=list)

    prefix: Optional[str] = None
    suffix: Optional[str] = None


# ============================================================
# =================== TRANSLATION UNIT =======================
# ============================================================

@dataclass
class TranslationUnit:
    """
    Represents one compiled unit (a .c file + headers + preprocessor result).
    """
    path: str  # path to the .c file that produced this TU

    includes: List[str] = field(default_factory=list)
    macros: List[MacroDefinition] = field(default_factory=list)
    types: List[TypeDecl] = field(default_factory=list)
    globals: List[Variable] = field(default_factory=list)
    functions: List[Function] = field(default_factory=list)

    # Symbol table for quick lookup in this TU: name -> Variable
    symbol_table: Dict[str, Variable] = field(default_factory=dict)

    # Active defines at TU scope
    active_defines: Set[str] = field(default_factory=set)

    # Optional module classification (like "drivers/uart")
    module: Optional[str] = None


# ============================================================
# ===================== PROJECT DATABASE =====================
# ============================================================

@dataclass
class ProjectDB:
    """
    Project-wide index built from all TranslationUnits.
    Enables cross-TU rules like:
    - "only one definition of g_schedulerState"
    - "only code in boot/ can call enter_bootloader()"
    - "no one else can include uart_internal.h"
    """
    translation_units: List[TranslationUnit] = field(default_factory=list)

    # Indexes
    functions_by_name: Dict[str, List[Function]] = field(default_factory=dict)
    globals_by_name: Dict[str, List[Variable]] = field(default_factory=dict)
    macros_by_name: Dict[str, List[MacroDefinition]] = field(default_factory=dict)

    include_usage: Dict[str, List[str]] = field(default_factory=dict)  # header -> list of TU paths

    # caller -> set(callee)
    call_graph: Dict[str, Set[str]] = field(default_factory=dict)

    # Policy / semantic metadata:
    # e.g. { "sleep_ms": {"is_blocking_api": True} }
    function_metadata: Dict[str, Dict[str, Any]] = field(default_factory=dict)

    # Optional: directory/module access rules
    module_rules: Dict[str, Any] = field(default_factory=dict)


# ============================================================
# ======================= RULE MODELS ========================
# ============================================================

@dataclass
class Rule:
    """
    Representation of a single rule loaded from YAML.
    This mirrors the YAML contract:

    - id: unique rule id
    - description: human text
    - severity: "error" | "warning" | ...
    - scope: which IR node type the rule iterates over,
             e.g. "Function", "IfStmt", "TranslationUnit", "ProjectDB"
    - select: textual DSL block that binds targets
    - assert_code: textual DSL block describing what must be true
    - message: template for violation message
    - exceptions: list of textual conditions that suppress violations
    - tags: optional list of strings
    - fixit: optional template for suggested fix
    """
    id: str
    description: str
    severity: str
    scope: str  # "Function", "IfStmt", "MacroDefinition", "ProjectDB", etc.

    select: str
    assert_code: str
    message: str

    exceptions: List[str] = field(default_factory=list)
    tags: List[str] = field(default_factory=list)
    fixit: Optional[str] = None


@dataclass
class ViolationContextFunction:
    name: str
    is_isr: bool
    linkage: str
    attributes: List[str]


@dataclass
class ViolationContextCall:
    callee: Optional[str]
    in_macro: bool
    in_branch: Optional[str]
    in_loop: Optional[str]


@dataclass
class ViolationContextCF:
    path_sensitive: bool
    all_paths_proven: bool


@dataclass
class ViolationContextPreproc:
    active_defines: List[str]
    guard_condition: Optional[str]


@dataclass
class ViolationExtrasSymbol:
    name: str
    decl_file: Optional[str]
    is_blocking_api: Optional[bool]


@dataclass
class Violation:
    """
    Rich violation object that we will eventually serialize to JSON.
    """
    rule_id: str
    severity: str
    message: str

    location: Dict[str, Any]  # {file, line_start, col_start, line_end, col_end}

    context: Dict[str, Any] = field(default_factory=dict)
    suggested_fix: Optional[str] = None
    extras: Dict[str, Any] = field(default_factory=dict)


# ============================================================
# ==================== RULE EVALUATION =======================
# ============================================================


def _mark_safe_callable(func: Any) -> Any:
    setattr(func, "_nomic_safe_callable", True)
    return func


def _wrap_safe_callable(func: Any) -> Any:
    @functools.wraps(func)
    def _safe_wrapper(*args: Any, **kwargs: Any) -> Any:
        return func(*args, **kwargs)

    return _mark_safe_callable(_safe_wrapper)


def _wrap_dynamic_callable(func: Any) -> Any:
    def _safe_wrapper(*args: Any, **kwargs: Any) -> Any:
        return func(*args, **kwargs)

    return _mark_safe_callable(_safe_wrapper)


_SAFE_BASE_CALLABLES: Dict[str, Any] = {
    "len": _wrap_safe_callable(len),
    "any": _wrap_safe_callable(any),
    "all": _wrap_safe_callable(all),
    "sum": _wrap_safe_callable(sum),
    "min": _wrap_safe_callable(min),
    "max": _wrap_safe_callable(max),
    "sorted": _wrap_safe_callable(sorted),
    "abs": _wrap_safe_callable(abs),
}

TEMPLATE_PATTERN = re.compile(r"\{\{\s*([^{}]+?)\s*\}\}")


class ExpressionEvalError(Exception):
    """Raised when the DSL expression evaluator encounters an unsafe or invalid construct."""


class _SafeExpressionInterpreter:
    """
    Evaluates a restricted subset of Python expressions by walking the AST.
    Supports boolean logic, arithmetic, comparisons, attribute access, indexing,
    safe function calls, and literals/containers.
    """

    _BIN_OPS = {
        ast.Add: operator.add,
        ast.Sub: operator.sub,
        ast.Mult: operator.mul,
        ast.Div: operator.truediv,
        ast.FloorDiv: operator.floordiv,
        ast.Mod: operator.mod,
        ast.Pow: operator.pow,
        ast.BitAnd: operator.and_,
        ast.BitOr: operator.or_,
        ast.BitXor: operator.xor,
        ast.LShift: operator.lshift,
        ast.RShift: operator.rshift,
    }
    _UNARY_OPS = {
        ast.Not: operator.not_,
        ast.USub: operator.neg,
        ast.UAdd: operator.pos,
        ast.Invert: operator.invert,
    }

    def __init__(self) -> None:
        self._cache: Dict[str, ast.AST] = {}

    def evaluate(self, expr: str, env: Dict[str, Any]) -> Any:
        expr = expr.strip()
        if not expr:
            return True

        try:
            tree = self._cache.get(expr)
            if tree is None:
                tree = ast.parse(expr, mode="eval")
                self._cache[expr] = tree
        except SyntaxError as exc:  # pragma: no cover
            raise ExpressionEvalError(f"invalid expression '{expr}': {exc}") from exc

        return self._eval_node(tree.body, env)

    def _eval_node(self, node: ast.AST, env: Dict[str, Any]) -> Any:
        if isinstance(node, ast.BoolOp):
            if isinstance(node.op, ast.And):
                result = True
                for value in node.values:
                    result = result and bool(self._eval_node(value, env))
                    if not result:
                        break
                return result
            if isinstance(node.op, ast.Or):
                result = False
                for value in node.values:
                    result = result or bool(self._eval_node(value, env))
                    if result:
                        break
                return result
            raise ExpressionEvalError("unsupported boolean operator")

        if isinstance(node, ast.UnaryOp):
            op = self._UNARY_OPS.get(type(node.op))
            if not op:
                raise ExpressionEvalError("unsupported unary operator")
            operand = self._eval_node(node.operand, env)
            return op(operand)

        if isinstance(node, ast.BinOp):
            op = self._BIN_OPS.get(type(node.op))
            if not op:
                raise ExpressionEvalError("unsupported binary operator")
            left = self._eval_node(node.left, env)
            right = self._eval_node(node.right, env)
            return op(left, right)

        if isinstance(node, ast.Compare):
            left = self._eval_node(node.left, env)
            for operator_node, comparator in zip(node.ops, node.comparators):
                right = self._eval_node(comparator, env)
                if not self._evaluate_comparison(operator_node, left, right):
                    return False
                left = right
            return True

        if isinstance(node, ast.IfExp):
            condition = self._eval_node(node.test, env)
            branch = node.body if condition else node.orelse
            return self._eval_node(branch, env)

        if isinstance(node, ast.Attribute):
            value = self._eval_node(node.value, env)
            if node.attr.startswith("_"):
                raise ExpressionEvalError("access to private attributes is not allowed")
            attr_value = getattr(value, node.attr)
            if callable(attr_value) and not getattr(attr_value, "_nomic_safe_callable", False):
                attr_value = _wrap_dynamic_callable(attr_value)
            return attr_value

        if isinstance(node, ast.Name):
            if node.id in env:
                return env[node.id]
            raise ExpressionEvalError(f"unknown identifier '{node.id}'")

        if isinstance(node, ast.Constant):
            return node.value

        if isinstance(node, ast.GeneratorExp):
            return self._build_generator(node, env)

        if isinstance(node, ast.ListComp):
            return list(self._comprehension_values(node.generators, env, node.elt))

        if isinstance(node, ast.SetComp):
            return set(self._comprehension_values(node.generators, env, node.elt))

        if isinstance(node, ast.DictComp):
            return {
                key: value
                for key, value in self._comprehension_items(node.generators, env, node.key, node.value)
            }

        if isinstance(node, ast.Call):
            func_obj = self._eval_node(node.func, env)
            if not getattr(func_obj, "_nomic_safe_callable", False):
                raise ExpressionEvalError("call to unsafe function is not allowed")
            args = [self._eval_node(arg, env) for arg in node.args]
            kwargs = {kw.arg: self._eval_node(kw.value, env) for kw in node.keywords if kw.arg}
            return func_obj(*args, **kwargs)

        if isinstance(node, ast.Subscript):
            value = self._eval_node(node.value, env)
            key = self._eval_slice(node.slice, env)
            return value[key]

        if isinstance(node, ast.List):
            return [self._eval_node(elt, env) for elt in node.elts]

        if isinstance(node, ast.Tuple):
            return tuple(self._eval_node(elt, env) for elt in node.elts)

        if isinstance(node, ast.Set):
            return {self._eval_node(elt, env) for elt in node.elts}

        if isinstance(node, ast.Dict):
            keys = [self._eval_node(k, env) if k is not None else None for k in node.keys]
            values = [self._eval_node(v, env) for v in node.values]
            return {k: v for k, v in zip(keys, values)}

        raise ExpressionEvalError(f"unsupported expression node: {type(node).__name__}")

    def _eval_slice(self, slice_node: ast.AST, env: Dict[str, Any]) -> Any:
        if isinstance(slice_node, ast.Slice):
            lower = self._eval_node(slice_node.lower, env) if slice_node.lower else None
            upper = self._eval_node(slice_node.upper, env) if slice_node.upper else None
            step = self._eval_node(slice_node.step, env) if slice_node.step else None
            return slice(lower, upper, step)
        return self._eval_node(slice_node, env)

    def _evaluate_comparison(self, operator_node: ast.AST, left: Any, right: Any) -> bool:
        if isinstance(operator_node, ast.Eq):
            return left == right
        if isinstance(operator_node, ast.NotEq):
            return left != right
        if isinstance(operator_node, ast.Gt):
            return left > right
        if isinstance(operator_node, ast.GtE):
            return left >= right
        if isinstance(operator_node, ast.Lt):
            return left < right
        if isinstance(operator_node, ast.LtE):
            return left <= right
        if isinstance(operator_node, ast.In):
            return left in right
        if isinstance(operator_node, ast.NotIn):
            return left not in right
        if isinstance(operator_node, ast.Is):
            return left is right
        if isinstance(operator_node, ast.IsNot):
            return left is not right
        raise ExpressionEvalError("unsupported comparison operator")

    def _build_generator(self, node: ast.GeneratorExp, env: Dict[str, Any]) -> Any:
        def generator() -> Any:
            yield from self._comprehension_values(node.generators, env, node.elt)

        return generator()

    def _comprehension_values(
        self,
        generators: List[ast.comprehension],
        env: Dict[str, Any],
        value_node: ast.AST,
    ) -> Any:
        yield from self._iterate_comprehension(generators, env, value_node, None)

    def _comprehension_items(
        self,
        generators: List[ast.comprehension],
        env: Dict[str, Any],
        key_node: ast.AST,
        value_node: ast.AST,
    ) -> Any:
        yield from self._iterate_comprehension(generators, env, value_node, key_node)

    def _iterate_comprehension(
        self,
        generators: List[ast.comprehension],
        env: Dict[str, Any],
        value_node: ast.AST,
        key_node: Optional[ast.AST],
    ):
        def recurse(index: int, current_env: Dict[str, Any]):
            if index == len(generators):
                if key_node is None:
                    yield self._eval_node(value_node, current_env)
                else:
                    yield (
                        self._eval_node(key_node, current_env),
                        self._eval_node(value_node, current_env),
                    )
                return

            comp = generators[index]
            iterable = self._eval_node(comp.iter, current_env)
            for item in iterable:
                new_env = dict(current_env)
                self._assign_comprehension_target(new_env, comp.target, item)
                if all(bool(self._eval_node(condition, new_env)) for condition in comp.ifs):
                    yield from recurse(index + 1, new_env)

        yield from recurse(0, dict(env))

    def _assign_comprehension_target(self, env: Dict[str, Any], target: ast.AST, value: Any) -> None:
        if isinstance(target, ast.Name):
            env[target.id] = value
            return
        if isinstance(target, (ast.Tuple, ast.List)):
            values = list(value) if not isinstance(value, (list, tuple)) else value
            if len(target.elts) != len(values):
                raise ExpressionEvalError("comprehension target length mismatch")
            for subtarget, subvalue in zip(target.elts, values):
                self._assign_comprehension_target(env, subtarget, subvalue)
            return
        raise ExpressionEvalError("unsupported comprehension target")


class RuleEngine:
    """
    The RuleEngine will:
    - take ProjectDB
    - take list[Rule]
    - walk the IR according to each rule's 'scope'
    - run the rule's select/assert/exception logic
    - emit Violations
    """

    def __init__(self, project_db: ProjectDB, rules: List[Rule]) -> None:
        self.project_db = project_db
        self.rules = rules
        self._scope_cache: Dict[str, List[Any]] = {}
        self._unknown_scopes: Set[str] = set()
        self._unknown_scope_reported: Set[str] = set()
        self._dsl_notice_emitted: bool = False
        self._expr_error_reported: Set[Tuple[str, str, str]] = set()
        self._expr_interpreter = _SafeExpressionInterpreter()

    def evaluate(self) -> List[Violation]:
        """
        Iterate every rule, gather the IR nodes for the declared scope,
        and run the lightweight Pythonic DSL that powers select/assert/exception.
        """
        violations: List[Violation] = []

        for rule in self.rules:
            nodes, recognized = self._collect_scope_objects(rule.scope)
            if not recognized:
                self._maybe_report_unknown_scope(rule, rule.scope)
                continue

            if not nodes:
                continue

            self._notify_dsl_unimplemented(rule)
            bindings, predicate = self._parse_select(rule.select, rule.scope)
            primary_alias, primary_scope = bindings[0]

            if primary_scope != rule.scope:
                nodes, recognized = self._collect_scope_objects(primary_scope)
                if not recognized:
                    self._maybe_report_unknown_scope(rule, primary_scope)
                    continue

            if not nodes:
                continue

            assert_expr = (rule.assert_code or "").strip()
            exception_exprs = [expr.strip() for expr in rule.exceptions if expr.strip()]

            binding_objects: List[List[Any]] = []
            binding_objects.append(nodes)

            scopes_valid = True
            for _, scope_name in bindings[1:]:
                objs, recognized = self._collect_scope_objects(scope_name)
                if not recognized:
                    self._maybe_report_unknown_scope(rule, scope_name)
                    scopes_valid = False
                    break
                binding_objects.append(objs)

            if not scopes_valid:
                continue

            if any(len(obj_list) == 0 for obj_list in binding_objects):
                continue

            for primary_obj in binding_objects[0]:
                env = self._create_base_env()
                self._assign_alias(env, primary_alias, primary_obj, primary=True)
                self._evaluate_binding_combinations(
                    rule=rule,
                    bindings=bindings,
                    binding_objects=binding_objects,
                    predicate=predicate,
                    assert_expr=assert_expr,
                    exception_exprs=exception_exprs,
                    env=env,
                    violations=violations,
                    start_index=1,
                    primary_node=primary_obj,
                )

        return violations

    def _parse_select(self, select_expr: Optional[str], scope: str) -> Tuple[List[Tuple[str, str]], str]:
        """
        Extract (bindings, predicate_expression) from the rule's select block.
        Supported forms:
          - "alias: Scope where expr"
          - "alias: Scope"
          - "alias1: Scope1, alias2: Scope2 where expr"
          - "where expr"
          - plain expression (defaults alias -> obj)
          - empty/None => predicate True
        """
        if not select_expr:
            return [("obj", scope)], "True"

        text = select_expr.strip()
        if not text:
            return [("obj", scope)], "True"

        predicate = "True"
        select_part = text
        lower = text.lower()
        if lower.startswith("where "):
            predicate = text[6:].strip() or "True"
            select_part = ""
        else:
            where_idx = lower.find(" where ")
            if where_idx != -1:
                predicate = text[where_idx + len(" where "):].strip() or "True"
                select_part = text[:where_idx].strip()

        binding_specs = [spec.strip() for spec in select_part.split(",") if spec.strip()]
        bindings: List[Tuple[str, str]] = []
        if not binding_specs:
            bindings.append(("obj", scope))
        else:
            for idx, spec in enumerate(binding_specs):
                if ":" in spec:
                    alias_part, scope_part = spec.split(":", 1)
                    alias_name = alias_part.strip() or f"obj{idx}"
                    scope_name = scope_part.strip() or scope
                else:
                    alias_name = spec.strip() or f"obj{idx}"
                    scope_name = scope
                bindings.append((alias_name, scope_name))

        return bindings, predicate

    def _create_base_env(self) -> Dict[str, Any]:
        env: Dict[str, Any] = dict(_SAFE_BASE_CALLABLES)
        env.update(
            {
                "project": self.project_db,
                "project_db": self.project_db,
                "functions_by_name": self.project_db.functions_by_name,
                "globals_by_name": self.project_db.globals_by_name,
                "call_edge": self._make_env_callable(self._call_edge_helper),
                "calls_function": self._make_env_callable(self._calls_function_helper),
                "call_path_exists": self._make_env_callable(self._call_path_helper),
            }
        )
        return env

    def _make_env_callable(self, func: Any) -> Any:
        @functools.wraps(func)
        def _wrapper(*args: Any, **kwargs: Any) -> Any:
            return func(*args, **kwargs)

        return _mark_safe_callable(_wrapper)

    def _assign_alias(self, env: Dict[str, Any], alias: str, node: Any, *, primary: bool = False) -> None:
        env[alias] = node
        if primary:
            env["obj"] = node
            env["_scope_name"] = getattr(node, "name", None)

    def _evaluate_binding_combinations(
        self,
        *,
        rule: Rule,
        bindings: List[Tuple[str, str]],
        binding_objects: List[List[Any]],
        predicate: str,
        assert_expr: str,
        exception_exprs: List[str],
        env: Dict[str, Any],
        violations: List[Violation],
        start_index: int,
        primary_node: Any,
    ) -> None:
        def recurse(index: int, current_env: Dict[str, Any]) -> None:
            if index == len(bindings):
                self._evaluate_rule_conditions(
                    rule=rule,
                    env=current_env,
                    predicate=predicate,
                    assert_expr=assert_expr,
                    exception_exprs=exception_exprs,
                    violations=violations,
                    node=primary_node,
                )
                return

            alias, _ = bindings[index]
            for obj in binding_objects[index]:
                next_env = dict(current_env)
                self._assign_alias(next_env, alias, obj, primary=False)
                recurse(index + 1, next_env)

        recurse(start_index, env)

    def _evaluate_rule_conditions(
        self,
        *,
        rule: Rule,
        env: Dict[str, Any],
        predicate: str,
        assert_expr: str,
        exception_exprs: List[str],
        violations: List[Violation],
        node: Any,
    ) -> None:
        if not self._evaluate_expression(predicate or "True", env, rule, stage="select"):
            return

        result = self._evaluate_expression(assert_expr or "True", env, rule, stage="assert")
        if result:
            return

        for exception_expr in exception_exprs:
            if self._evaluate_expression(exception_expr, env, rule, stage="exception"):
                return

        violations.append(self._build_violation(rule, env, node))

    def _call_edge_helper(self, caller: Any, callee: Any) -> bool:
        caller_name = self._extract_symbol_name(caller)
        callee_name = self._extract_symbol_name(callee)
        if not caller_name or not callee_name:
            return False
        return callee_name in self.project_db.call_graph.get(caller_name, set())

    def _calls_function_helper(self, function_obj: Any, callee: Any) -> bool:
        if not isinstance(function_obj, Function):
            return False
        target = self._extract_symbol_name(callee)
        if not target:
            return False
        for call in function_obj.calls:
            if target and target in {call.callee_symbol, call.callee_name}:
                return True
        return False

    def _call_path_helper(self, caller: Any, callee: Any, max_depth: int = 64) -> bool:
        start = self._extract_symbol_name(caller)
        target = self._extract_symbol_name(callee)
        if not start or not target:
            return False
        if start == target:
            return True
        visited = set([start])
        queue: deque[Tuple[str, int]] = deque([(start, 0)])
        while queue:
            symbol, depth = queue.popleft()
            if depth >= max_depth:
                continue
            for neighbor in self.project_db.call_graph.get(symbol, set()):
                if neighbor == target:
                    return True
                if neighbor not in visited:
                    visited.add(neighbor)
                    queue.append((neighbor, depth + 1))
        return False

    def _extract_symbol_name(self, obj: Any) -> Optional[str]:
        if obj is None:
            return None
        if isinstance(obj, str):
            return obj
        return getattr(obj, "name", None)

    def _eval_raw_expression(
        self,
        expr: str,
        env: Dict[str, Any],
        rule: Rule,
        *,
        stage: str,
    ) -> Any:
        expr = (expr or "").strip()
        if not expr:
            return True
        try:
            return self._expr_interpreter.evaluate(expr, env)
        except ExpressionEvalError as exc:
            self._report_eval_error(rule, stage, expr, exc)
            return False if stage in {"select", "assert", "exception"} else ""
        except Exception as exc:  # pragma: no cover - safeguard
            self._report_eval_error(rule, stage, expr, exc)
            return False if stage in {"select", "assert", "exception"} else ""

    def _evaluate_expression(
        self,
        expr: str,
        env: Dict[str, Any],
        rule: Rule,
        *,
        stage: str,
    ) -> bool:
        result = self._eval_raw_expression(expr, env, rule, stage=stage)
        return bool(result)

    def _report_eval_error(
        self,
        rule: Rule,
        stage: str,
        expr: str,
        exc: Exception,
    ) -> None:
        key = (rule.id, stage, expr)
        if key in self._expr_error_reported:
            return
        sys.stderr.write(
            f"[nomic] Failed to evaluate {stage} expression for rule '{rule.id}': "
            f"{expr!r} ({exc}).\n"
        )
        self._expr_error_reported.add(key)

    def _build_violation(self, rule: Rule, env: Dict[str, Any], node: Any) -> Violation:
        location = self._object_location(node)
        message = self._render_template(rule.message, env, rule, field="message")
        suggested_fix = None
        if rule.fixit:
            rendered_fix = self._render_template(rule.fixit, env, rule, field="fixit")
            if rendered_fix:
                suggested_fix = rendered_fix
        context = {
            "scope": rule.scope,
            "name": getattr(node, "name", None),
        }
        return Violation(
            rule_id=rule.id,
            severity=rule.severity,
            message=message,
            location=location,
            context=context,
            suggested_fix=suggested_fix,
            extras={"rule_tags": rule.tags},
        )

    def _object_location(self, node: Any) -> Dict[str, Any]:
        source_range = getattr(node, "source_range", None)
        if isinstance(source_range, SourceRange):
            return {
                "file": source_range.file,
                "line_start": source_range.line_start,
                "col_start": source_range.col_start,
                "line_end": source_range.line_end,
                "col_end": source_range.col_end,
            }

        location = getattr(node, "location", None)
        if isinstance(location, SourceLocation):
            return {
                "file": location.file,
                "line_start": location.line,
                "col_start": location.column,
                "line_end": location.line,
                "col_end": location.column,
            }

        file_hint = getattr(node, "path", None) or getattr(node, "file", None) or "<unknown>"
        return {
            "file": file_hint,
            "line_start": 0,
            "col_start": 0,
            "line_end": 0,
            "col_end": 0,
        }

    def _render_template(
        self,
        template: Optional[str],
        env: Dict[str, Any],
        rule: Rule,
        *,
        field: str,
    ) -> str:
        if not template:
            return ""

        def replace(match: re.Match[str]) -> str:
            expr = match.group(1)
            value = self._eval_raw_expression(expr, env, rule, stage=f"template:{field}")
            return "" if value is None else str(value)

        return TEMPLATE_PATTERN.sub(replace, template)

    def _collect_scope_objects(self, scope: str) -> Tuple[List[Any], bool]:
        """
        Map a rule scope string to a concrete list of IR objects.
        Returns (objects, recognized_scope_flag).
        """
        if scope in self._scope_cache:
            return self._scope_cache[scope], scope not in self._unknown_scopes

        known = True
        objects: List[Any] = []

        if scope == "ProjectDB":
            objects = [self.project_db]
        elif scope == "TranslationUnit":
            objects = list(self.project_db.translation_units)
        elif scope == "Function":
            funcs: List[Function] = []
            for tu in self.project_db.translation_units:
                funcs.extend(tu.functions)
            objects = funcs
        elif scope == "MacroDefinition":
            macros: List[MacroDefinition] = []
            for tu in self.project_db.translation_units:
                macros.extend(tu.macros)
            objects = macros
        elif scope == "TypeDecl":
            types: List[TypeDecl] = []
            for tu in self.project_db.translation_units:
                types.extend(tu.types)
            objects = types
        elif scope == "Variable":
            variables: List[Variable] = []
            for tu in self.project_db.translation_units:
                variables.extend(tu.globals)
                for fn in tu.functions:
                    variables.extend(fn.parameters)
                    variables.extend(fn.local_vars)
            objects = variables
        elif scope == "CallSite":
            callsites: List[CallSite] = []
            functions, _ = self._collect_scope_objects("Function")
            for fn in functions:
                callsites.extend(fn.calls)
                for block in fn.cfg.blocks.values():
                    callsites.extend(block.calls)
                for if_stmt in fn.if_stmts:
                    callsites.extend(if_stmt.then_block.calls)
                    if if_stmt.else_block:
                        callsites.extend(if_stmt.else_block.calls)
                for loop in fn.loops:
                    callsites.extend(loop.body.calls)
                for switch in fn.switches:
                    for case in switch.cases:
                        callsites.extend(case.body.calls)
            objects = callsites
        elif scope == "IfStmt":
            ifs: List[IfStmt] = []
            functions, _ = self._collect_scope_objects("Function")
            for fn in functions:
                ifs.extend(fn.if_stmts)
            objects = ifs
        elif scope == "LoopStmt":
            loops: List[LoopStmt] = []
            functions, _ = self._collect_scope_objects("Function")
            for fn in functions:
                loops.extend(fn.loops)
            objects = loops
        elif scope == "SwitchStmt":
            switches: List[SwitchStmt] = []
            functions, _ = self._collect_scope_objects("Function")
            for fn in functions:
                switches.extend(fn.switches)
            objects = switches
        elif scope == "SwitchCaseBlock":
            cases: List[SwitchCaseBlock] = []
            switches, _ = self._collect_scope_objects("SwitchStmt")
            for switch in switches:
                cases.extend(switch.cases)
            objects = cases
        elif scope == "BlockStmt":
            blocks: List[BlockStmt] = []
            ifs, _ = self._collect_scope_objects("IfStmt")
            for if_stmt in ifs:
                blocks.append(if_stmt.then_block)
                if if_stmt.else_block:
                    blocks.append(if_stmt.else_block)
            loops, _ = self._collect_scope_objects("LoopStmt")
            for loop in loops:
                blocks.append(loop.body)
            switches, _ = self._collect_scope_objects("SwitchStmt")
            for switch in switches:
                for case in switch.cases:
                    blocks.append(case.body)
            objects = blocks
        elif scope == "ControlFlowGraph":
            cfgs: List[ControlFlowGraph] = []
            functions, _ = self._collect_scope_objects("Function")
            for fn in functions:
                cfgs.append(fn.cfg)
            objects = cfgs
        elif scope == "BasicBlock":
            blocks: List[BasicBlock] = []
            cfgs, _ = self._collect_scope_objects("ControlFlowGraph")
            for cfg in cfgs:
                blocks.extend(cfg.blocks.values())
            objects = blocks
        else:
            known = False

        if not known:
            self._unknown_scopes.add(scope)

        self._scope_cache[scope] = objects
        return objects, known

    def _maybe_report_unknown_scope(self, rule: Rule, scope_name: Optional[str] = None) -> None:
        """
        Emit a single warning per unknown scope to help authors debug rules.
        """
        scope = scope_name or rule.scope
        if scope in self._unknown_scope_reported:
            return

        sys.stderr.write(
            f"[nomic] Unknown rule scope '{scope}' referenced by rule '{rule.id}'.\n"
        )
        self._unknown_scope_reported.add(scope)

    def _notify_dsl_unimplemented(self, rule: Rule, emitted: bool = True) -> None:
        """
        Inform the user once that the DSL interpreter currently supports only
        simple Pythonic expressions.
        """
        if self._dsl_notice_emitted:
            return
        sys.stderr.write(
            "[nomic] Rule evaluation currently supports simple Python-style "
            "expressions for select/assert/exceptions; complex DSL features "
            f"are not yet implemented (rule '{rule.id}').\n"
        )
        self._dsl_notice_emitted = True


# ============================================================
# ==================== YAML RULE LOADING =====================
# ============================================================

def load_rules_from_yaml(yaml_paths: List[str]) -> List[Rule]:
    """
    Load Rule objects from YAML rule files.

    This implementation keeps PyYAML optional: if it is not installed we emit a
    warning (when rule files were provided) and return an empty rule list so the
    rest of the pipeline can proceed.
    """
    if not yaml_paths:
        return []

    if yaml is None:
        sys.stderr.write(
            "[nomic] PyYAML is not installed; skipping rule files.\n"
        )
        return []

    def _to_str_list(value: Any) -> List[str]:
        if value is None:
            return []
        if isinstance(value, list):
            return [str(v) for v in value if v is not None]
        return [str(value)]

    def _normalize_rule_docs(doc: Any) -> List[Dict[str, Any]]:
        if doc is None:
            return []
        if isinstance(doc, list):
            return [item for item in doc if isinstance(item, dict)]
        if isinstance(doc, dict):
            if isinstance(doc.get("rules"), list):
                return [item for item in doc["rules"] if isinstance(item, dict)]
            return [doc]
        return []

    def _build_rule(raw_rule: Dict[str, Any], origin: str) -> Optional[Rule]:
        select_expr = raw_rule.get("select")
        assert_expr = raw_rule.get("assert_code", raw_rule.get("assert"))
        required_fields = {
            "id": raw_rule.get("id"),
            "severity": raw_rule.get("severity"),
            "scope": raw_rule.get("scope"),
            "select": select_expr,
            "assert_code": assert_expr,
            "message": raw_rule.get("message"),
        }
        missing = [name for name, value in required_fields.items() if value in (None, "")]
        if missing:
            sys.stderr.write(
                f"[nomic] Skipping rule from {origin}: missing required field(s) {missing}.\n"
            )
            return None

        return Rule(
            id=str(required_fields["id"]),
            description=str(raw_rule.get("description", "")),
            severity=str(required_fields["severity"]),
            scope=str(required_fields["scope"]),
            select=str(select_expr),
            assert_code=str(assert_expr),
            message=str(required_fields["message"]),
            exceptions=_to_str_list(raw_rule.get("exceptions")),
            tags=_to_str_list(raw_rule.get("tags")),
            fixit=str(raw_rule["fixit"]) if raw_rule.get("fixit") is not None else None,
        )

    rules: List[Rule] = []
    for path in yaml_paths:
        try:
            with open(path, "r", encoding="utf-8") as handle:
                documents = list(yaml.safe_load_all(handle))
        except FileNotFoundError:
            sys.stderr.write(f"[nomic] Rule file not found: {path}\n")
            continue
        except OSError as exc:
            sys.stderr.write(f"[nomic] Could not read rule file {path}: {exc}\n")
            continue

        for doc_index, doc in enumerate(documents):
            for raw_rule in _normalize_rule_docs(doc):
                origin = f"{path}#doc{doc_index + 1}"
                rule = _build_rule(raw_rule, origin)
                if rule:
                    rules.append(rule)

    return rules


# ============================================================
# ================= PROJECT BUILD PIPELINE ===================
# ============================================================

_CLANG_MISSING_WARNED = False


def build_translation_unit_from_clang(path: str) -> TranslationUnit:
    """
    Parse a C file with libclang (when available) and hydrate the core IR.
    Falls back to an empty TranslationUnit if libclang is unavailable or the
    file cannot be parsed so that the rest of the pipeline keeps running.
    """
    canonical_path = os.path.abspath(path)

    if clang_cindex is None:
        _warn_once_clang_missing()
        return _build_stub_translation_unit(path)

    if not os.path.exists(canonical_path):
        sys.stderr.write(f"[nomic] Input file not found: {path}\n")
        return _build_stub_translation_unit(path)

    try:
        index = clang_cindex.Index.create()
    except Exception as exc:  # pragma: no cover - libclang internal failure
        sys.stderr.write(f"[nomic] Failed to initialize libclang: {exc}\n")
        return _build_stub_translation_unit(path)

    args = _default_clang_args()
    options = _clang_parse_options()

    try:
        clang_tu = index.parse(canonical_path, args=args, options=options)
    except Exception as exc:
        sys.stderr.write(
            f"[nomic] libclang could not parse '{path}': {exc}\n"
        )
        return _build_stub_translation_unit(path)

    return _translate_clang_tu(clang_tu, canonical_path, path)


def stitch_project_db(tus: List[TranslationUnit]) -> ProjectDB:
    """
    Build ProjectDB indices from TranslationUnits.
    """
    db = ProjectDB(translation_units=tus)

    # Index functions
    for tu in tus:
        for fn in tu.functions:
            db.functions_by_name.setdefault(fn.name, []).append(fn)
            # call graph approximation:
            callerset = db.call_graph.setdefault(fn.name, set())
            for call in fn.calls:
                target = call.callee_symbol or call.callee_name
                if target:
                    callerset.add(target)

    # Index globals
    for tu in tus:
        for g in tu.globals:
            db.globals_by_name.setdefault(g.name, []).append(g)

    # Index macros
    for tu in tus:
        for m in tu.macros:
            db.macros_by_name.setdefault(m.name, []).append(m)

    # Includes
    for tu in tus:
        for header in tu.includes:
            db.include_usage.setdefault(header, []).append(tu.path)

    # You can pre-seed policy metadata here if desired.
    # Example: known blocking APIs, etc.
    # db.function_metadata["sleep_ms"] = {"is_blocking_api": True}

    return db


def _warn_once_clang_missing() -> None:
    global _CLANG_MISSING_WARNED
    if _CLANG_MISSING_WARNED:
        return
    sys.stderr.write(
        "[nomic] clang.cindex is not available; using stub parser.\n"
    )
    _CLANG_MISSING_WARNED = True


def _default_clang_args() -> List[str]:
    """
    Determine the clang arguments to use. Users can append additional flags
    via the NOMIC_CLANG_ARGS environment variable.
    """
    base = ["-x", "c", "-std=c11"]
    extra = os.environ.get("NOMIC_CLANG_ARGS")
    if extra:
        base.extend(shlex.split(extra))
    return base


def _clang_parse_options() -> int:
    """
    Choose libclang parse options that surface macros and preprocessing info.
    """
    options = 0
    if clang_cindex is None:
        return options
    options |= clang_cindex.TranslationUnit.PARSE_DETAILED_PROCESSING_RECORD
    options |= clang_cindex.TranslationUnit.PARSE_INCLUDE_BRIEF_COMMENTS_IN_CODE_COMPLETION
    return options


def _build_stub_translation_unit(
    path: str,
    *,
    includes: Optional[List[str]] = None,
    macros: Optional[List[MacroDefinition]] = None,
    types: Optional[List[TypeDecl]] = None,
    globals_list: Optional[List[Variable]] = None,
    functions: Optional[List[Function]] = None,
    active_defines: Optional[Set[str]] = None,
) -> TranslationUnit:
    globals_list = list(globals_list or [])
    symbol_table = {var.name: var for var in globals_list if var.name}
    return TranslationUnit(
        path=path,
        includes=list(includes or []),
        macros=list(macros or []),
        types=list(types or []),
        globals=globals_list,
        functions=list(functions or []),
        symbol_table=symbol_table,
        active_defines=set(active_defines or set()),
        module=None,
    )


def _translate_clang_tu(
    clang_tu: "clang_cindex.TranslationUnit",
    canonical_main: str,
    display_path: str,
) -> TranslationUnit:
    includes = _collect_includes(clang_tu)
    functions, globals_list, macros, types = _collect_ir_nodes(clang_tu, canonical_main)
    symbol_table = {var.name: var for var in globals_list if var.name}
    active_defines = {macro.name for macro in macros if macro.name}
    return TranslationUnit(
        path=display_path,
        includes=includes,
        macros=macros,
        types=types,
        globals=globals_list,
        functions=functions,
        symbol_table=symbol_table,
        active_defines=active_defines,
        module=None,
    )


def _collect_includes(clang_tu: "clang_cindex.TranslationUnit") -> List[str]:
    includes: List[str] = []
    seen: Set[str] = set()
    for inc in clang_tu.get_includes():
        if inc.include is None:
            continue
        inc_path = os.path.normpath(inc.include.name)
        if inc_path not in seen:
            includes.append(inc_path)
            seen.add(inc_path)
    return includes


def _collect_ir_nodes(
    clang_tu: "clang_cindex.TranslationUnit",
    canonical_main: str,
) -> Tuple[List[Function], List[Variable], List[MacroDefinition], List[TypeDecl]]:
    functions: List[Function] = []
    globals_list: List[Variable] = []
    macros: List[MacroDefinition] = []
    types: List[TypeDecl] = []

    def visit(cursor: "clang_cindex.Cursor", current_function: Optional[Function]) -> None:
        # Always traverse the TU root even if it has no explicit location.
        if cursor.kind != clang_cindex.CursorKind.TRANSLATION_UNIT:
            if not _cursor_in_main_file(cursor, canonical_main):
                return

        next_function = current_function

        if cursor.kind == clang_cindex.CursorKind.FUNCTION_DECL and cursor.is_definition():
            fn = _function_from_cursor(cursor, canonical_main)
            functions.append(fn)
            next_function = fn
        elif (
            cursor.kind == clang_cindex.CursorKind.VAR_DECL
            and cursor.semantic_parent
            and cursor.semantic_parent.kind == clang_cindex.CursorKind.TRANSLATION_UNIT
        ):
            globals_list.append(_global_from_cursor(cursor, canonical_main))
        elif cursor.kind == clang_cindex.CursorKind.MACRO_DEFINITION:
            macro = _macro_from_cursor(cursor, canonical_main)
            if macro:
                macros.append(macro)
        elif cursor.kind in (
            clang_cindex.CursorKind.TYPEDEF_DECL,
            clang_cindex.CursorKind.STRUCT_DECL,
            clang_cindex.CursorKind.UNION_DECL,
            clang_cindex.CursorKind.ENUM_DECL,
        ):
            type_decl = _type_decl_from_cursor(cursor, canonical_main)
            if type_decl:
                types.append(type_decl)
        elif cursor.kind == clang_cindex.CursorKind.CALL_EXPR and current_function is not None:
            callsite = _callsite_from_cursor(cursor, canonical_main)
            if callsite:
                current_function.calls.append(callsite)
                _append_call_to_cfg(current_function, callsite)
        elif cursor.kind == clang_cindex.CursorKind.IF_STMT and current_function is not None:
            if_stmt = _if_stmt_from_cursor(cursor, canonical_main, current_function.name)
            current_function.if_stmts.append(if_stmt)
            condition_block = _record_basic_block(
                current_function,
                f"If: {if_stmt.condition}",
                cursor,
                canonical_main,
                enclosing_construct="if_stmt",
                branch_condition=if_stmt.condition,
            )
            then_child = _find_child_of_kind(cursor, clang_cindex.CursorKind.COMPOUND_STMT, occurrence=0)
            else_child = _find_child_of_kind(cursor, clang_cindex.CursorKind.COMPOUND_STMT, occurrence=1)
            else_if_child = None
            if else_child is None:
                else_if_child = _find_child_of_kind(cursor, clang_cindex.CursorKind.IF_STMT, occurrence=0)
            branch_children: List["clang_cindex.Cursor"] = []
            if then_child is not None:
                branch_children.append(then_child)
            if else_child is not None:
                branch_children.append(else_child)
            elif else_if_child is not None:
                branch_children.append(else_if_child)

            branch_exit_blocks: List[int] = []
            if branch_children:
                for child in branch_children:
                    with _push_pending_predecessors(current_function, [condition_block.block_id]):
                        visit(child, next_function)
                        exit_id = _current_block_id(current_function)
                        if exit_id is not None:
                            branch_exit_blocks.append(exit_id)
            else:
                branch_exit_blocks.append(condition_block.block_id)

            if else_child is None and else_if_child is None and condition_block.block_id not in branch_exit_blocks:
                branch_exit_blocks.append(condition_block.block_id)

            _set_pending_predecessors(current_function, branch_exit_blocks or [condition_block.block_id])
        elif cursor.kind in (
            clang_cindex.CursorKind.FOR_STMT,
            clang_cindex.CursorKind.WHILE_STMT,
            clang_cindex.CursorKind.DO_STMT,
        ) and current_function is not None:
            loop_stmt = _loop_stmt_from_cursor(cursor, canonical_main, current_function.name)
            current_function.loops.append(loop_stmt)
            loop_block = _record_basic_block(
                current_function,
                f"Loop: {loop_stmt.kind}",
                cursor,
                canonical_main,
                enclosing_construct="loop_stmt",
                branch_condition=loop_stmt.condition,
            )
            body_child = _find_child_of_kind(cursor, clang_cindex.CursorKind.COMPOUND_STMT, occurrence=0) or cursor
            with _push_pending_predecessors(current_function, [loop_block.block_id]):
                visit(body_child, next_function)
                exit_id = _current_block_id(current_function)
                if exit_id is not None:
                    _connect_blocks(current_function, exit_id, loop_block.block_id)
            _set_pending_predecessors(current_function, [loop_block.block_id])
        elif cursor.kind == clang_cindex.CursorKind.SWITCH_STMT and current_function is not None:
            switch_stmt = _switch_stmt_from_cursor(cursor, canonical_main, current_function.name)
            current_function.switches.append(switch_stmt)
            switch_block = _record_basic_block(
                current_function,
                "switch",
                cursor,
                canonical_main,
                enclosing_construct="switch_stmt",
                branch_condition=switch_stmt.control_expr,
            )
            case_children = [
                child
                for child in cursor.get_children()
                if child.kind in (clang_cindex.CursorKind.CASE_STMT, clang_cindex.CursorKind.DEFAULT_STMT)
            ]
            case_exits: List[int] = []
            if case_children:
                for case_child in case_children:
                    with _push_pending_predecessors(current_function, [switch_block.block_id]):
                        visit(case_child, next_function)
                        exit_id = _current_block_id(current_function)
                        if exit_id is not None:
                            case_exits.append(exit_id)
            else:
                case_exits.append(switch_block.block_id)
            _set_pending_predecessors(current_function, case_exits or [switch_block.block_id])

        try:
            children = list(cursor.get_children())
        except Exception:
            children = []
        for child in children:
            visit(child, next_function)

    visit(clang_tu.cursor, None)
    return functions, globals_list, macros, types


def _cursor_in_main_file(cursor: "clang_cindex.Cursor", canonical_main: str) -> bool:
    loc = cursor.location
    if loc is None or loc.file is None:
        return False
    return os.path.abspath(loc.file.name) == canonical_main


def _make_source_location(
    location: "clang_cindex.SourceLocation",
    fallback_path: str,
) -> SourceLocation:
    file_name = location.file.name if location.file is not None else fallback_path
    return SourceLocation(file=file_name, line=location.line, column=location.column)


def _make_source_range(
    extent: "clang_cindex.SourceRange",
    fallback_path: str,
) -> SourceRange:
    start = extent.start
    end = extent.end
    return SourceRange(
        file=start.file.name if start.file is not None else fallback_path,
        line_start=start.line,
        col_start=start.column,
        line_end=end.line,
        col_end=end.column,
    )


def _cursor_attributes(cursor: "clang_cindex.Cursor") -> List[str]:
    getter = getattr(cursor, "get_attributes", None)
    if getter is None:
        return []
    try:
        return [attr.spelling for attr in getter()]
    except Exception:
        return []


def _storage_class_to_variable_storage(
    storage_class: "clang_cindex.StorageClass",
) -> str:
    if clang_cindex is None:
        return "unknown"
    mapping = {
        clang_cindex.StorageClass.INVALID: "unknown",
        clang_cindex.StorageClass.AUTO: "auto",
        clang_cindex.StorageClass.STATIC: "static",
        clang_cindex.StorageClass.EXTERN: "extern",
        clang_cindex.StorageClass.REGISTER: "register",
        clang_cindex.StorageClass.NONE: "unknown",
    }
    return mapping.get(storage_class, "unknown")


def _linkage_to_string(linkage: "clang_cindex.LinkageKind") -> str:
    if clang_cindex is None:
        return "external"
    kind = linkage
    lk = clang_cindex.LinkageKind
    none_kind = getattr(lk, "NONE", getattr(lk, "NO_LINKAGE", None))
    unique_external = getattr(lk, "UNIQUE_EXTERNAL", None)
    if kind == lk.INTERNAL:
        return "internal"
    if unique_external is not None and kind == unique_external:
        return "internal"
    if none_kind is not None and kind == none_kind:
        return "none"
    return "external"


def _function_from_cursor(
    cursor: "clang_cindex.Cursor",
    canonical_main: str,
) -> Function:
    source_range = _make_source_range(cursor.extent, canonical_main)
    storage = _storage_class_to_variable_storage(cursor.storage_class)
    linkage = "internal" if storage == "static" else _linkage_to_string(cursor.linkage)

    parameters: List[Variable] = []
    try:
        args = list(cursor.get_arguments())
    except Exception:
        args = []
    for index, arg in enumerate(args):
        parameters.append(_param_variable_from_cursor(arg, canonical_main, cursor.spelling or f"fn_{id(cursor)}", index))

    function = Function(
        name=cursor.spelling or cursor.displayname or "<anonymous>",
        return_type=cursor.result_type.spelling if cursor.result_type else "void",
        parameters=parameters,
        local_vars=[],
        linkage=linkage,
        storage=storage if storage != "unknown" else None,
        attributes=_cursor_attributes(cursor),
        annotations=[],
        is_isr=False,
        is_inline=bool(getattr(cursor, "is_inline_function", lambda: False)()),
        calls=[],
        cfg=ControlFlowGraph(),
        exit_points=[],
        if_stmts=[],
        loops=[],
        switches=[],
        globals_written=[],
        globals_read=[],
        source_range=source_range,
        preprocessor=PreprocessorContext(),
        prefix=None,
        suffix=None,
    )
    _initialize_function_cfg(function)
    return function


def _param_variable_from_cursor(
    cursor: "clang_cindex.Cursor",
    canonical_main: str,
    function_name: str,
    ordinal: int,
) -> Variable:
    source_range = _make_source_range(cursor.extent, canonical_main)
    name = cursor.spelling or cursor.displayname or f"param_{ordinal}"
    return Variable(
        name=name,
        ctype=cursor.type.spelling if cursor.type else "",
        storage="auto",
        linkage="none",
        is_const=_type_is_const(cursor.type),
        is_volatile=_type_is_volatile(cursor.type),
        is_atomic=_type_is_atomic(cursor.type),
        scope="param",
        decl_function=function_name,
        prefix=None,
        suffix=None,
        source_range=source_range,
        preprocessor=PreprocessorContext(),
        annotations=[],
        writes=[],
        reads=[],
    )


def _global_from_cursor(
    cursor: "clang_cindex.Cursor",
    canonical_main: str,
) -> Variable:
    source_range = _make_source_range(cursor.extent, canonical_main)
    storage = _storage_class_to_variable_storage(cursor.storage_class)
    linkage = _linkage_to_string(cursor.linkage)
    return Variable(
        name=cursor.spelling or cursor.displayname or "<anonymous>",
        ctype=cursor.type.spelling if cursor.type else "",
        storage=storage,
        linkage=linkage,
        is_const=_type_is_const(cursor.type),
        is_volatile=_type_is_volatile(cursor.type),
        is_atomic=_type_is_atomic(cursor.type),
        scope="file",
        decl_function=None,
        prefix=None,
        suffix=None,
        source_range=source_range,
        preprocessor=PreprocessorContext(),
        annotations=[],
        writes=[],
        reads=[],
    )


def _macro_from_cursor(
    cursor: "clang_cindex.Cursor",
    canonical_main: str,
) -> Optional[MacroDefinition]:
    name = cursor.spelling
    if not name:
        return None

    try:
        raw_tokens = [token.spelling for token in cursor.get_tokens()]
    except Exception:
        raw_tokens = []

    tokens = list(raw_tokens)
    # Trim the leading '# define' tokens if present.
    if len(tokens) >= 2 and tokens[0] == "#" and tokens[1] == "define":
        tokens = tokens[2:]

    body_tokens: List[str] = []
    params: List[str] = []
    kind: Literal["object_like", "function_like"] = "object_like"

    if tokens and tokens[0] == name:
        tokens = tokens[1:]

    if tokens and tokens[0] == "(":
        kind = "function_like"
        idx = 1
        current: List[str] = []
        while idx < len(tokens):
            tok = tokens[idx]
            if tok == ")":
                if current:
                    params.append("".join(current))
                    current = []
                idx += 1
                break
            elif tok == ",":
                if current:
                    params.append("".join(current))
                    current = []
            else:
                current.append(tok)
            idx += 1
        body_tokens = tokens[idx:]
    else:
        body_tokens = tokens

    body_statement = " ".join(body_tokens).strip()
    source_range = _make_source_range(cursor.extent, canonical_main)

    return MacroDefinition(
        name=name,
        kind=kind,
        params=params,
        body_tokens=body_tokens,
        body_statements=[body_statement] if body_statement else [],
        is_wrapped_single_stmt=False,
        source_range=source_range,
        preprocessor=PreprocessorContext(),
        annotations=[],
        prefix=None,
        suffix=None,
    )


def _callsite_from_cursor(
    cursor: "clang_cindex.Cursor",
    canonical_main: str,
) -> Optional[CallSite]:
    callee_name = cursor.displayname or cursor.spelling
    callee_symbol = None
    referenced = getattr(cursor, "referenced", None)
    if referenced is not None:
        callee_symbol = referenced.spelling or referenced.displayname
        if not callee_name:
            callee_name = callee_symbol

    args: List[str] = []
    arg_getter = getattr(cursor, "get_arguments", None)
    if arg_getter is not None:
        try:
            for arg in arg_getter():
                args.append(arg.displayname or arg.spelling or "")
        except Exception:
            pass

    if not callee_name:
        callee_name = "<unknown>"

    return CallSite(
        callee_name=callee_name,
        callee_symbol=callee_symbol,
        args=args,
        location=_make_source_range(cursor.extent, canonical_main),
        in_branch=None,
        in_loop=None,
        in_switch=False,
        in_macro_expansion=False,
        is_blocking_api=False,
        is_allocator=False,
        is_lock=False,
        is_unlock=False,
        preprocessor=PreprocessorContext(),
    )


def _type_decl_from_cursor(
    cursor: "clang_cindex.Cursor",
    canonical_main: str,
) -> Optional[TypeDecl]:
    kind_map = {
        clang_cindex.CursorKind.TYPEDEF_DECL: "typedef",
        clang_cindex.CursorKind.STRUCT_DECL: "struct",
        clang_cindex.CursorKind.UNION_DECL: "union",
        clang_cindex.CursorKind.ENUM_DECL: "enum",
    }
    kind = kind_map.get(cursor.kind)
    if kind is None:
        return None

    name = cursor.spelling or cursor.displayname
    if not name:
        # Skip anonymous declarations for now.
        return None

    if cursor.kind != clang_cindex.CursorKind.TYPEDEF_DECL and not cursor.is_definition():
        return None

    fields: List[FieldDecl] = []
    if cursor.kind in (
        clang_cindex.CursorKind.STRUCT_DECL,
        clang_cindex.CursorKind.UNION_DECL,
    ):
        for child in cursor.get_children():
            if child.kind != clang_cindex.CursorKind.FIELD_DECL:
                continue
            field_range = _make_source_range(child.extent, canonical_main)
            is_bitfield = bool(getattr(child, "is_bitfield", lambda: False)())
            bit_width = None
            if is_bitfield:
                try:
                    bit_width = child.get_bitfield_width()
                except Exception:
                    bit_width = None
            fields.append(
                FieldDecl(
                    name=child.spelling or child.displayname or "",
                    ctype=child.type.spelling if child.type else "",
                    is_bitfield=is_bitfield,
                    bit_width=bit_width,
                    source_range=field_range,
                )
            )
    elif cursor.kind == clang_cindex.CursorKind.ENUM_DECL:
        for child in cursor.get_children():
            if child.kind != clang_cindex.CursorKind.ENUM_CONSTANT_DECL:
                continue
            field_range = _make_source_range(child.extent, canonical_main)
            fields.append(
                FieldDecl(
                    name=child.spelling or "",
                    ctype="enum_constant",
                    is_bitfield=False,
                    bit_width=None,
                    source_range=field_range,
                )
            )

    underlying = None
    if cursor.kind == clang_cindex.CursorKind.TYPEDEF_DECL:
        try:
            underlying = cursor.underlying_typedef_type.spelling  # type: ignore[attr-defined]
        except Exception:
            underlying = None

    type_decl = TypeDecl(
        name=name,
        kind=kind,  # type: ignore[arg-type]
        fields=fields,
        underlying_type=underlying,
        attributes=_cursor_attributes(cursor),
        visibility="project_visible",
        source_range=_make_source_range(cursor.extent, canonical_main),
        preprocessor=PreprocessorContext(),
        annotations=[],
    )
    return type_decl


def _initialize_function_cfg(function: Function) -> None:
    if function.cfg.blocks:
        return
    entry_block = BasicBlock(
        block_id=0,
        statements=["entry"],
        successors=[],
        predecessors=[],
        source_range=function.source_range,
        preprocessor=function.preprocessor,
    )
    function.cfg.blocks[0] = entry_block
    function.cfg.entry_block = 0
    function.cfg.exit_blocks = [0]
    function.exit_points = [0]
    setattr(function, "_cfg_state", {"next_block_id": 1, "current_block_id": 0})


def _get_cfg_state(function: Function) -> Dict[str, Any]:
    state = getattr(function, "_cfg_state", None)
    if state is None:
        state = {"next_block_id": len(function.cfg.blocks) or 1, "current_block_id": function.cfg.entry_block or 0}
        setattr(function, "_cfg_state", state)
    return state


def _current_block_id(function: Function) -> Optional[int]:
    state = _get_cfg_state(function)
    return state.get("current_block_id")


def _set_pending_predecessors(function: Function, preds: List[int]) -> None:
    state = _get_cfg_state(function)
    state["pending_predecessors"] = list(preds)


@contextmanager
def _push_pending_predecessors(function: Function, preds: List[int]):
    state = _get_cfg_state(function)
    previous = list(state.get("pending_predecessors", []))
    state["pending_predecessors"] = list(preds)
    try:
        yield
    finally:
        state["pending_predecessors"] = previous


def _connect_blocks(function: Function, from_block_id: Optional[int], to_block_id: Optional[int]) -> None:
    if from_block_id is None or to_block_id is None:
        return
    from_block = function.cfg.blocks.get(from_block_id)
    to_block = function.cfg.blocks.get(to_block_id)
    if from_block is None or to_block is None:
        return
    if to_block_id not in from_block.successors:
        from_block.successors.append(to_block_id)
    if from_block_id not in to_block.predecessors:
        to_block.predecessors.append(from_block_id)


def _record_basic_block(
    function: Function,
    description: str,
    cursor: "clang_cindex.Cursor",
    canonical_main: str,
    *,
    enclosing_construct: Optional[str] = None,
    branch_condition: Optional[str] = None,
) -> BasicBlock:
    state = _get_cfg_state(function)
    block_id = state["next_block_id"]
    state["next_block_id"] += 1

    block = BasicBlock(
        block_id=block_id,
        statements=[description],
        enclosing_construct=enclosing_construct,
        branch_condition=branch_condition,
        source_range=_make_source_range(cursor.extent, canonical_main),
        preprocessor=PreprocessorContext(),
    )

    pending = state.get("pending_predecessors", [])
    if pending:
        predecessor_ids = list(pending)
    else:
        current_block_id = state.get("current_block_id")
        predecessor_ids = [current_block_id] if current_block_id is not None else []

    for pred_id in predecessor_ids:
        prev_block = function.cfg.blocks.get(pred_id)
        if prev_block is not None:
            if block_id not in prev_block.successors:
                prev_block.successors.append(block_id)
            if prev_block.block_id not in block.predecessors:
                block.predecessors.append(prev_block.block_id)

    state["pending_predecessors"] = []

    function.cfg.blocks[block_id] = block
    function.cfg.exit_blocks = [block_id]
    function.exit_points = [block_id]
    state["current_block_id"] = block_id
    return block


def _append_call_to_cfg(function: Function, callsite: CallSite) -> None:
    block = _current_block(function)
    if block is None:
        return
    block.calls.append(callsite)
    if block.source_range is None:
        block.source_range = callsite.location


def _current_block(function: Function) -> Optional[BasicBlock]:
    state = _get_cfg_state(function)
    block_id = state.get("current_block_id")
    if block_id is None:
        return None
    return function.cfg.blocks.get(block_id)


def _if_stmt_from_cursor(
    cursor: "clang_cindex.Cursor",
    canonical_main: str,
    function_name: str,
) -> IfStmt:
    condition = _cursor_condition_text(cursor)
    then_child = _find_child_of_kind(cursor, clang_cindex.CursorKind.COMPOUND_STMT, occurrence=0)
    else_child = _find_child_of_kind(cursor, clang_cindex.CursorKind.COMPOUND_STMT, occurrence=1)
    else_if_child = None
    if else_child is None:
        else_if_child = _find_child_of_kind(cursor, clang_cindex.CursorKind.IF_STMT, occurrence=0)

    then_block = _block_stmt_from_cursor(then_child or cursor, canonical_main)
    else_block_stmt = None
    if else_child is not None or else_if_child is not None:
        else_block_stmt = _block_stmt_from_cursor((else_child or else_if_child) or cursor, canonical_main)

    return IfStmt(
        condition=condition,
        then_block=then_block,
        else_block=else_block_stmt,
        parent_function=function_name,
        source_range=_make_source_range(cursor.extent, canonical_main),
        preprocessor=PreprocessorContext(),
        annotations=[],
    )


def _loop_stmt_from_cursor(
    cursor: "clang_cindex.Cursor",
    canonical_main: str,
    function_name: str,
) -> LoopStmt:
    if cursor.kind == clang_cindex.CursorKind.FOR_STMT:
        loop_kind: Literal["for", "while", "do_while"] = "for"
    elif cursor.kind == clang_cindex.CursorKind.WHILE_STMT:
        loop_kind = "while"
    else:
        loop_kind = "do_while"

    body_child = _find_child_of_kind(cursor, clang_cindex.CursorKind.COMPOUND_STMT, occurrence=0)
    loop_body = _block_stmt_from_cursor(body_child or cursor, canonical_main)

    return LoopStmt(
        kind=loop_kind,
        condition=_cursor_condition_text(cursor),
        body=loop_body,
        parent_function=function_name,
        source_range=_make_source_range(cursor.extent, canonical_main),
        preprocessor=PreprocessorContext(),
        annotations=[],
    )


def _switch_stmt_from_cursor(
    cursor: "clang_cindex.Cursor",
    canonical_main: str,
    function_name: str,
) -> SwitchStmt:
    cases: List[SwitchCaseBlock] = []
    has_default = False

    for child in cursor.get_children():
        if child.kind == clang_cindex.CursorKind.CASE_STMT:
            labels = [_cursor_condition_text(child)]
            body_child = _find_child_of_kind(child, clang_cindex.CursorKind.COMPOUND_STMT, occurrence=0)
            cases.append(
                SwitchCaseBlock(
                    labels=labels,
                    body=_block_stmt_from_cursor(body_child or child, canonical_main),
                    source_range=_make_source_range(child.extent, canonical_main),
                )
            )
        elif child.kind == clang_cindex.CursorKind.DEFAULT_STMT:
            has_default = True
            body_child = _find_child_of_kind(child, clang_cindex.CursorKind.COMPOUND_STMT, occurrence=0)
            cases.append(
                SwitchCaseBlock(
                    labels=["default"],
                    body=_block_stmt_from_cursor(body_child or child, canonical_main),
                    source_range=_make_source_range(child.extent, canonical_main),
                )
            )

    return SwitchStmt(
        control_expr=_cursor_condition_text(cursor),
        cases=cases,
        has_default=has_default,
        parent_function=function_name,
        source_range=_make_source_range(cursor.extent, canonical_main),
        preprocessor=PreprocessorContext(),
        annotations=[],
    )


def _block_stmt_from_cursor(
    cursor: Optional["clang_cindex.Cursor"],
    canonical_main: str,
) -> BlockStmt:
    if cursor is None:
        return BlockStmt(statements=[], source_range=None, preprocessor=PreprocessorContext())
    description = cursor.displayname or cursor.spelling or cursor.kind.name
    return BlockStmt(
        statements=[description] if description else [],
        source_range=_make_source_range(cursor.extent, canonical_main),
        preprocessor=PreprocessorContext(),
    )


def _find_child_of_kind(
    cursor: "clang_cindex.Cursor",
    kind: "clang_cindex.CursorKind",
    *,
    occurrence: int = 0,
) -> Optional["clang_cindex.Cursor"]:
    matches = [child for child in cursor.get_children() if child.kind == kind]
    if len(matches) > occurrence:
        return matches[occurrence]
    return None


def _cursor_condition_text(cursor: "clang_cindex.Cursor") -> str:
    text = cursor.displayname or cursor.spelling
    if text:
        return text
    try:
        tokens = [t.spelling for t in cursor.get_tokens()]
        if tokens:
            return " ".join(tokens[:32])
    except Exception:
        pass
    return "<expr>"
def _type_is_const(ctype: Optional["clang_cindex.Type"]) -> bool:
    if ctype is None:
        return False
    checker = getattr(ctype, "is_const_qualified", None)
    if checker is None:
        return False
    try:
        return bool(checker())
    except Exception:
        return False


def _type_is_volatile(ctype: Optional["clang_cindex.Type"]) -> bool:
    if ctype is None:
        return False
    checker = getattr(ctype, "is_volatile_qualified", None)
    if checker is None:
        return False
    try:
        return bool(checker())
    except Exception:
        return False


def _type_is_atomic(ctype: Optional["clang_cindex.Type"]) -> bool:
    if ctype is None:
        return False
    checker = getattr(ctype, "is_atomic_qualified", None)
    if checker is None:
        return False
    try:
        return bool(checker())
    except Exception:
        return False


# ============================================================
# ==================== VIOLATION OUTPUT ======================
# ============================================================

def violation_to_json_obj(v: Violation) -> Dict[str, Any]:
    """
    Convert a Violation dataclass into a JSON-friendly dict.
    We'll keep this explicit so we can guarantee stable field order over time if needed.
    """
    return {
        "rule_id": v.rule_id,
        "severity": v.severity,
        "message": v.message,
        "location": v.location,
        "context": v.context,
        "suggested_fix": v.suggested_fix,
        "extras": v.extras,
        "tool": "Nomic",
        "version": "0.1.0",
    }


def emit_violations_json(violations: List[Violation], out: Optional[str] = None) -> None:
    """
    Serialize all violations to JSON (list of violation objects).
    """
    as_json = [violation_to_json_obj(v) for v in violations]
    text = json.dumps(as_json, indent=2, sort_keys=False)
    if out:
        with open(out, "w", encoding="utf-8") as f:
            f.write(text + "\n")
    else:
        print(text)


# ============================================================
# ============================ CLI ===========================
# ============================================================

def main(argv: Optional[List[str]] = None) -> int:
    """
    CLI entry point for Nomic.
    Intended usage:
      python nomic.py analyze --rules rules.yaml src/file1.c src/file2.c ...

    This currently stubs out Clang integration and rule evaluation.
    """
    parser = argparse.ArgumentParser(
        prog="nomic",
        description="Nomic: Semantic rule enforcement for C"
    )

    subparsers = parser.add_subparsers(dest="command", required=True)

    analyze_p = subparsers.add_parser(
        "analyze",
        help="Analyze one or more C translation units and emit JSON violations."
    )
    analyze_p.add_argument(
        "--rules",
        nargs="+",
        metavar="RULE_FILE",
        help="YAML rule file(s).",
        required=False,
    )
    analyze_p.add_argument(
        "--out",
        metavar="OUT_JSON",
        help="Write violations to this JSON file instead of stdout.",
        required=False,
    )
    analyze_p.add_argument(
        "files",
        nargs="+",
        help="C source files to analyze."
    )

    args = parser.parse_args(argv)

    if args.command == "analyze":
        # 1. Build TranslationUnits (stubbed)
        tus: List[TranslationUnit] = []
        for path in args.files:
            tu = build_translation_unit_from_clang(path)
            tus.append(tu)

        # 2. Build ProjectDB
        project_db = stitch_project_db(tus)

        # 3. Load rules
        rule_list: List[Rule] = load_rules_from_yaml(args.rules or [])

        # 4. Run rule engine
        engine = RuleEngine(project_db, rule_list)
        violations = engine.evaluate()

        # 5. Emit results as JSON
        emit_violations_json(violations, out=args.out)
        return 0

    # unreachable if parser is correct
    return 1


if __name__ == "__main__":
    sys.exit(main())
