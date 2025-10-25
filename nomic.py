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
import json
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

SAFE_EVAL_GLOBALS: Dict[str, Any] = {
    "__builtins__": {},
    "len": len,
    "any": any,
    "all": all,
    "sum": sum,
    "min": min,
    "max": max,
    "sorted": sorted,
}

TEMPLATE_PATTERN = re.compile(r"\{\{\s*([^{}]+?)\s*\}\}")


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
        self._expr_error_reported: Set[Tuple[str, str]] = set()

    def evaluate(self) -> List[Violation]:
        """
        Iterate every rule, gather the IR nodes for the declared scope,
        and run the lightweight Pythonic DSL that powers select/assert/exception.
        """
        violations: List[Violation] = []

        for rule in self.rules:
            nodes, recognized = self._collect_scope_objects(rule.scope)
            if not recognized:
                self._maybe_report_unknown_scope(rule)
                continue

            if not nodes:
                continue

            self._notify_dsl_unimplemented(rule)
            alias, predicate = self._parse_select(rule.select, rule.scope)
            assert_expr = (rule.assert_code or "").strip()
            exception_exprs = [expr.strip() for expr in rule.exceptions if expr.strip()]

            for node in nodes:
                env = self._build_env(alias, node)

                if not self._evaluate_expression(predicate, env, rule, stage="select"):
                    continue

                result = self._evaluate_expression(assert_expr or "True", env, rule, stage="assert")
                if result:
                    continue

                exception_triggered = False
                for exception_expr in exception_exprs:
                    if self._evaluate_expression(exception_expr, env, rule, stage="exception"):
                        exception_triggered = True
                        break
                if exception_triggered:
                    continue

                violations.append(self._build_violation(rule, env, node))

        return violations

    def _parse_select(self, select_expr: Optional[str], scope: str) -> Tuple[str, str]:
        """
        Extract (alias, predicate_expression) from the rule's select block.
        Supported forms:
          - "alias: Scope where expr"
          - "alias: Scope"
          - "where expr"
          - plain expression (defaults alias -> obj)
          - empty/None => predicate True
        """
        if not select_expr:
            return "obj", "True"

        text = select_expr.strip()
        if not text:
            return "obj", "True"

        alias = "obj"
        remainder = text

        if ":" in text:
            alias_part, rest = text.split(":", 1)
            alias = alias_part.strip() or "obj"
            remainder = rest.strip()

        # Trim leading scope token if present (e.g. "Function where ...").
        if remainder:
            parts = remainder.split(None, 1)
            if parts and parts[0] == scope:
                remainder = parts[1] if len(parts) > 1 else ""

        predicate = "True"
        lower = remainder.lower()
        if lower.startswith("where "):
            predicate = remainder[6:].strip() or "True"
        else:
            where_idx = lower.find(" where ")
            if where_idx != -1:
                predicate = remainder[where_idx + len(" where "):].strip() or "True"
            elif remainder:
                predicate = remainder

        return alias, predicate

    def _build_env(self, alias: str, node: Any) -> Dict[str, Any]:
        env = {alias: node, "obj": node}
        env["_scope_name"] = getattr(node, "name", None)
        return env

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
            return eval(expr, SAFE_EVAL_GLOBALS, env)
        except Exception as exc:
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

    def _maybe_report_unknown_scope(self, rule: Rule) -> None:
        """
        Emit a single warning per unknown scope to help authors debug rules.
        """
        if rule.scope in self._unknown_scope_reported:
            return

        sys.stderr.write(
            f"[nomic] Unknown rule scope '{rule.scope}' referenced by rule '{rule.id}'.\n"
        )
        self._unknown_scope_reported.add(rule.scope)

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

    stack = [clang_tu.cursor]
    seen_hashes: Set[int] = set()

    while stack:
        cursor = stack.pop()
        try:
            cursor_hash = cursor.hash
        except AttributeError:
            cursor_hash = None
        if cursor_hash is not None:
            if cursor_hash in seen_hashes:
                continue
            seen_hashes.add(cursor_hash)

        # Always traverse the TU root even if it has no location.
        if cursor.kind != clang_cindex.CursorKind.TRANSLATION_UNIT:
            if not _cursor_in_main_file(cursor, canonical_main):
                continue

        if cursor.kind == clang_cindex.CursorKind.FUNCTION_DECL and cursor.is_definition():
            functions.append(_function_from_cursor(cursor, canonical_main))
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

        try:
            stack.extend(list(cursor.get_children()))
        except Exception:
            continue

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
    if linkage == clang_cindex.LinkageKind.INTERNAL:
        return "internal"
    if linkage == clang_cindex.LinkageKind.UNIQUE_EXTERNAL:
        return "internal"
    if linkage == clang_cindex.LinkageKind.NONE:
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
        is_const=bool(cursor.type and cursor.type.is_const_qualified()),
        is_volatile=bool(cursor.type and cursor.type.is_volatile_qualified()),
        is_atomic=bool(cursor.type and cursor.type.is_atomic_qualified()),
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
        is_const=bool(cursor.type and cursor.type.is_const_qualified()),
        is_volatile=bool(cursor.type and cursor.type.is_volatile_qualified()),
        is_atomic=bool(cursor.type and cursor.type.is_atomic_qualified()),
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
