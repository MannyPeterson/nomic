# Nomic Code Documentation

This file documents every major component inside `nomic.py`. It is organized top-to-bottom following the structure of the single-file implementation so that contributors can quickly locate definitions and understand their responsibilities.

---

## 1. File Header & Imports (`nomic.py:1-52`)

- Enables `#!/usr/bin/env python3` entry point and documents the high-level goals of Nomic.
- Uses `from __future__ import annotations` to allow forward references in type hints.
- Imports standard-library modules (`argparse`, `ast`, `functools`, `json`, `os`, `re`, `shlex`, `sys`, etc.) along with typing helpers, `dataclasses`, and optional dependencies (`yaml`, `clang.cindex`).
- Defines conditional symbols such as `_STATEMENT_CURSOR_KINDS` (when clang is available) for later use in CFG instrumentation.

---

## 2. Intermediate Representation (IR) Dataclasses (`nomic.py:64-353`)

### Source & Annotations
- `SourceLocation`, `SourceRange`: store file/line/column information.
- `PreprocessorContext`: captures active defines and guard conditions.
- `Annotation`: textual metadata (e.g., pragmas) with locations.

### Type & Field Declarations
- `FieldDecl`, `TypeDecl`: represent struct/union/enum/typedef declarations, their attributes, and fields.

### Memory Access Sites
- `WriteSite`, `ReadSite`: track where symbols are written/read, including branch/loop/macro context.

### Symbols
- `Variable`: describes globals/locals/parameters, type qualifiers, annotations, and associated read/write sites.
- `CallSite`: records callee names, semantic tags (blocking/allocator), and structural context.

### Control-Flow Graph (CFG)
- `BasicBlock`: contains statements, calls, reads, writes, successors/predecessors, dominance/post-dominance sets, and metadata.
- `ControlFlowGraph`: owns the block dictionary, entry/exit tracking, and provides `all_exit_paths_postdominated_by`.

### AST Blocks
- `BlockStmt`, `IfStmt`, `LoopStmt`, `SwitchCaseBlock`, `SwitchStmt`: describe structured control statements, including nested bodies, annotations, and preprocessor context.

### Functions & Macros
- `Function`: aggregates parameters, locals, calls, CFG, flow constructs, global access patterns, annotations, and naming helpers.
- `MacroDefinition`: captures macro kind, params, body tokens/statements, and heuristics (e.g., wrapped single statement).

### Translation Units & Project DB
- `TranslationUnit`: one parsed C file, contains includes/macros/types/globals/functions and per-TU symbol tables/defines.
- `ProjectDB`: aggregates TranslationUnits and builds indexes (functions/globals/macros, include usage, call graph, policy metadata).

---

## 3. Rule & Violation Models (`nomic.py:356-505`)

- `Rule`: YAML-backed representation with id, description, severity, scope, DSL fields (`select`, `assert_code`), message templates, exceptions, tags, and optional fixit.
- Violation context helpers (`ViolationContextFunction`, etc.) are placeholders for future extensions.
- `Violation`: final object serialized to JSON with rule metadata, message, location, contextual data, fix suggestions, and extras.

---

## 4. DSL & Rule Engine (`nomic.py:507-1125`)

### Safe DSL Interpreter
- `_mark_safe_callable`, `_wrap_safe_callable`, `_wrap_dynamic_callable`: flag callables as safe for the AST interpreter.
- Quantifier helpers `exists`, `forall`, `count` plus call-graph helpers (more below) are registered as safe builtins.
- `ExpressionEvalError`: custom exception raised for invalid constructs.
- `_SafeExpressionInterpreter`: parses expressions, caches ASTs, and recursively evaluates nodes while enforcing restrictions (no private attribute access, only safe callables, limited comprehension targets, etc.).

### RuleEngine Class
Responsibilities:
- Maintains scope caches, tracks unknown scopes, and centralizes DSL evaluation.
- `_parse_select`: parses selectors with multi-binding support (`alias: Scope where ...`).
- `_create_base_env`: seeds evaluation environment with project indexes, helper functions (`call_edge`, `calls_function`, `call_path_exists`, `reachable_functions`), and quantifiers.
- `_evaluate_binding_combinations`: iterates over Cartesian products of bindings to evaluate `select` predicates, assertions, and exceptions.
- `_call_edge_helper`, `_calls_function_helper`, `_call_path_helper`, `_reachable_functions_helper`: logic powering DSL helpers, operating on `ProjectDB.call_graph`.
- `_object_location`, `_build_violation`: convert IR nodes into `Violation` records with location/context.
- `_collect_scope_objects`: maps scope strings to actual IR lists (ProjectDB, Function, CallSite, etc.).
- `_notify_dsl_unimplemented`: currently warns once that only basic expressions are supported (placeholder for richer DSL).

---

## 5. Rule Loading (`nomic.py:688-780`)

- `load_rules_from_yaml`: reads one or more YAML files, supports both list and `rules:` formats, validates required fields (`id`, `severity`, `scope`, `select`, `assert_code`, `message`), and instantiates `Rule` objects.
- Handles missing PyYAML, file-not-found/OSError gracefully by warning and skipping files.

---

## 6. Clang Integration & Project Building (`nomic.py:782-2090`)

### Pipeline Functions
- `build_translation_unit_from_clang(path)`: orchestrates clang parsing, providing stub fallback if libclang is unavailable or file parsing fails.
- `_translate_clang_tu`: converts clang AST into IR lists; calls `_collect_ir_nodes`.
- `stitch_project_db`: aggregates TUs, builds function/global/macro indexes, include usage, and call graph.

### CFG & AST Helpers
- `_initialize_function_cfg`: establishes entry block and CFG state per function.
- `_get_cfg_state`, `_current_block_id`, `_set_pending_predecessors`, `_push_pending_predecessors`, `_connect_blocks`, `_record_basic_block`, `_current_block`: utilities for constructing CFGs with branch-merging support.
- `_finalize_function_cfg`: computes entry/exit blocks, marks exit nodes, and performs full dominance/post-dominance iterations.
- `_record_statement_in_block`, `_record_write_site`, `_record_read_site`, `_cursor_snippet`, `_looks_like_assignment`: capture block-level metadata while traversing clang cursors.

### Clang Cursor Translators
- `_collect_ir_nodes`: depth-first traversal capturing functions, globals, macros, type declarations, call sites, and control statements. Integrates CFG instrumentation by managing branch predecessors, loop back edges, and switch case successors.
- `_function_from_cursor`, `_param_variable_from_cursor`, `_global_from_cursor`, `_macro_from_cursor`, `_callsite_from_cursor`, `_if_stmt_from_cursor`, `_loop_stmt_from_cursor`, `_switch_stmt_from_cursor`, `_block_stmt_from_cursor`: convert clang cursors to IR dataclasses, populating statements/calls/writes/reads and preprocessor context.
- `_cursor_condition_text`, `_cursor_snippet`: textual extraction utilities for conditions/statements.
- Type qualifier helpers `_type_is_const`, `_type_is_volatile`, `_type_is_atomic` ensure compatibility across libclang versions.
- `_build_stub_translation_unit`: provides minimal TU for environments without clang.

---

## 7. Violation Serialization (`nomic.py:1335-1364`)

- `violation_to_json_obj`: deterministic dict representation, including `tool` and `version`.
- `emit_violations_json`: writes JSON either to stdout or to `--out` path (with newline termination for compatibility with CLI tools).

---

## 8. CLI (`nomic.py:1366-1660`)

- Uses `argparse` with subcommand `analyze`.
- CLI flow:
  1. Parse args (`--rules`, `--out`, positional C files).
  2. For each file, call `build_translation_unit_from_clang`.
  3. Stitch ProjectDB.
  4. Load rules (empty list allowed).
  5. Run `RuleEngine.evaluate`.
  6. Emit violations JSON.
- `if __name__ == "__main__": sys.exit(main())` ensures proper exit codes (0 on success, 1 on unexpected command).

---

## 9. Support Files

- `RULES_SPEC.md`: detailed specification of the rule YAML format (separate document).
- `NOMIC_CODE_OVERVIEW.md` (this file): architectural documentation for `nomic.py`.

---

## 10. Development Notes

- The entire tool is intentionally in a single file to simplify prototyping.
- Optional dependencies (PyYAML, libclang) are guarded so stub functionality keeps the CLI usable without them.
- Apply future changes consistently:
  - Add new IR scopes to `_collect_scope_objects`, `_parse_select`, and documentation.
  - Update the DSL helper list when exposing new capabilities.
  - Keep CFG helper utilities in sync with clang traversal logic.

With this overview, contributors should be able to navigate each component, understand data flow, and extend Nomic confidently.

