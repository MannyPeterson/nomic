# Nomic Rules YAML Specification

This document explains the complete specification for the rule files consumed by `nomic.py`. After reading it you should be able to author rules that reference the intermediate representation (IR), reason about control-flow, describe exceptions, and emit rich violation messages.

---

## 1. High-Level Structure

A Nomic rule file is a YAML stream containing either:

```yaml
- id: RULE_ID
  ... rule fields ...
- id: ANOTHER_RULE
  ...
```

or

```yaml
rules:
  - id: RULE_ID
    ...
```

Each entry must expand to a `Rule` object. When multiple YAML documents are present in a single file, every document is processed independently.

---

## 2. Fields

| Field          | Required | Type                 | Description                                                                                                               |
|----------------|----------|----------------------|---------------------------------------------------------------------------------------------------------------------------|
| `id`           | ✅       | string               | Unique identifier. Suggested format: `<ORG>-<CATEGORY>-<NUMBER>`, e.g. `NOMIC-CFG-001`.                                   |
| `description`  | ❌       | string               | Human readable explanation shown in docs or tooling.                                                                      |
| `severity`     | ✅       | string               | Typically `error`, `warning`, or `info`. No hard enforcement—pipelines may map to their own severity model.               |
| `scope`        | ✅       | string               | IR type the rule iterates over. Valid values include `ProjectDB`, `TranslationUnit`, `Function`, `Variable`, `CallSite`, `MacroDefinition`, `TypeDecl`, `IfStmt`, `LoopStmt`, `SwitchStmt`, `SwitchCaseBlock`, `BlockStmt`, `ControlFlowGraph`, `BasicBlock`. |
| `select`       | ✅       | string               | DSL clause binding alias names to scopes and defining the candidate set.                                                  |
| `assert_code`  | ✅       | string               | DSL expression that must evaluate to `True` for every selected binding.                                                   |
| `message`      | ✅       | string / template    | Violation message supporting `{{ placeholder }}` expressions.                                                             |
| `exceptions`   | ❌       | list of expressions  | If any expression evaluates truthy, the violation is suppressed.                                                          |
| `fixit`        | ❌       | string template      | Suggested fix shown in JSON output. Allows `{{ placeholder }}` expressions.                                               |
| `tags`         | ❌       | list of strings      | Arbitrary strings to categorize the rule (`style`, `safety`, `cfg`, ...).                                                 |

### Templates

`message` and `fixit` support `{{ expr }}` insertions. Expressions reuse the DSL interpreter (safe AST-based execution). Example:

```yaml
message: "ISR {{fn.name}} calls blocking API {{call.callee_name}}"
```

---

## 3. `select` Syntax

`select` drives which objects the rule inspects. Grammar:

```
select := [binding {',' binding}] ['where' predicate]
binding := alias ':' ScopeName | alias
```

- Omitted bindings default to `obj: <scope>`.
- Scopes can differ from the rule’s `scope`. Example:

```yaml
select: "caller: Function, callee: Function where call_edge(caller, callee)"
```

Bindings are made available as variables inside `assert_code`, `exceptions`, and templates. The primary binding (the first entry) also becomes `obj`.

---

## 4. DSL (Expression Language)

The DSL is a restricted subset of Python evaluated via an AST interpreter—no `eval`, no direct access to builtins beyond the approved helpers.

### Supported Constructs

- Literals: numbers, strings, booleans, lists, tuples, dicts, sets.
- Boolean / arithmetic operators (`and`, `or`, `not`, `+`, `-`, `*`, `/`, `%`, comparisons) plus the infix `implies`.
- Conditional expressions: `a if condition else b`.
- Attribute access (blocks private names `_foo`).
- Indexing / slicing (`array[0]`, `array[1:3]`).
- List / set / dict comprehensions and generator expressions.
- Lambda expressions (e.g. `(lambda x, y=0: x + y)(5)`).
- Safe function calls (helpers automatically marked as safe).

### Blocking Rules

- No loops or assignments inside DSL expressions.
- Access to module/global state outside the provided environment is forbidden.
- Methods/attributes resolved at runtime must be explicitly marked safe, otherwise they are rejected.
- Helpers that perform I/O or mutate state are not exposed; everything is pure/side-effect free.

### Built-In Helpers

| Helper                        | Description                                                                 |
|-------------------------------|-----------------------------------------------------------------------------|
| `len`, `any`, `all`, `sum`, `min`, `max`, `sorted`, `abs`, `hasattr`, `safe_getattr(obj, name, default)` | Python-like helpers. |
| `exists(iterable)`            | Returns `True` if any element of the iterable is truthy.                   |
| `forall(iterable)`            | Returns `True` if every element is truthy.                                 |
| `count(iterable)`             | Counts truthy elements (alias for `sum(1 for ...)`).                       |
| `call_edge(caller, callee)`   | Checks whether `caller` directly calls `callee` (by symbol name).         |
| `calls_function(function, callee)` | True if the `Function` contains a `CallSite` whose resolved name matches `callee`. |
| `call_path_exists(caller, callee, max_depth=64)` | Breadth-first search over the ProjectDB call graph looking for a path. |
| `reachable_functions(caller, max_depth=64)` | Sorted list of function names transitively reachable from `caller`. |
| `has_blocking_call_path(start, max_depth=32)` | Returns `True` if a blocking call is reachable from `start`.       |
| `pattern_search(target, pattern, **kwargs)` / `pattern_matches` / `pattern_findall` / `pattern_extract` | Multi-line regex toolkit that accepts strings, `Statement`, `BasicBlock`, `Function`, or iterables. |
| `semantic_pattern_matches(target, predicate=None, pattern=None, pattern_scope='statement', max_results=256, pattern_kwargs=None)` | Combined semantic + regex search returning rich match objects. |
| `semantic_pattern_any(...)`   | Boolean convenience wrapper around `semantic_pattern_matches`.             |
| `semantic_statements(target, predicate=None)` | Returns statements that satisfy a predicate.                                  |

Every helper is exposed as a safe callable; you can pass either objects (e.g. `Function`) or strings. When a string is provided, it is interpreted as a symbol name.

#### Pattern & Semantic Matching

- **Pattern helpers** (`pattern_search`, `pattern_matches`, `pattern_findall`, `pattern_extract`) accept arbitrary text, `Statement`, `BasicBlock`, `BlockStmt`, `Function`, or iterables of these. They default to multi-line + dot-all regexes so you can span multiple IR statements. `pattern_search` returns `PatternMatch` objects with spans, line numbers, capture groups, and the original context.
- **Semantic helpers** (`semantic_pattern_matches`, `semantic_pattern_any`, `semantic_statements`) iterate the statements inside functions/blocks and apply predicates (e.g., `lambda stmt, block, fn: stmt.contains_call and block.loop_depth > 0`). Optional regex filters operate over the statement text, the enclosing block, or the entire function via `pattern_scope`.
- These helpers enable extremely expressive rules such as “find every statement that writes to `g_state` inside a loop and ensure there is a `mutex_lock` statement in the same block.”

### Example Assertions

```yaml
assert_code: "forall(call in caller.calls for call in caller.calls if call.is_blocking_api implies caller.is_isr)"

assert_code: "exists(reachable_functions(caller) if fn.startswith('crit_') for fn in reachable_functions(caller))"

assert_code: "not call_path_exists(caller, 'malloc')"
```

---

## 5. Scopes and IR Data

All scopes map to concrete objects built by `nomic.py`. Summary:

- **ProjectDB**: Aggregates translation units and indexes (`functions_by_name`, `globals_by_name`, `call_graph`, etc.).
- **TranslationUnit**: Includes macros, globals, functions, includes list.
- **Function**: Has parameters, locals, calls, CFG (with blocks), lists of `if_stmts`, `loops`, `switches`, `globals_written`, `globals_read`, and rich metadata (annotations, storage, `is_isr`, `is_api`, `is_callback`, `is_syscall`, `has_mutex_protection`, complexity metrics, etc.).
- **Variable**: Tracks storage/linkage, qualifiers, read/write sites, initialization flags, `type_name`, `base_type`, `is_custom_type`, `parent_function`, `read_by`, `written_by`, and mutex protection information.
- **CallSite**: Provides callee names, branch/loop context, classification flags (`is_blocking_api`, `is_lock`, `is_unlock`, `is_allocator`), arguments, and source ranges.
- **Statement**: Each CFG block statement stores parsed text plus derived facts: `contains_call`, `contains_assignment`, `contains_macro`, `variables_read`, `variables_written`, macro names, and line spans.
- **ControlFlowGraph / BasicBlock**: Blocks store statements, calls, reads, writes, dominators/postdominators, successor/predecessor lists, loop depth, `parent_loop`, and `is_loop_header`/`is_loop_exit` flags. `CFG.all_exit_paths_postdominated_by(lambda block: ...)`, `CFG.has_path_without`, and `CFG.get_paths_to_exit` enable structural reasoning.
- **BlockStmt / IfStmt / LoopStmt / SwitchStmt / SwitchCaseBlock**: Capture sub-block statements, writes, calls, locals, parent function names, annotations, and preprocessor context for structured constructs.

When binding alias names, the rule can navigate these dataclasses (e.g. `fn.calls`, `block.successors`, `loop.body.statements`, etc.).

---

## 6. Exceptions

`exceptions` is a list of expressions. If any evaluates to truthy, the violation is skipped. Example:

```yaml
exceptions:
  - "fn.annotations and 'ALLOW_BLOCKING' in [ann.text for ann in fn.annotations]"
  - "call.preprocessor.guard_condition == 'DEBUG'"
```

---

## 7. Messages & Fix-Its

Templated fields allow any DSL expression. Good practices:

```yaml
message: >
  ISR {{ fn.name }} calls blocking API {{ call.callee_name }}
  inside block {{ block.block_id }}
fixit: "Replace {{ call.callee_name }} with {{ safe_api }}"
```

If `fixit` renders to an empty string, the field is omitted from the emitted JSON.

---

## 8. Example Rule

```yaml
- id: NOMIC-CFG-001
  description: "ISR functions must not call blocking APIs (including transitive paths)."
  severity: error
  scope: Function
  select: "fn: Function where fn.is_isr"
  assert_code: >
    not exists(
      call.is_blocking_api
      for call in fn.calls
    ) and not call_path_exists(fn, 'sleep_ms')
  exceptions:
    - "'ALLOW_BLOCKING' in [ann.text for ann in fn.annotations]"
  message: >
    ISR {{ fn.name }} makes a blocking call
  fixit: >
    Consider moving {{ fn.name }} onto a deferred worker thread.
  tags: [safety, isr]
```

---

## 9. Validation Tips

1. **YAML schema**: All required fields must be present. Unknown fields are ignored but should be avoided.
2. **Scope sanity**: Ensure `scope` and `select` refer to valid IR types; otherwise `nomic.py` logs a warning.
3. **DSL correctness**: Use `python nomic.py analyze ... --rules ...` on sample C files to smoke test complex expressions.
4. **Performance**: Favor selectors that narrow candidates early; expensive transitive helpers (`call_path_exists`) should be guarded by simpler checks first.
5. **Determinism**: Avoid non-deterministic constructs; rule evaluation should depend only on IR content.

---

## 10. Future Extensions (Reserved Keywords)

The following identifiers are built-ins inside the DSL environment—avoid using them as alias names to prevent conflicts:

- `project`, `project_db`, `functions_by_name`, `globals_by_name`
- `call_edge`, `calls_function`, `call_path_exists`, `reachable_functions`, `has_blocking_call_path`
- Quantifiers: `exists`, `forall`, `count`
- Pattern helpers: `pattern_search`, `pattern_matches`, `pattern_findall`, `pattern_extract`
- Semantic helpers: `semantic_pattern_matches`, `semantic_pattern_any`, `semantic_statements`
- Misc helpers: `all_paths_reach`, `any_path_reaches`, `paths_between`, `detect_isr`, `check_type_compliance`, `is_api_function`, `is_callback_function`, `is_syscall_function`

As the DSL grows (e.g., explicit quantifier keywords, path matchers, policy modules), updates will be published in this document.

---

## Appendix: Helper Reference

| Helper | Signature | Notes |
|--------|-----------|-------|
| `call_edge(caller, callee)` | `bool` | Accepts `Function`/`CallSite` objects or strings. Uses `ProjectDB.call_graph`. |
| `calls_function(function_obj, callee)` | `bool` | Scans `function_obj.calls`. |
| `call_path_exists(caller, callee, max_depth=64)` | `bool` | BFS; pass `max_depth` to limit search. |
| `reachable_functions(caller, max_depth=64)` | `List[str]` | Returns sorted reachable names. |
| `has_blocking_call_path(start, max_depth=32)` | `bool` | Returns true if a blocking `CallSite` is reachable. |
| `all_paths_reach(block, predicate)` | `bool` | Ensures every CFG path from `block` hits a node satisfying `predicate`. |
| `any_path_reaches(block, predicate)` | `bool` | True if at least one path satisfies the predicate. |
| `paths_between(block1, block2, max_paths=256)` | `List[List[BasicBlock]]` | Enumerates CFG paths. |
| `pattern_search/ pattern_matches/ pattern_findall/ pattern_extract` | see above | Multi-line regex helpers returning `PatternMatch` metadata. |
| `semantic_pattern_matches(...)` / `semantic_pattern_any(...)` / `semantic_statements(...)` | see above | Combine statement predicates with regexes to yield `SemanticMatch` objects. |
| `exists(iterable)` / `forall(iterable)` / `count(iterable)` | quantifier helpers | Work well with comprehensions: `exists(call for call in fn.calls if call.is_blocking_api)` |

All helpers are exposed via `_make_env_callable`, ensuring they comply with the safe-eval environment.

---

By following this specification you can create expressive, safe, and maintainable rules for Nomic. When the IR or DSL evolves, this document will be updated accordingly.
