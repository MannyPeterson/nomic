import unittest

import nomic


def build_sample_function() -> nomic.Function:
    fn = nomic.Function(name="process_packets", return_type="void")
    entry = nomic.BasicBlock(
        block_id=1,
        statements=[
            nomic.Statement("LOG_INFO(\"start\")"),
            nomic.Statement("packet = dequeue();"),
            nomic.Statement("if (!packet) { return; }"),
        ],
        function=fn,
    )
    exit_block = nomic.BasicBlock(
        block_id=2,
        statements=[
            nomic.Statement("process(packet);"),
            nomic.Statement("LOG_DEBUG(\"done\")"),
        ],
        function=fn,
    )
    entry.successors.append(exit_block.block_id)
    exit_block.predecessors.append(entry.block_id)
    fn.cfg.blocks = {entry.block_id: entry, exit_block.block_id: exit_block}
    fn.cfg.entry_block = entry
    fn.cfg.exit_blocks = [exit_block]
    return fn


class SemanticPatternTests(unittest.TestCase):
    def setUp(self) -> None:
        self.fn = build_sample_function()

    def test_semantic_statements_filter(self) -> None:
        statements = nomic.semantic_statements(
            self.fn, predicate=lambda stmt, *_: stmt.contains_call
        )
        self.assertGreaterEqual(len(statements), 2)
        self.assertTrue(any("process" in stmt.text for stmt in statements))

    def test_semantic_pattern_matches_statement_scope(self) -> None:
        matches = nomic.semantic_pattern_matches(
            self.fn,
            predicate=lambda stmt, *_: "LOG" in stmt.text,
            pattern=r"LOG_(?P<level>\w+)",
            pattern_scope="statement",
        )
        self.assertEqual(len(matches), 2)
        levels = {m.pattern_match.groupdict["level"] for m in matches if m.pattern_match}
        self.assertSetEqual(levels, {"INFO", "DEBUG"})
        for match in matches:
            self.assertIs(match.function, self.fn)

    def test_semantic_pattern_matches_function_scope(self) -> None:
        matches = nomic.semantic_pattern_matches(
            self.fn,
            predicate=lambda stmt, *_: "packet" in stmt.text,
            pattern=r"process\((?P<var>\w+)\)",
            pattern_scope="function",
        )
        self.assertGreaterEqual(len(matches), 1)
        first = matches[0]
        self.assertIsNotNone(first.pattern_match)
        self.assertEqual(first.pattern_match.groupdict["var"], "packet")

    def test_semantic_pattern_any_convenience(self) -> None:
        has_info_log = nomic.semantic_pattern_any(
            self.fn,
            predicate=lambda stmt, *_: "LOG_INFO" in stmt.text,
            pattern=r"LOG_INFO",
        )
        self.assertTrue(has_info_log)


class RuleEngineEnvironmentTests(unittest.TestCase):
    def test_pattern_helpers_available_in_env(self) -> None:
        project = nomic.ProjectDB()
        engine = nomic.RuleEngine(project, [])
        env = engine._create_base_env()
        for helper_name in (
            "pattern_search",
            "pattern_matches",
            "semantic_pattern_matches",
            "semantic_pattern_any",
        ):
            self.assertIn(helper_name, env)
            self.assertTrue(getattr(env[helper_name], "_nomic_safe_callable", False))


if __name__ == "__main__":
    unittest.main()
