import unittest

import nomic


class PatternEngineTests(unittest.TestCase):
    def test_pattern_search_basic(self) -> None:
        text = "alpha\nbeta foo\nGamma Foo"
        matches = nomic.pattern_search(text, r"foo", ignore_case=True)
        self.assertGreaterEqual(len(matches), 2)
        self.assertTrue(all("foo" in m.match.lower() for m in matches))
        # validate line span metadata
        span_lines = {m.line_span for m in matches}
        self.assertIn((2, 2), span_lines)

    def test_pattern_matches_literal(self) -> None:
        stmt = nomic.Statement("call(zero == 0)")
        self.assertTrue(
            nomic.pattern_matches(stmt, "call(zero == 0)", literal=True, multiline=False)
        )

    def test_pattern_extract_groups(self) -> None:
        text = "API_call(foo)\nAPI_call(bar)"
        extracted = nomic.pattern_extract(
            text,
            r"API_call\((?P<name>\w+)\)",
            group="name",
        )
        self.assertEqual(extracted, ["foo", "bar"])


class PatternFunctionsAsTargets(unittest.TestCase):
    def setUp(self) -> None:
        self.fn = nomic.Function(name="processor", return_type="void")
        block = nomic.BasicBlock(
            block_id=1,
            statements=[
                nomic.Statement("LOG_INFO(\"start\")"),
                nomic.Statement("value = compute();"),
            ],
            function=self.fn,
        )
        self.fn.cfg.blocks = {1: block}
        self.fn.cfg.entry_block = block
        self.fn.cfg.exit_blocks = [block]

    def test_pattern_search_function(self) -> None:
        matches = nomic.pattern_search(self.fn, r"LOG_(\w+)")
        self.assertEqual(len(matches), 1)
        self.assertEqual(matches[0].groups[0], "INFO")


if __name__ == "__main__":
    unittest.main()
