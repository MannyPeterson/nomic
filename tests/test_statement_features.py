import unittest

import nomic


class StatementFeatureTests(unittest.TestCase):
    def test_assignment_and_call_detection(self) -> None:
        stmt = nomic.Statement("result = compute_value(input);")
        self.assertTrue(stmt.contains_assignment)
        self.assertTrue(stmt.contains_call)
        self.assertEqual(stmt.variables_written, ["result"])
        self.assertIn("compute_value", stmt.variables_read)

    def test_macro_and_return_detection(self) -> None:
        stmt = nomic.Statement("ASSERT_STATE(ok);\nreturn status;")
        self.assertTrue(stmt.contains_macro)
        self.assertIn("ASSERT_STATE", stmt.macro_names)
        self.assertTrue(stmt.contains_return)


class CoercePatternTextTests(unittest.TestCase):
    def test_coerce_statement_and_collection(self) -> None:
        stmt = nomic.Statement("debug log")
        block = nomic.BasicBlock(
            block_id=7,
            statements=[stmt, nomic.Statement("next line")],
            function=None,
        )
        text = nomic._coerce_pattern_text(block)
        self.assertIn("debug log", text)
        self.assertIn("next line", text)
        combined = nomic._coerce_pattern_text([stmt, "plain text"])
        self.assertIn("plain text", combined)


if __name__ == "__main__":
    unittest.main()
