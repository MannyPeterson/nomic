import unittest

import nomic


class ExpressionEngineTests(unittest.TestCase):
    def setUp(self) -> None:
        self.engine = nomic._SafeExpressionInterpreter()
        self.base_env = dict(nomic._SAFE_BASE_CALLABLES)

    def test_implies_operator_desugaring(self) -> None:
        env = dict(self.base_env)
        env.update({"a": True, "b": False})
        self.assertFalse(self.engine.evaluate("a implies b", env))
        self.assertTrue(self.engine.evaluate("not a implies b", env))

    def test_lambda_support_in_expressions(self) -> None:
        result = self.engine.evaluate("(lambda x: x * 3)(5)", {})
        self.assertEqual(result, 15)

    def test_hasattr_is_accessible(self) -> None:
        class Dummy:
            value = 10

        env = dict(self.base_env)
        env["obj"] = Dummy()
        self.assertTrue(self.engine.evaluate("hasattr(obj, 'value')", env))


if __name__ == "__main__":
    unittest.main()
