import unittest

import nomic


def build_project_with_two_functions():
    fn_alpha = nomic.Function(name="alpha_service", return_type="void")
    fn_beta = nomic.Function(name="beta_handler", return_type="void")
    call = nomic.CallSite(callee_name="beta_handler")
    fn_alpha.calls.append(call)

    blocking_call = nomic.CallSite(callee_name="sleep_ms")
    blocking_call.is_blocking_api = True
    fn_beta.calls.append(blocking_call)

    project = nomic.ProjectDB()
    project.functions_by_name = {
        "alpha_service": [fn_alpha],
        "beta_handler": [fn_beta],
    }
    project.call_graph = {"alpha_service": {"beta_handler"}, "beta_handler": set()}
    project.module_rules = {"drivers/alpha": {"allow_blocking": False}}
    project.function_metadata = {
        "alpha_service": {"allow_blocking": False},
        "beta_handler": {"allow_blocking": True},
    }
    return project, fn_alpha, fn_beta


class RuleHelperTests(unittest.TestCase):
    def setUp(self) -> None:
        project, self.fn_alpha, self.fn_beta = build_project_with_two_functions()
        self.engine = nomic.RuleEngine(project, [])
        self.env = self.engine._create_base_env()

    def test_call_graph_helpers(self) -> None:
        self.assertTrue(self.env["call_edge"](self.fn_alpha, "beta_handler"))
        self.assertTrue(self.env["call_path_exists"](self.fn_alpha, "beta_handler"))
        self.assertEqual(self.env["reachable_functions"](self.fn_alpha), ["beta_handler"])

    def test_blocking_call_path_helper(self) -> None:
        helper = self.env["has_blocking_call_path"]
        self.assertTrue(helper(self.fn_alpha))
        self.assertTrue(helper("alpha_service"))
        self.assertTrue(helper(self.fn_beta))

    def test_pattern_helpers_exposed(self) -> None:
        pattern_helpers = ["pattern_search", "pattern_matches", "semantic_pattern_matches"]
        for helper_name in pattern_helpers:
            helper = self.env[helper_name]
            self.assertTrue(getattr(helper, "_nomic_safe_callable", False))

    def test_policy_helpers(self) -> None:
        policy_lookup = self.env["policy_lookup"]
        function_policy = self.env["function_policy"]
        self.assertFalse(policy_lookup("drivers/alpha", "allow_blocking", True))
        self.assertFalse(function_policy(self.fn_alpha, "allow_blocking", True))
        self.assertTrue(function_policy(self.fn_beta, "allow_blocking", False))

    def test_compute_call_graph_dominators(self) -> None:
        dominators = nomic._compute_call_graph_dominators(self.engine.project_db.call_graph)
        self.assertIn("alpha_service", dominators)
        self.assertIn("beta_handler", dominators["beta_handler"])


if __name__ == "__main__":
    unittest.main()
