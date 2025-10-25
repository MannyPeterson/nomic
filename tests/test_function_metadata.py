import unittest

import nomic


class FunctionMetadataTests(unittest.TestCase):
    def test_api_callback_syscall_detection(self) -> None:
        api_fn = nomic.Function(name="api_handle_request", return_type="void")
        cb_fn = nomic.Function(name="work_callback", return_type="void")
        sys_fn = nomic.Function(name="sys_enter", return_type="void", attributes=["syscall"])
        self.assertTrue(api_fn.is_api)
        self.assertTrue(cb_fn.is_callback)
        self.assertTrue(sys_fn.is_syscall)

    def test_mutex_protection_detection(self) -> None:
        fn = nomic.Function(name="guarded", return_type="void")
        lock = nomic.CallSite(callee_name="mutex_lock")
        lock.is_lock = True
        unlock = nomic.CallSite(callee_name="mutex_unlock")
        unlock.is_unlock = True
        fn.calls.extend([lock, unlock])
        self.assertTrue(fn.has_mutex_protection)

    def test_statement_collections(self) -> None:
        fn = nomic.Function(name="stmt_fn", return_type="void")
        entry = nomic.BasicBlock(
            block_id=0,
            statements=[nomic.Statement("line 1"), nomic.Statement("line 2")],
            function=fn,
        )
        fn.cfg.blocks = {0: entry}
        fn.cfg.entry_block = entry
        fn.cfg.exit_blocks = [entry]
        self.assertEqual([stmt.text for stmt in fn.all_statements], ["line 1", "line 2"])
        self.assertEqual([stmt.text for stmt in fn.entry_statements], ["line 1", "line 2"])
        self.assertEqual([stmt.text for stmt in fn.exit_statements], ["line 1", "line 2"])


class VariableMetadataTests(unittest.TestCase):
    def test_variable_properties(self) -> None:
        var = nomic.Variable(
            name="data",
            ctype="custom_t",
            storage="static",
            scope="file",
            decl_function="main",
        )
        nomic.register_custom_types(["custom_t"])
        self.assertTrue(var.is_static)
        self.assertFalse(var.is_global)
        self.assertEqual(var.base_type, "custom_t")
        self.assertEqual(var.parent_function, "main")
        self.assertTrue(var.is_custom_type)
        self.assertEqual(nomic.get_type_equivalent("u8"), "uint8_t")


if __name__ == "__main__":
    unittest.main()
