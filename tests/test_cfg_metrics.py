import unittest

import nomic


def build_cfg_function() -> nomic.Function:
    fn = nomic.Function(name="cfg_demo", return_type="void")
    entry = nomic.BasicBlock(
        block_id=0,
        statements=[nomic.Statement("entry"), nomic.Statement("branch")],
        function=fn,
    )
    cleanup = nomic.BasicBlock(
        block_id=1,
        statements=[nomic.Statement("cleanup resources")],
        function=fn,
    )
    fastpath = nomic.BasicBlock(
        block_id=2,
        statements=[nomic.Statement("fast path")],
        function=fn,
    )
    exit_block = nomic.BasicBlock(
        block_id=3,
        statements=[nomic.Statement("return done")],
        function=fn,
    )

    entry.successors = [cleanup.block_id, fastpath.block_id]
    cleanup.predecessors = [entry.block_id]
    cleanup.successors = [exit_block.block_id]
    fastpath.predecessors = [entry.block_id]
    fastpath.successors = [exit_block.block_id]
    exit_block.predecessors = [cleanup.block_id, fastpath.block_id]

    fn.cfg.blocks = {
        entry.block_id: entry,
        cleanup.block_id: cleanup,
        fastpath.block_id: fastpath,
        exit_block.block_id: exit_block,
    }
    fn.cfg.entry_block = entry
    fn.cfg.exit_blocks = [exit_block]
    nomic._finalize_function_cfg(fn)
    return fn


class CFGMetricsTests(unittest.TestCase):
    def setUp(self) -> None:
        self.fn = build_cfg_function()
        self.entry = self.fn.cfg.entry_block
        self.cleanup = self.fn.cfg.blocks[1]
        self.exit_block = self.fn.cfg.blocks[3]

    def test_exit_path_helpers(self) -> None:
        contains_cleanup = lambda block: any("cleanup" in stmt.text for stmt in block.statements)
        self.assertFalse(self.fn.cfg.all_exit_paths_postdominated_by(contains_cleanup))
        self.assertTrue(self.fn.cfg.has_path_without(contains_cleanup))
        paths = self.fn.cfg.get_paths_to_exit()
        self.assertEqual(len(paths), 2)

    def test_path_analysis_helpers(self) -> None:
        contains_cleanup = lambda block: any("cleanup" in stmt.text for stmt in block.statements)
        self.assertTrue(nomic.any_path_reaches(self.entry, contains_cleanup))
        self.assertFalse(nomic.all_paths_reach(self.entry, contains_cleanup))
        between = nomic.paths_between(self.entry, self.exit_block)
        self.assertEqual(len(between), 2)


if __name__ == "__main__":
    unittest.main()
