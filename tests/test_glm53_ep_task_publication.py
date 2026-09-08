"""CPU byte/slot oracle for EP-local's four-task vector publication."""
import ast
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import Mock


ROOT = Path(__file__).resolve().parents[1]
KERNEL = ROOT / "overlay/modules/glm53_moe/moe_dynamic_ep_local.py"


class TaskArray:
    def __init__(self, base, memory):
        self.base = base
        self.memory = memory

    def __setitem__(self, index, value):
        self.memory[self.base + 4 * index] = value & 0xFFFFFFFF


def scalar_publish(experts, rows, gate_tiles, chunk, expert, tile, valid):
    """Pinned stock publish_uniform_deferred_tasks' scalar descriptor oracle."""
    groups = (gate_tiles + chunk - 1) // chunk
    for group in range(groups):
        slot = tile * groups + group
        begin = group * chunk
        count = min(chunk, gate_tiles - begin)
        experts[slot] = expert | (tile << 16)
        rows[slot] = valid | (begin << 8) | (count << 20)


def load_publisher(memory):
    tree = ast.parse(KERNEL.read_text())
    cls = next(node for node in tree.body if isinstance(node, ast.ClassDef)
               and node.name == "MoEGatedEPLocalKernel")
    method = next(node for node in cls.body if isinstance(node, ast.FunctionDef)
                  and node.name == "publish_ep_local_uniform_tasks")
    method.decorator_list = []
    calls = []

    def vector_store(address, *words):
        if address % 16:
            raise AssertionError("unaligned vector store")
        if len(words) != 4:
            raise AssertionError("vector store width differs")
        calls.append(address)
        for offset, word in enumerate(words):
            memory[address + 4 * offset] = word & 0xFFFFFFFF

    namespace = dict(
        Int32=int, Int64=int, Uint32=lambda value: value & 0xFFFFFFFF,
        get_ptr_as_int64=lambda tensor, index: tensor.base + 4 * index,
        st_global_v4_u32=vector_store,
    )
    exec(compile(ast.Module(body=[method], type_ignores=[]), str(KERNEL), "exec"),
         namespace)
    fallback = Mock(side_effect=scalar_publish)
    owner = SimpleNamespace(publish_uniform_deferred_tasks=fallback)
    return lambda *args: namespace[method.name](owner, *args), calls, fallback


class TaskPublicationTests(unittest.TestCase):
    def test_all_valid_rows_and_experts_match_scalar_words(self):
        memory, reference = {}, {}
        publish, calls, fallback = load_publisher(memory)
        experts, rows = TaskArray(0x10000000, memory), TaskArray(0x20000000, memory)
        ref_experts, ref_rows = TaskArray(experts.base, reference), TaskArray(rows.base, reference)
        for tile in (0, 1, 71, 1095, 65535):
            for expert in range(72):
                for valid in range(1, 129):
                    memory.clear()
                    reference.clear()
                    calls.clear()
                    publish(experts, rows, 16, 4, expert, tile, valid)
                    scalar_publish(ref_experts, ref_rows, 16, 4, expert, tile, valid)
                    self.assertEqual(memory, reference)
                    self.assertEqual(calls, [experts.base + tile * 16,
                                             rows.base + tile * 16])
        fallback.assert_not_called()

    def test_complete_tile_prefix_has_no_gap_overlap_or_tail_write(self):
        for tokens in (4096, 4097, 6912, 8192, 16384):
            pairs = tokens * 8
            fixtures = (
                [pairs] + [0] * 71,
                [pairs // 72 + (expert < pairs % 72) for expert in range(72)],
                [pairs - 71] + [1] * 71,
                [0] * 72,
            )
            for counts in fixtures:
                memory = {}
                publish, calls, fallback = load_publisher(memory)
                experts = TaskArray(0x10000000, memory)
                rows = TaskArray(0x20000000, memory)
                tile = 0
                expected = []
                for expert, count in enumerate(counts):
                    for start in range(0, count, 128):
                        valid = min(128, count - start)
                        publish(experts, rows, 16, 4, expert, tile, valid)
                        expected.extend((expert, tile, valid, begin, 4)
                                        for begin in (0, 4, 8, 12))
                        tile += 1
                # Decode the actual wire fields independently of the store
                # expressions; every active task must cover its four slices.
                actual = []
                for slot in range(tile * 4):
                    expert_word = memory[experts.base + slot * 4]
                    row_word = memory[rows.base + slot * 4]
                    actual.append((expert_word & 0xFFFF, expert_word >> 16,
                                   row_word & 0xFF, (row_word >> 8) & 0xFFF,
                                   row_word >> 20))
                self.assertEqual(actual, expected)
                self.assertEqual(len(memory), tile * 8)
                self.assertEqual(len(calls), tile * 2)
                self.assertEqual(len(set(calls)), len(calls))
                self.assertNotIn(experts.base + tile * 16, memory)
                self.assertNotIn(rows.base + tile * 16, memory)
                fallback.assert_not_called()

    def test_four_byte_pointer_abi_uses_scalar_fallback(self):
        for expert_offset, rows_offset in ((4, 0), (0, 4), (8, 12), (12, 8)):
            memory, reference = {}, {}
            publish, calls, fallback = load_publisher(memory)
            experts = TaskArray(0x10000000 + expert_offset, memory)
            rows = TaskArray(0x20000000 + rows_offset, memory)
            publish(experts, rows, 16, 4, 71, 1095, 127)
            scalar_publish(TaskArray(experts.base, reference),
                           TaskArray(rows.base, reference), 16, 4, 71, 1095, 127)
            self.assertEqual(memory, reference)
            self.assertEqual(calls, [])
            fallback.assert_called_once_with(experts, rows, 16, 4, 71, 1095, 127)

    def test_other_slice_contracts_keep_stock_scalar_publication(self):
        for gate_tiles, chunk in ((16, 2), (12, 4), (17, 4), (3, 8)):
            memory, reference = {}, {}
            publish, calls, fallback = load_publisher(memory)
            experts, rows = TaskArray(0x10000000, memory), TaskArray(0x20000000, memory)
            publish(experts, rows, gate_tiles, chunk, 71, 1095, 1)
            scalar_publish(TaskArray(experts.base, reference),
                           TaskArray(rows.base, reference), gate_tiles, chunk, 71, 1095, 1)
            self.assertEqual(memory, reference)
            self.assertEqual(calls, [])
            fallback.assert_called_once()


if __name__ == "__main__":
    unittest.main()
