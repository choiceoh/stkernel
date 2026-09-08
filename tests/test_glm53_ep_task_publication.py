"""CPU row/scale-address and byte/slot oracles for EP-local publication."""
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


def load_physical_row():
    tree = ast.parse(KERNEL.read_text())
    method = next(node for node in ast.walk(tree) if isinstance(node, ast.FunctionDef)
                  and node.name == "initialize_route_q0_and_publish")
    # Execute the actual allocation expression, not a test-side copy of the
    # new formula. Other phys_row assignments load already allocated rows.
    assignment = next(node for node in ast.walk(method) if isinstance(node, ast.Assign)
                      and any(isinstance(target, ast.Name) and target.id == "phys_row"
                              for target in node.targets)
                      and any(isinstance(item, ast.Name) and item.id == "expert_tile_base"
                              for item in ast.walk(node.value)))
    expression = compile(ast.Expression(body=assignment.value), str(KERNEL), "eval")
    namespace = dict(Int32=int, self=SimpleNamespace(tile_shape_mnk=(128, 128, 128)))

    def physical_row(tile_base, expert, row):
        namespace.update(expert_tile_base=tile_base, expert_id=expert, row=row)
        return eval(expression, namespace)

    return physical_row


class PhysicalRowAddressTests(unittest.TestCase):
    def test_every_admissible_tile_boundary_matches_original_address(self):
        physical_row = load_physical_row()
        max_pairs = 16384 * 8
        # Expert 0 has one preceding row; expert 71 can then receive every
        # remaining pair. Its nonzero base exercises the prefix addition.
        local_count = max_pairs - 1
        bases = [0] + [1] * 71 + [1 + (local_count + 127) // 128]
        boundary_rows = {0, local_count - 1}
        for start in range(0, local_count, 128):
            boundary_rows.update(row for row in (start - 1, start, start + 1)
                                 if 0 <= row < local_count)
        for row in sorted(boundary_rows):
            expected = (bases[71] + row // 128) * 128 + row % 128
            self.assertEqual(physical_row(bases, 71, row), expected)
            self.assertLess(expected * (4096 // 2), 1 << 31)

    def test_histogram_rows_match_original_without_overlap_or_padding_writes(self):
        physical_row = load_physical_row()
        for tokens in (4096, 4097, 6912, 8192, 16384):
            pairs = tokens * 8
            edges = [0, 1, 127, 128, 129, 255, 256, 257, 511, 512, 513]
            fixtures = (
                [pairs] + [0] * 71,
                [0] * 71 + [pairs],
                [pairs // 72 + (expert < pairs % 72) for expert in range(72)],
                [pairs - 71] + [1] * 71,
                edges + [pairs - sum(edges)] + [0] * (71 - len(edges)),
                [0] * 72,
            )
            for counts in fixtures:
                bases = [0]
                for count in counts:
                    bases.append(bases[-1] + (count + 127) // 128)
                self.assertEqual(len(bases), 73)
                self.assertLessEqual(sum(counts), pairs)
                self.assertLessEqual(bases[-1] * 128, pairs + 72 * 127)
                self.assertLess(bases[-1] * 128 * (4096 // 2), 1 << 31)
                addresses = set()
                for expert, count in enumerate(counts):
                    lower, upper = bases[expert] * 128, bases[expert + 1] * 128
                    for row in range(count):
                        address = physical_row(bases, expert, row)
                        expected = (bases[expert] + row // 128) * 128 + row % 128
                        self.assertEqual(address, expected)
                        self.assertTrue(lower <= address < upper)
                        self.assertNotIn(address, addresses)
                        addresses.add(address)
                    self.assertTrue(all(address not in addresses
                                        for address in range(lower + count, upper)))
                self.assertEqual(len(addresses), sum(counts))


def scalar_scale_offset(physical_row, sf_index):
    """Original M128/H4096 scale layout, independently using divmod."""
    physical_tile, tile_row = divmod(physical_row, 128)
    k_tile, inner_k = divmod(sf_index, 4)
    inner_m, outer_m = divmod(tile_row, 32)
    return (physical_tile * 64 * 512 + k_tile * 512
            + outer_m * 16 + inner_m * 4 + inner_k)


def load_scale_offsets():
    tree = ast.parse(KERNEL.read_text())
    method = next(node for node in ast.walk(tree) if isinstance(node, ast.FunctionDef)
                  and node.name == "initialize_route_q0_and_publish")
    assignments = [node for node in ast.walk(method) if isinstance(node, ast.Assign)
                   and any(isinstance(target, ast.Name) and target.id == "scale_offset"
                           for target in node.targets)]
    if len(assignments) != 2:
        raise AssertionError("must inspect both equal and varied scale store paths")

    def uint32(value):
        if not 0 <= value < 1 << 32:
            raise AssertionError("scale address leaves the unsigned 32-bit range")
        return value

    def int32(value):
        if not -(1 << 31) <= value < 1 << 31:
            raise AssertionError("scale address leaves the signed 32-bit range")
        return value

    def compile_expression(expression):
        function = ast.parse("def offset(phys_row, sf_idx): return 0").body[0]
        function.body[0].value = expression
        namespace = dict(Int32=int32, Uint32=uint32, num_k_tiles=64)
        module = ast.fix_missing_locations(ast.Module(body=[function], type_ignores=[]))
        exec(compile(module, str(KERNEL), "exec"), namespace)
        return namespace[function.name]

    def addition_terms(expression):
        if isinstance(expression, ast.BinOp) and isinstance(expression.op, ast.Add):
            return addition_terms(expression.left) + addition_terms(expression.right)
        return [expression]

    result = []
    for assignment in assignments:
        expression = assignment.value
        if not (isinstance(expression, ast.Call)
                and isinstance(expression.func, ast.Name)
                and expression.func.id == "Int32" and len(expression.args) == 1):
            raise AssertionError("final scale address must retain checked Int32 conversion")
        # Prove row/SF separability before using exhaustive per-axis checks:
        # every additive term must depend on exactly one of the two axes.
        # No sampled Cartesian subset can hide a cross-axis interaction.
        row_terms, sf_terms = [], []
        for term in addition_terms(expression.args[0]):
            axes = {node.id for node in ast.walk(term)
                    if isinstance(node, ast.Name) and node.id in ("phys_row", "sf_idx")}
            if axes == {"phys_row"}:
                row_terms.append(term)
            elif axes == {"sf_idx"}:
                sf_terms.append(term)
            else:
                raise AssertionError("scale layout is no longer additive per axis")
        if not row_terms or not sf_terms:
            raise AssertionError("scale layout must retain both axes")
        result.append(compile_expression(expression))
    return result


class ScaleOffsetAddressTests(unittest.TestCase):
    def test_all_physical_rows_and_sf_indices_match_original_layout(self):
        # Even all 72 experts' maximum padding is included. The actual row
        # allocator cannot reach this conservative exclusive upper bound.
        rows_bound = 16384 * 8 + 72 * 127
        allocation_bytes = ((rows_bound + 127) // 128) * 128 * 256
        self.assertLess(allocation_bytes, 1 << 31)
        for offset in load_scale_offsets():
            # Combined with the AST separability proof, exhaustive checks of
            # each axis cover every admitted (physical row, SF index) pair.
            for row in range(rows_bound):
                expected = scalar_scale_offset(row, 0)
                self.assertEqual(offset(row, 0), expected)
                self.assertEqual(offset(row, 255), scalar_scale_offset(row, 255))
                self.assertTrue(0 <= expected < allocation_bytes)
            for sf_index in range(256):
                self.assertEqual(offset(0, sf_index), scalar_scale_offset(0, sf_index))
                self.assertEqual(offset(rows_bound - 1, sf_index),
                                 scalar_scale_offset(rows_bound - 1, sf_index))
            self.assertLess(offset(rows_bound - 1, 255), allocation_bytes)

    def test_m128_tiles_cover_every_scale_byte_without_overlap(self):
        rows_bound = 16384 * 8 + 72 * 127
        for offset in load_scale_offsets():
            # Every within-tile row and K boundary, plus low/high tile bits.
            for tile in (0, 1, (rows_bound - 1) // 128):
                start = tile * 128 * 256
                addresses = {
                    offset(tile * 128 + tile_row, sf_index)
                    for tile_row in range(128) for sf_index in range(256)
                }
                self.assertEqual(addresses, set(range(start, start + 128 * 256)))


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
