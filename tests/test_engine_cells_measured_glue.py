"""A glue cell with a GPU judgment and a measurement record is admitted (engine/QWEN38_CARRY.md C7, the operator's Q3).

engine/kernels/cells.py admitted a lane only inside a compiled cell reached directly and a measured cell. An exact adapter
(glue) puts another shape on the same compiled kernel -- zero-padded dense columns, a per-head decay read through a
stride-0 channel axis, the V4.1 split-sinkhorn seam -- and was refused whatever had been judged or timed. The records
live in four tuples (DENSE_GLUE_MEASURED_COLUMNS, KDA_DECAY_MEASURED_CELLS, MHC_V41_MEASURED_HIDDEN, MOE_MEASURED_CELLS),
empty until the single-GPU lane's records land. With them empty nothing changes; with a cell in them, that cell's lanes
are admitted and serve the same kernel marked judged. Nothing else moves: a cell not in a tuple is judged as before.
"""
from dataclasses import replace
import unittest
from unittest import mock

from engine.kernels import cells
from tests.test_engine_kernel_shape import dsv41_shape, qwen_shape


def verdicts(shape):
    return {v.lane: v for v in cells.admission(shape)}


class MeasuredGlueTests(unittest.TestCase):
    def test_the_tuples_start_empty_and_nothing_moves(self):
        self.assertEqual((cells.DENSE_GLUE_MEASURED_COLUMNS, cells.KDA_DECAY_MEASURED_CELLS, cells.MHC_V41_MEASURED_HIDDEN,
                          cells.MOE_MEASURED_CELLS), ((), (), (), ()))
        q = verdicts(qwen_shape())
        self.assertEqual({lane: v.status for lane, v in q.items() if lane in ("dense", "kda_ring", "kda_chunk", "moe")},
                         {"dense": cells.REFUSED, "kda_ring": cells.REFUSED, "kda_chunk": cells.REFUSED,
                          "moe": cells.UNMEASURED})

    def test_measured_dense_columns_admit_the_padded_lane(self):
        shape = qwen_shape()
        with mock.patch.object(cells, "DENSE_GLUE_MEASURED_COLUMNS", (160,)), \
                mock.patch.object(cells, "DENSE_MEASURED_HIDDEN", (4096, shape.hidden)):
            v = verdicts(shape)["dense"]
        self.assertEqual((v.status, v.serve.tier, v.serve.judged), (cells.ADMITTED, cells.GLUE, True))
        self.assertIn("PaddedDenseLinear", v.serve.kernel)
        self.assertIsNone(v.recipe)

    def test_a_column_or_a_hidden_without_a_record_stays_refused(self):
        shape = qwen_shape()
        with mock.patch.object(cells, "DENSE_GLUE_MEASURED_COLUMNS", (160,)):          # the hidden 2560 dispatch untimed
            self.assertEqual(verdicts(shape)["dense"].status, cells.REFUSED)
        with mock.patch.object(cells, "DENSE_MEASURED_HIDDEN", (4096, shape.hidden)):    # the padded width untimed
            self.assertEqual(verdicts(shape)["dense"].status, cells.REFUSED)

    def test_a_measured_decay_cell_admits_all_three_linear_lanes(self):
        shape = qwen_shape()
        l = shape.linear
        with mock.patch.object(cells, "KDA_DECAY_MEASURED_CELLS", ((l.heads, l.v_heads, l.k_dim, l.v_dim),)):
            v = verdicts(shape)
        for lane in ("kda_recurrent", "kda_ring", "kda_chunk"):
            with self.subTest(lane=lane):
                self.assertEqual((v[lane].status, v[lane].serve.tier, v[lane].serve.judged),
                                 (cells.ADMITTED, cells.GLUE, True))
                self.assertNotIn("unjudged", v[lane].serve.note)

    def test_a_measured_moe_cell_is_admitted_with_its_pin(self):
        shape = qwen_shape()
        cell = replace(shape.moe, dynamic_tile_m=None)
        pinned = replace(shape, moe=replace(shape.moe, dynamic_tile_m=32))
        with mock.patch.object(cells, "MOE_MEASURED_CELLS", (cell,)):
            v = verdicts(pinned)["moe"]
        self.assertEqual((v.status, v.serve.tier, v.serve.judged), (cells.ADMITTED, cells.SPECIALIZED, True))
        self.assertIn("tile pinned at 32", v.why)

    def test_a_measured_v41_hidden_admits_the_seam_for_both_mhc_lanes(self):
        shape = dsv41_shape()
        with mock.patch.object(cells, "MHC_V41_MEASURED_HIDDEN", (shape.hidden,)):
            v = verdicts(shape)
        for lane in ("mhc_decode", "mhc_prefill"):
            with self.subTest(lane=lane):
                self.assertEqual((v[lane].status, v[lane].serve.tier, v[lane].serve.judged),
                                 (cells.ADMITTED, cells.GLUE, True))
                self.assertIn("MHCV41", v[lane].serve.kernel)

    def test_the_v41_record_does_not_admit_another_hyper_connection_form(self):
        shape = qwen_shape()                                                          # gated residual, not split-sinkhorn
        with mock.patch.object(cells, "MHC_V41_MEASURED_HIDDEN", (shape.hidden,)):
            self.assertEqual(verdicts(shape)["mhc_decode"].status, cells.REFUSED)


if __name__ == "__main__":
    unittest.main()
