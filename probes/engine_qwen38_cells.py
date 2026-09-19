"""Qwen3.8's cells on one GB10 before any boot: its served lanes qualified, and the glue's GPU cases (carry campaign C1).

The wizard (engine/kernels/cells.py) marks every Qwen3.8 lane that is not admitted as unjudged on a GPU: the KDA decay
glue at the per-rank cell (4 key / 12 value heads x 128), the padded dense lane, the V4.1 mHC seam, the MLA glue. Their
GPU cases already exist in tests/test_engine_kernel_glue.py and skip without CUDA; the lanes that own arithmetic the
glue does not cover (the gated residual, GDN's gates and norm, QSA's head norm with its partial rotation) qualify
themselves at boot through engine/profiles/qwen38/lanes.qualify. This runs both on the single-GPU lane, where the
Qwen3.8 checkpoint is absent (srv2 holds it): the facts come from its config, copied beside this file.

qwen38_config.json is /home/choiceoh/models/qwen38-flash-next-nvfp4/config.json on srv2, byte for byte
(sha256 e765305daba0951974308f4d32c075b52a6a45974730d273f2216718a994d624, read 2026-09-17).

    bash bench/fleet.sh run --gpu qwen38-cells 20 'Qwen3.8 cells: lanes qualified, glue GPU cases' -- \\
      bash probes/run_engine_probe.sh probes/engine_kernel_check.py --lanes qwen38_cells

Correctness only: no timing here, so nothing in it is a speed claim. The glue cells become admitted only with a
measurement record beside the judgment (engine/QWEN38_CARRY.md, C2-C5).
"""
import json
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

CONFIG = Path(__file__).with_name('qwen38_config.json')
CONFIG_SHA256 = 'e765305daba0951974308f4d32c075b52a6a45974730d273f2216718a994d624'
# the glue's GPU cases, and the Qwen3.8 folds whose last step (a BF16 rounding) the CPU interpreter cannot judge
GLUE_CASES = ('tests.test_engine_kernel_glue.KdaDecayKernelTests', 'tests.test_engine_kernel_glue.GlueOnTheGpuTests',
              'tests.test_engine_qwen38_moe_finish.GatedSumTests', 'tests.test_engine_swiglu_pad.SwigluPadTests',
              'tests.test_engine_gdn_ring_gate.GdnRingGateTests', 'tests.test_engine_qk_norm_strided_heads.OnTheGpuTests',
              'tests.test_engine_qwen38_qsa_inputs.QsaInputsTests',
              'tests.test_engine_qwen38_attention_gate.AttentionGateTests',
              'tests.test_engine_qwen38_qsa_blocks.BlockAttentionTests',
              'tests.test_engine_qwen38_kernels.SparseAttentionTests',
              'tests.test_engine_qwen38_kernels.NormRopeTests',
              'tests.test_engine_qwen38_moe_route.SoftmaxTopkTests', 'tests.test_engine_qwen38_moe_route.LayerTests',
              'tests.test_engine_gdn_chunk_native.NativeChunkTests',
              # the QSA selection's folds on the served kernels (carry Q8, Q11): on a GB10 the split step is the
              # wide one, where both sides take the native radix select -- the case no other box can run
              'tests.test_engine_qwen38_qsa_group_scores.GroupScoreTests',
              'tests.test_engine_qwen38_covered_blocks.ServedSelectionTests',
              'tests.test_engine_qwen38_covered_attention.CoveredAttentionTests',
              'tests.test_engine_qwen38_query_shards.ServedSelectionTests',
              # the decode step's selection in one launch (carry Q7): the rule, and torch.topk's set without ties
              'tests.test_engine_qwen38_qsa_select.SelectTests',
              # the compact MoE's combine in one launch: each row's pairs in their order, the CPU's sequential sum
              'tests.test_engine_qwen38_moe_pairs.PairSumTests',
              # the short conv (GDN's, and GLM-5.3's KDA): byte for byte the frozen legacy adapter, prefill and ring,
              # with its token count an argument -- one kernel for every prompt length
              'tests.test_engine_causal_conv.SingleConvTests', 'tests.test_engine_conv_ring.ConvRingTests')
GLUE_LEFT_OUT = ('tests.test_engine_causal_conv.SingleConvTests.test_graph_replay_changed_inputs_state_and_independent_streams',
                 'tests.test_engine_conv_ring.ConvRingTests.test_declared_reference_conv_disables_direct_ring')
"""Cases of those classes this lane does not run: they build GLM-5.3's served lane table (glm53.lanes.served), which arms
the MLA lane's prefill mode for the process -- after the MLA glue case above has armed it with another, so they refuse
(`configure_prefill: the MLA lane is already armed`, the 2026-09-19 run). They judge GLM's wiring, which GLM's own check
runs in a process of its own; the conv's arithmetic is the rest of their classes."""


def facts():
    """The served facts of the checkpoint this config describes (engine/profiles/qwen38/facts.load's checks included)."""
    import hashlib
    import tempfile
    raw = CONFIG.read_bytes()
    if hashlib.sha256(raw).hexdigest() != CONFIG_SHA256:
        raise RuntimeError(f'{CONFIG.name} is not the checkpoint config it records')
    from engine.profiles.qwen38 import facts as qwen38
    with tempfile.TemporaryDirectory() as ckpt:            # the probe's /repo is mounted read-only
        (Path(ckpt) / 'config.json').write_bytes(raw)
        return qwen38.load(ckpt)


def held(qualified: dict) -> dict:
    """What lanes.qualify returned, as JSON: a lane's worst a key is a (max, rms) pair -- a list here -- or, the skinny
    GEMV's, one number. (Taking every value for a pair killed this lane at its first line when that lane was added.)"""
    return {name: {key: list(worst) if isinstance(worst, (tuple, list)) else worst for key, worst in lane.items()}
            for name, lane in qualified.items()}


def _cases(suite):
    """The test cases of a loaded suite, flattened."""
    for item in suite:
        if isinstance(item, unittest.TestSuite):
            yield from _cases(item)
        else:
            yield item


def run(output=None):
    import torch
    assert torch.cuda.get_device_capability() == (12, 1), 'requires GB10'
    rows = []

    def report(lane, **values):
        rows.append(dict(lane=lane, **values))
        print(json.dumps(rows[-1]), flush=True)

    from engine.profiles.qwen38 import lanes
    F = facts()
    report('qwen38_qualify', config_sha256=CONFIG_SHA256, **held(lanes.qualify(torch.device('cuda'), F)))
    suite = unittest.TestSuite(case for case in _cases(unittest.defaultTestLoader.loadTestsFromNames(GLUE_CASES))
                               if case.id() not in GLUE_LEFT_OUT)
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    if not result.wasSuccessful() or result.skipped:
        raise RuntimeError(f'glue GPU cases failed or skipped: {len(result.failures)} failed, {len(result.errors)} errors, '
                           f'{len(result.skipped)} skipped')
    report('glue_gpu', passed=True, tests=result.testsRun, cases=list(GLUE_CASES), device=torch.cuda.get_device_name(),
           torch=torch.__version__, cuda=torch.version.cuda)
    if output:
        Path(output).write_text(''.join(json.dumps(row) + '\n' for row in rows))
