"""Judge ST kernel imports and numerical contracts with vLLM imports forbidden.

Use --imports-only in a CPU container, or --lanes to run a bounded GPU subset.
The default checks engine references and graph replay. MoE migration is judged
against the seed's original FlashInfer kernel, with the torch oracle's error
reported separately because FP4 quantization has discontinuous rounding.
"""
from __future__ import annotations

import argparse
import importlib
import importlib.abc
import json
from pathlib import Path
import pkgutil
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


class ForbidVllm(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname == "vllm" or fullname.startswith("vllm."):
            raise AssertionError(f"ST kernel attempted a vLLM import: {fullname}")
        return None


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--imports-only", action="store_true")
    parser.add_argument("--lanes", default="conv,kda,mhc,indexer,kpool,mla,moe")
    parser.add_argument("--ranks", help="exact consumer rank directory for real router validation")
    parser.add_argument("--output", type=Path, help="artifact path for the moe_pair comparison")
    parser.add_argument("--moe-experts", type=int, choices=(8, 288), default=8,
                        help="8 for bounded smoke; 288 for GLM's full TP4 expert geometry")
    parser.add_argument("--moe-static", default="stock", help="served b12x static-lane spec (STK_moe_static): stock | t,r,sf6[,q0]")
    parser.add_argument("--mla-prefill", default="stock", help="served MLA prefill mode (STK_mla_prefill): stock | tile32 | pair | pair4")
    parser.add_argument("--seqs", help="dense_cells: concurrencies to compare, 1 -> 8 rows, 2 -> 16 rows (default 1,2)")
    parser.add_argument("--samples", help="dense_cells: B/A/A/B brackets per comparison (default 2)")
    args = parser.parse_args()
    if args.lanes in ('fp4_scale_search', 'fp4_scale_search_compile', 'fp4_scale_search_quant'):
        from probes.engine_fp4_scale_search import run
        run(args.output, args.ranks, compile_only=args.lanes.endswith('_compile'),
            quant_only=args.lanes.endswith('_quant'))
        return
    if args.lanes == 'next_k_compile' or args.lanes == 'next_k_cost' or args.lanes.startswith('next_k_cost:'):
        from probes.engine_fixed_k_next import run
        run(args.output, args.ranks, compile_only=args.lanes == 'next_k_compile', sections=args.lanes.split(':')[1:])
        return
    if args.lanes in ('fixed_k_compile', 'fixed_k_cost'):
        from probes.engine_fixed_k_cost import run
        run(args.output, args.ranks, compile_only=args.lanes == 'fixed_k_compile')
        return
    if args.lanes == 'boundary_stage':
        from probes.engine_boundary_stage import run as boundary_stage_check
        boundary_stage_check(args.output)
        return
    if args.lanes == 'vocab_merge':
        from probes.engine_vocab_merge import run as vocab_merge_check
        vocab_merge_check(args.output)
        return
    if args.lanes == 'vocab_selection':
        from probes.engine_vocab_selection import run as vocab_selection_check
        vocab_selection_check(args.output)
        return
    if args.lanes == 'vocab_argmax':
        # component timings: the greedy pick's two launches at 4, 2 and 1 warps, exact against the CPU key first (D5)
        from probes.engine_vocab_selection import argmax_run as vocab_argmax_check
        vocab_argmax_check(args.output)
        return
    if args.lanes == 'mhc_c1_tails':
        from probes.engine_mhc_c1_tails import main as mhc_c1_tails_check
        mhc_c1_tails_check(args.ranks, samples=args.samples, output=args.output)
        return
    if args.lanes == 'producer_pack':
        from probes.engine_producer_pack import main as producer_pack_check
        producer_pack_check(args.ranks, seqs=args.seqs, samples=args.samples, output=args.output)
        return
    if args.lanes == 'moe_c2_cells' or args.lanes.startswith('moe_c2_cells:'):
        # the routed experts' same-build cells: tile-major w13 chunk, stamped timeline, prefill (real rank weights)
        from probes.engine_moe_c2_cells import main as moe_c2_cells
        moe_c2_cells(args.ranks, sections=args.lanes.split(':')[1:], samples=args.samples, output=args.output)
        return
    if args.lanes in ('moe_input_reuse', 'moe_input_reuse_compile'):
        from probes.engine_moe_input_reuse import run
        run(args.output, args.ranks, compile_only=args.lanes.endswith('_compile'))
        return
    if args.lanes == 'router_cells':
        # the decode router's launch fold: the served seven-launch chain against one fused launch (real rank gates)
        from probes.engine_router_cells import main as router_cells
        router_cells(args.ranks, samples=args.samples, output=args.output)
        return
    if args.lanes == 'dense_cells' or args.lanes.startswith('dense_cells:'):
        from probes.engine_dense_cells import main as dense_cells_check
        dense_cells_check(args.ranks, cells=args.lanes.split(':')[1:], seqs=args.seqs, samples=args.samples,
                          output=args.output)
        return
    if args.lanes == 'forward_pipeline':
        from probes.engine_forward_pipeline import main as forward_pipeline_check
        forward_pipeline_check(args.ranks)
        return
    if args.lanes == 'forward_reduce':
        from probes.engine_forward_reduce import main as forward_reduce_check
        forward_reduce_check(args.ranks)
        return
    if args.lanes == 'mhc_c2_packed':
        from probes.engine_mhc_c2_packed import main as mhc_c2_packed_check
        mhc_c2_packed_check(args.ranks, args.output)
        return
    if args.lanes == 'dsa_inputs':
        from probes.engine_decode_dsa_inputs import check as dsa_inputs_check
        dsa_inputs_check(args.ranks)
        return
    if args.lanes == 'oneshot_consumer':
        # The PDL consumer sum against the ordinary kernel at C=1/C=2 rows, and the MoE packet ring at the same
        # rows, on the production transport source behind a CPU proxy: bytes and tickets, not NIC latency.
        import unittest
        suite = unittest.defaultTestLoader.loadTestsFromNames(('tests.test_engine_oneshot_consumer_cuda',
                                                               'tests.test_engine_moe_output_transport'))
        result = unittest.TextTestRunner(verbosity=2).run(suite)
        if not result.wasSuccessful() or result.skipped:
            raise RuntimeError('one-shot consumer transport gates failed or skipped')
        import torch
        row = dict(lane='oneshot_consumer', passed=True, tests=result.testsRun, device=torch.cuda.get_device_name(),
                   torch=torch.__version__, cuda=torch.version.cuda)
        if args.output:
            args.output.write_text(json.dumps(row) + '\n')
        print(json.dumps(row), flush=True)
        return
    if args.lanes == 'oneshot_consumer_timing':
        # GPU-side only: peers are landed before each chain publishes, so no RDMA time is in these numbers.
        from probes.engine_oneshot_consumer_timing import check as consumer_timing
        rows = []
        def report(name, **values):
            rows.append(dict(lane=name, **values))
            print(json.dumps(rows[-1]), flush=True)
        consumer_timing(report)
        if args.output:
            args.output.write_text(''.join(json.dumps(row) + '\n' for row in rows))
        return
    if args.lanes == 'topk_params':
        # which host choice (bin cache, stash, block width) sets st_dsa_select's time on this device
        from probes.engine_topk_hpcops import sweep as topk_params_sweep
        topk_params_sweep(args.output)
        return
    if args.lanes == 'topk_hpcops':
        # component only: HPC-Ops' exact top-k (vendored, MIT) against st_dsa_select / prefill_topk and a read floor
        from probes.engine_topk_hpcops import run as topk_hpcops_check
        topk_hpcops_check(args.output)
        return
    if args.lanes == 'qwen38_cells':
        # correctness only: Qwen3.8's lanes qualified and the glue's GPU cases, from its config (engine/QWEN38_CARRY.md C1)
        from probes.engine_qwen38_cells import run as qwen38_cells
        qwen38_cells(args.output)
        return
    if args.lanes == 'qwen38_dense':
        # component timings: the W4A8/FP8 switch at Qwen3.8's projection shapes and the padded widths, gated first (C2)
        from probes.engine_qwen38_dense import run as qwen38_dense
        qwen38_dense(args.output)
        return
    if args.lanes == 'qwen38_kda':
        # component timings: Qwen3.8's GDN on the KDA kernels at its 4/12 x 128 cell, value tiles 8/16/32, exact gate first (C3)
        from probes.engine_qwen38_kda import run as qwen38_kda
        qwen38_kda(args.output)
        return
    if args.lanes == 'qwen38_step':
        # component timings: a captured Qwen3.8 decode step's kernels on one rank's own weights, solved from small nets
        # (fixed, GDN, QSA, PLE) and summed to 48 layers -- the decode levers ranked by the step, not by guesses
        from probes.engine_qwen38_step import run as qwen38_step
        qwen38_step(args.output, args.ranks)
        return
    if args.lanes == 'qwen38_step_where':
        # one layer set's build only: its graphs, the served loop, and the Python stack each eager kernel came from
        from probes.engine_qwen38_step import LAYER_SETS, run as qwen38_step
        qwen38_step(args.output, args.ranks, layer_sets=LAYER_SETS[:1])
        return
    if args.lanes == 'qwen38_mtp_experts':
        # the MTP head's experts: the side-file kernel (kernels/moe_rows) on the rank's real weights against its torch
        # form, and BF16 / FP8 / the NVFP4 re-encoding each against the checkpoint's original BF16
        from probes.engine_qwen38_mtp_experts import run as qwen38_mtp_experts
        qwen38_mtp_experts(args.output, args.ranks)
        return
    if args.lanes == 'qwen38_draft_head':
        # the draft argmax from an inverted-file index over the rank's real head: agreement with the full head on three
        # stand-in query sets at each (clusters, probes), the index's build time, and its cost against the full head's
        from probes.engine_qwen38_draft_head import run as qwen38_draft_head
        qwen38_draft_head(args.output, args.ranks)
        return
    if args.lanes == 'qwen38_step_mtp':
        # the draft graph under the MTP head's dense projections in BF16 (served), W4A8 and FP8, one layer set each
        from probes.engine_qwen38_step import LAYER_SETS, MTP_ARMS, run as qwen38_step
        qwen38_step(args.output, args.ranks, layer_sets=LAYER_SETS[1:2], arms=MTP_ARMS)
        return
    if args.lanes == 'qwen38_step_mtp_gemv':
        # the draft graph with the MTP head's BF16 projections on the skinny GEMV (served) and on torch.mm, one layer set,
        # three rounds in alternating order: a production that comes or goes mid-ticket lands on both arms, and shows in
        # each build's free memory (q38mtpgemv-0919a built its arms once each and production booted between them --
        # 80.7 against 39.7 GiB free, the draft graph 4.6 against 10.2 ms: contention, not the kernels)
        from probes.engine_qwen38_step import GEMV_ARMS, LAYER_SETS, run as qwen38_step
        rounds = [qwen38_step(None, args.ranks, layer_sets=LAYER_SETS[1:2], arms=arms)
                  for arms in (GEMV_ARMS, GEMV_ARMS[::-1], GEMV_ARMS)]
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        Path(args.output).write_text(json.dumps({"rounds": rounds}, indent=1) + "\n")
        return
    if args.lanes == 'qwen38_mtp_window':
        # the MTP head's draft graph at every context bucket, its QSA selection scored against a sink-and-recent window
        # of groups (fleet --mtp-window): what the draft's index scoring costs as the context grows
        from probes.engine_qwen38_mtp_window import run as qwen38_mtp_window
        qwen38_mtp_window(args.output, args.ranks)
        return
    if args.lanes == 'qwen38_step_overlap':
        # one rank's captured step with the shared expert forked beside the routed experts, against the served step (M5)
        from probes.engine_qwen38_step import LAYER_SETS, OVERLAP_ARMS, run as qwen38_step
        qwen38_step(args.output, args.ranks, layer_sets=LAYER_SETS[:1], arms=OVERLAP_ARMS)
        return
    if args.lanes == 'qwen38_step_ab':
        # the same step, each layer set built under the served lanes and again with the skinny GEMV's shapes on
        # torch.mm -- what the router's and the mixers' down projections on it change in a replayed step
        from probes.engine_qwen38_step import ARMS, run as qwen38_step
        qwen38_step(args.output, args.ranks, arms=ARMS)
        return
    if args.lanes == 'qwen38_step_ahead':
        # the served model's decode loop on one rank, synchronous and with the draft step launched behind the verify
        # step on the device (fleet --draft-ahead), six runs in turn: each run's wall a step, and the same tokens
        from probes.engine_qwen38_step import ahead as qwen38_step_ahead
        qwen38_step_ahead(args.output, args.ranks)
        return
    if args.lanes == 'qwen38_gemv':
        # component timings: a skinny BF16 GEMV (one weight read for all rows) against cuBLAS at the decode step's mixer
        # and router shapes, interleaved in CUDA graphs -- whether the 11.2 ms of BF16 GEMM a step has a faster kernel
        from probes.engine_qwen38_gemv import run as qwen38_gemv
        qwen38_gemv(args.output)
        return
    if args.lanes == 'qwen38_head':
        # component timings: the vocabulary head at decode rows -- deep_gemm against dense/fp8_rows (the served lane and
        # a tile sweep) and a pure read of the weight, with each one's error and argmax agreement
        from probes.engine_qwen38_head import run as qwen38_head
        qwen38_head(args.output)
        return
    if args.lanes in ('glm53_head', 'glm53_gemv'):
        # component timings at GLM-5.3's shapes and rows: whether Qwen3.8's decode-row kernels (dense/fp8_rows,
        # common/skinny_gemv) beat what GLM serves -- its cuBLASLt head reader, torch.mm on the indexer pair
        from probes.engine_glm53_decode_rows import run_gemv, run_head
        (run_head if args.lanes == 'glm53_head' else run_gemv)(args.output)
        return
    if args.lanes == 'sm121_inventory':
        # what the seed image carries for engine/SM121_INTAKE.md (U0): files, imports and signatures -- nothing compiled
        from probes.engine_sm121_inventory import run as sm121_inventory
        sm121_inventory(args.output)
        return
    if args.lanes == 'qwen38_site_components':
        # mix_block's two launches one at a time against the cuBLAS product + elementwise launch each replaces, a tile
        # sweep each -- which tiles the table takes, and from how many rows the fold wins
        from probes.engine_qwen38_gemv import run_site_components
        run_site_components(args.output)
        return
    if args.lanes == 'qwen38_site_norm_in':
        # mix_block's two launches over the normalised streams and over the streams normalised as read (site's way),
        # the down fold at several tiles, many interleaved rounds -- the tiles the served table takes for site
        from probes.engine_qwen38_gemv import run_site_norm_in
        run_site_norm_in(args.output)
        return
    if args.lanes == 'qwen38_site_whole':
        # a whole site at a prefill step's rows -- leave, norm, mixer -- as main served it, with the normalised streams
        # written for mix_block, and as gated_residual.site serves it (stream scales kept, the tiles normalised as read)
        from probes.engine_qwen38_gemv import run_site_whole
        run_site_whole(args.output)
        return
    if args.lanes == 'qwen38_site_prefill':
        # the mixer at a prefill step's rows: the five launches it served before against gated_residual.mix_block's two
        # (the up product never written), with a tile sweep -- what the fold is worth where the chunk spends 43%
        from probes.engine_qwen38_gemv import run_site_prefill
        run_site_prefill(args.output)
        return
    if args.lanes == 'qwen38_site':
        # component timings: a hyper-connection site's mixer as four launches on cuBLAS and as gated_residual.mix serves
        # a decode step's rows (two launches, carry H2), 16 sites a graph -- what the fold is worth on a GB10
        from probes.engine_qwen38_gemv import run_site as qwen38_site
        qwen38_site(args.output)
        return
    if args.lanes == 'qwen38_leave':
        # component timings: a decode site's leave behind a stand-in TP sum that waits like the fleet's -- launched
        # after it, as its programmatic dependent, and prefetching the mixer's down projection (H4); bytes checked first
        from probes.engine_qwen38_leave import run as qwen38_leave
        qwen38_leave(args.output)
        return
    if args.lanes == 'qwen38_moe_precision':
        from probes.engine_qwen38_moe_precision import run
        run(args.output)
        return
    if args.lanes == 'qwen38_moe':
        # the b12x EP cell held to its oracle within 2%, then micro tile x MAC and prefill tile_m timings (C4)
        from probes.engine_qwen38_moe import run as qwen38_moe
        qwen38_moe(args.output)
        return
    if args.lanes == 'qwen38_mix_tiles':
        # component timings: the mixer mean's hidden axis in tiles, every tile the one-block launch's bytes first (H3)
        from probes.engine_qwen38_mix_tiles import run as qwen38_mix_tiles
        qwen38_mix_tiles(args.output)
        return
    if args.lanes == 'qwen38_qsa_runs':
        # the sparse attention's run launch (a prefill segment's rows in runs over the union of their blocks) against
        # the split launch, over selections whose neighbours share more or less (probes/engine_qwen38_qsa_geometry)
        from probes.engine_qwen38_qsa_geometry import run_runs
        run_runs(args.output)
        return
    if args.lanes == 'qwen38_qsa_stacked':
        # the covered attention's stacked launch against the run launch, and a first chunk across the reach split
        # between the covered and the sparse launch (probes/engine_qwen38_qsa_geometry.run_stacked)
        from probes.engine_qwen38_qsa_geometry import run_stacked
        run_stacked(args.output)
        return
    if args.lanes == 'qwen38_qsa_geometry':
        # component timings: the QSA launches' geometry at Qwen3.8's cell -- the attention's split profile, the scorer's
        # tiles, the decode selection against its torch form, the input launches' warps -- each gated first (Q9)
        from probes.engine_qwen38_qsa_geometry import run as qwen38_qsa_geometry
        qwen38_qsa_geometry(args.output)
        return
    if args.lanes == 'qwen38_qualify_soak' or args.lanes.startswith('qwen38_qualify_soak:'):
        # correctness only: the boot's lane qualify again and again -- how often it fails on this card, and whose
        # failure it is (gated_residual.blame); `:N` sets the repeats
        from probes.engine_qwen38_qualify_soak import REPEATS, run as qwen38_qualify_soak
        qwen38_qualify_soak(args.output, repeats=int((args.lanes.split(':')[1:] or [REPEATS])[0]))
        return
    if args.lanes == 'qwen38_ple_conv':
        # the PLE conv, its silu and the gated add: the torch form against ngram_gate.conv_add, many rounds
        from probes.engine_qwen38_ple import run as qwen38_ple_conv
        qwen38_ple_conv(args.output)
        return
    if args.lanes == 'qwen38_prefill' or args.lanes.startswith('qwen38_prefill:'):
        # component census: a prefill chunk's wall, device and host time by kernel family, solved from four small
        # nets of the rank file's own weights (--ranks); `:N` sets the chunk's tokens
        from probes.engine_qwen38_prefill import CHUNK, run as qwen38_prefill
        qwen38_prefill(args.output, args.ranks, chunk=int((args.lanes.split(':')[1:] or [CHUNK])[0]))
        return
    if args.lanes == 'qwen38_serve_compiles' or args.lanes.startswith('qwen38_serve_compiles:'):
        # compile census: the served model built in the fleet boot's order on one rank (--ranks), the serving window's
        # requests through the runner, and every kernel a step added after the door
        # (`:K` sets the drafts a step: the census boots at the operator's K=3 unless told)
        from probes.engine_qwen38_serve_compiles import SPEC_K, run as qwen38_serve_compiles
        qwen38_serve_compiles(args.output, args.ranks, spec_k=int((args.lanes.split(':')[1:] or [SPEC_K])[0]))
        return
    if args.lanes == 'qwen38_eager_moe':
        # compile counts: the eager MoE's decode-sized launches after the boot's warm pass (warmup.eager_moe) --
        # no request may add a micro kernel -- and whether the workspace's capacity changes a launch's bytes (--ranks)
        from probes.engine_qwen38_eager_moe import run as qwen38_eager_moe
        qwen38_eager_moe(args.output, args.ranks)
        return
    if args.lanes == 'select_rows':
        # a captured step's joined C=2 indexer selection against its per-row control, then bounded timings
        from probes.engine_decode_select_rows import run as select_rows_check
        select_rows_check(args.output)
        return
    if args.lanes in ('scatter_bundle', 'batch_fusions', 'batch_boundaries', 'batch_integration', 'k7_commit_bundle', 'k7_output_bundle'):
        from probes.engine_decode_bundle import check as decode_bundle
        decode_bundle(args.ranks, bundle=args.lanes)
        return
    sys.meta_path.insert(0, ForbidVllm())
    assert not any(n == "vllm" or n.startswith("vllm.") for n in sys.modules)

    import torch
    import engine.kernels
    from engine.profiles.glm53 import lanes

    imported = [m.name for m in pkgutil.walk_packages(engine.kernels.__path__, "engine.kernels.")]
    for name in imported:
        importlib.import_module(name)
    native, ref = lanes.served(moe_static=args.moe_static, mla_prefill=args.mla_prefill), lanes.reference()
    rows = []

    def report(name, **values):
        row = dict(lane=name, **values)
        rows.append(row)
        print(json.dumps(row), flush=True)

    report("imports", modules=len(imported), table=native.name, vllm_loaded=False)
    if args.imports_only:
        return
    assert torch.cuda.get_device_capability() == (12, 1), "requires GB10"
    torch.manual_seed(29)
    selected = set(args.lanes.split(","))
    assert selected <= {"conv", "kda", "kda-storage", "mhc", "indexer", "kpool", "mla", "moe", "moe_route_scatter", "moe_direct_scatter", "moe_route_direct", "moe_fc1_reuse", "moe_compact_staging", "moe_register_scales", "moe_sync_cleanup", "paired_projection", "indexer_boundary", "wide_input", "direct_producer", "calibration", "pointwise", "residency", "latency", "shared_mlp", "kda_ring", "decode7", "decode_rows", "kda_ring_bench", "decode_k7", "moe_output", "moe_pair", "moe_pair_serial", "moe_pair_overlap", "moe_pair_direct", "moe_pair_reuse", "moe_pair_prefetch", "moe_pair_sync", "moe_pair_vec4", "moe_pair_packed_load"}, selected

    if 'moe_pair_packed_load' in selected:
        from probes.engine_moe_pair_check import check as moe_pair_check
        target = args.output.with_stem(args.output.stem+'-packed-load') if args.output else None
        moe_pair_check(report, args.ranks, scatter_packed_load_only=True, output=target)

    if 'moe_pair_vec4' in selected:
        from probes.engine_moe_pair_check import check as moe_pair_check
        target = args.output.with_stem(args.output.stem+'-vec4') if args.output else None
        moe_pair_check(report, args.ranks, scatter_vec4_only=True, output=target)

    if 'moe_pair_sync' in selected:
        from probes.engine_moe_pair_check import check as moe_pair_check
        path = args.output or Path('/cache/c2-moe.json')
        moe_pair_check(report, args.ranks, sync_cleanup_only=True,
                       output=path.with_stem(path.stem+'-sync'))

    if 'moe_pair_prefetch' in selected:
        from probes.engine_moe_pair_check import check as moe_pair_check
        path = args.output or Path('/cache/c2-moe.json')
        moe_pair_check(report, args.ranks, fc2_prefetch_only=True,
                       output=path.with_stem(path.stem+'-prefetch'))

    if 'moe_pair_reuse' in selected:
        from probes.engine_moe_pair_check import check as moe_pair_check
        path = args.output or Path('/cache/c2-moe.json')
        moe_pair_check(report, args.ranks, scatter_reuse_only=True,
                       output=path.with_stem(path.stem+'-reuse'))

    if 'moe_pair_direct' in selected:
        from probes.engine_moe_pair_check import check as moe_pair_check
        path = args.output or Path('/cache/c2-moe.json')
        moe_pair_check(report, args.ranks, direct_scatter_only=True,
                       output=path.with_stem(path.stem+'-direct'))

    if 'moe_pair' in selected:
        from probes.engine_moe_pair_check import check as moe_pair_check
        moe_pair_check(report, args.ranks, output=args.output)

    for mode in ('serial', 'overlap'):
        if 'moe_pair_'+mode in selected:
            from probes.engine_moe_pair_check import check as moe_pair_check
            path = args.output or Path('/cache/c2-moe.json')
            moe_pair_check(report, args.ranks, shared_mode=mode,
                           output=path.with_stem(path.stem+'-'+mode))

    if 'moe_output' in selected:
        from probes.engine_moe_output_check import check as moe_output_check
        moe_output_check(report, args.ranks)

    if 'decode_k7' in selected:
        from probes.engine_decode_k7 import check as k7_check
        k7_check(report, args.ranks)

    if selected & {'moe_route_scatter', 'moe_direct_scatter', 'moe_route_direct', 'moe_fc1_reuse', 'moe_compact_staging', 'moe_register_scales', 'moe_sync_cleanup', 'paired_projection', 'indexer_boundary', 'wide_input', 'direct_producer'}:
        from probes.engine_decode_bundle import require_current_probe
        require_current_probe()
    if 'direct_producer' in selected:
        import unittest
        suite = unittest.defaultTestLoader.loadTestsFromNames(('tests.test_engine_direct_mhc_cuda',
                                                               'tests.test_engine_direct_producer_cuda'))
        result = unittest.TextTestRunner(verbosity=2).run(suite)
        if not result.wasSuccessful() or result.skipped:
            raise RuntimeError('merged direct producer/MHC gates failed or skipped')
        report('direct_producer', passed=True, tests=result.testsRun)
    if 'indexer_boundary' in selected:
        from probes.engine_decode_batch import indexer_check
        indexer_check(report, args.ranks)
    if 'wide_input' in selected:
        from probes.engine_decode_batch import wide_check
        wide_check(report, args.ranks)
    if selected & {'moe_route_scatter', 'moe_direct_scatter', 'moe_route_direct', 'moe_fc1_reuse', 'moe_compact_staging', 'moe_register_scales', 'moe_sync_cleanup'}:
        from probes.engine_decode_scatter_check import moe_check
        for lane in ('moe_route_scatter', 'moe_direct_scatter', 'moe_route_direct', 'moe_fc1_reuse', 'moe_compact_staging', 'moe_register_scales', 'moe_sync_cleanup'):
            if lane in selected:
                moe_check(report, args.ranks, lane)

    if 'paired_projection' in selected:
        from probes.engine_decode_projection import paired_check
        if 'paired_projection' in selected:
            paired_check(report, args.ranks)

    if "kda_ring_bench" in selected:
        # timing only: the KDA ring launch at 1..4 rows, FP32 vs FP16 storage, cold and warm -- bytes or programs?
        from probes.engine_kda_ring_bench import bench
        bench(report)

    if "decode_rows" in selected:
        import unittest
        # every kernel a captured decode step folds over its rows (45차, the C=4 question) against its one-row
        # launches, byte for byte: the KDA rings, the conv ring, the pool/tail writers and the slot finalizer
        suite = unittest.defaultTestLoader.loadTestsFromNames(["tests.test_engine_kda_ring", "tests.test_engine_conv_ring",
                                                               "tests.test_engine_state", "tests.test_engine_pool_slots",
                                                               "tests.test_engine_indexer_rows"])
        result = unittest.TextTestRunner(verbosity=2).run(suite)
        assert result.wasSuccessful() and not result.skipped, "decode row-fold checks did not pass"
        report("decode_rows", passed=True, tests=result.testsRun)

    if "decode7" in selected:
        import unittest
        suite = unittest.defaultTestLoader.loadTestsFromName("tests.test_engine_decode_seven")
        result = unittest.TextTestRunner(verbosity=2).run(suite)
        assert result.wasSuccessful() and not result.skipped, "seven-row dense/router numerical gates did not pass"
        report("decode7", passed=True, tests=result.testsRun)
        from probes.engine_decode_fusions import seven_row_dense, tensorcore_router
        seven_row_dense(report)
        tensorcore_router(report, ranks=args.ranks)

    if "kda_ring" in selected:
        import unittest
        # the conv ring rides with the recurrent ring: a decode step folds both over its rows (net._kda)
        suite = unittest.defaultTestLoader.loadTestsFromNames(["tests.test_engine_kda_ring", "tests.test_engine_conv_ring"])
        result = unittest.TextTestRunner(verbosity=2).run(suite)
        assert result.wasSuccessful() and not result.skipped, "KDA ring numerical/replay checks did not pass"
        report("kda_ring", passed=True, tests=result.testsRun)

    if "shared_mlp" in selected:
        import unittest
        suite = unittest.defaultTestLoader.loadTestsFromName("tests.test_engine_shared_mlp")
        result = unittest.TextTestRunner(verbosity=2).run(suite)
        assert result.wasSuccessful() and not result.skipped, "shared MLP numerical/replay checks did not pass"
        report("shared_mlp", passed=True, tests=result.testsRun)
        from probes.engine_decode_fusions import shared_mlp
        shared_mlp(report, native)


    if "kda-storage" in selected:
        import unittest
        suite = unittest.defaultTestLoader.loadTestsFromNames(
            ["tests.test_engine_kda_ring", "tests.test_engine_boundary_stage"])
        result = unittest.TextTestRunner(verbosity=2).run(suite)
        assert result.wasSuccessful() and not result.skipped, "KDA FP32/FP16 storage checks did not pass"
        report("kda-storage", passed=True, tests=result.testsRun, arithmetic="fp32", storage=["fp32", "fp16"])

    if "residency" in selected:
        import unittest
        suite = unittest.defaultTestLoader.loadTestsFromName("tests.test_engine_decode_residency")
        result = unittest.TextTestRunner(verbosity=2).run(suite)
        assert result.wasSuccessful() and not result.skipped, "decode residency checks did not pass"
        report("residency", passed=True, tests=result.testsRun)
        from probes.engine_decode_fusions import residency
        residency(report)

    if "calibration" in selected:
        import unittest
        suite = unittest.defaultTestLoader.loadTestsFromName("tests.test_engine_calibration_gram")
        result = unittest.TextTestRunner(verbosity=2).run(suite)
        assert result.wasSuccessful() and not result.skipped, "calibration GPU checks did not pass"
        report("calibration", passed=True, tests=result.testsRun)

    if "pointwise" in selected:
        import unittest
        suite = unittest.defaultTestLoader.loadTestsFromName("tests.test_engine_glm_pointwise")
        result = unittest.TextTestRunner(verbosity=2).run(suite)
        assert result.wasSuccessful() and not result.skipped, "pointwise GPU checks did not pass"
        report("pointwise", passed=True, tests=result.testsRun)

    # Numerical/replay contracts always precede timings. These explicitly
    # scoped component results do not replace the four-node onepass gate.
    if selected & {"calibration", "pointwise"}:
        from probes.engine_decode_fusions import calibration, pointwise
        if "pointwise" in selected:
            pointwise(report)
        if "calibration" in selected:
            calibration(report)

    if 'latency' in selected:
        from probes.engine_latency_check import check as latency_check
        report('latency', **latency_check())

    def rand(*shape, scale=1.):
        return torch.randn(*shape, device="cuda", dtype=torch.bfloat16) * scale

    def check(name, got, expected, tolerance=.02):
        error = ((got.float() - expected.float()).abs().max() /
                 expected.float().abs().max().clamp_min(1e-6)).item()
        assert torch.isfinite(got).all().item() and error <= tolerance, (name, error, tolerance)
        return error

    if "conv" in selected:
        for t in (1, 6, 96):
            x, w = rand(t, 512), rand(512, 4, scale=.3).float()
            for seeded in (False, True):
                initial = rand(512, 3) if seeded else None
                saved = initial.clone() if seeded else None
                expected, state = ref.conv_prefill(x, w, initial)
                out, actual = native.conv_prefill(x, w, initial)
                assert not seeded or torch.equal(initial, saved)
                report("conv", tokens=t, seeded=seeded,
                       output=check("conv", out, expected), state=check("conv state", actual, state, 0.))

    if "kda" in selected:
        from probes.engine_kda_check import main as check_kda
        check_kda()
        report("kda", passed=True, cases=6, every_verify_state=True)

    if "mhc" in selected:
        fn, scale, base, norm = rand(24, 16384, scale=.01).float(), rand(3).float(), rand(24).float(), rand(4096)
        for t in (1, 6, 8, 65):
            residual = rand(t, 4, 4096)
            inputs = (residual, fn, scale, base, 1e-6, 1e-6, 2., 20, norm, 1e-6)
            expected = ref.mhc_pre(*inputs)
            actual = native.mhc_pre(*inputs)
            pre = [check("mhc pre", a, b) for a, b in zip(actual, expected)]
            x = rand(t, 4096)
            post = check("mhc post", native.mhc_post(x, residual, *actual[:2]),
                         ref.mhc_post(x, residual, *actual[:2]))
            report("mhc", tokens=t, pre=pre, post=post)

    if "indexer" in selected:
        q = rand(8, 16, 128).to(torch.float8_e4m3fn)
        k = rand(256, 128).to(torch.float8_e4m3fn)
        scale, weight = rand(256).float().abs(), rand(8, 16).float().abs()
        ends = torch.tensor([0, 1, 2, 8, 16, 64, 128, 256], device="cuda", dtype=torch.int32)
        actual = native.indexer_logits(q, k, scale, weight, ends)
        expected = ref.indexer_logits(q, k, scale, weight, ends)
        mask = torch.arange(256, device="cuda")[None, :] < ends[:, None]
        report("indexer", valid_logits=check("indexer logits", actual[mask], expected[mask]))

    if "kpool" in selected:
        for pools in (1, 3, 17):
            k, score, ape = rand(pools, 4, 128), rand(pools, 4, 128), rand(4, 128).float()
            actual = native.kpool_compress(k, score, ape)
            expected = ref.kpool_compress(k, score, ape)
            assert torch.equal(actual[0].view(torch.uint8), expected[0].view(torch.uint8)), "kpool fp8 bytes"
            assert torch.equal(actual[1], expected[1]), "kpool scales"
            report("kpool", pools=pools, bytes_equal=True, scales_equal=True)

    if "mla" in selected:
        from engine.kernels import mla
        mla.maybe_arm()  # Includes six ragged/decode/prefill numerical fixtures.
        q = rand(1, 16, 512, scale=.3)
        cache = rand(64, 512, scale=.5).to(torch.float8_e4m3fn)
        slots = torch.arange(64, device="cuda", dtype=torch.int32)[None, :]
        valid = torch.tensor([64], device="cuda", dtype=torch.int32)
        expected = ref.mla_sparse(q, cache, slots, valid, 512**-.5, .7)
        native.mla_sparse(q, cache, slots, valid, 512**-.5, .7)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            out = native.mla_sparse(q, cache, slots, valid, 512**-.5, .7)
        for _ in range(3):
            graph.replay()
        report("mla", boot_selftest=True, graph_replays=3, output=check("mla graph", out, expected))

    if "moe" in selected:
        torch.manual_seed(29)  # Independent of which other lanes were selected.
        # Only the migration judge calls the original library implementation;
        # the served table always calls engine.kernels.b12x.
        from flashinfer.fused_moe import b12x_fused_moe as original_b12x
        from engine.modules.nvfp4_sf import mma_sf_view, swizzle_sf
        e, h, intermediate = args.moe_experts, 4096, 512
        topk = 8 if e == 288 else 2
        w13 = torch.randint(0, 256, (e, 2*intermediate, h//2), device="cuda", dtype=torch.uint8)
        w2 = torch.randint(0, 256, (e, h, intermediate//2), device="cuda", dtype=torch.uint8)
        s13 = torch.stack([swizzle_sf(torch.full((2*intermediate, h//16), .015625, device="cuda").to(torch.float8_e4m3fn)) for _ in range(e)])
        s2 = torch.stack([swizzle_sf(torch.full((h, intermediate//16), .015625, device="cuda").to(torch.float8_e4m3fn)) for _ in range(e)])
        sf13, sf2 = mma_sf_view(s13, 2*intermediate, h), mma_sf_view(s2, h, intermediate)
        ones = torch.ones(e, device="cuda")
        for t in (1, 8, 129):
            x = rand(t, h, scale=.3)
            sel = (torch.arange(t*topk, device="cuda").reshape(t, topk) % e).to(torch.int32)
            weights = torch.full((t, topk), 1/topk, device="cuda")
            inputs = (x, sel, weights, w13, s13, w2, s2, 10.)
            out = native.moe(*inputs)
            expected = original_b12x(x=x, w1_weight=w13, w1_weight_sf=sf13, w2_weight=w2, w2_weight_sf=sf2,
                                    token_selected_experts=sel, token_final_scales=weights,
                                    num_experts=e, top_k=topk, w1_alpha=ones, w2_alpha=ones, fc2_input_scale=ones,
                                    activation="swigluoai_uninterleave", swiglu_alpha=1., swiglu_beta=0.,
                                    swiglu_limit=10., activation_precision="fp4", quant_mode="nvfp4")
            oracle = ref.moe(*inputs)
            oracle_error = ((out.float()-oracle.float()).abs().max()/oracle.float().abs().max().clamp_min(1e-6)).item()
            report("moe", tokens=t, experts=e, topk=topk, original_kernel=check("b12x migration", out, expected),
                   torch_oracle_relative=oracle_error)
            zero = native.moe(x, sel, weights * 0, w13, s13, w2, s2, 10.)
            assert torch.count_nonzero(zero).item() == 0, "zero-weight routes must not contribute"
            if t == 1:
                graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph):
                    captured = native.moe(*inputs)
                for _ in range(3):
                    graph.replay()
                report("moe_graph", replays=3, output=check("b12x graph", captured, expected))

    assert not any(n == "vllm" or n.startswith("vllm.") for n in sys.modules)
    print(json.dumps(dict(passed=True, vllm_loaded=False, checks=rows)), flush=True)


if __name__ == "__main__":
    main()
