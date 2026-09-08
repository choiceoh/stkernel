#!/usr/bin/env python3
"""Correctness only: sf6 CuTe layout/compile, exact expansion and MoE replay.

Explicit --cpu compiles without a CUDA context; --gpu requires the ordinary
fleet GPU hold. There are no timers, serving requests or reservation helpers.
Both modes compile the actual production method used by both FC1 and FC2.
"""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import sys


def check_launch_observations(events, arm, rows, *, raw_fallback=False):
    """Require the kernel actually selected during warmup AND capture."""
    if len(events) < 3:
        raise AssertionError((arm, rows, "missing two warmups and capture", events))
    expected = dict(kind="stock" if arm == "stock" else "static_v2", rows=rows,
                    reform=arm != "stock" and 1 <= rows <= 8,
                    sf6=arm == "sf6" and not raw_fallback)
    if any(event != expected for event in events):
        raise AssertionError((arm, rows, "wrong executed lane", expected, events))
    return expected


def correctness_limit(peak, noise):
    """Preserve the existing stock gate, refusing nonfinite/negative bounds."""
    if not math.isfinite(peak) or not math.isfinite(noise) or peak < 0 or noise < 0:
        raise AssertionError(("invalid stock bound", peak, noise))
    limit = max(4*noise, .01*peak)
    if not math.isfinite(limit):
        raise AssertionError(("overflowed stock bound", peak, noise))
    return limit


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    mode = ap.add_mutually_exclusive_group(required=True)
    mode.add_argument("--cpu", action="store_true")
    mode.add_argument("--gpu", action="store_true")
    ap.add_argument("--out", type=Path)
    args = ap.parse_args()
    os.environ.setdefault("CUTE_DSL_ARCH", "sm_121a")
    os.environ["VLLM_GLM53_B12X_FORCE_BACKEND"] = "static"
    os.environ["VLLM_GLM53_B12X_STATIC_V2"] = "0"
    sys.path.insert(0, os.environ.get("MK_PKG_PATH", "/usr/local/lib/python3.12/dist-packages"))
    import torch
    if args.cpu:
        torch.cuda.is_available = lambda: True
        torch.cuda.get_device_capability = lambda *a, **kw: (12, 1)
    import cutlass
    import cutlass.cute as cute
    import cuda.bindings.driver as cuda
    from flashinfer.fused_moe.cute_dsl.blackwell_sm12x import moe_dispatch as md
    from flashinfer.fused_moe.cute_dsl.blackwell_sm12x import moe_sf_pack as old_pack
    from flashinfer.fused_moe.cute_dsl.blackwell_sm12x import moe_reform_sf_pack as sf6
    from flashinfer.fused_moe.cute_dsl.blackwell_sm12x.moe_static_kernel_v4 import MoEStaticKernelV4
    from flashinfer.cute_dsl.fp4_common import shared_ptr_to_u32
    assert md._GLM53_B12X_FORCE_BACKEND == "static", "forced static path was not latched"
    assert md._GLM53_B12X_STATIC_V2 is None, "stock base must explicitly be cell 0"
    assert md._FORCED_BACKEND is None, "unexpected private backend override"

    if args.cpu:
        md.get_num_sm = lambda dev=None: 48
        md.get_max_active_clusters = lambda n=1: 48
    owner = MoEStaticKernelV4(sf_vec_size=16, output_tile_count_n=4,
                              decode_reform=True, reform_sf_pack=True)

    def compile_expand(block):
        stage = block * 3 // 4 + 16

        @cute.kernel
        def expand(src: cute.Tensor, dst: cute.Tensor):
            tid, _, _ = cute.arch.thread_idx()
            bid, _, _ = cute.arch.block_idx()
            smem = cutlass.utils.SmemAllocator()

            @cute.struct
            class Storage:
                data: cute.struct.Align[cute.struct.MemRange[cutlass.Uint8, block+32], 16]

            storage = smem.allocate(Storage)
            data = storage.data.get_tensor(cute.make_layout((block+32,)))
            i = cutlass.Int32(tid)
            while i < block+32:
                data[i] = cutlass.Uint8(0xA5)
                i += 128
            cute.arch.sync_threads()
            i = cutlass.Int32(tid)
            while i < stage:
                data[i+16] = src[bid, i]
                i += 128
            cute.arch.sync_threads()
            # This is the actual production helper, not a copied oracle.
            owner._sf_expand_stage(shared_ptr_to_u32(storage.data.data_ptr()) + 16,
                                   cutlass.Int32(tid), block)
            i = cutlass.Int32(tid)
            while i < block+32:
                dst[bid, i] = data[i]
                i += 128

        @cute.jit
        def entry(src: cute.Tensor, dst: cute.Tensor, stream: cuda.CUstream):
            expand(src, dst).launch(grid=(8, 1, 1), block=(128, 1, 1), stream=stream)

        src = cute.runtime.make_fake_compact_tensor(cutlass.Uint8, (8, stage),
                                                     stride_order=(1, 0), assumed_align=16)
        dst = cute.runtime.make_fake_compact_tensor(cutlass.Uint8, (8, block+32),
                                                     stride_order=(1, 0), assumed_align=16)
        return cute.compile(entry, src, dst,
                            cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=True),
                            options="--opt-level 2 --enable-tvm-ffi")

    expanders = {block: compile_expand(block) for block in (1024, 2048, 4096)}
    report = {"mode": "cpu" if args.cpu else "gpu", "expand_blocks": [1024, 2048, 4096],
              "source_sha256": {Path(p).name: hashlib.sha256(Path(p).read_bytes()).hexdigest()
                                for p in md._kernel_source_files()}, "gates": []}
    if args.cpu:
        for m in (1, 2, 6, 8, 16):
            cfg = md._parse_glm53_static_v2("t,r,sf6")
            md._get_static_kernel_v2(288, 288, m, 4096, 512, 8,
                                     md._align_up(m*8, 128), config=cfg, mac_override=48,
                                     activation="swigluoai_uninterleave", swiglu_alpha=1.,
                                     swiglu_beta=0., swiglu_limit=10.)
            effective = md._static_v2_decode_config(cfg, m)
            assert effective["reform_sf_pack"], (m, "direct SF6 disabled")
            assert effective["decode_reform"] == (m <= 8), (m, "wrong tile geometry")
            report["gates"].append({"m": m, "compiled": True,
                                    "sf6": True, "reform": m <= 8, "direct_prefill": m > 8})
        assert not torch.cuda.is_initialized(), "CPU compile initialized CUDA"
    else:
        # Exact stage expansion, canaries and reused CUDA graphs. Alternate
        # all four code quadrants, including byte 255, on the SAME pointers.
        for block, kernel in expanders.items():
            raw = torch.empty((8, block), dtype=torch.uint8, device="cuda")
            packed = torch.empty((8, block*3//4+16), dtype=torch.uint8, device="cuda")
            output = torch.empty((8, block+32), dtype=torch.uint8, device="cuda")
            raw.zero_()
            packed.copy_(old_pack.pack_sf_inline(raw, block))
            kernel(packed, output)
            torch.cuda.synchronize()
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                kernel(packed, output)
            for replay in range(32):
                idx = torch.arange(raw.numel(), device="cuda").reshape_as(raw)
                raw.copy_(((idx * 13 + replay * 7) % 64 + (replay % 4)*64).to(torch.uint8))
                packed.copy_(old_pack.pack_sf_inline(raw, block))
                graph.replay()
                torch.cuda.synchronize()
                assert torch.equal(output[:, 16:-16], raw), (block, replay, "expanded bytes")
                assert bool((output[:, :16] == 0xA5).all()), (block, "prefix canary")
                assert bool((output[:, -16:] == 0xA5).all()), (block, "suffix canary")
            report["gates"].append({"block": block, "exact_expand_replays": 32})
        # Serving wrapper and both complete FC pipelines, including original
        # M>8 layout. Same source baseline, independent stock oracle, mutated
        # inputs/routes and repeated graph replay; no timing output.
        import moe_decode_stream_probe as fixture
        torch.manual_seed(906)
        w13, sf13, w2, sf2 = fixture.expert_set(torch.Generator().manual_seed(906))
        scales = torch.ones(fixture.E, device="cuda")
        wrapper = fixture.served_wrapper()
        owners, executables = [], []
        # Different tensor/storage identity protects captured normal-case
        # graphs. Both byte codes are finite, positive E4M3; their span is
        # 66, so sf6 must decline while the ordinary kernel remains valid.
        fallback_sf2 = torch.empty_like(sf2)
        fallback_bytes = fallback_sf2.view(torch.uint8).reshape(-1)
        fallback_bytes[0::2] = 0x01
        fallback_bytes[1::2] = 0x42
        assert bool(torch.isfinite(fallback_sf2).all()), "fallback fixture must remain finite"
        get_v2, get_stock = md._get_static_kernel_v2, md._get_static_kernel
        observed = []

        def observe_v2(*call_args, **kwargs):
            result = get_v2(*call_args, **kwargs)
            rows = int(call_args[2])
            config = md._static_v2_decode_config(kwargs["config"], rows)
            observed.append(dict(kind="static_v2", rows=rows,
                                 reform=bool(config["decode_reform"]),
                                 sf6=bool(config.get("reform_sf_pack"))))
            executables.append(result)  # Own every captured function handle.
            return result

        def observe_stock(*call_args, **kwargs):
            result = get_stock(*call_args, **kwargs)
            observed.append(dict(kind="stock", rows=int(call_args[2]), reform=False, sf6=False))
            executables.append(result)
            return result

        md._get_static_kernel_v2, md._get_static_kernel = observe_v2, observe_stock
        cases = [(m, u, False) for m, u in ((1,8), (2,16), (6,8), (6,40), (6,48),
                                             (8,8), (8,40), (8,64), (16,40))]
        cases.append((6, 40, True))
        for m, unique, raw_fallback in cases:
            fixture.T = m
            ids, weights = fixture._routing(unique)
            original_ids, original_weights = ids.clone(), weights.clone()
            x = torch.randn(m, 4096, device="cuda", dtype=torch.bfloat16)*.5
            outputs = {arm: torch.empty_like(x) for arm in ("stock", "baseline", "sf6")}
            graphs = {}
            lane_proof = {}
            current_sf2 = fallback_sf2 if raw_fallback else sf2
            for arm, spec in (("stock", "0"), ("baseline", "t,r"), ("sf6", "t,r,sf6")):
                md._STATIC_V2_OVERRIDE = md._parse_glm53_static_v2(spec)
                before = len(observed)

                def run(arm=arm):
                    wrapper.run(x,w13,sf13,w2,current_sf2,ids,weights,w1_alpha=scales,
                                w2_alpha=scales,fc2_input_scale=scales,out=outputs[arm])

                graphs[arm] = fixture._graph(run, torch.cuda.Stream())
                lane_proof[arm] = check_launch_observations(
                    observed[before:], arm, m, raw_fallback=raw_fallback)
                views = wrapper._weight_views
                owners.append(views)
                # The independent stock kernel must always read the untouched
                # row-major weights. The two tile consumers receive exact
                # copies; no arm may relayout the shared source in place.
                assert not getattr(w13, "_b12x_tile_major", False), "stock W13 source was retiled"
                assert not getattr(w2, "_b12x_tile_major", False), "stock W2 source was retiled"
                assert views.tiled == (arm != "stock"), (arm, "wrong weight layout")
                if arm == "stock":
                    assert views.w13_fp4.data_ptr() == w13.data_ptr(), "stock W13 pointer changed"
                    assert views.down_fp4.data_ptr() == w2.data_ptr(), "stock W2 pointer changed"
                else:
                    for source, tiled, width in ((w13, views.w13_tiled_storage, 256),
                                                 (w2, views.w2_tiled_storage, 64)):
                        e, rows, kb = source.shape
                        expected = source.view(e, rows, kb//width, width).permute(0,2,1,3)
                        assert torch.equal(tiled, expected), (arm, "tiled weight bytes differ")
                if arm == "sf6":
                    scale_owner = views.reform_scales
                    assert scale_owner is not None, "sf6 owner was never prepared"
                    assert scale_owner.enabled != raw_fallback, (m, raw_fallback, scale_owner.reason)
                    if raw_fallback:
                        assert "fc2" in scale_owner.reason, "wrong fallback reason"
                        assert views.sfb1_packed is None and views.sfb2_packed is None
            numeric = []
            for replay in range(8):
                x.copy_(torch.randn_like(x)*(.125+replay*.1))
                ids.copy_((original_ids + 17*(replay+1)) % fixture.E)
                weights.copy_(original_weights.roll(replay, dims=1))
                if replay == 6:
                    weights.zero_()
                graphs["stock"].replay(); torch.cuda.synchronize()
                ref = outputs["stock"].clone()
                assert bool(torch.isfinite(ref).all()), (m, unique, "stock finite")
                graphs["stock"].replay(); torch.cuda.synchronize()
                assert bool(torch.isfinite(outputs["stock"]).all()), (m, unique, "stock replay finite")
                noise = float((ref.float()-outputs["stock"].float()).abs().max())
                peak = float(ref.float().abs().max())
                limit = correctness_limit(peak, noise)
                if replay == 6:
                    assert torch.count_nonzero(ref) == 0, "stock zero-route reference"
                else:
                    assert peak > 0, "stock collapsed to zero with nonzero routes"
                for arm in ("baseline", "sf6"):
                    graphs[arm].replay(); torch.cuda.synchronize()
                    got = outputs[arm]
                    assert bool(torch.isfinite(got).all()), (m, unique, arm, "finite")
                    error = float((got.float()-ref.float()).abs().max())
                    assert math.isfinite(error) and error <= limit, (m, unique, arm, error, limit)
                    if replay == 6:
                        assert torch.count_nonzero(got) == 0, "zero route output"
                    else:
                        assert torch.count_nonzero(got) > 0, "zero-state was not repopulated"
                    numeric.append(dict(replay=replay, arm=arm, max_error=error,
                                        stock_noise=noise, limit=limit))
            report["gates"].append({"m": m, "unique": unique, "raw_fallback": raw_fallback,
                                    "moe_replays": 8, "lanes": lane_proof, "numeric": numeric})
        md._get_static_kernel_v2, md._get_static_kernel = get_v2, get_stock
        md._STATIC_V2_OVERRIDE = None
        report["max_allocated_bytes"] = torch.cuda.max_memory_allocated()
    report["status"] = "PASS"
    report["scope"] = "correctness/compile only; no serving or speed verdict"
    text = json.dumps(report, indent=2)
    if args.out:
        args.out.write_text(text+"\n")
    print(text, flush=True)
    print("REFORM_SF6_CORRECTNESS_PASS", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
