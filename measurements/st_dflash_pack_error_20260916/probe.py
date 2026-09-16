"""CPU-only DFlash dense-pack audit; no engine imports, boot, or cache writes.

Score real cached weights against BF16 with the current calibration Gram matrix.
This is an in-sample, layer-local weight-only diagnostic, NOT acceptance or tok/s.
Production packing/smoothing modules are loaded as standalone Python files.
"""
import argparse
import ast
import gc
import hashlib
import importlib.util
import json
import math
import platform
import time
from pathlib import Path

import torch
from safetensors import safe_open

PREFIX = "DFlash2Qwen3ForCausalLM/outputs-5-14-24-33-42/model."


def digest(t):
    return hashlib.sha256(t.detach().contiguous().view(torch.uint8).numpy()).hexdigest()


def module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    result = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(result)
    return result


def algorithm_hash(root):
    # inspect.getsource(pack_w4), including decorator, without importing CUDA code.
    source = (root / "dense_init.py").read_text()
    node = next(n for n in ast.parse(source).body
                if isinstance(n, ast.FunctionDef) and n.name == "pack_w4")
    start = min([node.lineno] + [n.lineno for n in node.decorator_list])
    function = "".join(source.splitlines(keepends=True)[start - 1:node.end_lineno])
    return hashlib.sha256((root / "packing.py").read_bytes() + function.encode()).hexdigest()


def load(path):
    return torch.load(path, map_location="cpu", mmap=True, weights_only=True)


def energies(m, h, tile=512):
    """Exact FP64 row energies, bounded scratch even for the 20480-wide FC.

    No Hessian symmetrization is needed: v H v^T cancels the antisymmetric part.
    All input columns and all off-diagonal terms are included.
    """
    m = m.double()
    energy = torch.zeros(len(m), dtype=torch.float64)
    for start in range(0, h.shape[0], tile):
        stop = min(start + tile, h.shape[0])
        energy += ((m @ h[start:stop].double().T) * m[:, start:stop]).sum(1)
    if not torch.isfinite(energy).all() or energy.min() < -1e-6:
        raise ValueError("non-finite or negative Gram energy")
    return energy.clamp_min(0)


def rows_dequant(blob, rows, packing):
    if "q" in blob:
        scales = blob["scale"][rows // 128].repeat_interleave(128, 1)
        return blob["q"].view(torch.uint8)[rows].view(torch.float8_e4m3fn).float() * scales
    # Read only selected output rows; preserve actual tile layout and row scales.
    data = blob["data"][rows // 128, :, rows % 128, :].reshape(len(rows), -1)
    scale = blob["scale"][rows // 128, :, rows % 128, :].reshape(len(rows), -1)
    return packing.mk_w4_dequant_rowmajor(data, scale, rgs=blob["rowscale"][rows])


def fp8_scales(w):
    """Same 128x128 power-of-two scaling as packing.fp8_rtn."""
    blocks = w.float().view(w.shape[0] // 128, 128, w.shape[1] // 128, 128)
    return torch.exp2(torch.ceil(torch.log2(blocks.abs().amax((1, 3)).clamp_min(1e-4) / 448)))


def fp8_rtn_rows(w, rows, scale=None):
    scale = fp8_scales(w) if scale is None else scale
    per = scale[rows // 128].repeat_interleave(128, 1)
    return (w[rows].float() / per).to(torch.float8_e4m3fn).float() * per


def numerical_checks(packing):
    torch.manual_seed(7)
    w = torch.randn(256, 256).bfloat16()
    rows = torch.tensor([0, 7, 127, 128, 255])
    q, s = packing.fp8_rtn(w)
    expected = q.float() * s.repeat_interleave(128, 0).repeat_interleave(128, 1)
    assert torch.equal(fp8_rtn_rows(w, rows), expected[rows])
    # Distinct tile/row scales expose indexing mistakes in the cached W4 reader.
    data = torch.randint(0, 256, (2, 2, 128, 64), dtype=torch.uint8)
    scale = torch.randint(-16, 0, (2, 2, 128, 8), dtype=torch.int8)
    rgs = torch.linspace(0.5, 2, 256)
    blob = dict(data=data, scale=scale, rowscale=rgs)
    expected = packing.mk_w4_dequant(data, scale, 256, rgs=rgs)
    assert torch.equal(rows_dequant(blob, rows, packing), expected[rows])
    x = torch.randn(73, 256, dtype=torch.float64)
    m = torch.randn(9, 256, dtype=torch.float64)
    h = x.T @ x
    assert torch.allclose(energies(m, h, 31), (x @ m.T).square().sum(0), rtol=1e-12, atol=1e-9)
    return {"fp8_reference": True, "w4_tile_reference": True, "gram_vs_explicit_rows": True}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", type=Path, default=Path("/cache"))
    ap.add_argument("--model", type=Path, default=Path("/model"))
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--rank", type=int, default=0)
    ap.add_argument("--world", type=int, default=4)
    ap.add_argument("--rows", type=int, default=128, help="0 means all output rows")
    ap.add_argument("--only", nargs="*", help="reader names to refine; omit for all 31")
    ap.add_argument("--threads", type=int, default=4)
    args = ap.parse_args()
    torch.set_num_threads(args.threads)
    torch.set_num_interop_threads(1)
    root = Path(__file__).resolve().parent
    packing = module("audit_packing", root / "packing.py")
    smoothing = module("audit_smoothing", root / "smoothing.py")
    algorithm = algorithm_hash(root)
    checks = numerical_checks(packing)
    assert not torch.cuda.is_initialized()
    config = json.loads((args.model / "config.json").read_text())
    assert config["dflash_config"]["target_layer_ids"] == [5, 14, 24, 33, 42]
    assert config["num_hidden_layers"] == 5 and config["hidden_size"] == 4096
    entries = []
    for path in sorted((args.cache / "st-dense-packs").glob("*.pt")):
        identity = load(path).get("identity", {})
        if identity.get("name", "").startswith(PREFIX):
            entries.append({"path": str(path), "identity": identity})
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "pack-index.json").write_text(json.dumps(entries, indent=2) + "\n")
    header = dict(torch=torch.__version__, machine=platform.machine(), rank=args.rank, world=args.world,
                  rows=args.rows, algorithm=algorithm, checks=checks, prefix=PREFIX,
                  cuda_initialized=torch.cuda.is_initialized(), cache_entries=len(entries),
                  source_hashes={p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                                 for p in (root / "probe.py", root / "packing.py", root / "smoothing.py", root / "dense_init.py")},
                  model_config_sha256=hashlib.sha256((args.model / "config.json").read_bytes()).hexdigest(),
                  measurement="in_sample_weight_only_gram_relative_rms")
    (args.out / "run.json").write_text(json.dumps(header, indent=2) + "\n")
    print(json.dumps(header), flush=True)
    results = []
    with safe_open(args.model / "model.safetensors", framework="pt", device="cpu") as model:
        def weight(key):
            return model.get_tensor(key)

        def calibration(key):
            path = args.cache / "mkcalib" / f"rank{args.rank}" / (PREFIX + key + ".pt")
            blob = load(path)
            assert blob["name"] == PREFIX + key and blob["ntok"] > 0
            for part in blob["H"].split(128):
                assert torch.isfinite(part).all()
            return path, blob

        factors = {}
        for layer in range(5):
            pre = f"layers.{layer}."
            for norm, key, readers in (
                ("input_layernorm.weight", "self_attn.qkv_proj", ["attention_conv.kernel_projection.weight"] + [f"self_attn.{s}_proj.weight" for s in ("q", "k", "v")]),
                ("post_attention_layernorm.weight", "mlp.gate_up_proj", ["mlp_conv.kernel_projection.weight"] + [f"mlp.{s}_proj.weight" for s in ("gate", "up")]),
            ):
                _, blob = calibration(pre + key)
                s = smoothing.scales(blob["amax"], [weight(pre + k) for k in readers])
                factors[pre + norm] = smoothing.fold(weight(pre + norm).clone(), s)
                del blob

        def prepared(key):
            if key == "fc.committed-decode-v1":
                return weight("fc.weight"), None
            pre = ".".join(key.split(".")[:2]) + "."
            suffix = key[len(pre):]
            factor = None
            if suffix in ("self_attn.qkv_proj", "attention_conv.kernel_projection"):
                factor = factors[pre + "input_layernorm.weight"]
            if suffix in ("mlp.gate_up_proj", "mlp_conv.kernel_projection"):
                factor = factors[pre + "post_attention_layernorm.weight"]
            def transformed(k):
                w = weight(pre + k)
                return smoothing.smooth_weight(w, factor) if factor is not None else w
            if suffix in ("self_attn.qkv_proj", "mlp.gate_up_proj"):
                names = [f"self_attn.{s}_proj.weight" for s in ("q", "k", "v")] if suffix.startswith("self_attn") else [f"mlp.{s}_proj.weight" for s in ("gate", "up")]
                w = torch.cat([transformed(k).chunk(args.world, 0)[args.rank] for k in names])
            elif suffix in ("self_attn.o_proj", "mlp.down_proj"):
                w = transformed(suffix + ".weight").chunk(args.world, 1)[args.rank].contiguous()
            else:
                w = transformed(suffix + ".weight")
            return w, factor

        names = [f"layers.{l}.{s}" for l in range(5) for s in (
            "self_attn.qkv_proj", "self_attn.o_proj", "mlp.gate_up_proj", "mlp.down_proj",
            "attention_conv.kernel_projection", "mlp_conv.kernel_projection")]
        names.append("fc.committed-decode-v1")
        if args.only:
            assert all(n in names for n in args.only)
            names = args.only
        for key in names:
            started = time.monotonic()
            w, smooth = prepared(key)
            path, calib = calibration(key)
            h = calib["H"]
            if smooth is not None:
                h = smoothing.smooth_hessian(h.float(), smooth)
            assert h.shape == (w.shape[1], w.shape[1])
            expected = dict(name=PREFIX + key, weight=digest(w), shape=list(w.shape),
                            calibration=digest(h), smooth="none" if smooth is None else digest(smooth.float()),
                            algorithm=algorithm)
            is_fc = key.startswith("fc.")
            expected.update(dict(kind="fp8") if is_fc else dict(per_row=False))
            candidates = [e for e in entries if e["identity"].get("name") == PREFIX + key]
            matches = [e for e in candidates if all(e["identity"].get(k) == v or
                        (k == "shape" and list(e["identity"].get(k, [])) == v)
                        for k, v in expected.items()) and e["identity"].get("gptq_damping", .01) == .01]
            record = dict(reader=key, expected=expected, calibration_path=str(path), ntok=calib["ntok"],
                          calibration_weights_id=calib.get("weights_id"), candidate_count=len(candidates))
            if len(matches) != 1:
                record.update(status="unmatched", match_count=len(matches),
                              differences=[dict(path=e["path"], fields=[k for k,v in expected.items()
                                           if (list(e["identity"].get(k, [])) if k == "shape" else e["identity"].get(k)) != v]) for e in candidates])
            else:
                selected = matches[0]
                blob = load(selected["path"])
                assert blob["identity"]["version"] == 2
                n = w.shape[0]
                rng = torch.Generator().manual_seed(20260916)
                rows = torch.arange(n) if args.rows == 0 else torch.randperm(n, generator=rng)[:min(n, args.rows)].sort().values
                sums = dict(signal=0., cached_error=0., fp8_rtn_error=0., weight=0., cached_weight_error=0.)
                row_stats = []
                alternative_scale = fp8_scales(w)
                for batch in rows.split(128):
                    ref = w[batch].float()
                    q = rows_dequant(blob, batch, packing)
                    alt = fp8_rtn_rows(w, batch, alternative_scale)
                    den = energies(ref, h)
                    error = energies(ref - q, h)
                    alt_error = energies(ref - alt, h)
                    sums["signal"] += den.sum().item()
                    sums["cached_error"] += error.sum().item()
                    sums["fp8_rtn_error"] += alt_error.sum().item()
                    sums["weight"] += ref.double().square().sum().item()
                    sums["cached_weight_error"] += (ref.double()-q.double()).square().sum().item()
                    row_stats.extend([dict(row=int(r), signal=float(d), cached_error=float(e), fp8_rtn_error=float(a))
                                      for r,d,e,a in zip(batch,den,error,alt_error)])
                record.update(status="matched", cache_path=selected["path"], identity=selected["identity"],
                              evaluated_rows=len(rows), total_rows=n, sums=sums,
                              cached_relative_rms=math.sqrt(sums["cached_error"] / sums["signal"]),
                              fp8_rtn_relative_rms=math.sqrt(sums["fp8_rtn_error"] / sums["signal"]),
                              cached_weight_relative_rms=math.sqrt(sums["cached_weight_error"] / sums["weight"]),
                              fp8_rtn_error_energy_reduction=1-sums["fp8_rtn_error"] / sums["cached_error"])
                groups = [("all", 0, n)]
                if key.endswith("qkv_proj"):
                    groups += [("q", 0, n*2//3), ("k", n*2//3, n*5//6), ("v", n*5//6, n)]
                if key.endswith("gate_up_proj"):
                    groups += [("gate", 0, n//2), ("up", n//2, n)]
                record["groups"] = {}
                for label, lo, hi in groups:
                    selected_rows = [r for r in row_stats if lo <= r["row"] < hi]
                    total = {field: sum(r[field] for r in selected_rows)
                             for field in ("signal", "cached_error", "fp8_rtn_error")}
                    record["groups"][label] = dict(evaluated_rows=len(selected_rows), **total)
                if args.rows:
                    record["sample_row_ids"] = rows.tolist()
                del blob, q, alt, ref
            record["elapsed_seconds"] = time.monotonic() - started
            results.append(record)
            (args.out / "results.json").write_text(json.dumps(results, indent=2) + "\n")
            print(json.dumps({k:v for k,v in record.items() if k not in ("row_stats", "differences", "expected", "identity")}), flush=True)
            del w, h, calib
            gc.collect()
    assert not torch.cuda.is_initialized()
    print(json.dumps(dict(done=True, matched=sum(r["status"] == "matched" for r in results), total=len(results),
                          cuda_initialized=torch.cuda.is_initialized())), flush=True)


if __name__ == "__main__":
    with torch.inference_mode():
        main()
