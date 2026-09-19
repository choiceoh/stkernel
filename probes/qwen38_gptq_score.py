"""Score actual RTN/GPTQ weight packs under separate real-input Hessians.

sqrt(tr((W-Q) H (W-Q)^T) / tr(W H W^T)) is projection output relative RMSE
under the empirical input distribution. It isolates the weight packing error;
activation quantization, native accumulation, and whole-model quality are not
measured here. The serving comparison remains bench/onepass.py.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import statistics


def output_error(weight, quantized, hessian, chunk=128):
    """Use FP64 throughout the energy calculation, including the row-wise sum."""
    import torch
    h = hessian.double()
    h = (h + h.T) * 0.5
    error, reference = h.new_zeros(()), h.new_zeros(())
    for first in range(0, weight.shape[0], chunk):
        w = weight[first:first + chunk].double()
        d = w - quantized[first:first + chunk].double()
        error += ((d @ h) * d).sum()
        reference += ((w @ h) * w).sum()
    if not torch.isfinite(error) or not torch.isfinite(reference) or reference <= 0:
        raise ValueError("nonfinite or empty reference output energy")
    if error < -reference * 1e-10:
        raise ValueError("negative error energy: held-out Gram statistics are not accurate enough")
    return dict(relative_rmse=float((error.clamp_min(0) / reference).sqrt()),
                error_energy=float(error), reference_energy=float(reference))


def score(args):
    import torch
    from safetensors import safe_open
    from engine.profiles.qwen38.net import HEAD_NAME
    from engine.kernels.dense.packing import fp8_rtn, mk_w4_dequant, FP8_BLOCK
    from probes.qwen38_gptq_feed import verify_owner
    if args.device == "cuda":
        verify_owner(args.lease, args.owner)
    torch.set_num_threads(2)
    report = json.loads(args.audit.read_bytes())
    if not report["serving_gptq_verified"]:
        raise ValueError("need the candidate boot's verified pack-consumption audit")
    records = {r["name"]: r for r in report["records"]}
    packs = {}
    for path in (args.fit / "st-dense-packs").glob("*.pt"):
        blob = torch.load(path, map_location="cpu", mmap=True, weights_only=True)
        ident = blob["identity"]
        if ident["name"] not in records:
            continue
        kind = "fp8" if ident.get("kind") == "fp8" else "w4"
        key = ident["name"], kind, ident["calibration"]
        if key in packs:
            raise ValueError("ambiguous pack identity in the isolated experiment store")
        packs[key] = path
    rows = []
    with safe_open(str(args.weights), framework="pt", device="cpu") as weights:
        for name, row in sorted(records.items()):
            if args.device == "cuda":
                verify_owner(args.lease, args.owner)
            raw = weights.get_tensor(row["key"])
            weight = torch.nn.functional.pad(raw, (0, row["width"] - raw.shape[1]))
            weight_sha = hashlib.sha256(weight.contiguous().view(torch.uint8).numpy()).hexdigest()
            held_path = args.heldout / "mkcalib" / f"rank{report['rank']}" / (name + ".pt")
            held = torch.load(held_path, map_location="cpu", mmap=True, weights_only=True)
            if held["weights_id"] != report["weights_id"] or int(held["ntok"]) < 4096:
                raise ValueError("held-out statistics have the wrong model identity or too few rows")
            h = held["H"].to(args.device)
            weight = weight.to(args.device)
            for kind in (("fp8",) if name == HEAD_NAME else ("w4", "fp8")):
                values = {}
                for arm in ("rtn", "gptq"):
                    calib = "rtn" if arm == "rtn" else row["hessian_sha256"]
                    if kind == "fp8" and arm == "rtn":
                        q, scale = fp8_rtn(weight)
                        dequant = q.float() * scale.repeat_interleave(FP8_BLOCK, 0).repeat_interleave(FP8_BLOCK, 1)
                    else:
                        blob = torch.load(packs[name, kind, calib], map_location=args.device,
                                          weights_only=True)
                        if blob["identity"]["weight"] != weight_sha:
                            raise ValueError("pack's source weight differs from the checkpoint projection")
                        if kind == "w4":
                            dequant = mk_w4_dequant(blob["data"], blob["scale"], weight.shape[0],
                                                  rgs=blob["rowscale"])
                        else:
                            dequant = blob["q"].float() * blob["scale"].repeat_interleave(FP8_BLOCK, 0).repeat_interleave(FP8_BLOCK, 1)
                    values[arm] = output_error(weight, dequant[:weight.shape[0], :weight.shape[1]], h)
                    del dequant
                rows.append(dict(name=name, key=row["key"], lane=kind, heldout_rows=int(held["ntok"]),
                                 **values))
            print(json.dumps(dict(done=len(rows), site=row["key"])), flush=True)
    summary = {}
    for kind in ("w4", "fp8"):
        cases = [r for r in rows if r["lane"] == kind]
        summary[kind] = dict(sites=len(cases),
                            median_rtn_rmse=statistics.median(r["rtn"]["relative_rmse"] for r in cases),
                            median_gptq_rmse=statistics.median(r["gptq"]["relative_rmse"] for r in cases),
                            improved=sum(r["gptq"]["relative_rmse"] < r["rtn"]["relative_rmse"] for r in cases),
                            worsened=sum(r["gptq"]["relative_rmse"] > r["rtn"]["relative_rmse"] for r in cases))
    result = dict(scope=__doc__, rank=report["rank"], device=args.device, torch=torch.__version__,
                  weights_id=report["weights_id"], summary=summary, cases=rows)
    args.out.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(summary, indent=2), flush=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--fit", type=Path, required=True)
    ap.add_argument("--heldout", type=Path, required=True)
    ap.add_argument("--weights", type=Path, required=True)
    ap.add_argument("--audit", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    ap.add_argument("--owner", default="")
    ap.add_argument("--lease", type=Path, default=Path("/home/choiceoh/glm53-logs/st-fleet.lock"))
    score(ap.parse_args())


if __name__ == "__main__":
    main()
