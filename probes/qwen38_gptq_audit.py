"""CPU audit of every Qwen target Hessian and the packs a later boot consumed.

Run once per rank after the collector has finished writing. The serving boot's
record supplies the expected weight identity, so files alone cannot prove that
a reboot used GPTQ. Output contains model site names and digests, never input text.
"""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path


def counters(span):
    result = dict(span.get("counters", {}))
    for child in span.get("children", []):
        result.update(counters(child))
    return result


def validate_blob(blob, name, width, weights_id, min_rows):
    import torch
    h, a = blob["H"], blob["amax"]
    if blob.get("weights_id") != weights_id or blob.get("name") != name:
        raise ValueError("calibration provenance differs from the serving boot")
    if tuple(h.shape) != (width, width) or tuple(a.shape) != (width,):
        raise ValueError("calibration shape differs from the served padded projection")
    if h.dtype != torch.float32 or a.dtype != torch.float32:
        raise ValueError("calibration statistics must retain FP32")
    if not torch.isfinite(h).all() or not torch.isfinite(a).all():
        raise ValueError("nonfinite calibration statistics")
    if int(blob["ntok"]) < min_rows or (h.diag() < 0).any() or (a < 0).any() or not (h.diag() > 0).any():
        raise ValueError("insufficient or invalid real-input coverage")
    return dict(name=name, width=width, ntok=int(blob["ntok"]),
                dead_columns=int((h.diag() == 0).sum()),
                hessian_sha256=hashlib.sha256(h.contiguous().numpy()).hexdigest(),
                amax_sha256=hashlib.sha256(a.contiguous().numpy()).hexdigest())


def audit(args):
    import torch
    from engine.profiles.qwen38 import facts, specs
    from engine.profiles.qwen38.net import Qwen38Net, HEAD_NAME
    from engine.kernels.dense import padded_columns
    torch.set_num_threads(2)
    boot = json.loads(args.boot.read_bytes())
    seen = counters(boot["root"])
    if args.expect_row_target is not None and seen.get("calibration_row_target") != args.expect_row_target:
        raise ValueError("collector row target differs from the declared experiment")
    weights_id = seen["calibration_weights_id"]
    if seen["dense_pack_root"] != str(args.root):
        raise ValueError("audit root differs from the boot's declared pack store")
    if args.expect_rtn:
        if (seen.get("packs_rtn") != 192 or seen.get("packs_gptq", 0) != 0
                or seen.get("packs_fp8_gptq", 0) != 0 or seen.get("calibration_GiB") != 0):
            raise ValueError("baseline must serve RTN with collection disabled")
        report = dict(rank=args.rank, weights_id=weights_id, serving_rtn_verified=True,
                      collection_disabled=True, w4_sites=192,
                      boot_sha256=hashlib.sha256(args.boot.read_bytes()).hexdigest())
        args.out.write_text(json.dumps(report, indent=2) + "\n")
        print(json.dumps(report, indent=2))
        return
    shape = {s.name: s.shape for s in specs.all_specs(facts.load(args.ckpt), mtp=False)}
    names = Qwen38Net.dense_names(shape)
    names["head"] = HEAD_NAME
    records = []
    for key, name in sorted(names.items()):
        path = args.root / "mkcalib" / f"rank{args.rank}" / (name + ".pt")
        blob = torch.load(path, map_location="cpu", mmap=True, weights_only=True)
        width = shape[key][1] if key == "head" else padded_columns(shape[key][1])
        records.append(dict(validate_blob(blob, name, width, weights_id, args.min_rows), key=key))
    if len(records) != 193:
        raise ValueError("the frozen Qwen target must have 193 calibration sites")
    report = dict(rank=args.rank, sites=len(records), weights_id=weights_id,
                  minimum_rows=min(r["ntok"] for r in records),
                  maximum_rows=max(r["ntok"] for r in records), statistics_valid=True,
                  serving_gptq_verified=False, boot_sha256=hashlib.sha256(args.boot.read_bytes()).hexdigest(),
                  records=records)
    if args.expect_gptq:
        if seen.get("packs_gptq") != 192 or seen.get("packs_fp8_gptq") != 193:
            raise ValueError("the serving boot did not consume every target W4/FP8 GPTQ pack")
        if seen.get("calibration_GiB") != 0 or seen.get("calibration_deferred") != 0:
            raise ValueError("the measured boot still collects or defers target calibration")
        wanted = {r["name"]: r["hessian_sha256"] for r in records}
        found = Counter()
        for path in sorted((args.root / "st-dense-packs").glob("*.pt")):
            blob = torch.load(path, map_location="cpu", mmap=True, weights_only=True)
            identity = blob["identity"]
            name = identity["name"]
            if name not in wanted or identity["calibration"] != wanted[name]:
                continue
            kind = "fp8" if identity.get("kind") == "fp8" else "w4"
            found[name, kind] += 1
        for name in wanted:
            for kind in (("fp8",) if name == HEAD_NAME else ("w4", "fp8")):
                if found[name, kind] != 1:
                    raise ValueError("missing or ambiguous calibrated pack in the isolated store")
        report.update(serving_gptq_verified=True, w4_sites=192, fp8_sites=193)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({k: v for k, v in report.items() if k != "records"}, indent=2))


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--expect-row-target", type=int, default=None)
    ap.add_argument("--root", type=Path, required=True)
    ap.add_argument("--rank", type=int, choices=range(4), required=True)
    ap.add_argument("--ckpt", type=Path, required=True)
    ap.add_argument("--boot", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--min-rows", type=int, default=131072)
    mode = ap.add_mutually_exclusive_group()
    mode.add_argument("--expect-gptq", action="store_true")
    mode.add_argument("--expect-rtn", action="store_true")
    audit(ap.parse_args())


if __name__ == "__main__":
    main()
