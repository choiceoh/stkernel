"""Verify the exact evaluated 5050 packs in an isolated fleet cache (CPU only)."""
import argparse
import json
from pathlib import Path

from probes.qwen38_gptq_offline import manifest_files
from probes.qwen38_gptq_subset import file_sha


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root', type=Path, required=True)
    p.add_argument('--manifest', type=Path, required=True)
    p.add_argument('--audit', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--engine-tree', required=True)
    args = p.parse_args()
    if args.out.exists():
        raise FileExistsError('the installation receipt must not be overwritten')
    result = json.loads(args.manifest.read_bytes())
    audit = json.loads(args.audit.read_bytes())
    if (result['rank'] != audit['rank'] or result['weights_id'] != audit['weights_id']
            or result['minimum_fit_rows'] < 330000 or audit['minimum_rows'] < 330000
            or result['fit_audit_sha256'] != file_sha(args.audit)
            or result['engine_tree'] != args.engine_tree):
        raise ValueError('5050 artifacts differ from the frozen fleet collection')
    manifest_files(result, args.root / 'st-dense-packs')
    receipt = dict(stage='offline_packs_installed', rank=result['rank'], weights_id=result['weights_id'],
                   engine_tree=args.engine_tree, offline_manifest_sha256=file_sha(args.manifest),
                   pack_count=len(result['packs']), serving_gptq_verified=False)
    args.out.write_text(json.dumps(receipt, indent=2) + '\n')
    print(json.dumps(receipt), flush=True)


if __name__ == '__main__':
    main()
