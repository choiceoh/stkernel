"""CPU-only PTX/SASS and resource audit of the instruction qualification cubins."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import re
import subprocess


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--root', type=Path, required=True)
    ap.add_argument('--nvdisasm', required=True)
    ap.add_argument('--cuobjdump', default='cuobjdump')
    ap.add_argument('--out', type=Path, required=True)
    args = ap.parse_args()
    rows = []
    for directory in ('quant-compiled', 'compiled'):
        for cubin in sorted((args.root/directory).glob('*.cubin')):
            ptx = cubin.with_suffix('.ptx').read_text()
            sass = subprocess.check_output([args.nvdisasm, '-c', str(cubin)], text=True)
            cubin.with_suffix('.sass').write_text(sass)
            resources = subprocess.check_output([args.cuobjdump, '--dump-resource-usage', str(cubin)], text=True)
            inst = []
            for line in sass.splitlines():
                m = re.match(r'\s*/\*[0-9a-f]+\*/\s+(?:@!?P\d+\s+)?([A-Z][A-Z0-9_.]*)\s', line)
                if m:
                    inst.append(m[1])
            counts = Counter(op.split('.')[0] for op in inst)
            relevant = {op: [line.strip() for line in sass.splitlines() if op in line][:16]
                        for op in ('FMNMX', 'FMUL2', 'F2FP', 'HMMA', 'OMMA', 'QMMA')}
            rows.append(dict(file=str(cubin.relative_to(args.root)),
                ptx_sha256=hashlib.sha256(ptx.encode()).hexdigest(),
                cubin_sha256=hashlib.sha256(cubin.read_bytes()).hexdigest(),
                sass_sha256=hashlib.sha256(sass.encode()).hexdigest(),
                ptx_max3=ptx.count('max.abs.f32'), ptx_mul2=ptx.count('mul.rn.f32x2'),
                ptx_nvfp4_mma=ptx.count('kind::mxf4nvf4.block_scale.scale_vec::4X'),
                sass_opcode_counts=dict(sorted(counts.items())), resources=resources,
                relevant_sass=relevant))
    args.out.write_text(json.dumps(dict(nvdisasm_version=subprocess.check_output(
        [args.nvdisasm, '--version'], text=True), kernels=rows), indent=2)+'\n')


if __name__ == '__main__':
    main()
