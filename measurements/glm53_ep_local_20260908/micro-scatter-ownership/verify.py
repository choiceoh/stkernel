"""Pure address interpreter for the original, receipt-bound CPU13 PTX.

Only integer address operations in explicitly bounded ranges are evaluated.
No CUDA/CuTe/Torch imports, compilation, or device execution occurs.
"""
from collections import Counter, defaultdict
import gzip
import hashlib
import json
from pathlib import Path
import re


def evaluate(lines, ranges, registers, tid):
    values = dict(registers)
    values['%tid.x'] = tid
    pattern = re.compile(r'\s*(mov|add|shl|shr|and|or|xor)\.(?:b32|s32|u32)\s+(%r\d+),\s*(.*);$')
    for start, end in ranges:
        for line in lines[start-1:end]:
            match = pattern.fullmatch(line)
            if not match:
                continue
            op, dst, operands = match.groups()
            args = []
            for operand in operands.split(','):
                operand = operand.strip()
                if operand in values:
                    args.append(values[operand])
                elif re.fullmatch(r'-?\d+', operand):
                    args.append(int(operand))
                else:
                    break  # Unrelated token/weight values are not addresses.
            else:
                if op == 'mov': result = args[0]
                elif op == 'add': result = args[0] + args[1]
                elif op == 'shl': result = args[0] << args[1]
                elif op == 'shr': result = args[0] >> args[1]
                elif op == 'and': result = args[0] & args[1]
                elif op == 'or': result = args[0] | args[1]
                elif op == 'xor': result = args[0] ^ args[1]
                values[dst] = result & 0xffffffff
    return values


def verify(root=None):
    root = Path(root) if root else Path(__file__).resolve().parent
    identity = json.loads((root/'identity.json').read_text())
    results = {}
    for name, spec in identity['variants'].items():
        compressed = (root/(name+'.ptx.gz')).read_bytes()
        assert hashlib.sha256(compressed).hexdigest() == spec['compressed_sha256']
        raw = gzip.decompress(compressed)
        assert hashlib.sha256(raw).hexdigest() == spec['ptx_sha256']
        assert len(raw) == spec['ptx_bytes']
        lines = raw.decode().splitlines()
        start, end = spec['store_range']
        stores = []
        for line in lines[start-1:end]:
            match = re.fullmatch(r'\s*st.shared.b32\s+\[(%r\d+)\], %r\d+;', line)
            assert match, line
            stores.append(match[1])
        assert 'fence.proxy.async.shared::cta;' in lines[spec['fence_line']-1]
        assert 'bar.sync' in lines[spec['post_barrier_line']-1]
        assert ', 128;' in lines[spec['post_barrier_line']-1]
        between = lines[end:spec['post_barrier_line']-1]
        assert not any('bar.sync' in line or 'bar.warp.sync' in line for line in between)
        assert sum('ld.shared.b16' in line for line in between) == 2
        writers = defaultdict(list)
        for tid in range(128):
            values = evaluate(lines, spec['producer_ranges'],
                              {spec['base_register']: 0, spec['tid_register']: tid}, tid)
            for reg in stores:
                address = values[reg]
                assert address % 4 == 0
                for offset in (0, 2):
                    writers[address+offset].append(tid)
        assert all(len(owners) == 1 for owners in writers.values())
        counts, witnesses = Counter(), []
        for tid in range(64):  # Existing scalar scatter uses warps 0 and 1.
            for row in range(8):
                values = evaluate(lines, spec['reader_ranges'],
                                  {spec['base_register']: 0, spec['tid_register']: tid,
                                   spec['reader_seed']: row*32+(tid % 32)}, tid)
                addresses = [values[r] for r in spec['read_registers']]
                owners = [writers[a][0] for a in addresses]
                assert owners[0] == owners[1]  # Both halves of the stored b32.
                producer_warp = owners[0] // 32
                counts[(tid//32, producer_warp)] += 1
                if producer_warp != tid//32 and len(witnesses) < 4:
                    witnesses.append(dict(reader_tid=tid, producer_tid=owners[0], row=row,
                                          column=(tid//32)*64+(tid%32)*2,
                                          shared_offsets=addresses))
        assert counts == {(0,0):128, (0,2):128, (1,0):128, (1,2):128}, counts
        results[name] = {
            'ptx_sha256': spec['ptx_sha256'],
            'unique_producer_bf16_locations': len(writers),
            'pair_reads': sum(counts.values()),
            'cross_warp_pair_reads': sum(n for (reader, writer), n in counts.items() if reader != writer),
            'reader_to_writer_warp_pairs': {f'{r}->{w}': n for (r,w), n in counts.items()},
            'witnesses': witnesses,
            'store_lines': spec['store_range'], 'fence_line': spec['fence_line'],
            'load_lines': [i+1 for i in range(end, spec['post_barrier_line']-1)
                           if 'ld.shared.b16' in lines[i]],
            'post_barrier_line': spec['post_barrier_line'],
        }
    return {'verdict': 'PASS', 'scope': identity['scope'], 'variants': results}


if __name__ == '__main__':
    print(json.dumps(verify(), indent=2))
