#!/usr/bin/env python3
"""Static PTX comparison only; instruction counts are not GPU timing."""
from collections import Counter
import argparse
import gzip
import hashlib
import json
from pathlib import Path
import re

REPO = Path('/Users/ost/.worktrees/stkernel2/glm53-prefill-moe-pipeline-20260908')
OLD_ROOT = REPO/'measurements/glm53_ep_tiled_20260909/ring_cpu3'
NEW_ROOT = Path('/tmp/glm53-ep-word-cpu4-readback')


def sha(data):
    return hashlib.sha256(data).hexdigest()


def inspect(data):
    lines = data.decode().splitlines()
    barriers = [i for i,line in enumerate(lines) if re.search(r'\bbar\.sync\s+3,\s*128;',line)]
    assert len(barriers) == 12
    pairs = []
    for start,end in zip(barriers[::2],barriers[1::2]):
        instructions = []
        for line in lines[start+1:end]:
            match = re.match(r'^\s*(?:@\S+\s+)?([a-z][\w.]*)\s+.*;',line)
            if match:
                instructions.append(match.group(1))
        counts = Counter(instructions)
        assert counts['st.shared.s32'] == 4
        assert not any(op.startswith(('bra','call','ld.','mma','bar.')) for op in counts)
        integer = sum(n for op,n in counts.items() if op.split('.')[0] in
                      {'add','sub','and','or','xor','shl','shr','mul','mad','bfe','bfi','lop3','mov','cvt'})
        assert integer + counts['st.shared.s32'] == len(instructions)
        pairs.append(dict(read_barrier_line=start+1,write_barrier_line=end+1,
                          instructions=len(instructions),integer_alu_and_address=integer,
                          stores=counts['st.shared.s32'],opcodes=dict(sorted(counts.items()))))
    return dict(sha256=sha(data),barrier3_count=len(barriers),restore_regions=pairs,
                first_restore_excerpt='\n'.join(lines[barriers[0]:barriers[1]+1])+'\n')


def compare(old_root=OLD_ROOT, new_root=NEW_ROOT):
    old_receipt=(old_root/'result.json').read_bytes()
    new_receipt=(new_root/'result.json').read_bytes()
    before=json.loads(old_receipt)['static_passes'][0]
    after=json.loads(new_receipt)['static_passes'][0]
    assert before['arm']==after['arm']=='static/M6'
    old=gzip.decompress((old_root/(before['artifacts'][0]['path']+'.gz')).read_bytes())
    archived_new=new_root/(after['artifacts'][0]['path']+'.gz')
    new=gzip.decompress(archived_new.read_bytes()) if archived_new.exists() else (new_root/'M6.ptx').read_bytes()
    assert sha(old)==before['artifacts'][0]['sha256']
    assert sha(new)==after['artifacts'][0]['sha256']
    return dict(scope='static emitted PTX only; each region strictly between successive read/write bar.sync 3,128; excludes loads before the first barrier; not SASS cycles, occupancy, numerical or speed acceptance',
                old_result_sha256=sha(old_receipt),new_result_sha256=sha(new_receipt),
                old=inspect(old),new=inspect(new),
                old_resources=before['resources'][0]['resources'],
                new_resources=after['resources'][0]['resources'],
                old_cache_key=before['cache_key'],new_cache_key=after['cache_key'])


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--old-root',type=Path,default=OLD_ROOT)
    parser.add_argument('--new-root',type=Path,default=NEW_ROOT)
    args=parser.parse_args()
    print(json.dumps(compare(args.old_root,args.new_root),indent=2))
