"""Read an existing ELF/disassembly and map recorded sanitizer PCs; no execution."""
import argparse
import gzip
import hashlib
import json
from pathlib import Path
import re
import struct

ap=argparse.ArgumentParser(description=__doc__)
ap.add_argument("--artifacts",type=Path,required=True)
ap.add_argument("--log",type=Path,required=True)
args=ap.parse_args()
ROOT=args.artifacts
ELF = ROOT/'cydriver.cpython-312-aarch64-linux-gnu.so'
raw = ELF.read_bytes()
assert raw[:6] == b'\x7fELF\x02\x01'
header = struct.unpack_from('<16sHHIQQQIHHHHHH',raw)
assert header[2] == 183  # AArch64
sections = [struct.unpack_from('<IIQQQQIIQQ',raw,header[6]+i*header[11]) for i in range(header[12])]

def c_string(address):
    for section in sections:
        _,typ,_,vaddr,offset,size,*_ = section
        if typ!=8 and vaddr<=address<vaddr+size:
            start=offset+address-vaddr
            return raw[start:raw.index(b'\0',start)].decode('ascii')
    raise ValueError('string address not backed by ELF bytes: '+hex(address))

instructions=[]
for line in (ROOT/'cydriver.disassembly.txt').read_text().splitlines():
    match=re.match(r'\s*([0-9a-f]+):\s+[0-9a-f]{8}\s+(\S+)\s*(.*)',line)
    if match:
        address=int(match[1],16)
        instructions.append((address,match[2],match[3].split('//',1)[0].strip(),line.strip()))
positions={item[0]:index for index,item in enumerate(instructions)}
# The default-stream resolver block is reached by this forward branch. The
# intervening text is the other stream/error paths, not linear predecessors.
default_entry=positions[0x165ac]
default_predecessor=positions[0x14ef4]
assert instructions[default_predecessor][1:3]==('cbz','w1, 165ac <PyInit_cydriver@@Base+0x41f4>')
assert instructions[default_entry-1][1]=='b'
incoming=[item for item in instructions if item[1] in ('b','cbz','cbnz','tbz','tbnz') or item[1].startswith('b.')]
assert [item[0] for item in incoming if re.search(r'\b165ac\s+<',item[2])]==[0x14ef4]

def normalize(reg):
    return 'x'+reg[1:] if reg.startswith('w') else reg

def resolve(reg,before,trail,depth=0):
    if depth>20:
        raise ValueError('register expression too deep')
    reg=normalize(reg)
    if reg in ('xzr','wzr'):
        return 0
    writes={'adrp','adr','add','mov','sub','ldr','ldur','ldp','and','orr','movk','movz','csel'}
    for index in range(before-1,-1,-1):
        address,op,operand,line=instructions[index]
        if index==default_entry:
            trail.extend([line,instructions[default_predecessor][3]])
            return resolve(reg,default_predecessor+1,trail,depth+1)
        if op in ('blr','bl') and reg.startswith('x') and reg[1:].isdigit() and int(reg[1:])<19:
            raise ValueError('value crosses caller-saved register call: '+reg+' '+line)
        if op not in writes:
            continue
        args=[a.strip() for a in operand.split(',')]
        if normalize(args[0])!=reg and not (op=='ldp' and normalize(args[1])==reg):
            continue
        trail.append(line)
        if op in ('adrp','adr'):
            return int(args[1].split()[0],16)
        if op=='mov':
            if args[1].startswith('#'):
                return int(args[1][1:],0)
            return resolve(args[1],index,trail,depth+1)
        if op in ('add','sub') and args[2].startswith('#'):
            value=int(args[2][1:],0)
            if len(args)>3:
                shift=re.fullmatch(r'lsl #(\d+)',args[3])
                if shift is None:
                    raise ValueError('unsupported shift: '+line)
                value <<= int(shift[1])
            return resolve(args[1],index,trail,depth+1)+(value if op=='add' else -value)
        raise ValueError('unresolved register '+reg+' from '+line)
    raise ValueError('no register definition: '+reg)

log=args.log.read_bytes()
if args.log.suffix == '.gz':
    log=gzip.decompress(log)
blocks=re.split(r'(?=^========= Program hit )',log.decode(),flags=re.M)[1:]
mapped=[]
for block in blocks:
    frame=int(re.search(r'Host Frame:  \[(0x[0-9a-f]+)\] in cydriver',block)[1],16)
    call=frame-3
    index=positions[call]
    assert instructions[index][1:3]==('blr','x23')
    row=dict(sanitizer_frame=hex(frame),call_address=hex(call),call=instructions[index][3],
             boundary='sanitizer frame + 1 is the instruction after the 4-byte BLR')
    trail=[]
    pointer=resolve('x0',index,trail)
    row.update(symbol=c_string(pointer),symbol_address=hex(pointer),
               requested_version=resolve('x2',index,trail),flags=resolve('x3',index,trail),
               query_result_pointer=resolve('x4',index,trail),argument_instructions=trail,
               return_check=[item[3] for item in instructions[index+1:index+3]])
    assert row['symbol'].startswith('cu'), row
    mapped.append(row)
result=dict(binary=str(ELF),sha256=hashlib.sha256(raw).hexdigest(),
            log_sha256=hashlib.sha256(log).hexdigest(),mapping='static AArch64 immediate/address dataflow at each recorded call site',
            observed_driver_api=13000,lookups=mapped,
            branch_resolution=dict(branch=instructions[default_predecessor][3],target=instructions[default_entry][3],
                                   fallthrough_predecessor=instructions[default_entry-1][3],
                                   unique_direct_predecessor=True,reason='Avoid interpreting intervening stream/error blocks as linear predecessors of default-stream entry'),
            limitations=['Raw sanitizer log does not record call arguments; these are read from the copied binary.',
                         'Image identity and container provenance are in copy.json; this does not identify every runtime-loaded library.',
                         'A request version above the observed driver API is a mismatch candidate, not an A/B-proven sole cause.'])
(ROOT/'lookup-map.json').write_text(json.dumps(result,indent=2)+'\n')
rows=['| Frame | Call | Symbol | Requested version | Flags | Query result |','|---|---|---|---:|---:|---|']
for row in mapped:
    rows.append(f"| {row['sanitizer_frame']} | {row['call_address']} | {row['symbol']} | {row['requested_version']} | {row['flags']} | {'NULL' if row['query_result_pointer']==0 else row['query_result_pointer']} |")
(ROOT/'lookup-map.md').write_text('\n'.join(rows)+'\n')
print(json.dumps(dict(mapped=len(mapped),versions={str(v):sum(r['requested_version']==v for r in mapped) for v in sorted({r['requested_version'] for r in mapped})},
                     symbols=[r['symbol'] for r in mapped],all_above_driver=all(r['requested_version']>13000 for r in mapped)),indent=2))
