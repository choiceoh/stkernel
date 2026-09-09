#!/usr/bin/env python3
"""Pure, strict archive inspection. Does not admit or modify CPU17 evidence."""
import argparse
import copy
import datetime
import gzip
import hashlib
import json
from pathlib import Path
import struct

SOURCE_SHA = '9d83120fb20e4aae382e3971a86c1a3b6970e591dd31ee5d7fc3e1a2ca521339'
SOURCE_NAME = 'glm53_ep_route_remap.py'
SOURCE_DIR = '/usr/local/lib/python3.12/dist-packages/flashinfer/fused_moe/cute_dsl/blackwell_sm12x'
DEBUG_NAMES = {'.debug_line', '.nv.merc.debug_line'}
DWARF_SPEC = 'https://dwarfstd.org/doc/dwarf-2.0.0.pdf'


def need(value, reason):
    if not value:
        raise ValueError(reason)


def sha(data):
    return hashlib.sha256(data).hexdigest()


def elf(data):
    need(len(data) >= 64, 'short ELF')
    head = struct.unpack_from('<16sHHIQQQIHHHHHH', data)
    need(head[0][:7] == b'\x7fELF\x02\x01\x01' and head[2] == 190, 'expected ELF64 little-endian CUDA')
    need(head[8] == 64 and head[11] == 64 and 0 < head[12] < 1000, 'unexpected ELF layout')
    need(head[6]+head[11]*head[12] <= len(data), 'section table outside ELF')
    need(head[5]+head[9]*head[10] <= len(data), 'program table outside ELF')
    headers = [struct.unpack_from('<IIQQQQIIQQ', data, head[6]+i*64) for i in range(head[12])]
    need(0 <= head[13] < len(headers), 'bad section string index')
    table = headers[head[13]]
    need(table[4]+table[5] <= len(data), 'section name table outside ELF')
    strings = data[table[4]:table[4]+table[5]]
    sections = []
    for i, h in enumerate(headers):
        need(h[0] < len(strings), 'invalid section name')
        end = strings.index(0, h[0]); name = strings[h[0]:end].decode('ascii')
        need(h[1] == 8 or h[4]+h[5] <= len(data), 'section outside ELF')
        sections.append(dict(index=i, name=name, header=list(h), offset=h[4], size=h[5],
                             flags=h[2], data=b'' if h[1] == 8 else data[h[4]:h[4]+h[5]]))
    need(len({s['name'] for s in sections}) == len(sections), 'duplicate section names')
    return head, sections


def dwarf_line(data):
    need(len(data) >= 16, 'short DWARF line table')
    length, version, header_length = struct.unpack_from('<IHI', data)
    need(length+4 == len(data) and version == 2, 'expected one DWARF32 v2 line table')
    header_end = 10+header_length
    need(15 <= header_end <= len(data) and data[14] > 0, 'invalid line header')
    pos = 15+data[14]-1
    need(pos <= header_end, 'invalid standard opcodes')

    def string():
        nonlocal pos
        end = data.index(0, pos, header_end)
        result = data[pos:end].decode('utf-8'); pos = end+1
        return result

    def uleb():
        nonlocal pos
        start = pos; result = shift = 0
        while True:
            need(pos < header_end and shift < 64, 'invalid ULEB128')
            byte = data[pos]; pos += 1
            result |= (byte & 127) << shift
            if not byte & 128:
                return dict(value=result, start=start, end=pos, hex=data[start:pos].hex())
            shift += 7

    directories = []
    while True:
        directory = string()
        if not directory:
            break
        directories.append(directory)
    files = []
    while True:
        filename = string()
        if not filename:
            break
        directory, mtime, size = uleb(), uleb(), uleb()
        files.append(dict(filename=filename, directory=directory, mtime=mtime, size=size))
    need(pos == header_end, 'unparsed line-header bytes')
    need(directories == [SOURCE_DIR] and len(files) == 1, 'unexpected source directory/file table')
    entry = files[0]
    need(entry['filename'] == SOURCE_NAME and entry['directory']['value'] == 1
         and entry['size']['value'] == 5493, 'unexpected source filename/directory/length')
    return dict(version=version, header_end=header_end, directories=directories, file=entry)


def compare(old, new, *, expected_new_mtime):
    need(len(old) == len(new), 'ELF length differs')
    h1, sections1 = elf(old); h2, sections2 = elf(new)
    need(h1 == h2, 'ELF header differs')
    need(len(sections1) == len(sections2), 'ELF section count differs')
    normalized1, normalized2 = bytearray(old), bytearray(new)
    rows, permitted, timestamps = [], set(), []
    for a, b in zip(sections1, sections2):
        need(a['name'] == b['name'] and a['header'] == b['header'], 'ELF section metadata differs')
        name = a['name']
        row = {k: v for k, v in a.items() if k != 'data'}
        row.update(cpu16_sha256=sha(a['data']), cpu17_sha256=sha(b['data']), exact_equal=a['data'] == b['data'])
        if name in DEBUG_NAMES:
            need(a['flags'] & 2 == 0 and b['flags'] & 2 == 0, 'debug section has SHF_ALLOC')
            d1, d2 = dwarf_line(a['data']), dwarf_line(b['data'])
            t1, t2 = d1['file']['mtime'], d2['file']['mtime']
            need((t1['start'], t1['end']) == (t2['start'], t2['end']), 'mtime encoding span changed')
            need(t1['value'] == 1788872668, 'CPU16 mtime differs from pinned source stat')
            need(t2['value'] == expected_new_mtime, 'CPU17 mtime differs from archived source stat')
            for i in range(t1['start'], t1['end']):
                offset = a['offset']+i
                permitted.add(offset); normalized1[offset] = normalized2[offset] = 0
            row['dwarf'] = dict(cpu16=d1, cpu17=d2)
            row['changed_section_offsets'] = [i for i, (x, y) in enumerate(zip(a['data'], b['data'])) if x != y]
            timestamps.append((t1['value'], t2['value']))
        rows.append(row)
    need({r['name'] for r in rows if 'dwarf' in r} == DEBUG_NAMES, 'missing required debug sections')
    need(len(set(timestamps)) == 1, 'mtime values disagree between line tables')
    need(normalized1 == normalized2, 'bytes other than the two source mtimes differ')
    differences = [dict(offset=i, cpu16=x, cpu17=y) for i, (x, y) in enumerate(zip(old, new)) if x != y]
    need(differences and all(d['offset'] in permitted for d in differences), 'unexpected raw differences')
    return dict(elf_header=list(h1[1:]), elf_ident_hex=h1[0].hex(),
                raw_differences=differences, all_non_mtime_bytes_exact=True,
                all_section_headers_exact=True, all_program_headers_exact=True,
                executable_sections_exact=all(r['exact_equal'] for r in rows if r['flags'] & 4),
                all_alloc_sections_exact=all(r['exact_equal'] for r in rows if r['flags'] & 2),
                cpu16_mtime=timestamps[0][0], cpu17_mtime=timestamps[0][1], sections=rows)


def read_artifact(root, rel, want):
    stored = root/(rel+'.gz')
    data = gzip.decompress(stored.read_bytes()) if stored.exists() else (root/rel).read_bytes()
    if not stored.exists():
        stored = root/rel
    need(sha(data) == want, 'artifact receipt SHA mismatch: '+str(stored))
    return data, dict(path=str(stored), raw_sha256=sha(data), raw_bytes=len(data),
                      stored_sha256=sha(stored.read_bytes()), stored_bytes=stored.stat().st_size)


def mutation_checks(old, new, mtime):
    _, sections = elf(new); by_name = {s['name']:s for s in sections}
    debug = by_name['.debug_line']; dwarf = dwarf_line(debug['data'])
    probes = dict(
        executable_text=next(s['offset'] for s in sections if s['flags'] & 4),
        debug_filename=debug['offset']+debug['data'].index(SOURCE_NAME.encode()),
        debug_line_program=debug['offset']+dwarf['header_end'],
        debug_file_size=debug['offset']+dwarf['file']['size']['start'],
        source_mtime_mismatch=debug['offset']+dwarf['file']['mtime']['start'],
        elf_flags=48)
    checks = []
    for name, offset in probes.items():
        changed = bytearray(new); changed[offset] ^= 1
        try:
            compare(old, bytes(changed), expected_new_mtime=mtime)
        except ValueError as exc:
            checks.append(dict(mutation=name, offset=offset, rejected=True, reason=str(exc)))
        else:
            raise ValueError('mutation was incorrectly accepted: '+name)
    return checks


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    roots = [args.root/'measurements/glm53_ep_local_20260908'/x for x in ('cpu16', 'cpu17.collecting')]
    proof_bytes = [(r/'local/result.json').read_bytes() for r in roots]
    proofs = [json.loads(x) for x in proof_bytes]
    need(sha(proof_bytes[0]) == '70c06279f97f2cb951606ae64be65038c957a9274bb3885d2371dc998a16578c', 'CPU16 receipt pin differs')
    maps = [{r['label']:r for r in p['remap_compilation']} for p in proofs]
    need(len(maps[0]) == len(maps[1]) == 24 and set(maps[0]) == set(maps[1]), 'expected matched 24-case matrix')
    identity_bytes = (roots[1]/'source-identity.json').read_bytes()
    identity = json.loads(identity_bytes)
    source_meta = identity['files']['frozen-source/build/glm53/'+SOURCE_NAME]
    need(source_meta['sha256'] == SOURCE_SHA and source_meta['bytes'] == 5493, 'archived source identity differs')
    source_data = gzip.decompress((roots[1]/('frozen-source/build/glm53/'+SOURCE_NAME+'.gz')).read_bytes())
    need(sha(source_data) == SOURCE_SHA, 'archived source bytes differ')
    for proof in proofs:
        need(proof['mounted_sources'][SOURCE_DIR+'/'+SOURCE_NAME] == SOURCE_SHA, 'compiled source differs')
    mtime = source_meta['mtime_ns']//10**9
    need(mtime == 1788880342, 'CPU17 archived source mtime differs')
    cases = []; mutations = None
    for label in sorted(maps[0]):
        artifacts = []
        cubins = []
        ptx = []
        for root, mapping in zip(roots, maps):
            binary, binding = read_artifact(root, 'local/remap/'+label+'/kernel.cubin', mapping[label]['cubin_sha256'])
            text, text_binding = read_artifact(root, 'local/remap/'+label+'/kernel.ptx', mapping[label]['ptx_sha256'])
            artifacts.append(dict(cubin=binding, ptx=text_binding)); cubins.append(binary); ptx.append(text)
        need(ptx[0] == ptx[1], 'PTX differs: '+label)
        result = compare(*cubins, expected_new_mtime=mtime)
        if mutations is None:
            mutations = mutation_checks(*cubins, mtime)
        cases.append(dict(label=label, artifacts=artifacts, ptx_exact=True, **result))
    old_times = {case['cpu16_mtime'] for case in cases}
    need(len(old_times) == 1, 'CPU16 mtime differs across cases')
    report = dict(schema=1, scope='Read-only static classification; original REJECT_DIFFERENCE retained',
                  diagnostic_only=True, accepted_compile_proof=False,
                  classification='ONLY_DWARF_SOURCE_MTIME_BYTES_DIFFER',
                  cpu16_receipt_sha256=sha(proof_bytes[0]), cpu17_receipt_sha256=sha(proof_bytes[1]),
                  cpu17_source_identity_sha256=sha(identity_bytes), source_sha256=SOURCE_SHA,
                  source_basename=SOURCE_NAME, cpu17_source_stat=source_meta,
                  cpu16_source_mtime_stat_independently_verified=False,
                  dwarf_v2_file_table_spec=DWARF_SPEC,
                  cases_count=len(cases), total_changed_raw_bytes=sum(len(c['raw_differences']) for c in cases),
                  mutation_checks=mutations, cases=cases)
    args.output.write_text(json.dumps(report, indent=2)+'\n')
    lines = [
        'All 24 remap cubins differ only in the source-file modification timestamp in `.debug_line` and `.nv.merc.debug_line`.',
        'Each cubin has four changed bytes: section-relative offsets 136 and 137 in each of those two sections. Both sections have SHF_ALLOC clear.',
        'ELF headers, complete section metadata, program headers, executable and allocated sections, relocations, source paths, line programs, and every other byte are identical. All 24 PTX artifacts match their own receipts and each other.',
        f"The DWARF v2 file-table mtime decodes from {next(iter(old_times))} to {mtime}. CPU17 matches the archived source stat (`mtime_ns={source_meta['mtime_ns']}`). CPU16's decoded timestamp has not been independently checked against a source stat by this inspector.",
        f"Source: `{SOURCE_DIR}/{SOURCE_NAME}`, length 5493, SHA-256 `{SOURCE_SHA}`. The source hash is bound by both compilation receipts and CPU17 archived source bytes.",
        f'[DWARF v2 section 6.2 file table specification]({DWARF_SPEC}) identifies the ULEB128 timestamp field after the filename and directory index.',
        'Six local byte-mutation checks were rejected: executable text, filename, line program, file size, mtime inconsistent with the archived source stat, and ELF flags. No accelerator or repository module is imported.',
        'This report classifies the binary difference; it does not replace the original exact-byte rejection, admit the collecting directory, or establish GPU numerics/performance acceptance.',
        'Per-file original/stored hashes, exact differing bytes, all section hashes and decoded DWARF fields are in the adjacent JSON.',
    ]
    args.output.with_suffix('.md').write_text('\n\n'.join(lines)+'\n')
    print(json.dumps({k:report[k] for k in ('classification','cases_count','total_changed_raw_bytes','accepted_compile_proof')}))


if __name__ == '__main__':
    main()
