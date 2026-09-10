#!/usr/bin/env python3
"""Reproduce CPU13/14 Q0 compiler inspection using only archived local bytes.

Default: verify the saved JSON/Markdown. --write: create those two derived
artifacts. No imports of Torch/CUDA, compilation, subprocesses or GPU work.
All line numbers are one-based lines of decompressed, receipt-bound PTX.
"""
import argparse
import ast
import gzip
import hashlib
import json
from pathlib import Path
import re

HERE = Path(__file__).resolve().parent
PREFIX = '/usr/local/lib/python3.12/dist-packages/flashinfer/'
KERNEL_SOURCE = PREFIX+'fused_moe/cute_dsl/blackwell_sm12x/moe_dynamic_ep_local.py'
STOCK_SOURCE = PREFIX+'fused_moe/cute_dsl/blackwell_sm12x/_moe_dynamic/gated.py'
PIN = {
    13: dict(revision='18148116a7abb242741d5b112c8d735353d9fc71',
             receipt='b68bcead10b78d19c3df09f0d765cf485dffd79b0ce3377ca379921d1c4ee7c2',
             submission='8f8cfdafa4ee341b18404c29c9d79534128aa3dfe9384b5e6d14e1d5a42fb75f',
             ptx='21a5c860b7f02aa887dd1cd8e90a5953094842b83ee24b118697c2f4d334d2de',
             cubin='397eaf35853780c85b37ea3a9c8491f4e4fd9f0c3c9380e1ca298b419fe0f4a3',
             source='c52757d150f890f302593aa936a015ed2e6f1a6f0bf0211db5c9dd02c0cc32f1', tests=68),
    14: dict(revision='881456a1f43fcf61de1bb5822b883dd2fd32e693',
             receipt='5ee89ff895b3f99076732c25d465a6e6fe318afa28e2319ac376eec401a2052f',
             submission='bc86c2075227ede68ef4d4110ca4d3a695b5a2c7c30ec95765bfba334bbf31ee',
             ptx='7c4bdc1d65f84314089dc399c39e91c90b12c699b34257de697c9fda151d1cc8',
             cubin='a657728bfb29e02e032b0fa43b69cf169fa5ac63b0cb74bf3954100211720dd4',
             source='664a2481b9f2065a4e8f08842a30e7840b83c78b79bcebbb6187f9a085acba55', tests=71),
}


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def cite(lines, first, last=None):
    last = first if last is None else last
    return dict(first_line=first, last_line=last, text='\n'.join(lines[first-1:last]))


def load(version):
    directory = HERE.parent/('cpu'+str(version))
    raw = (directory/'local/result.json').read_bytes()
    assert sha(raw) == PIN[version]['receipt'], 'compile receipt changed'
    receipt = json.loads(raw)
    raw = (directory/'submission.json').read_bytes()
    assert sha(raw) == PIN[version]['submission'], 'submission reference changed'
    submission = json.loads(raw)
    assert submission['revision'] == PIN[version]['revision']
    assert '--cpu' in submission['command'] and '--gpu' not in submission['command']
    assert receipt['arm'] == 'local' and receipt['cuda_initialized'] is False
    assert receipt['sources'][KERNEL_SOURCE] == PIN[version]['source']
    assert receipt['mounted_sources'][KERNEL_SOURCE] == PIN[version]['source']
    assert {k: v for k, v in receipt['contracts'].items() if k != 'files'} == dict(
        tests_run=PIN[version]['tests'], failures=0, errors=0, skips=0)
    assert len(receipt['artifacts']) == len(receipt['resources']) == 1
    identity = dict(revision=submission['revision'], receipt_sha256=PIN[version]['receipt'],
                    submission_sha256=PIN[version]['submission'], kernel_source_sha256=PIN[version]['source'])
    for kind, row in (('ptx', receipt['artifacts'][0]), ('cubin', receipt['resources'][0])):
        assert Path(row['file']).name == row['file'] and row['file'].endswith('.'+kind)
        path = directory/'local'/(row['file']+'.gz')
        compressed = path.read_bytes()
        payload = gzip.decompress(compressed)
        assert len(payload) == row['bytes'] and sha(payload) == row['sha256'] == PIN[version][kind]
        identity[kind] = dict(path=str(path.relative_to(HERE.parent)), bytes=len(payload),
                              sha256=sha(payload), gzip_sha256=sha(compressed))
        if kind == 'ptx':
            lines = payload.decode().splitlines()
    resources = receipt['resources'][0]['resources']
    selected = {key: int(re.search(r'\b'+key+r':(\d+)', resources)[1])
                for key in ('REG', 'STACK', 'SHARED')}
    assert selected == dict(REG=168, STACK=112, SHARED=1024)
    identity['resources'] = selected
    return identity, receipt, lines


def scan_pointers(lines):
    """Track parameter-root plus byte-offset terms through PTX address adds.

    This is intentionally bounded to the first Q0 phase. Only recognized
    ld.param/add pointer definitions propagate origins; other definitions
    clear them. It is not a general PTX interpreter or dynamic proof.
    """
    origins, ids, weights, stores, parameters = {}, [], [], [], {}
    for line_number, line in enumerate(lines, 1):
        stripped = line.strip()
        if line_number > 1900:
            break
        match = re.fullmatch(r'ld\.param\.b64\s+(%rd\d+), \[\w+_param_(\d+)\];', stripped)
        if match:
            register, parameter = match[1], int(match[2])
            origins[register] = (parameter, ())
            parameters[parameter] = cite(lines, line_number)
            continue
        match = re.fullmatch(r'(?:add\.s64|add\.u64)\s+(%rd\d+), (%rd\d+), ([^;]+);', stripped)
        if match:
            destination, base, offset = match.groups()
            if base in origins:
                root, terms = origins[base]
                origins[destination] = (root, terms+(offset,))
            else:
                origins.pop(destination, None)
            continue
        match = re.fullmatch(r'ld\.global\.b32\s+(%r\d+), \[(%rd\d+)(\+\d+)?\];', stripped)
        if match and match[2] in origins:
            root, terms = origins[match[2]]
            row = dict(line=line_number, register=match[1],
                       byte_offset_terms=list(terms)+(list([match[3][1:]]) if match[3] else []))
            if root == 1:
                ids.append(row)
            elif root == 2:
                weights.append(row)
        # Both old adaptive inline asm and new plain-store asm use this form.
        matches = re.findall(r'st\.global\.u64\s+\[(%rd\d+)\], (%rd\d+);', stripped)
        for address, payload in matches:
            if address in origins and origins[address][0] == 3:
                stores.append(dict(line=line_number, payload=payload,
                                   kind='adaptive' if 'setp.le.s32 persist' in line else 'plain',
                                   instruction=cite(lines, line_number)))
        # Keep address definitions trustworthy when a register gets reused.
        defined = re.match(r'(?:[a-z]\w*\.)[^\s]*\s+(%rd\d+),', stripped)
        if defined:
            origins.pop(defined[1], None)
    assert len(ids) == len(weights) == 8
    assert len(stores) == 10
    assert len({s['payload'] for s in stores[:7]}) == 1
    assert len({s['payload'] for s in stores}) == 4
    return ids, weights, stores, {str(i): parameters[i] for i in (1, 2, 3)}


def inspect_weights(lines, ids, weights, guarded):
    result = []
    labels = {line.strip()[:-1]: index for index, line in enumerate(lines, 1)
              if re.fullmatch(r'\$L__BB\d+_\d+:', line.strip())}
    for index, (id_load, weight_load) in enumerate(zip(ids, weights)):
        assert id_load['byte_offset_terms'] == weight_load['byte_offset_terms']
        lo, weight_line = id_load['line'], weight_load['line']
        local_check = next((i, m) for i in range(lo, lo+8)
                           if (m := re.fullmatch(r'\s*setp\.gt\.u32\s+(%p\d+), '
                                                +re.escape(id_load['register'])+r', 71;', lines[i-1])))
        check_line, match = local_check
        local_predicate = match[1]
        zero = next((i, m) for i in range(weight_line+1, weight_line+8)
                    if (m := re.fullmatch(r'\s*setp\.eq\.f32\s+(%p\d+), '
                                         +re.escape(weight_load['register'])+r', 0f00000000;', lines[i-1])))
        zero_line, zero_match = zero
        zero_predicate = zero_match[1]
        if guarded:
            branch_line, branch = next((i, m) for i in range(check_line+1, weight_line)
                                      if (m := re.fullmatch(r'\s*@'+re.escape(local_predicate)
                                                           +r' bra\s+(\$L__BB\d+_\d+);', lines[i-1])))
            target = branch[1]
            assert lo < check_line < branch_line < weight_line < zero_line < labels[target]
            zero_branch_line = zero_line+1
            assert re.fullmatch(r'\s*@'+re.escape(zero_predicate)+r' bra\s+'
                                +re.escape(target)+r';', lines[zero_branch_line-1])
        else:
            assert lo < weight_line < check_line < zero_line
            # No branch or label can make the old weight load conditional on
            # the ID between these paired, same-index global reads.
            assert not any('bra' in line or line.strip().endswith(':')
                           for line in lines[lo:weight_line-1])
            merged = re.fullmatch(r'\s*or\.pred\s+(%p\d+), '+re.escape(local_predicate)
                                  +', '+re.escape(zero_predicate)+r';', lines[zero_line])
            assert merged
            branch_line = zero_line+2
            branch = re.fullmatch(r'\s*@'+re.escape(merged[1])+r' bra\s+(\$L__BB\d+_\d+);',
                                 lines[branch_line-1])
            assert branch
            target, zero_branch_line = branch[1], branch_line
            assert labels[target] > branch_line
        result.append(dict(role='histogram' if index == 0 else 'producer', static_copy=index,
                           id_load=cite(lines, lo), weight_load=cite(lines, weight_line),
                           id_check=cite(lines, check_line), skip_branch=cite(lines, branch_line),
                           skip_target=cite(lines, labels[target]), zero_check=cite(lines, zero_line),
                           instruction_window=cite(lines, lo, zero_branch_line),
                           weight_load_guarded_by_valid_id=guarded))
    return result


def build_report():
    data = {version: load(version) for version in (13, 14)}
    identities = {str(version): values[0] for version, values in data.items()}
    old_receipt, new_receipt = data[13][1], data[14][1]
    assert set(old_receipt['sources']) == set(new_receipt['sources'])
    changed = [key for key in old_receipt['sources']
               if old_receipt['sources'][key] != new_receipt['sources'][key]]
    assert changed == [KERNEL_SOURCE]
    stock_path = HERE.parent/'cpu13/stock-gated.py.gz'
    stock_raw = gzip.decompress(stock_path.read_bytes())
    assert sha(stock_raw) == old_receipt['sources'][STOCK_SOURCE] == new_receipt['sources'][STOCK_SOURCE]
    stock_tree = ast.parse(stock_raw.decode())
    kernel = next(node for node in ast.walk(stock_tree) if isinstance(node, ast.FunctionDef) and node.name == 'kernel')
    assert [arg.arg for arg in kernel.args.args[:5]] == ['self', 'a_input', 'topk_ids', 'topk_weights', 'packed_a_storage']
    versions = {}
    for version in (13, 14):
        lines = data[version][2]
        ids, weights, stores, parameters = scan_pointers(lines)
        weight_evidence = inspect_weights(lines, ids, weights, guarded=version == 14)
        expected = 'adaptive' if version == 13 else 'plain'
        assert all(store['kind'] == expected for store in stores)
        assert sum('setp.le.s32 persist' in line for line in lines) == (10 if version == 13 else 0)
        for index, store in enumerate(stores):
            store['role'] = 'equal-scales' if index < 7 else 'varied-scales'
            store['static_copy'] = index
            if version == 13:
                assert '2048;' in store['instruction']['text'] and '@!persist st.global.u64' in store['instruction']['text']
            else:
                assert re.fullmatch(r'\s*st\.global\.u64 \[%rd\d+\], %rd\d+;', store['instruction']['text'])
        versions[str(version)] = dict(parameter_origins=parameters, weight_loads=weight_evidence, q0_stores=stores)
    return dict(schema=1, verdict='VERIFIED_STATIC_COMPILER_OUTPUT', performance_acceptance=False,
                gpu_acceptance=False, verifier_sha256=sha(Path(__file__).read_bytes()),
                artifacts=identities, stock_parameter_reference=dict(path='cpu13/stock-gated.py.gz',
                    sha256=sha(stock_raw), kernel_definition_line=kernel.lineno,
                    parameter_order=['a_input', 'topk_ids', 'topk_weights', 'packed_a_storage']),
                changed_compiler_source=changed, instruction_evidence=versions,
                counts=dict(histogram_weight_load_sites=[1, 1], producer_weight_load_sites=[7, 7],
                    valid_id_guarded_weight_load_sites=[0, 8], q0_adaptive_store_sites=[10, 0],
                    q0_plain_store_sites=[0, 10], equal_scale_store_sites=[7, 7], varied_scale_store_sites=[3, 3]),
                artifact_byte_deltas=dict(ptx=identities['14']['ptx']['bytes']-identities['13']['ptx']['bytes'],
                    cubin=identities['14']['cubin']['bytes']-identities['13']['cubin']['bytes']),
                limitations=[
                    'Counts are static instruction sites, not executed instructions, transactions or bytes saved.',
                    'Invalid signed Int32 IDs compare above 71 under unsigned comparison and bypass the CPU14 weight load; valid NaNs remain unequal to zero and signed zeros compare equal.',
                    'The dynamic quantization loops and all ten payload store sites remain. Seven equal-scale and three varied-scale sites are compiler unroll copies, not route-count limits.',
                    'Plain Q0 stores match the old executed arm only for the admitted T4096..16384 range; no defaults or admission range were changed.',
                    'Resource and cubin-byte facts come from archived CPU compilation. No GPU numerics, race freedom, memory-traffic reduction, throughput or TTFT claim is established.'])


def markdown(report):
    old, new = (report['artifacts'][str(n)] for n in (13, 14))
    rows = ['CPU14 retains eight static route-weight load sites, and all eight now follow a valid-ID branch. '
            'The ten Q0 adaptive store sites became ten plain `st.global.u64` sites.', '',
            '| Compiler fact | CPU13 | CPU14 |', '|---|---:|---:|',
            '| Histogram / producer weight-load sites | 1 / 7 | 1 / 7 |',
            '| Weight loads guarded by valid ID | 0 | 8 |',
            '| Adaptive / plain Q0 store sites | 10 / 0 | 0 / 10 |',
            '| Equal / varied scale store sites | 7 / 3 | 7 / 3 |',
            f"| PTX bytes | {old['ptx']['bytes']} | {new['ptx']['bytes']} |",
            f"| Cubin bytes | {old['cubin']['bytes']} | {new['cubin']['bytes']} |",
            '| REG / STACK / SHARED | 168 / 112 / 1024 | 168 / 112 / 1024 |', '',
            'The static weight-load count did not fall: the invalid-ID path now branches around those loads. '
            'These are compiler facts, not measured traffic or speed improvements.', '']
    for label, index in (('Histogram', 0), ('First producer copy', 1)):
        for version in (13, 14):
            excerpt = report['instruction_evidence'][str(version)]['weight_loads'][index]['instruction_window']
            rows += [f"{label}, CPU{version}, decompressed PTX lines {excerpt['first_line']}–{excerpt['last_line']}:",
                     '', '```ptx', excerpt['text'], '```', '']
    for version in (13, 14):
        stores = report['instruction_evidence'][str(version)]['q0_stores']
        rows += [f"CPU{version} Q0 store lines: "+', '.join(str(store['line']) for store in stores)+'.', '']
    rows += ['The JSON records exact instructions, skip targets and parameter origins for all eight load sites and ten stores in each version. '
             'The source parameter reference is the SHA-verified inherited kernel; only the EP-local source differs in the two compiler source maps.', '']
    for version in (13, 14):
        item = report['artifacts'][str(version)]
        rows += [f"CPU{version} source `{item['revision']}`; PTX SHA256 `{item['ptx']['sha256']}`; cubin SHA256 `{item['cubin']['sha256']}`.", '']
    rows += ['Reproduce without compilation or GPU access:', '',
             '```sh', 'python3 measurements/glm53_ep_local_20260908/cpu14/verify_q0_load_store_inspection.py', '```', '',
             'This checks archived receipt and payload hashes before regenerating and comparing the derived JSON/Markdown. '
             'It does not establish GPU numerics, race freedom, throughput or TTFT.', '']
    return '\n'.join(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--write', action='store_true')
    args = parser.parse_args()
    report = build_report()
    json_path, md_path = HERE/'q0-load-store-inspection.json', HERE/'q0-load-store-inspection.md'
    md = markdown(report)
    if args.write:
        json_path.write_text(json.dumps(report, indent=2)+'\n')
        md_path.write_text(md)
    else:
        assert json.loads(json_path.read_text()) == report, 'saved JSON does not reproduce'
        assert md_path.read_text() == md, 'saved Markdown does not reproduce'
    print(json.dumps(dict(verdict=report['verdict'], counts=report['counts'],
                          artifact_byte_deltas=report['artifact_byte_deltas'])))


if __name__ == '__main__':
    main()
