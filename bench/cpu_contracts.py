# SPDX-License-Identifier: Apache-2.0
"""Audited pure CPU contracts and bounded, in-memory fault sensitivity checks."""
import argparse
import ast
import contextlib
import copy
import hashlib
import importlib.util
import io
import json
import re
from pathlib import Path
import subprocess
import __future__

# Re-audited after merging SF6/boot-memory assertions with EP short-decode,
# compact warmup and structured proof assertions. The three audited helper
# tests and load_defs AST match main. PR #508 adds the video-slot assertion;
# PR #509 parses the already-read boot-stamp phase table and TARGETS strings.
# Dependency access graphs and source closure are unchanged.
LOGIC_AUDIT = '40e681a1e7a3c5e99e1c14b6df4105395c75cb154e5ae52b77cc0ff881d5f244'
CONTRACTS = {
    'math': ('test_prefill_chunker','overlay/modules/mla_indexer/indexer.py','split_indexer_prefill_chunks'),
    'layout': ('test_sp_ranges','overlay/modules/dsv4_attention/attention.py','_indexer_sp_owned_ranges'),
    'dispatch': ('test_skip_topk','overlay/modules/dsv4_attention/attention.py','_resolve_skip_topk'),
}
# Reviewed name/attribute/import/call-target graph. Arithmetic edits can reuse
# the narrow dependency audit; new access paths require a full gate/cache key.
DEPENDENCY_AUDITS = {
    'math': '3f8d6c2465c38716edf111c2bae9f5ef3eab5e6ec8b96d3a2bf664e2ad8ad867',
    'layout': '95d6a18cda0b95cf59dfc9fd0f07b652c7417836fefce95e62fbf43d586c5103',
    'dispatch': '68b8fe423fb6e884956320fb00c9b7f78f58f293628122cdeb2d8b533d98d47c',
}
MUTATIONS = {
    'logits-byte-width': ('math','max_logits_bytes // 4','max_logits_bytes'),
    'request-boundary': ('math','end + request_offset','end + request_offset - 1'),
    'shard-stride': ('layout','rank * shard','rank + shard'),
    'reuse-mask': ('dispatch','c4a_idx % getattr(config, "index_topk_freq", 1) != 0',
                   'c4a_idx % getattr(config, "index_topk_freq", 1) == 0'),
}


def audited(root):
    path = root/'tests/test_logic.py'
    return path.is_file() and hashlib.sha256(path.read_bytes()).hexdigest() == LOGIC_AUDIT


def dependencies(root, names):
    if not audited(root):
        return None
    try:
        for name in names:
            _,source,symbol = CONTRACTS[name]
            functions = [n for n in ast.parse((root/source).read_text()).body
                         if isinstance(n,ast.FunctionDef) and n.name == symbol]
            if len(functions) != 1:
                return None
            accesses = sorted(set(ast.dump(n.func) if isinstance(n,ast.Call) else ast.dump(n)
                                  for n in ast.walk(functions[0])
                                  if isinstance(n,(ast.Name,ast.Attribute,ast.Import,ast.ImportFrom,
                                                   ast.Call,ast.Global,ast.Nonlocal))))
            if hashlib.sha256(json.dumps(accesses).encode()).hexdigest() != DEPENDENCY_AUDITS[name]:
                return None
    except (OSError,SyntaxError):
        return None
    sources = {CONTRACTS[name][1] for name in names}
    for source in sources:
        path = Path(source)
        # load_defs resolves legacy flat paths first, then uniquely by filename.
        if (root/'overlay'/path.name).exists() or list((root/'overlay/modules').glob('*/'+path.name)) != [root/path]:
            return None
    return sources | {'tests/test_logic.py','.gitignore'}


def changed_contracts(root, base):
    """Narrow automatically only if every changed AST node is an audited helper."""
    if not base or not audited(root):
        return None
    try:
        paths = subprocess.check_output(['git','-C',str(root),'diff','--name-only',base,'HEAD'],text=True).splitlines()
        selected = set()
        for path in paths:
            candidates = {symbol:name for name,(_,source,symbol) in CONTRACTS.items() if source == path}
            if not candidates:
                return None
            old = subprocess.check_output(['git','-C',str(root),'show',base+':'+path],text=True,stderr=subprocess.PIPE)
            new = subprocess.check_output(['git','-C',str(root),'show','HEAD:'+path],text=True)
            a,b = ast.parse(old),ast.parse(new)
            def partition(tree):
                known = {n.name:ast.dump(n) for n in tree.body if isinstance(n,ast.FunctionDef) and n.name in candidates}
                rest = [ast.dump(n) for n in tree.body if not isinstance(n,ast.FunctionDef) or n.name not in candidates]
                return known,rest
            before,before_rest = partition(a)
            after,after_rest = partition(b)
            if before_rest != after_rest or before.keys() != after.keys():
                return None
            selected.update(candidates[name] for name in before if before[name] != after[name])
        return sorted(selected) if selected and dependencies(root,selected) else None
    except (OSError,SyntaxError,subprocess.SubprocessError):
        return None


def run(name, root, mutation=None):
    test_name, source, symbol = CONTRACTS[name]
    spec = importlib.util.spec_from_file_location('_cpu_contract_logic',root/'tests/test_logic.py')
    logic = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(logic)
    checks,applied = 0,0
    original_check = logic.check
    def check(condition,message):
        nonlocal checks
        checks += 1
        original_check(condition,message)
    logic.check = check
    if mutation:
        _,needle,replacement = MUTATIONS[mutation]
        needle = ast.dump(ast.parse(needle,mode='eval').body)
        replacement = ast.parse(replacement,mode='eval').body
        class Fault(ast.NodeTransformer):
            def visit(self,node):
                nonlocal applied
                if ast.dump(node) == needle:
                    applied += 1
                    return ast.copy_location(copy.deepcopy(replacement),node)
                return super().visit(node)
        original = logic.load_defs
        def load_defs(path,names,namespace):
            if symbol not in names:
                return original(path,names,namespace)
            tree = ast.parse((root/source).read_text())
            functions = [n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name == symbol]
            if len(functions) != 1:
                raise ValueError('mutation target is unavailable')
            before = applied
            body = [Fault().visit(functions[0])]
            if applied-before != 1:
                raise ValueError('mutation must change exactly one audited expression per extraction')
            module = ast.fix_missing_locations(ast.Module(body=body,type_ignores=[]))
            exec(compile(module,source,'exec',flags=__future__.annotations.compiler_flag),namespace)
            return namespace
        logic.load_defs = load_defs
    output = io.StringIO()
    error = None
    try:
        with contextlib.redirect_stdout(output):
            getattr(logic,test_name)()
    except (Exception,SystemExit) as exc:
        error = dict(type=type(exc).__name__,message=str(exc),exit_code=exc.code if isinstance(exc,SystemExit) else None)
    skipped = [line for line in output.getvalue().splitlines() if re.search(r'\bskip(?:ped)?\b',line,re.I)]
    complete = checks > 0 and not skipped
    passed = error is None and complete
    return dict(contract=name,tests_run=checks,count_unit='assertions',passed=passed,
                coverage_complete=complete,skipped=skipped,error=error,mutation=mutation,applications=applied,
                scope='declared CPU helper contract',full_suite=False)


def sensitivity(root):
    controls = [run(name,root) for name in CONTRACTS]
    cases = []
    for name,(contract,_,_) in MUTATIONS.items():
        result = run(contract,root,name)
        status = ('detected' if result['applications'] and result['tests_run'] > 0 and ((result['error'] or {}).get('type') == 'AssertionError' or ((result['error'] or {}).get('type') == 'SystemExit' and result['error']['exit_code'] == 1))
                  else 'survived' if result['passed'] else 'invalid')
        cases.append(dict(name=name,status=status,result=result))
    passed = all(c['passed'] for c in controls) and all(c['status']=='detected' for c in cases)
    return dict(passed=passed,coverage_complete=passed,tests_run=len(controls)+len(cases),skipped=[],
                scope='fault sensitivity of three declared CPU helpers only',controls=controls,mutations=cases,
                detected=sum(c['status']=='detected' for c in cases),total=len(cases))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument('--contract',choices=sorted(CONTRACTS))
    mode.add_argument('--mutation-audit',action='store_true')
    parser.add_argument('--report',type=Path,required=True)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    report = sensitivity(root) if args.mutation_audit else run(args.contract,root)
    args.report.write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report))
    raise SystemExit(0 if report['passed'] else 3)
