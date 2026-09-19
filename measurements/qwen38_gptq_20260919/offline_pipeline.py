"""One frozen campaign: fleet collect330 -> RTX 5050 packs -> fleet serve330.

Run on srv4. Transfers and packing never hold a fleet lease. This worker fails
closed and retains partial evidence; it does not retry a failed GPU experiment,
replace another worker, or promote the result into the production pack store.
"""
from __future__ import annotations

import argparse
import fcntl
import json
import os
from pathlib import Path
import shlex
import socket
import subprocess
import time

NODES = ('10.10.10.2', '10.10.10.1', '10.10.10.3', None)
LOG = '/home/choiceoh/glm53-logs/qwen38-gptq-330k-20260919'
OLDLOG = '/home/choiceoh/glm53-logs/qwen38-gptq-20260919'
PACK = '/home/choiceoh/glm53-cache/qwen38-gptq-330k-20260919'
OLDPACK = '/home/choiceoh/glm53-cache/qwen38-gptq-20260919'
COLLECT_SHA = '6253afe7779fee1601607e0842376d58b86d4d98'
HEAD_TREE = '/home/choiceoh/st-worktrees/qwen38-gptq-offline-4436'
LANE_ROOT = '/home/choiceoh/st-qwen38-gptq-330k-5050-4436'
SSH = ['ssh', '-n', '-o', 'BatchMode=yes', '-o', 'ConnectTimeout=10',
       '-o', 'ServerAliveInterval=30', '-o', 'ServerAliveCountMax=6']


def command(host, argv, *, capture=False):
    cmd = argv if host is None else SSH + ['choiceoh@' + host, shlex.join(map(str, argv))]
    return subprocess.run(cmd, check=True, text=True, stdout=subprocess.PIPE if capture else None).stdout


def read_json(host, path):
    return json.loads(command(host, ['cat', path], capture=True))


def copy(source, destination):
    subprocess.run(['rsync', '-a', '--partial', '--ignore-existing', '-e', shlex.join(SSH[:1] + SSH[2:]),
                    str(source), str(destination)], check=True)


def remote(host, path):
    return str(path) if host is None else 'choiceoh@' + host + ':' + str(path)


def run(args, report):
    if socket.gethostname().split('.')[0] != 'srv4':
        raise ValueError('run this transfer controller on srv4')
    from probes.qwen38_gptq_subset import file_sha
    from probes.qwen38_gptq_offline import check_fit, manifest_files
    code = args.code.resolve()
    if (code / 'source.sha').read_text().strip() != args.sha:
        raise ValueError('controller source receipt differs')
    if (code / 'engine.tree').read_text().strip() != args.engine_tree:
        raise ValueError('controller engine receipt differs')
    state = read_json(NODES[0], LOG + '/deferred-collect330/waiter-status.json')
    if state['source_sha'] != COLLECT_SHA or state['mode'] != 'collect330':
        raise ValueError('wrong fleet collection worker')
    actual = command(NODES[0], ['git', '-C', state['source_tree'], 'rev-parse', 'HEAD:engine'], capture=True).strip()
    if actual != args.engine_tree:
        raise ValueError('5050 and collection must use the exact same engine code')
    command(args.lane, ['install', '-d', '-m', '700', LANE_ROOT])
    command(args.lane, ['mkdir', '-p', LANE_ROOT + '/inputs', LANE_ROOT + '/code'])
    copy(str(code) + '/', remote(args.lane, LANE_ROOT + '/code/'))

    # This useful CPU/network work can finish while another session owns fleet.
    # The old audit supplies the tensor list/model identity, never old fit rows.
    report('preparing_dense_weights_and_heldout')
    for rank, host in enumerate(NODES):
        command(host, ['sudo', '-n', 'true'])
        output = args.root / 'inputs' / f'rank{rank}'
        output.mkdir(parents=True, exist_ok=True)
        export = f'/home/choiceoh/st-qwen38-gptq-330k-export-4436/rank{rank}'
        command(host, ['install', '-d', '-m', '700', export])
        copy(code / 'probes/qwen38_gptq_subset.py', remote(host, export + '/export.py'))
        command(host, ['python3', export + '/export.py', '--source',
                      f'/home/choiceoh/models/st-qwen38-tep4/rank{rank}of4.safetensors',
                      '--audit', OLDLOG + f'/fit-audit-rank{rank}.json', '--out', export + '/dense.safetensors'])
        copy(remote(host, export + '/dense.safetensors'), output / 'dense.safetensors')
        copy(remote(host, export + '/dense.json'), output / 'dense.json')
        copy(remote(host, OLDLOG + f'/heldout-audit-rank{rank}.json'), output / 'heldout-audit.json')
        hdir = output / 'heldout/mkcalib' / f'rank{rank}'
        hdir.mkdir(parents=True, exist_ok=True)
        copy(remote(host, OLDPACK + f'/heldout/mkcalib/rank{rank}/'), str(hdir) + '/')
        metadata = json.loads((output / 'dense.json').read_bytes())
        if file_sha(output / 'dense.safetensors') != metadata['sha256']:
            raise ValueError('checkpoint transfer failed')
        report('preparing_dense_weights_and_heldout', ranks_prepared=rank + 1)
    copy(str(args.root / 'inputs') + '/', remote(args.lane, LANE_ROOT + '/inputs/'))

    report('waiting_for_330k_statistics')
    deadline = time.time() + 12 * 3600
    while time.time() < deadline:
        state = read_json(NODES[0], LOG + '/deferred-collect330/waiter-status.json')
        if state['source_sha'] != COLLECT_SHA:
            raise ValueError('collection source changed')
        if state['state'] == 'finished':
            break
        if state['state'] in ('failed', 'refused', 'expired', 'cancelled'):
            raise RuntimeError('330K collection did not finish: ' + state['state'])
        report('waiting_for_330k_statistics', collector=state['state'], reason=state.get('reason', ''))
        time.sleep(30)
    else:
        raise TimeoutError('330K collection wait expired')
    completed = read_json(NODES[0], LOG + '/collection-complete.json')
    if completed['source_sha'] != COLLECT_SHA or min(completed['minimum_rows_by_rank']) < 330000:
        raise ValueError('incomplete 330K collection')
    report('transferring_330k_statistics', collection=completed)
    for rank, host in enumerate(NODES):
        output = args.root / 'inputs' / f'rank{rank}'
        copy(remote(host, LOG + f'/fit330-audit-rank{rank}.json'), output / 'fit330-audit.json')
        audit = json.loads((output / 'fit330-audit.json').read_bytes())
        check_fit(audit)
        if audit['weights_id'] != json.loads((output / 'dense.json').read_bytes())['weights_id']:
            raise ValueError('checkpoint changed between export and collection')
        (output / 'fit330').mkdir()
        copy(remote(host, PACK + '/fit330/'), str(output / 'fit330') + '/')
    copy(str(args.root / 'inputs') + '/', remote(args.lane, LANE_ROOT + '/inputs/'))
    report('packing_and_scoring_on_5050')
    command(args.lane, ['bash', LANE_ROOT + '/code/measurements/qwen38_gptq_20260919/run_5050.sh',
                        LANE_ROOT, args.sha, args.engine_tree])

    report('retrieving_and_installing_evaluated_packs')
    (args.root / 'results').mkdir()
    copy(remote(args.lane, LANE_ROOT + '/out/'), str(args.root / 'results') + '/')
    for rank, host in enumerate(NODES):
        output = args.root / 'results' / f'rank{rank}'
        manifest = json.loads((output / 'result.json').read_bytes())
        manifest_files(manifest, output / 'fit330/st-dense-packs')
        # Container-created cache directories are root-owned. Grant this task's
        # operator access to this isolated directory only; existing files stay.
        command(host, ['sudo', '-n', 'install', '-d', '-o', 'choiceoh', '-g', 'choiceoh',
                      '-m', '755', PACK + '/fit330/st-dense-packs'])
        # New GPTQ names have calibration hashes; --ignore-existing plus the
        # verifier refuses any collision rather than overwriting another file.
        copy(str(output / 'fit330/st-dense-packs') + '/', remote(host, PACK + '/fit330/st-dense-packs/'))
        copy(output / 'result.json', remote(host, LOG + f'/offline-result-rank{rank}.json'))
        toolroot = LOG + '/offline-tools'
        command(host, ['mkdir', '-p', toolroot + '/probes'])
        for filename in ('qwen38_gptq_install.py', 'qwen38_gptq_offline.py', 'qwen38_gptq_subset.py'):
            copy(code / 'probes' / filename, remote(host, toolroot + '/probes/' + filename))
        command(host, ['env', 'PYTHONPATH=' + toolroot, 'python3', '-m', 'probes.qwen38_gptq_install',
                      '--root', PACK + '/fit330', '--manifest', LOG + f'/offline-result-rank{rank}.json',
                      '--audit', LOG + f'/fit330-audit-rank{rank}.json',
                      '--out', LOG + f'/offline-installed-rank{rank}.json', '--engine-tree', args.engine_tree])
        receipt = args.root / 'results' / f'offline-installed-rank{rank}.json'
        copy(remote(host, LOG + f'/offline-installed-rank{rank}.json'), receipt)
        if rank != 0:
            copy(receipt, remote(NODES[0], LOG + f'/offline-installed-rank{rank}.json'))
            copy(output / 'result.json', remote(NODES[0], LOG + f'/offline-result-rank{rank}.json'))

    report('waiting_for_fleet_consumer_validation')
    # This separate clean worktree was prepared before dispatch. The collection
    # checkout stays at COLLECT_SHA throughout its own waiting/running window.
    actual = command(NODES[0], ['git', '-C', HEAD_TREE, 'rev-parse', 'HEAD'], capture=True).strip()
    if actual != args.sha:
        raise ValueError('consumer source checkout changed')
    command(NODES[0], ['python3', '-u', HEAD_TREE + '/measurements/qwen38_gptq_20260919/wait_for_window.py',
                      '--tree', HEAD_TREE, '--sha', args.sha, '--out', LOG + '/deferred-serve330',
                      '--mode', 'serve330', '--deadline-hours', '12'])
    ledger = command(NODES[0], ['cat', LOG + '/onepass.jsonl'], capture=True)
    (args.root / 'results/onepass.jsonl').write_text(ledger)
    rcs = [int(command(NODES[0], ['cat', LOG + f'/{arm}-onepass-{run}.rc'], capture=True))
           for arm in ('A1', 'B330', 'A2') for run in (1, 2)]
    report('finished', consumer_returncodes=rcs, consumer_all_passed=all(rc == 0 for rc in rcs),
           note='Results retained; no production pack promotion performed')


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root', type=Path, required=True)
    p.add_argument('--code', type=Path, required=True)
    p.add_argument('--lane', required=True)
    p.add_argument('--sha', required=True)
    p.add_argument('--engine-tree', required=True)
    args = p.parse_args()
    args.root.mkdir(parents=True, exist_ok=True, mode=0o700)
    lock = (args.root / 'pipeline.lock').open('w')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    def report(stage, **more):
        state = dict(stage=stage, time=time.time(), pid=os.getpid(), source_sha=args.sha,
                     engine_tree=args.engine_tree, lane=args.lane, **more)
        path = args.root / 'status.json'
        path.with_suffix('.tmp').write_text(json.dumps(state, indent=2) + '\n')
        path.with_suffix('.tmp').replace(path)
        print(json.dumps(state), flush=True)
    try:
        run(args, report)
    except BaseException as exc:
        report('failed', error=type(exc).__name__ + ': ' + str(exc))
        raise


if __name__ == '__main__':
    main()
