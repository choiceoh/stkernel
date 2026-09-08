#!/usr/bin/env python3
"""One bounded local SF6 correctness gate under an owned supervised boot hold.

No reservation, serving control, timing, or remote execution is performed.
The owning campaign must stop the model containers on all four nodes first.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import re
import subprocess
import threading
import time

from run_ar_consumer_gpu import atomic_json
from run_gemm_input_reuse import memory_available
from run_moe_reform_cpu import IMAGE, ROOT, mounts

GIB = 1024 ** 3
SCHEMA = 'decode-sf6-gpu-v2'
HOLDER = Path('/home/choiceoh/glm53-logs/fleet/holder')
CASES = {(1, 8, False), (2, 16, False), (6, 8, False), (6, 40, False),
         (6, 48, False), (8, 8, False), (8, 40, False), (8, 64, False),
         (16, 40, False), (6, 40, True)}
HARNESS = ('probes/moe_reform_sf6_check.py', 'probes/run_decode_sf6_gpu.py',
           'probes/moe_decode_stream_probe.py',
           'probes/run_moe_reform_cpu.py', 'probes/run_gemm_input_reuse.py',
           'probes/run_ar_consumer_gpu.py', 'build/glm53/manifest.tsv')
KERNEL_FILES = ('moe_dispatch.py', 'moe_activation.py', 'moe_static_kernel.py',
                'moe_static_common.py', 'moe_static_kernel_v4.py', 'moe_static_kernel_v5.py',
                'moe_reform_sf_pack.py', 'moe_sf_pack.py', 'moe_dynamic_gated_tiled.py',
                'moe_dynamic_gated_sf6.py',
                'moe_dynamic_prefill.py', 'moe_dynamic_prefill_n128.py',
                'moe_micro_kernel.py', 'moe_dynamic_kernel.py')


def require(condition, message):
    if not condition:
        raise ValueError(message)


def digest(data):
    return hashlib.sha256(data).hexdigest()


def read_json(path):
    def unique(pairs):
        value = {}
        for key, item in pairs:
            require(key not in value, 'duplicate JSON field: ' + key)
            value[key] = item
        return value
    return json.loads(Path(path).read_bytes(), object_pairs_hook=unique)


def source_identity():
    def git(*args):
        return subprocess.check_output(['git', '-C', str(ROOT), *args], text=True).strip()
    require(not git('status', '--porcelain', '--untracked-files=all'), 'clean frozen source required')
    hashes = {name: digest((ROOT/name).read_bytes()) for name in HARNESS}
    kernels = {}
    for original in sorted((ROOT/'overlay/modules/glm53_moe').glob('*.py')):
        built = ROOT/'build/glm53'/original.name
        require(original.read_bytes() == built.read_bytes(), 'stale composed source: ' + original.name)
        kernels[original.name] = digest(built.read_bytes())
        for path in (original, built):
            name = str(path.relative_to(ROOT))
            require(git('ls-files', '--error-unmatch', name), 'untracked input: ' + name)
            hashes[name] = digest(path.read_bytes())
    return dict(revision=git('rev-parse', 'HEAD'), inputs_sha256=hashes, kernels_sha256=kernels)


def held_session():
    session = os.environ.get('FLEET_SESSION', '')
    require(re.fullmatch('[A-Za-z0-9_-]+', session), 'invalid fleet session')
    text = HOLDER.read_text().strip()
    fields = text.split('|')
    require(len(fields) >= 2 and fields[0] == session and fields[-1] == 'boot'
            and os.environ.get('FLEET_RESTORE_MANAGED') == '1',
            'matching owned supervised boot hold required')
    return session, text


def model_containers_stopped():
    names = subprocess.check_output(['docker', 'ps', '--format', '{{.Names}}'],
                                    text=True, timeout=10).splitlines()
    # Include suffixed workers/replicas, not just the leader's exact name.
    active = [name for name in names if re.match(r'^(glm53|dsv4)(?:[-_]|$)', name)]
    require(not active, 'model containers must already be stopped: ' + ','.join(active))


def check_environment(hold, minimum):
    require(held_session() == hold, 'owned supervised hold changed')
    model_containers_stopped()
    available = memory_available()
    require(available >= minimum, f'host MemAvailable below {minimum // GIB} GiB: {available}')
    return available


def validate_report(report, source):
    require(report.get('status') == 'PASS' and report.get('mode') == 'gpu', 'GPU PASS report required')
    require(report.get('expand_blocks') == [1024, 2048, 4096], 'all three device expansion layouts required')
    observed = report.get('source_sha256', {})
    require(isinstance(observed, dict) and all(isinstance(v, str) and re.fullmatch('[0-9a-f]{64}', v)
            for v in observed.values()), 'invalid probe source hashes')
    for name in KERNEL_FILES:
        require(observed.get(name) == source['kernels_sha256'][name], 'executed kernel mismatch: ' + name)
    gates = report.get('gates', [])
    require(isinstance(gates, list) and len(gates) == 13, 'complete 3 expansion + 10 MoE cases required')
    expansion, cases = set(), set()
    for gate in gates:
        if 'block' in gate:
            block = gate['block']
            require(block in (1024, 2048, 4096) and block not in expansion
                    and gate.get('exact_expand_replays') == 32, 'invalid device expansion evidence')
            expansion.add(block)
            continue
        key = gate.get('m'), gate.get('unique'), gate.get('raw_fallback')
        require(key in CASES and key not in cases and type(key[2]) is bool, 'invalid/duplicate MoE case')
        cases.add(key)
        m, _, fallback = key
        require(gate.get('moe_replays') == 8, 'all eight changing graph replays required')
        expected_lanes = {arm: dict(kind='stock' if arm == 'stock' else 'static_v2', rows=m,
                            reform=arm != 'stock' and 1 <= m <= 8,
                            sf6=arm == 'sf6' and not fallback)
                          for arm in ('stock', 'baseline', 'sf6')}
        require(gate.get('lanes') == expected_lanes, 'actual stock/baseline/SF6 lanes not proven')
        numeric = gate.get('numeric', [])
        require(isinstance(numeric, list) and len(numeric) == 16, 'all numerical comparisons required')
        pairs = set()
        for record in numeric:
            pair = record.get('replay'), record.get('arm')
            require(pair[0] in range(8) and pair[1] in ('baseline', 'sf6') and pair not in pairs,
                    'invalid/duplicate numerical comparison')
            pairs.add(pair)
            values = [record.get(k) for k in ('max_error', 'stock_noise', 'limit')]
            require(all(type(v) in (int, float) and math.isfinite(v) and v >= 0 for v in values),
                    'nonfinite or negative numerical evidence')
            require(values[0] <= values[2], 'GPU numerical bound exceeded')
    require(expansion == {1024, 2048, 4096} and cases == CASES, 'missing correctness cases')


def container_command(out, name, token):
    return ['docker', 'create', '--name', name, '--cidfile', str(out/'container.id'),
            '--label', 'decode.sf6.owner=' + token,
            '--gpus', 'device=0', '--network=none', '--cpuset-cpus=14-17',
            '--memory=16g', '--memory-swap=16g', '--shm-size=1g',
            '-e', 'MAX_JOBS=1', '-e', 'OMP_NUM_THREADS=1', '-e', 'MKL_NUM_THREADS=1',
            '-e', 'CUTE_DSL_ARCH=sm_121a', '-e', 'MK_PKG_PATH=/usr/local/lib/python3.12/dist-packages',
            '-e', 'XDG_CACHE_HOME=/evidence/cache', '-e', 'VLLM_CACHE_ROOT=/evidence/cache/vllm',
            '-e', 'CUDA_CACHE_PATH=/evidence/cache/cuda', '-e', 'TRITON_CACHE_DIR=/evidence/cache/triton',
            '-e', 'FLASHINFER_WORKSPACE_BASE=/evidence/cache/flashinfer',
            '-e', 'CUTE_DSL_CACHE_DIR=/evidence/cache/cute',
            '-e', 'TORCH_EXTENSIONS_DIR=/evidence/build', '-e', 'PYTHONDONTWRITEBYTECODE=1',
            '--mount', f'type=bind,src={ROOT},dst=/repo,readonly', *mounts(),
            '--mount', f'type=bind,src={out},dst=/evidence',
            '--mount', f'type=bind,src={out}/cache,dst=/root/.cache',
            '--workdir', '/repo', '--entrypoint', 'python3', IMAGE,
            '/repo/probes/moe_reform_sf6_check.py', '--gpu', '--out', '/evidence/result.json']


def inspect_owned(cid, token):
    data = json.loads(subprocess.check_output(['docker', 'inspect', cid], text=True, timeout=15))[0]
    require(data['Id'] == cid and data['Config']['Labels'].get('decode.sf6.owner') == token,
            'container identity/ownership mismatch')
    return data


def stop_owned(cid, token):
    inspect_owned(cid, token)
    subprocess.run(['docker', 'stop', '-t', '1', cid], check=True,
                   stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=15)


def validate_container(data, cid, token):
    require(data['Id'] == cid and data['Config']['Labels'].get('decode.sf6.owner') == token,
            'wrong container receipt')
    state, config = data['State'], data['HostConfig']
    require(data['Image'] == IMAGE and not state['Running'] and not state['OOMKilled']
            and state['ExitCode'] == 0 and not state.get('Error'), 'container failed or OOM killed')
    require(config['Memory'] == config['MemorySwap'] == 16 * GIB
            and config['CpusetCpus'] == '14-17' and config['NetworkMode'] == 'none'
            and config['ShmSize'] == GIB, 'container isolation/limits changed')
    requests = config.get('DeviceRequests', [])
    require(len(requests) == 1 and requests[0].get('DeviceIDs') == ['0'], 'GPU0 isolation required')


def verify_admission(out):
    """Read-only contract for a subsequent onepass; requires this frozen source."""
    out = Path(out)
    receipt = read_json(out/'admission.json')
    require(receipt.get('schema') == SCHEMA and receipt.get('status') == 'PASS'
            and receipt.get('image') == IMAGE and receipt.get('returncode') == 0
            and not receipt.get('issues'), 'fresh SF6 GPU admission required')
    source = read_json(out/'source.json')
    require(source == source_identity(), 'tested source changed before onepass')
    for name in ('source.json', 'result.json', 'probe.log', 'container.json', 'samples.json'):
        require(receipt.get('artifacts_sha256', {}).get(name) == digest((out/name).read_bytes()),
                'changed evidence: ' + name)
    validate_container(read_json(out/'container.json'), receipt['container_id'], receipt['owner_token'])
    validate_report(read_json(out/'result.json'), source)
    require('REFORM_SF6_CORRECTNESS_PASS' in (out/'probe.log').read_text(), 'missing probe PASS marker')
    samples = read_json(out/'samples.json')
    require(samples and samples[0]['available_bytes'] >= 24 * GIB
            and all(row['available_bytes'] >= 12 * GIB for row in samples), 'invalid memory guard evidence')
    return receipt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    out = args.out.resolve()
    hold = held_session()
    source = source_identity()
    image = json.loads(subprocess.check_output(['docker', 'image', 'inspect', IMAGE], text=True))[0]
    require(image['Id'] == IMAGE, 'pinned serving image identity required')
    initial = check_environment(hold, 24 * GIB)
    out.mkdir(parents=True, exist_ok=False)
    (out/'cache').mkdir()
    (out/'build').mkdir()
    atomic_json(out/'source.json', source)
    token = hold[0] + '-' + str(os.getpid()) + '-' + str(time.time_ns())
    name = 'decode-sf6-' + token
    receipt = dict(schema=SCHEMA, status='RUNNING', image=IMAGE, owner_token=token,
                   session=hold[0], holder=hold[1], source_commit=source['revision'], issues=[],
                   started_utc=datetime.now(timezone.utc).isoformat(), returncode=None)
    samples = [dict(elapsed_seconds=0, available_bytes=initial)]
    issues = receipt['issues']
    cid, thread = None, None
    done = threading.Event()
    started = time.monotonic()

    def watch():
        while not done.wait(.5):
            try:
                available = check_environment(hold, 12 * GIB)
                samples.append(dict(elapsed_seconds=time.monotonic() - started, available_bytes=available))
            except Exception as exc:
                issues.append('continuous guard: ' + repr(exc))
                try:
                    stop_owned(cid, token)
                except Exception as stop_error:
                    issues.append('owned stop: ' + repr(stop_error))
                return

    atomic_json(out/'admission.json', receipt)
    try:
        cid = subprocess.check_output(container_command(out, name, token), text=True, timeout=30).strip()
        require(re.fullmatch('[0-9a-f]{64}', cid), 'invalid owned container ID')
        receipt['container_id'] = cid
        inspect_owned(cid, token)
        check_environment(hold, 24 * GIB)
        thread = threading.Thread(target=watch, daemon=True)
        thread.start()
        require(not issues, 'continuous guard failed before launch')
        with (out/'probe.log').open('w') as log:
            receipt['returncode'] = subprocess.run(['docker', 'start', '--attach', cid], stdout=log,
                stderr=subprocess.STDOUT, timeout=900).returncode
        require(receipt['returncode'] == 0 and not issues, 'GPU probe or continuous guard failed')
        check_environment(hold, 12 * GIB)
        require(source_identity() == source, 'frozen source changed during GPU probe')
        validate_report(read_json(out/'result.json'), source)
        require('REFORM_SF6_CORRECTNESS_PASS' in (out/'probe.log').read_text(), 'missing probe PASS marker')
    except BaseException as exc:
        issues.append(repr(exc))
    finally:
        done.set()
        if thread is not None:
            thread.join(timeout=45)
            if thread.is_alive():
                issues.append('guard thread failed to finish')
        if cid is None and (out/'container.id').is_file():
            # docker create may time out after creating the container but before
            # returning stdout. Recover its ID, then require the owner label.
            recovered = (out/'container.id').read_text().strip()
            if re.fullmatch('[0-9a-f]{64}', recovered):
                cid = recovered
                receipt['container_id'] = cid
        if cid is not None:
            try:
                data = inspect_owned(cid, token)
                if data['State']['Running']:
                    stop_owned(cid, token)
                    data = inspect_owned(cid, token)
                # Preserve exit/OOM/identity before removing this exact owned ID.
                atomic_json(out/'container.json', data)
                validate_container(data, cid, token)
            except Exception as exc:
                issues.append('container diagnostics: ' + repr(exc))
            finally:
                try:
                    cleanup_data = inspect_owned(cid, token)
                    if cleanup_data['State']['Running']:
                        stop_owned(cid, token)
                        cleanup_data = inspect_owned(cid, token)
                    if not (out/'container.json').is_file():
                        atomic_json(out/'container.json', cleanup_data)
                    subprocess.run(['docker', 'rm', '-f', cid], check=True,
                        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=15)
                except Exception as exc:
                    issues.append('owned cleanup: ' + repr(exc))
        atomic_json(out/'samples.json', samples)
        receipt['artifacts_sha256'] = {p.name: digest(p.read_bytes()) for p in
            (out/'source.json', out/'result.json', out/'probe.log', out/'container.json', out/'samples.json')
            if p.is_file()}
        receipt.update(status='FAIL' if issues else 'PASS',
                       finished_utc=datetime.now(timezone.utc).isoformat())
        atomic_json(out/'admission.json', receipt)
    require(not issues, 'SF6 GPU gate failed; see ' + str(out/'admission.json'))
    try:
        verify_admission(out)
    except BaseException as exc:
        issues.append('final verification: ' + repr(exc))
        receipt['status'] = 'FAIL'
        atomic_json(out/'admission.json', receipt)
        raise
    print('PASS fresh SF6 GPU correctness: 3 expansion + 10 MoE cases; no speed verdict', flush=True)


if __name__ == '__main__':
    main()
