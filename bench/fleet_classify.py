#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Explain fleet.sh's supplied classification without changing its policy."""
import argparse
import json
import os
from pathlib import Path
import re
import stat

GPU_PATTERN = r'ab-lever|start-glm53|deploy-overlays|run_mk_probe|run_megakernel_bench|docker run|--gpus|onepass\.py|bracket\.py|bench-dec|torch\.cuda|nvidia-smi|\.cu\b|cuda_'
CPU_PATTERN = r'MK_PROBE_NO_GPU=1|head_pack_accuracy_cpu|baseline\.py|judge\.py|test_logic\.py|b12x_static_compile_check|compile\.sh|nvcc |bash -n|^git |md5sum|proof\.py'
PYTHON_TOKENS = re.compile(r'torch\.cuda|\.cuda\(|device=.cuda|--gpus|docker run')
MAX_FILE_BYTES = 1024 * 1024
MAX_REASONS = 8


def explain(classification, command, environ=None):
    env = os.environ if environ is None else environ
    answer = dict(classification=classification, authoritative='fleet.sh classify_cmd',
                  evidence=[], limits=dict(reasons=MAX_REASONS, file_bytes=MAX_FILE_BYTES),
                  truncated_files=[], skipped_files=[])
    if classification == 'nogpu' and env.get('FLEET_REHEARSE') == '1':
        from fleet_onepass import validate
        try:
            reviewed_rehearsal = validate(command, os.getcwd(),
                env.get('FLEET_RUNNER_REPO') or env.get('REPO', os.getcwd()), env,
                rehearsal_only=True)
        except (OSError, ValueError):
            pass
        else:
            answer.update(reason='FLEET_REHEARSE=1 selects CPU execution for this verified canonical helper.',
                          evidence=[dict(source='environment', name='FLEET_REHEARSE', match='1',
                                         entry=reviewed_rehearsal['entry'])])
            return answer
    reviewed = {'bench/cpu_compile.py'}
    if env.get('REPO'):
        reviewed.add(str(Path(env['REPO']) / 'bench/cpu_compile.py'))
    if (len(command) >= 2 and re.fullmatch(r'python|python3|python3\..*', Path(command[0]).name)
            and command[1] in reviewed):
        answer.update(reason='The reviewed cpu_compile.py entrypoint compiles without device execution.',
                      evidence=[dict(source='argv', argv_indices=[0, 1], match=command[1])])
        return answer

    chunks, spans, offset = [], [], 0
    def append(content, location=None):
        nonlocal offset
        chunks.append(content)
        if location and content:
            spans.append((offset, offset + len(content), location))
        offset += len(content)

    for index, argument in enumerate(command):
        if index:
            append(' ')
        append(argument, dict(source='argv', argv_indices=[index], snippet=argument[:240]))
    for index, argument in enumerate(command):
        path = Path(argument)
        try:
            if not path.is_file():
                continue
            with path.open('rb') as stream:
                metadata = os.fstat(stream.fileno())
                if not stat.S_ISREG(metadata.st_mode):
                    continue
                data = stream.read(MAX_FILE_BYTES)
        except OSError as exc:
            answer['skipped_files'].append(dict(path=argument, reason=type(exc).__name__))
            continue
        if metadata.st_size > MAX_FILE_BYTES:
            answer['truncated_files'].append(argument)
        if b'\0' in data:
            answer['skipped_files'].append(dict(path=argument, reason='binary file'))
            continue
        append(' ')
        extracted = 0
        for number, line in enumerate(data.decode(errors='replace').splitlines(keepends=True), 1):
            if re.match(r'^\s*#', line):
                continue
            location = dict(source='file', path=str(path.resolve()), line=number,
                            argv_indices=[index], snippet=line.rstrip('\r\n')[:240])
            if argument.endswith('.py'):
                for match in PYTHON_TOKENS.finditer(line):
                    if extracted:
                        append('\n')
                    append(match.group(), location)
                    extracted += 1
                    if extracted == 3:
                        break
                if extracted == 3:
                    break
            else:
                append(line, location)
        # Shell command substitution removes trailing newlines. They do not
        # contain evidence but can otherwise change a following ^git match.
        while chunks and chunks[-1].endswith('\n'):
            removed = len(chunks[-1]) - len(chunks[-1].rstrip('\n'))
            chunks[-1] = chunks[-1].rstrip('\n')
            offset -= removed
            if spans and spans[-1][1] > offset:
                start, _, location = spans.pop()
                if start < offset:
                    spans.append((start, offset, location))
            if chunks[-1]:
                break
            chunks.pop()

    document = ''.join(chunks)
    pattern = GPU_PATTERN if classification == 'gpu' else CPU_PATTERN if classification == 'nogpu' else None
    if pattern:
        for match in re.finditer(pattern, document, re.MULTILINE):
            locations = [location for start, end, location in spans if start < match.end() and end > match.start()]
            evidence = dict(match=match.group())
            if locations and all(location['source'] == 'argv' for location in locations):
                evidence.update(source='argv', argv_indices=[i for location in locations for i in location['argv_indices']],
                                snippet=' '.join(command[i] for location in locations for i in location['argv_indices'])[:240])
            elif len(locations) == 1:
                evidence.update(locations[0])
            else:
                evidence.update(source='combined', locations=locations)
            if evidence not in answer['evidence']:
                answer['evidence'].append(evidence)
            if len(answer['evidence']) == MAX_REASONS:
                break
    if classification == 'gpu':
        answer['reason'] = 'GPU evidence takes precedence over CPU hints.'
    elif classification == 'nogpu':
        answer['reason'] = 'The classifier selected CPU evidence after checking for GPU evidence.'
    else:
        answer['reason'] = 'The classifier found no GPU or CPU evidence; fleet queues this as GPU unless --cpu is explicitly selected.'
    if pattern and not answer['evidence']:
        answer['reason'] += ' No matching reason was recovered within these bounded reads; the supplied classification remains authoritative.'
    return answer


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--classification', required=True, choices=('gpu', 'nogpu', 'unknown'))
    parser.add_argument('command', nargs=argparse.REMAINDER)
    args = parser.parse_args(argv)
    command = args.command[1:] if args.command[:1] == ['--'] else args.command
    print(json.dumps(explain(args.classification, command), ensure_ascii=False))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
