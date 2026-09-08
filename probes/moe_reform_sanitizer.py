"""Keep sanitizer faults fatal; identify pre-kernel CUDA symbol lookup probes."""
import re


def sanitizer_result(log, returncode):
    if returncode == 0:
        return dict(returncode=0, pre_kernel_api_lookup_errors=0)
    assert returncode == 77, returncode
    known = ('Program hit CUDA_ERROR_INVALID_VALUE (error 1) due to "invalid argument" '
             'on CUDA API call to cuGetProcAddress_v2.')
    errors = []
    summary = []
    compiled = []
    zero_races = False
    for index, line in enumerate(log.splitlines()):
        if 'Compiling CuTe-DSL kernel' in line:
            compiled.append(index)
        if not line.startswith('========= '):
            continue
        message = line[len('========= '):]
        if message == known:
            errors.append(index)
        elif re.fullmatch(r'ERROR SUMMARY: \d+ errors?', message):
            summary.append(int(message.split()[2]))
        elif message == 'RACECHECK SUMMARY: 0 hazards displayed (0 errors, 0 warnings)':
            zero_races = True
        elif (not message or message == 'COMPUTE-SANITIZER'
              or message.startswith('    Saved host backtrace ')
              or message.startswith('        Host Frame: ')):
            continue
        else:
            raise AssertionError(('unrecognized sanitizer diagnostic', message))
    assert errors and (summary == [len(errors)] or (not summary and zero_races)), (len(errors), summary)
    assert compiled and max(errors) < min(compiled), 'API error after kernel compilation'
    return dict(returncode=returncode, pre_kernel_api_lookup_errors=len(errors),
                scope='Only cuGetProcAddress_v2 lookup failures before the first kernel compile; full log retained.')
