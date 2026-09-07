"""Strict parsing of API reports and detector positive controls."""
import hashlib
import json
from pathlib import Path
import re


def error_count(lines):
    values=[int(m.group(1)) for l in lines if (m:=re.fullmatch(r'========= ERROR SUMMARY: (\d+) errors?',l))]
    if len(values)!=1:raise ValueError('exactly one sanitizer error summary required')
    return values[0]


def validate_canaries(text,source):
    records=[json.loads(l) for l in text.splitlines() if l.startswith('{')]
    api=[r for r in records if r.get('mode')=='driver-torch-invalid']
    kernels=[r for r in records if 'tool' in r]
    source=Path(source)
    if len(api)!=1 or [r['tool'] for r in kernels]!=['memcheck','racecheck']:
        raise ValueError('API, memory and race detector controls required')
    a=api[0]
    expected={'========= Program hit CUDA_ERROR_INVALID_DEVICE (error 101) due to "invalid device ordinal" on CUDA API call to cuDeviceGet.':1}
    if (a['exit_code']!=99 or a['error_count']!=1 or a['api_reports']!=expected
            or a['program']['positive_control']!=dict(call='cuDeviceGet',code=101,intentional=True)):
        raise ValueError('deliberately invalid driver API was not detected exactly')
    lookup=hashlib.sha256((source/'glm53_cuda_driver_lookup_check.py').read_bytes()).hexdigest()
    kernel=hashlib.sha256((source/'glm53_sanitizer_order_canary.py').read_bytes()).hexdigest()
    if a['program']['source_sha256']!=lookup:
        raise ValueError('driver control source mismatch')
    for r in kernels:
        if r['exit_code']!=99 or r['deliberate_error_detected'] is not True or r['source_sha256']!=kernel:
            raise ValueError('deliberately bad device kernel was not detected or source differs')
    return dict(verdict='SANITIZER_DETECTOR_CONTROLS_PASS',controls=records)
