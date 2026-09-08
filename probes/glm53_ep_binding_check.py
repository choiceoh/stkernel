#!/usr/bin/env python3
"""Separate post-context CUDA binding initialization from MoE execution."""
from __future__ import annotations

import argparse
import importlib.metadata
import json
from pathlib import Path


def observe(result):
    import torch
    result['cuda_initialized_before'] = torch.cuda.is_initialized()
    if result['cuda_initialized_before']:
        raise RuntimeError('fresh process with no existing CUDA context required')
    result['phase'] = 'torch-context'
    print('EP_BINDING_CONTEXT_BEGIN', flush=True)
    # Allocation opens the same Torch context needed by the original probe.
    # No model, MoE, CuTe or arithmetic kernel is invoked.
    keepalive = torch.empty((1,), device='cuda')
    torch.cuda.synchronize()
    print('EP_BINDING_CONTEXT_READY', flush=True)
    result['cuda_initialized_after'] = torch.cuda.is_initialized()
    result['phase'] = 'binding-device-count'
    print('EP_BINDING_DEVICE_COUNT_BEGIN', flush=True)
    from cuda.bindings import driver
    count_code, count = driver.cuDeviceGetCount()
    print('EP_BINDING_DEVICE_COUNT_END', flush=True)
    version_code, version = driver.cuDriverGetVersion()
    result.update(cuda_bindings=importlib.metadata.version('cuda-bindings'),
                  count_result=[int(count_code), count],
                  version_result=[int(version_code), version],
                  tensor_numel=keepalive.numel(), verdict='OBSERVED', phase='complete')
    if int(count_code) or int(version_code):
        raise RuntimeError('binding device-count/version call returned an error')


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--output', type=Path, required=True)
    args = ap.parse_args()
    result = dict(verdict='RUNNING', phase='prepare', performance_acceptance=False,
                  scope='Torch context then first binding device-count; no MoE or CuTe')
    try:
        observe(result)
    except BaseException as exc:
        result.update(verdict='FAIL', error=repr(exc))
        raise
    finally:
        args.output.write_text(json.dumps(result, indent=2)+'\n')
        print(json.dumps(result), flush=True)


if __name__ == '__main__':
    main()
