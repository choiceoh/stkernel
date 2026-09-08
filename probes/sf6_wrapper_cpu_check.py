#!/usr/bin/env python3
"""CPU-only: call the installed, decorated SF6 wrapper with a mocked launch.

Run inside the ordinary isolated serving-image CPU gate with all MoE overlays
mounted. This checks Python API/logging compatibility and owner forwarding;
it does not execute kernels or establish numerical or CUDA-graph correctness.
"""
import argparse
from contextlib import ExitStack
import hashlib
import importlib
import inspect
import json
import os
from pathlib import Path
import sys
from types import SimpleNamespace
from unittest.mock import Mock, patch


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--cpu', action='store_true', required=True)
    parser.add_argument('--out', type=Path)
    args = parser.parse_args()
    assert not os.environ.get('CUDA_VISIBLE_DEVICES'), 'CPU gate must hide CUDA devices'
    assert os.environ.get('NVIDIA_VISIBLE_DEVICES', 'void') in ('void', 'none', '')
    os.environ.setdefault('CUTE_DSL_ARCH', 'sm_121a')
    sys.path.insert(0, os.environ.get('MK_PKG_PATH', '/usr/local/lib/python3.12/dist-packages'))
    import torch
    assert not torch.cuda.is_initialized(), 'CUDA already initialized before CPU gate'
    # Match the existing device-free CuTe compile gate's import-only capability
    # shim. Every actual CUDA initialization is forbidden throughout this probe.
    with ExitStack() as context:
        context.enter_context(patch.object(torch.cuda, 'is_available', return_value=True))
        context.enter_context(patch.object(torch.cuda, 'get_device_capability', return_value=(12, 1)))
        context.enter_context(patch.object(torch.cuda, '_lazy_init', side_effect=AssertionError('CPU gate initialized CUDA')))
        wrapper_module = importlib.import_module('flashinfer.fused_moe.cute_dsl.b12x_moe')
        dispatch = importlib.import_module('flashinfer.fused_moe.cute_dsl.blackwell_sm12x.moe_dispatch')
        api_logging = importlib.import_module('flashinfer.api_logging')
        trace_module = importlib.import_module('flashinfer.trace.templates.moe')
        wrapper_class = wrapper_module.B12xMoEWrapper
        signature = inspect.signature(wrapper_class.run)
        assert '_weight_views' in signature.parameters, 'installed wrapper lacks explicit packed owner ABI'
        report = dict(mode='cpu', status='RUNNING', cases=[],
            scope='actual decorated wrapper, mocked dispatcher; no kernel execution',
            run_signature=str(signature),
            trace_template_type=type(trace_module.b12x_moe_wrapper_run_trace).__qualname__,
            source_sha256={Path(p).name: hashlib.sha256(Path(p).read_bytes()).hexdigest()
                           for p in (wrapper_module.__file__, api_logging.__file__, trace_module.__file__, __file__)})

        def raw_path(*args, **kwargs):
            raise AssertionError('packed-only wrapper touched raw scale preparation')

        for name in ('_get_weight_views', 'static_v2_weights_layout', 'static_v2_weights_sf_pack',
                     'static_v2_weights_reform_sf_pack', '_sf6_tensor_version',
                     '_pad_intermediate_to_tile', 'is_gated_activation'):
            context.enter_context(patch.object(dispatch, name, side_effect=raw_path))
        selection = context.enter_context(patch.object(dispatch, 'select_sm120_moe_backend'))
        launch = context.enter_context(patch.object(dispatch, 'launch_sm120_moe',
            side_effect=lambda **kwargs: kwargs['scatter_output']))
        # No constructor, capability query, workspace allocation, JIT or launch.
        wrapper = wrapper_class.__new__(wrapper_class)
        for name, value in dict(use_cuda_graph=True, max_num_tokens=8192,
                hidden_size=16, output_dtype=torch.bfloat16, quant_mode='nvfp4',
                _dynamic_workspace=object(), _static_workspace=object(), device=torch.device('cpu'),
                num_experts=2, num_local_experts=2, top_k=2, activation='swigluoai_uninterleave',
                swiglu_alpha=1., swiglu_beta=0., swiglu_limit=10., activation_precision='fp4',
                source_format='modelopt', intermediate_size=128,
                _weight_views=None, _weight_key=None, _padded_weights=None,
                _padded_weight_key=None, _folded_w1_alpha=None, _folded_w1_alpha_key=None).items():
            setattr(wrapper, name, value)
        scales = SimpleNamespace(enabled=True, fc1=torch.zeros((1,1552), dtype=torch.uint8),
                                 fc2=torch.zeros((1,1552), dtype=torch.uint8))
        owner = SimpleNamespace(packed_only=True, reform_scales=scales,
            sfb1_packed=scales.fc1, sfb2_packed=scales.fc2,
            w1_scale_storage=None, w2_scale_storage=None, _w13_sf_storage=None,
            _down_sf_storage=None, sfb_w13_ptr=None, sfb_down_ptr=None)
        weights = dict(w1_weight=torch.zeros((2,256,8), dtype=torch.uint8), w1_weight_sf=None,
            w2_weight=torch.zeros((2,16,64), dtype=torch.uint8), w2_weight_sf=None,
            w1_alpha=torch.ones(2), w2_alpha=torch.ones(2), fc2_input_scale=torch.ones(2),
            _weight_views=owner)
        for rows, backend in ((6, 'static'), (16, 'static'), (513, 'dynamic'), (4096, 'dynamic')):
            selection.return_value = backend
            arguments = dict(weights, x=torch.zeros((rows,16), dtype=torch.bfloat16),
                token_selected_experts=torch.zeros((rows,2), dtype=torch.int32),
                token_final_scales=torch.ones((rows,2)),
                out=torch.empty((rows,16), dtype=torch.bfloat16))
            launch.reset_mock()
            with patch.object(torch, 'empty', side_effect=AssertionError('wrapper allocated torch.empty')), \
                 patch.object(torch, 'zeros', side_effect=AssertionError('wrapper allocated torch.zeros')):
                for replay in range(3):
                    arguments['x'].fill_(replay)
                    arguments['token_selected_experts'].fill_(replay % 2)
                    result = wrapper.run(**arguments)
                    assert result.data_ptr() == arguments['out'].data_ptr()
            assert launch.call_count == 3, (rows, launch.call_count)
            for call in launch.call_args_list:
                kwargs = call.kwargs
                assert kwargs['_weight_views'] is owner
                assert kwargs['_workspace'] is getattr(wrapper, '_' + backend + '_workspace')
                assert kwargs['w1_weight_sf'] is None and kwargs['w2_weight_sf'] is None
                assert kwargs['a'] is arguments['x']
                assert kwargs['topk_ids'] is arguments['token_selected_experts']
            assert wrapper._weight_views is None and wrapper._weight_key is None
            assert wrapper._padded_weights is None and wrapper._folded_w1_alpha is None
            report['cases'].append(dict(rows=rows, backend=backend, repeats=3,
                                        raw_scales=None, explicit_owner=True, status='PASS'))
        # Fail-closed argument checks through the same installed decorator.
        for replacement in ({'_weight_views': None}, {'w1_weight_sf': torch.ones(1)},
                            {'w2_weight_sf': torch.ones(1)}, {'input_global_scale': torch.ones(1)}):
            before = launch.call_count
            try:
                wrapper.run(**dict(arguments, **replacement))
            except ValueError:
                pass
            else:
                raise AssertionError(('invalid packed-only call accepted', tuple(replacement)))
            assert launch.call_count == before, 'invalid packed-only call reached dispatcher'
        report['rejected_invalid_cases'] = 4
        assert len(report['cases']) == 4
        assert not torch.cuda.is_initialized(), 'CPU wrapper check initialized CUDA'
        report.update(status='PASS', cuda_initialized=False)
    encoded = json.dumps(report, indent=2, sort_keys=True) + '\n'
    if args.out:
        args.out.write_text(encoded)
    print(encoded, end='', flush=True)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
