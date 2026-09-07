#!/usr/bin/env python3
"""Real GLM CPU preprocessing equality before/after early warmup.

Run in the pinned image under a fleet reservation. This creates renderers,
not an LLM engine; all inspected preprocessing tensors must remain on CPU.
"""
import argparse
import base64
from collections.abc import Mapping
import dataclasses
import hashlib
import io
import json
import os
from pathlib import Path
import time
import traceback

import numpy as np
from PIL import Image
import torch


def normalized(value, tensors, path='root'):
    if isinstance(value, torch.Tensor):
        assert value.device.type == 'cpu', (path, value.device)
        raw = value.detach().contiguous().reshape(-1).view(torch.uint8).numpy()
        item = {'dtype': str(value.dtype), 'shape': list(value.shape),
                'stride': list(value.stride()), 'sha256': hashlib.sha256(raw).hexdigest()}
        tensors[path] = item
        return item
    if isinstance(value, np.ndarray):
        return {'dtype': str(value.dtype), 'shape': list(value.shape),
                'sha256': hashlib.sha256(np.ascontiguousarray(value)).hexdigest()}
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Mapping):
        # Request arrival time is metadata, not a token or preprocessing value.
        return {str(k): normalized(v, tensors, f'{path}.{k}') for k, v in sorted(value.items())
                if k != 'arrival_time'}
    if dataclasses.is_dataclass(value):
        return {'class': type(value).__qualname__, **{field.name: normalized(getattr(value, field.name), tensors, f'{path}.{field.name}')
                                                   for field in dataclasses.fields(value)}}
    if isinstance(value, (list, tuple)):
        return [normalized(v, tensors, f'{path}.{i}') for i, v in enumerate(value)]
    if isinstance(value, slice):
        return {name: normalized(getattr(value, name), tensors, f'{path}.{name}')
                for name in ('start', 'stop', 'step')}
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    raise TypeError(f'unsupported value {path}: {type(value)}')


def fingerprint(value):
    tensors = {}
    data = normalized(value, tensors)
    return {'sha256': hashlib.sha256(json.dumps(data, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest(),
            'tensors': tensors}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', default='/models/glm-5.3-flash-nvfp4')
    parser.add_argument('--template', required=True)
    parser.add_argument('--out', required=True)
    args = parser.parse_args()
    os.environ['VLLM_WORKER_MULTIPROC_METHOD'] = 'spawn'
    from vllm.engine.arg_utils import AsyncEngineArgs
    from vllm.renderers import ChatParams, renderer_from_config
    from vllm.renderers import glm53_renderer_warmup as helper
    from vllm.utils.torch_utils import set_default_torch_num_threads

    config = AsyncEngineArgs(model=args.model, trust_remote_code=True,
        max_model_len=1048576, tensor_parallel_size=4, nnodes=4,
        distributed_executor_backend='mp', limit_mm_per_prompt={'image': 4, 'video': 1}).create_engine_config()
    params = ChatParams(chat_template=Path(args.template).read_text(),
                        chat_template_content_format='auto', chat_template_kwargs={'thinking': False})
    buffer = io.BytesIO()
    Image.new('RGB', (64, 64), (255, 0, 0)).save(buffer, format='PNG')
    image_uri = 'data:image/png;base64,' + base64.b64encode(buffer.getvalue()).decode()
    messages = [[{'role': 'user', 'content': [{'type': 'text', 'text': '색상을 말하세요.'},
                   {'type': 'image_url', 'image_url': {'url': image_uri}}]}]]
    report = {'arms': {}, 'checks': 0, 'source_sha256': hashlib.sha256(Path(helper.__file__).read_bytes()).hexdigest()}
    for early in (0, 1):
        os.environ['VLLM_GLM53_EARLY_MM_WARMUP'] = str(early)
        renderer = renderer_from_config(config)
        results = {'warmup': [], 'requests': []}
        try:
            assert renderer.mm_processor is not None and renderer._readonly_mm_processor is not None
            for label, processor in (('main', renderer.mm_processor), ('readonly', renderer._readonly_mm_processor)):
                apply = processor.apply
                def recorded(*a, _apply=apply, _label=label, **kw):
                    try:
                        out = _apply(*a, **kw)
                        results['warmup'].append({'processor': _label, **fingerprint(out)})
                    except Exception:
                        traceback.print_exc()
                        raise
                    return out
                processor.apply = recorded
            before_threads = torch.get_num_threads()
            started = time.perf_counter()
            assert helper.start_renderer_warmup(renderer) == bool(early)
            renderer.warmup(params)
            results['warmup_wall_s'] = time.perf_counter() - started
            assert torch.get_num_threads() == before_threads
            report['checks'] += 1
            assert len(results['warmup']) == 2, results['warmup']
            for processor in (renderer.mm_processor, renderer._readonly_mm_processor):
                # Stop capturing warmup: ordinary requests are compared separately.
                processor.apply = processor.apply.__kwdefaults__['_apply']
            with set_default_torch_num_threads(1):
                for readonly in (False, True):
                    result = renderer.render_chat(messages, params, skip_mm_cache=readonly)
                    results['requests'].append({'readonly': readonly, **fingerprint(result)})
            assert all(row['tensors'] for row in results['warmup'] + results['requests'])
            report['arms'][str(early)] = results
        finally:
            renderer.shutdown()
    for key in ('warmup', 'requests'):
        baseline = report['arms']['0'][key]
        candidate = report['arms']['1'][key]
        assert baseline == candidate, f'{key} preprocessing differs'
        report['checks'] += len(baseline)
    report['ok'] = True
    Path(args.out).write_text(json.dumps(report, indent=2, sort_keys=True)+'\n')
    print(json.dumps({'ok': True, 'checks': report['checks'],
                      'tensor_fields': sum(len(x['tensors']) for x in report['arms']['0']['warmup'] + report['arms']['0']['requests'])}))


if __name__ == '__main__':
    main()
