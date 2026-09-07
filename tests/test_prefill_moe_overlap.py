"""CPU checks for row routing, stream dependencies and vLLM layer identity."""
import ast
from contextlib import contextmanager
from dataclasses import dataclass, replace
from pathlib import Path
from types import SimpleNamespace, ModuleType
import sys
import unittest
from unittest.mock import patch

SOURCE = Path(__file__).resolve().parents[1] / 'overlay/modules/glm53_runtime/glm53_prefill_collectives.py'

@dataclass
class Context:
    no_compile_layers: dict
    all_moe_layers: object = None
    moe_layer_index: int = 0
    dp_metadata: object = None
    ubatch_slices: object = None

class OverlapTests(unittest.TestCase):
    def setUp(self):
        names = {'_mlp_overlap_slices', '_moe_overlap_context', 'prefill_moe_overlap'}
        tree = ast.parse(SOURCE.read_text())
        self.ns = dict(_TP=4, _MLP_OVERLAP=True, _FP8_MODE='3', _MLP_COMM_STREAMS={}, replace=replace)
        exec(compile(ast.Module([n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name in names], []), str(SOURCE), 'exec'), self.ns)
        self.runner = SimpleNamespace(layer_name='layer.7')
        self.parent = Context({'layer.7': self.runner}, ['layer.7', 'layer.8'])
        self.active = self.parent
        fc = ModuleType('vllm.forward_context')
        fc.get_forward_context = lambda: self.active
        @contextmanager
        def override(ctx):
            previous, self.active = self.active, ctx
            try: yield
            finally: self.active = previous
        fc.override_forward_context = override
        tu = ModuleType('vllm.utils.torch_utils'); tu._USE_LAYERNAME = False
        self.modules = patch.dict(sys.modules, {'vllm.forward_context': fc, 'vllm.utils.torch_utils': tu})
        self.modules.start(); self.addCleanup(self.modules.stop)

    def test_every_ragged_chunk_preserves_rank_row_order(self):
        for total in range(4096, 8193):
            stripes = self.ns['_mlp_overlap_slices'](total)
            rows = stripes[-1][1]
            for rank in range(4):
                restored = []
                for lo, hi, actual in stripes:
                    full = [r*rows+i for r in range(4) for i in range(lo,hi)]
                    valid = full[:actual]
                    self.assertTrue(all(x < total for x in valid))
                    self.assertEqual(len(valid), actual)
                    received = (valid+[None]*(4*(hi-lo)-actual))[rank*(hi-lo):(rank+1)*(hi-lo)]
                    restored.extend(received)
                self.assertEqual(restored, [r if r < total else None for r in range(rank*rows,(rank+1)*rows)])

    def test_layer_context_uses_explicit_identity_without_advancing_parent(self):
        parent, child, indexed = self.ns['_moe_overlap_context'](SimpleNamespace(experts=self.runner))
        self.assertTrue(indexed); self.assertIs(parent, self.parent)
        self.assertIsNone(child.all_moe_layers); self.assertIs(child.no_compile_layers['layer.7'], self.runner)
        self.assertEqual(parent.moe_layer_index, 0)
        self.parent.all_moe_layers = ['layer.8']
        with self.assertRaisesRegex(RuntimeError, 'out of order'):
            self.ns['_moe_overlap_context'](SimpleNamespace(experts=self.runner))
        self.parent.dp_metadata = object()
        self.assertIsNone(self.ns['_moe_overlap_context'](SimpleNamespace(experts=self.runner)))

    def test_schedule_and_lifetime_on_success_and_failure(self):
        for failing in (False, True):
            self.parent.moe_layer_index = 0
            events = []; current = ['compute']; outer = self
            class Stream:
                def __init__(self, name): self.name = name
                def wait_stream(self, s): events.append((self.name, 'wait_stream', s.name))
                def wait_event(self, e): events.append((self.name, 'wait_event', e.name))
            compute, comm = Stream('compute'), Stream('comm')
            @contextmanager
            def stream(s):
                previous, current[0] = current[0], s.name
                try: yield
                finally: current[0] = previous
            class Event:
                count = 0
                def __init__(self): self.name = Event.count; Event.count += 1
                def record(self, s): events.append((s.name, 'record', self.name))
            class Tensor:
                def __init__(self, rows, name): self.shape=(rows,4096); self.device='cuda:0'; self.name=name
                def __getitem__(self, s): return Tensor(s.stop-s.start, f'input:{s.start}:{s.stop}')
                def record_stream(self, s): events.append((self.name, 'retain', s.name))
            class MLP:
                experts = outer.runner
                calls = 0
                def __call__(self, x):
                    self.calls += 1
                    outer.assertIsNone(outer.active.all_moe_layers)
                    outer.assertIs(outer.active.no_compile_layers['layer.7'], self.experts)
                    events.append((current[0], 'moe', x.shape[0]))
                    if failing and self.calls == 2: raise RuntimeError('MoE failed')
                    return Tensor(x.shape[0], 'partial'+str(self.calls))
            @contextmanager
            def scope(**kw): yield
            def gather(x, **kw):
                events.append((current[0], 'ag', kw['num_tokens'], kw['_transport_tokens']))
                return Tensor(kw['num_tokens'], x.name.replace('input','gather'))
            def scatter(x, **kw):
                events.append((current[0], 'rs', x.shape[0], kw['_transport_tokens']))
                return Tensor((x.shape[0]+3)//4, 'reduced'+x.name[-1])
            self.ns.update(torch=SimpleNamespace(cuda=SimpleNamespace(
                is_current_stream_capturing=lambda:False, current_stream=lambda:compute,
                Stream=lambda **kw:comm, stream=stream, Event=Event),
                cat=lambda values,dim:Tensor(sum(v.shape[0] for v in values),'out')),
                _MLP_COMM_STREAMS={}, _check=lambda x:None, partial_tp_output=scope,
                prefill_all_gather=gather,prefill_reduce_scatter=scatter,
                logger=SimpleNamespace(info_once=lambda s:events.append(('launch',))))
            if failing:
                with self.assertRaisesRegex(RuntimeError,'MoE failed'):
                    self.ns['prefill_moe_overlap'](MLP(),Tensor(1729,'input'),num_tokens=6913)
                self.assertEqual(self.parent.moe_layer_index,0)
                self.assertNotIn(('launch',),events)
            else:
                result=self.ns['prefill_moe_overlap'](MLP(),Tensor(1729,'input'),num_tokens=6913)
                self.assertEqual(result.shape,(1729,4096)); self.assertEqual(self.parent.moe_layer_index,1)
                self.assertEqual([e[1] for e in events if len(e)>1 and e[1] in ('ag','rs')],['ag','ag','rs','rs'])
                self.assertEqual([e for e in events if len(e)>1 and e[1]=='moe'],[('compute','moe',3456),('compute','moe',3457)])
                self.assertIn(('comm','wait_stream','compute'),events)
                self.assertIn(('compute','wait_stream','comm'),events)
                self.assertIn(('partial1','retain','comm'),events)
                self.assertIn(('reduced1','retain','compute'),events)
            self.assertIs(self.active,self.parent)

    def test_disabled_short_capture_and_unsupported_transport_do_not_enter_context(self):
        self.ns['torch']=SimpleNamespace(cuda=SimpleNamespace(is_current_stream_capturing=lambda:False))
        self.ns['_moe_overlap_context']=lambda mlp: self.fail('context was touched')
        for flag, mode, size, capture in [(False,'3',6912,False),(True,'3',2128,False),(True,'3',4096,False),(True,'3',6143,False),(True,'3',8193,False),(True,'2',6912,False),(True,'3',6912,True)]:
            self.ns.update(_MLP_OVERLAP=flag,_FP8_MODE=mode)
            self.ns['torch'].cuda.is_current_stream_capturing=lambda:capture
            self.assertIsNone(self.ns['prefill_moe_overlap'](None,None,num_tokens=size))

if __name__ == '__main__': unittest.main()
