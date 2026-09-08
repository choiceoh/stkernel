"""Host reclaim contracts; real CPU tensors run in the pinned CPU-only image."""
import asyncio
import ctypes
import gc
import json
import os
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'probes'))
sys.path.insert(0, str(ROOT/'bench'))
import glm53_prefill_observer as observer
import prefill_observation_run as runner


def sample(initialized=True):
    return dict(process_kib={'Pss': 100}, host_kib={'MemTotal': 1000, 'MemAvailable': 200},
                cuda_initialized=initialized,
                pinned={'allocated_bytes.current': 128, 'active_bytes.current': 64} if initialized else {},
                device={'allocated': 1024, 'reserved': 2048} if initialized else {})


def report():
    def row(initialized=True):
        return dict(active=False, source_sha256='a'*64,
                    memory=dict(before=sample(initialized), after=sample(initialized),
                                pinned_release_called=initialized, gc_collected=0,
                                malloc_trim_result=1, gpu_cache_flushed=False))
    return dict(op='reclaim', ranks=[dict(rank=r, **row()) for r in range(4)], api=row(False))


class ReclaimTests(unittest.TestCase):
    def fake_torch(self, initialized=True, capturing=False):
        return SimpleNamespace(cuda=SimpleNamespace(is_initialized=lambda:initialized,
            is_current_stream_capturing=lambda:capturing, synchronize=Mock()),
            _C=SimpleNamespace(_host_emptyCache=Mock()))

    def test_reclaim_orders_synchronization_and_only_returns_unused_host_pages(self):
        t=self.fake_torch();calls=[]
        t.cuda.synchronize.side_effect=lambda:calls.append('sync')
        t._C._host_emptyCache.side_effect=lambda:calls.append('host')
        trim=Mock(side_effect=lambda _:calls.append('trim') or 1)
        with (patch.object(ctypes,'CDLL',return_value=SimpleNamespace(malloc_trim=trim)),
              patch.object(gc,'collect',side_effect=lambda:calls.append('gc') or 0),
              patch.object(observer,'memory_snapshot',side_effect=lambda _:calls.append('snapshot') or sample())):
            result=observer.reclaim_host_memory(t)
        self.assertEqual(calls,['sync','snapshot','gc','host','trim','snapshot'])
        self.assertFalse(result['gpu_cache_flushed'])
        trim.assert_called_once_with(0)

    def test_missing_api_and_capture_refuse_before_collection_or_release(self):
        for missing in ('pinned','trim','capture'):
            t=self.fake_torch(capturing=missing=='capture')
            if missing=='pinned':t._C._host_emptyCache=None
            trim=Mock();libc=SimpleNamespace() if missing=='trim' else SimpleNamespace(malloc_trim=trim)
            with patch.object(ctypes,'CDLL',return_value=libc),patch.object(gc,'collect') as collect:
                with self.assertRaises(RuntimeError):observer.reclaim_host_memory(t)
                collect.assert_not_called();trim.assert_not_called();t.cuda.synchronize.assert_not_called()

    def test_api_process_does_not_initialize_or_flush_cuda(self):
        t=self.fake_torch(False);trim=Mock(return_value=0)
        with (patch.object(ctypes,'CDLL',return_value=SimpleNamespace(malloc_trim=trim)),
              patch.object(gc,'collect',return_value=0),
              patch.object(observer,'memory_snapshot',return_value=sample(False))):
            result=observer.reclaim_host_memory(t)
        self.assertFalse(result['pinned_release_called'])
        t.cuda.synchronize.assert_not_called();t._C._host_emptyCache.assert_not_called()

    def test_exact_worker_source_and_complete_counters_required(self):
        observer.validate_memory_report(report(),'a'*64)
        cases=[]
        x=report();x['ranks'][0]['rank']=False;cases.append(x)
        x=report();x['ranks'][0]['active']=True;cases.append(x)
        x=report();x['api']['source_sha256']='b'*64;cases.append(x)
        x=report();x['ranks'][1]['memory']['after']['pinned']={};cases.append(x)
        x=report();x['ranks'][1]['memory']['after']['host_kib']['MemAvailable']=-1;cases.append(x)
        x=report();x['ranks'][1]['memory']['gpu_cache_flushed']=True;cases.append(x)
        for bad in cases:
            with self.assertRaises(ValueError):observer.validate_memory_report(bad,'a'*64)

    def test_real_cpu_tensor_survives_heap_reclaim_without_cuda_initialization(self):
        import torch
        self.assertFalse(torch.cuda.is_initialized())
        self.assertTrue(callable(torch._C._host_emptyCache))
        x=torch.arange(65536,dtype=torch.int64);expected=x.clone();pointer=x.data_ptr()
        result=observer.reclaim_host_memory(torch)
        self.assertEqual(x.data_ptr(),pointer)
        self.assertTrue(torch.equal(x,expected))
        self.assertFalse(torch.cuda.is_initialized())
        self.assertFalse(result['pinned_release_called'])
        self.assertIn('Pss',result['after']['process_kib'])

    def test_middleware_reclaims_api_only_after_successful_worker_rpc(self):
        for failed in (False,True):
            calls=[]
            class Engine:
                async def collective_rpc(self,*args,**kwargs):
                    calls.append('workers')
                    if failed:raise RuntimeError('partial worker RPC failure')
                    return report()['ranks']
            class Request:
                url=SimpleNamespace(path='/glm53/prefill-observe')
                method='POST'
                client=SimpleNamespace(host='127.0.0.1')
                app=SimpleNamespace(state=SimpleNamespace(engine_client=Engine()))
                async def json(self):return {'op':'reclaim'}
            def reclaim(_):calls.append('api');return report()['api']['memory']
            with patch.object(observer,'reclaim_host_memory',side_effect=reclaim):
                response=asyncio.run(observer.middleware(Request(),Mock()))
            self.assertEqual(response.status_code,500 if failed else 200)
            self.assertEqual(calls,['workers'] if failed else ['workers','api'])
            if not failed:self.assertEqual(json.loads(response.body)['api']['memory'],report()['api']['memory'])

    def test_default_skips_reclaim_and_bad_report_never_reaches_prime(self):
        for enabled in (False,True):
            with tempfile.TemporaryDirectory() as tmp,patch.dict(os.environ,{'FLEET_SESSION':'cpu'}):
                run=runner.Run(ROOT,'frozen',Path(tmp),reclaim_host_memory=enabled)
                run.sha='a'*64;ops=[]
                class API:
                    def post(self,path,payload):
                        ops.append(payload['op'])
                        if payload['op']=='reclaim':return dict(report(),ranks=[])
                        return dict(ranks=[dict(rank=r,active=False,source_sha256='a'*64) for r in range(4)])
                with (patch.object(run,'host'),patch.object(run,'ready'),patch.object(run,'snapshot',return_value={}),
                      patch.object(run,'attest_clone'),patch.object(run,'cleanup') as cleanup,
                      patch.object(runner.lifecycle,'idle'),patch.object(runner,'PrivateObserverAPI',return_value=API()),
                      patch.object(run,'phase',side_effect=RuntimeError('stop at PRIME')) as phase):
                    with self.assertRaises((RuntimeError,ValueError)):run.collect({}, {})
                    cleanup.assert_called_once()
                    if enabled:
                        phase.assert_not_called();self.assertEqual(ops,['status','reclaim'])
                        self.assertTrue((Path(tmp)/'host-memory-reclaim.json').exists())
                    else:
                        phase.assert_called_once_with('baseline','PRIME');self.assertEqual(ops,['status'])


if __name__=='__main__':unittest.main()
