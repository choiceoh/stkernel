"""EP tiled canary CPU contracts; no Torch/CUDA imports or GPU work."""
import ast
from contextlib import redirect_stdout
import importlib.util
import io
import json
from pathlib import Path
import sys
import tempfile
from types import ModuleType, SimpleNamespace
import unittest
from unittest.mock import Mock, patch

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT/'overlay/modules/glm53_moe/glm53_ep_tiled_selftest.py'
PACKAGE = '_ep_tiled_canary_cpu'
package = ModuleType(PACKAGE)
package.__path__ = [str(SOURCE.parent)]
sys.modules[PACKAGE] = package
spec = importlib.util.spec_from_file_location(PACKAGE+'.glm53_ep_tiled_selftest', SOURCE)
canary = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = canary
spec.loader.exec_module(canary)


def runtime():
    return dict(torch=SimpleNamespace(cuda=SimpleNamespace(synchronize=Mock())),
        md=SimpleNamespace(_WORKSPACE_CACHE={'old': object()}, _WEIGHT_CACHE={}),
        device='cuda:0', provenance=dict(source='fixed'))


class LifecycleTests(unittest.TestCase):
    def setUp(self):
        canary._STATES.clear()
        self.owner, self.layer, self.rt = object(), object(), runtime()

    def capture(self, context, owner, layer, receipt):
        return dict(references=['CPU B1/B2'], buffers=['shared input'], original={})

    def test_two_hooks_publish_once_after_validation_sync_and_release(self):
        output = io.StringIO()
        events = []
        self.rt['torch'].cuda.synchronize.side_effect = lambda _: events.append('sync')
        with redirect_stdout(output), patch.object(canary,'_runtime',return_value=self.rt), patch.object(
                canary,'_memory',return_value={}), patch.object(
                canary,'_capture_references',side_effect=self.capture) as capture, patch.object(
                canary,'_validate_candidate',side_effect=lambda *a: events.append('validate')) as validate:
            handle = canary.before_relayout(self.owner,self.layer)
            self.assertEqual(output.getvalue(),'')
            self.assertEqual(handle['receipt']['verdict'],'RUNNING')
            receipt = canary.after_relayout(self.owner,self.layer,handle)
            cached = canary.before_relayout(object(),object())
            self.assertIs(canary.after_relayout(object(),object(),cached),receipt)
        self.assertEqual((capture.call_count,validate.call_count),(1,1))
        self.assertEqual(events,['validate','sync'])
        self.assertNotIn('references',handle)
        self.assertNotIn('buffers',handle)
        lines=output.getvalue().splitlines()
        self.assertEqual(len(lines),1)
        self.assertEqual(json.loads(lines[0].removeprefix('[ep-tiled-selftest] PASS ')),receipt)
        self.assertFalse(receipt['performance_acceptance'])
        self.assertFalse(receipt['full_sanitizer_acceptance'])

    def test_first_candidate_failure_retained_when_cleanup_also_fails(self):
        original=AssertionError('original peak numerical rejection')
        with redirect_stdout(io.StringIO()) as output, patch.object(canary,'_runtime',return_value=self.rt), patch.object(
                canary,'_memory',return_value={}), patch.object(canary,'_capture_references',side_effect=self.capture), patch.object(
                canary,'_validate_candidate',side_effect=original) as validate:
            handle=canary.before_relayout(self.owner,self.layer)
            self.rt['torch'].cuda.synchronize.side_effect=RuntimeError('cleanup sync failure')
            with self.assertRaisesRegex(RuntimeError,'readiness refused') as caught:
                canary.after_relayout(self.owner,self.layer,handle)
            with self.assertRaisesRegex(RuntimeError,'previously failed'):
                canary.before_relayout(self.owner,self.layer)
        self.assertIs(caught.exception.__cause__,original)
        self.assertEqual(validate.call_count,1)
        self.assertIn('original peak',handle['receipt']['error'])
        self.assertIn('cleanup sync',handle['receipt']['cleanup_error'])
        self.assertEqual(handle['receipt']['verdict'],'FAIL')
        self.assertNotIn('] PASS ',output.getvalue())

    def test_reference_failure_blocks_relayout_and_retry(self):
        with redirect_stdout(io.StringIO()), patch.object(canary,'_runtime',return_value=self.rt), patch.object(
                canary,'_memory',return_value={}), patch.object(
                canary,'_capture_references',side_effect=AssertionError('stock repeat unstable')) as capture:
            with self.assertRaisesRegex(RuntimeError,'stock reference failed'):
                canary.before_relayout(self.owner,self.layer)
            with self.assertRaisesRegex(RuntimeError,'previously failed'):
                canary.before_relayout(self.owner,self.layer)
        self.assertEqual(capture.call_count,1)
        self.assertEqual(next(iter(canary._STATES.values()))['verdict'],'FAIL')

    def test_pending_handle_cannot_be_reused_for_another_owner(self):
        with patch.object(canary,'_runtime',return_value=self.rt), patch.object(canary,'_memory',return_value={}), patch.object(
                canary,'_capture_references',side_effect=self.capture), patch.object(canary,'_validate_candidate') as validate:
            handle=canary.before_relayout(self.owner,self.layer)
            with self.assertRaisesRegex(RuntimeError,'pending owner'):
                canary.after_relayout(object(),self.layer,handle)
            with self.assertRaisesRegex(RuntimeError,'remains in progress'):
                canary.before_relayout(self.owner,self.layer)
        validate.assert_not_called()


class AdmissionTests(unittest.TestCase):
    def test_runtime_accepts_new_owner_without_old_micro_workspaces_or_old_knob(self):
        owner=SimpleNamespace(_use_ep=True,_ep_no_dummy=True,global_num_experts=288,
            num_local_experts=72,hidden_dim=4096,intermediate_size_per_partition=2048,
            topk=8,_kernel_num_experts=72,_activation_str='swigluoai_uninterleave',
            _swiglu_alpha=1.,_swiglu_beta=0.,_swiglu_limit=10.)
        torch=SimpleNamespace(__file__=str(SOURCE),__version__='test',cuda=SimpleNamespace(
            get_device_capability=lambda d:(12,1),is_current_stream_capturing=lambda:False))
        md=SimpleNamespace(__file__=str(SOURCE),_GLM53_EP_TILED=True,_GLM53_EP_PREFILL_LOCAL=False)
        local=SimpleNamespace(__file__=str(SOURCE),_stock=SimpleNamespace(__file__=str(SOURCE)),stock_contract_matches=lambda:True)
        decode=SimpleNamespace(__file__=str(SOURCE),ep_tiled_source_contract=Mock())
        prefix='flashinfer.fused_moe.cute_dsl.blackwell_sm12x.'
        modules={prefix+name:SimpleNamespace(__file__=str(SOURCE)) for name in
                 ('glm53_ep_tiled','moe_static_kernel_v4','moe_static_kernel_v5')}
        modules.update({prefix+'moe_static_ep_tiled':decode,'torch':torch,
            'flashinfer':SimpleNamespace(__file__=str(SOURCE),__version__='test'),
            'cuda.bindings':SimpleNamespace(__file__=None)})
        with tempfile.TemporaryDirectory() as temp:
            relative=Path('cuda_bindings-13.3.1.dist-info/METADATA')
            metadata=Path(temp)/relative;metadata.parent.mkdir();metadata.write_text('Version: 13.3.1\n')
            distribution=SimpleNamespace(files=[relative],version='13.3.1',locate_file=lambda p:Path(temp)/p)
            with patch.dict(sys.modules,{'torch':torch,PACKAGE+'.moe_dispatch':md,PACKAGE+'.moe_dynamic_ep_local':local}), patch.object(
                    canary.importlib,'import_module',side_effect=lambda name:modules[name]), patch.object(
                    canary.importlib.metadata,'distribution',return_value=distribution), patch.object(canary.inspect,'getfile',return_value=str(SOURCE)):
                context=canary._runtime(owner,SimpleNamespace(w13_weight=SimpleNamespace(device=SimpleNamespace(type='cuda'))))
                self.assertEqual(context['provenance']['versions']['cuda.bindings']['version'],'13.3.1')
                self.assertIsNone(context['provenance']['versions']['cuda.bindings']['path'])
                self.assertEqual(len(context['provenance']['source']),10)
                md._GLM53_EP_TILED=False
                with self.assertRaisesRegex(RuntimeError,'selection/source'):
                    canary._runtime(owner,SimpleNamespace(w13_weight=SimpleNamespace(device=SimpleNamespace(type='cuda'))))
        decode.ep_tiled_source_contract.assert_called_once()

    def test_routes_all_ranks_remote_changed_bounds_duplicates_and_full_cases(self):
        self.assertEqual([c[1] for c in canary.CASES],[6,12,24,32,33,2128,4096,6912,8192])
        for offset in (0,72,144,216):
            for changed in (False,True):
                remote=canary.route_rows(33,'remote',offset,changed)
                self.assertTrue(all(len(r)==8 and all(0<=e<288 and not offset<=e<offset+72 for e in r) for r in remote))
                concentrated=canary.route_rows(6912,'concentrated',offset,changed)
                self.assertTrue(all(offset<=e<offset+72 for r in concentrated for e in r))
                mixed=canary.route_rows(6,'mixed',offset,changed)[0]
                self.assertEqual(mixed[-1],mixed[-2]);self.assertIn(-1,mixed);self.assertIn(288,mixed)
        with self.assertRaises(ValueError):canary.route_rows(6,'unknown',0)

    def test_only_weight_byte_permutation_allowed_without_storage_replacement(self):
        before={name:dict(shape=[4],dtype='uint8',data_ptr=i,sha256='old')
                for i,name in enumerate(('w13','w2','sf1','fc1_alpha'))}
        after={name:dict(value) for name,value in before.items()}
        after['w13']['sha256']='permuted';after['w2']['sha256']='permuted'
        canary._same_weights_and_scales(before,after,relayout=True)
        for name,field,value in [('w13','data_ptr',999),('sf1','sha256','bad'),('fc1_alpha','shape',[8])]:
            changed={n:dict(v) for n,v in after.items()};changed[name][field]=value
            with self.assertRaises(AssertionError):canary._same_weights_and_scales(before,changed,relayout=True)
        with self.assertRaises(AssertionError):canary._same_weights_and_scales(before,after,relayout=False)

    def test_cache_namespace_and_native_shape_must_match(self):
        geom=lambda m,r,c:dict(fc1=(16,128,256),fc2=(16,256,128))
        decode=SimpleNamespace(ep_tiled_geometry=geom,_EP_TILED_KERNEL_CACHE={})
        owner=SimpleNamespace(_ep_tiled_workspace=SimpleNamespace(static=SimpleNamespace(max_rows=256),scratch=SimpleNamespace(max_active_clusters=48)))
        context=dict(decode=decode,md=SimpleNamespace(_DYNAMIC_KERNEL_CACHE={}))
        key=('glm53_ep_static_tiled_fp32_v1',6,256,48,'torch.int32',False,True,(16,128,256),(16,256,128),'nvfp4','raw_mma_scales','swigluoai_uninterleave',1.,0.,10.,'fp32_scatter')
        decode._EP_TILED_KERNEL_CACHE[key]=object()
        self.assertEqual(canary._cache_evidence(context,owner,6)['keys'],[repr(key)])
        with self.assertRaises(AssertionError):canary._cache_evidence(context,owner,7)
        dynamic=('dynamic','fp4','nvfp4',72,4096,2048,8,48,(128,128),'torch.int32',False,True,'swigluoai_uninterleave',1.,0.,10.,False,True,'glm53_ep_prefill_local_fp32_v2')
        context['md']._DYNAMIC_KERNEL_CACHE[dynamic]=object()
        self.assertEqual(canary._cache_evidence(context,owner,8192)['keys'],[repr(dynamic)])
        context['md']._DYNAMIC_KERNEL_CACHE={dynamic[:-2]+(False,dynamic[-1]):object()}
        with self.assertRaises(AssertionError):canary._cache_evidence(context,owner,8192)

    def test_source_uses_established_numeric_contract_and_bounded_graph_flow(self):
        old=sys.modules[PACKAGE+'.glm53_ep_local_selftest']
        self.assertIs(canary.compare,old.compare);self.assertIs(canary.check_control,old.check_control)
        self.assertEqual((canary.ROW_L2_FLOOR,canary.ROW_PEAK_FLOOR),(.02,.04))
        tree=ast.parse(SOURCE.read_text())
        calls=[ast.unparse(n.func) for n in ast.walk(tree) if isinstance(n,ast.Call)]
        for forbidden in ('torch.manual_seed','torch.cuda.manual_seed','torch.cuda.reset_peak_memory_stats','md.clear_sm120_moe_caches'):
            self.assertNotIn(forbidden,calls)
        for node in ast.walk(tree):
            if isinstance(node,ast.Call) and ast.unparse(node.func) in ('torch.rand','torch.randn'):
                self.assertIn('generator',[kw.arg for kw in node.keywords])
                self.assertIn('dtype',[kw.arg for kw in node.keywords])
        text=SOURCE.read_text()
        self.assertNotIn('from probes',text)
        self.assertIn('graph.replay() if graph is not None else call()',text)
        self.assertIn('identity != cell["inputs"][int(changed)]',text)


if __name__=='__main__':unittest.main()
