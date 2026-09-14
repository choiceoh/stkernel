import hashlib,json,os,subprocess,sys,unittest
from pathlib import Path
import torch
from probes.engine_ffn_packets_compile import fingerprint as packet_fingerprint
from probes.engine_mixed_tickets_check import fingerprint as mixed_fingerprint
modules = "test_engine_decode_bundle test_engine_decode_dsa_inputs test_engine_decode_indexer_gate test_engine_moe_activation_store test_engine_mixed_tickets test_engine_mixed_shared test_engine_moe_output test_engine_shared_mlp test_engine_native_execution test_engine_mixed_completion test_engine_moe_sf6_staging test_engine_mixed_experts test_engine_moe_scatter_config test_engine_moe_scatter_owner test_moe_static_sf6_direct test_engine_decode_projection test_engine_burst_decode test_engine_decode_fastpaths test_glm53_modelopt_serving test_engine_ffn_packets test_engine_prefill_fp8_consumer test_engine_compact_serving test_engine_execution_plans test_engine_execution test_engine_prefill_route_cache test_engine_token_shards test_engine_prefill_outputs test_engine_knobs test_engine_graph_contracts test_engine_warmup_draws test_engine_draft_tuning_integration test_engine_drafter test_engine_kernel_common test_engine_prefill_tiles test_glm53_prefill_collectives test_prefill_sum_pack test_fleet_onepass test_fleet_onepass_integration test_engine_early_observe test_engine_router_widths test_engine_graph_profile test_engine_prefill_covered_queries test_engine_prefill_absorb_tiles test_prefill_dense_prefix".split()
child = "import json,sys,unittest,torch; torch.set_num_threads(1); r=unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromName('tests.'+sys.argv[1])); print(json.dumps(dict(tests=r.testsRun,passed=r.testsRun-len(r.skipped)-len(r.errors)-len(r.failures),skipped=[(str(t),why) for t,why in r.skipped],failures=[(str(t),why) for t,why in r.errors+r.failures],status='PASS' if r.wasSuccessful() else 'FAIL'))); raise SystemExit(not r.wasSuccessful())"
results=[]
for module in modules:
 try:
  p=subprocess.run([sys.executable,'-c',child,module],capture_output=True,text=True,timeout=150,env=dict(os.environ,OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1'))
  Path('/out/'+module+'.log').write_text(p.stdout+p.stderr)
  row=json.loads(p.stdout.strip().splitlines()[-1]); row['module']=module
  if p.returncode: row['status']='FAIL'
 except Exception as error:
  row=dict(module=module,status='FAIL',tests=0,passed=0,skipped=[],failures=[str(error)])
 results.append(row)
 print(json.dumps({k:v for k,v in row.items() if k not in ('skipped','failures')}),flush=True)
sources=dict(packet_fingerprint(), **mixed_fingerprint())
sources['probes/engine_mixed_completion_check.py']=hashlib.sha256(Path('probes/engine_mixed_completion_check.py').read_bytes()).hexdigest()
for n in modules:
 p='tests/'+n+'.py'
 sources[p]=hashlib.sha256(Path(p).read_bytes()).hexdigest()
report=dict(scope='M2 ticket ownership and actual four-process Gloo agreement, bound shared readers, M1/S/P integration, rebased SF6 paths and fleet admission; isolated modules, no GPU exposed',source_revision=os.environ['TEST_REVISION'],torch=torch.__version__,gpu_available=torch.cuda.is_available(),cuda_initialized=torch.cuda.is_initialized(),tests=sum(r['tests'] for r in results),passed=sum(r['passed'] for r in results),skipped=sum(len(r['skipped']) for r in results),modules=results,source_sha256=sources,status='PASS' if all(r['status']=='PASS' for r in results) else 'FAIL')
Path('/out/cpu-postmerge.json').write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps({k:v for k,v in report.items() if k not in ('source_sha256','modules')}),flush=True)
raise SystemExit(report['status']!='PASS')
