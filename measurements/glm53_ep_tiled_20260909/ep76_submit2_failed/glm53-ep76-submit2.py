#!/usr/bin/env python3
"""Unbound EP76 normal fleet submission; bind descriptor only after new CPU receipt passes."""
import json, re, shlex, subprocess
from pathlib import Path
CONFIG=Path('/tmp/glm53-ep76-onepass2-submit.json')
p=json.loads(CONFIG.read_text())
for key in ('source','session','revision','cpu_output','cpu_host'):
    if not isinstance(p.get(key),str) or '_TO_BIND' in p[key]:
        raise SystemExit('bind '+key+' before submission')
if (not re.fullmatch('[0-9a-f]{40}',p['revision'])
    or not re.fullmatch('/home/choiceoh/[A-Za-z0-9._/-]+',p['source'])
    or not re.fullmatch('/home/choiceoh/[A-Za-z0-9._/-]+',p['cpu_output'])
    or '..' in Path(p['source']).parts or '..' in Path(p['cpu_output']).parts
    or not re.fullmatch('[A-Za-z0-9][A-Za-z0-9_-]{1,63}',p['session'])
    or p['cpu_host'] not in ('10.10.10.1','10.10.10.2','10.10.10.3','10.10.10.4')):
    raise SystemExit('invalid exact source/CPU/session binding')
if p.get('candidate_cache_tag') != 'glm53_ep_static_sf6_fc1_register_v2':
    raise SystemExit('bind the reviewed v2 candidate cache tag')
cmd=p['command']
if (p['job']!='/tmp/glm53-ep76-onepass-0910-2'
    or p['sequence']!=['B','A'] or p['optional_normal_chain_arm']!='ABASE'
    or cmd[:6]!=['bash',p['source']+'/bench/fleet.sh','run','--gpu','--detach',p['session']]
    or any('_TO_BIND' in arg for arg in cmd)
    or cmd[-4:]!=['bash',p['source']+'/bench/chain.sh','EPDECODE76B=','EPDECODE76A=VLLM_GLM53_EP_DECODE_OPT=1']
    or any(cmd.count(x)!=1 for x in ('SPEC_K=5','ENABLE_EP=1','VLLM_GLM53_EP_TILED=1','VLLM_GLM53_TP_SF6_Q0=0','VLLM_GLM53_EP_DECODE_OPT=0','VLLM_GLM53_PREP_FUSED=1','ONEPASS_REQUIRE_EP_TILED=1','ONEPASS_REQUIRE_PREP_FUSED=1','ONEPASS_FIXED_DECODE_TOKENS=1024','ONEPASS_FIXED_DECODE_REPS=3','REPO='+p['source']))):
    raise SystemExit('bound EP76 canonical command differs')
if (type(p.get('cpu_expected_tests')) is not int or p['cpu_expected_tests']<=0
    or type(p.get('cpu_expected_lowerings')) is not int or p['cpu_expected_lowerings']<=0
    or not isinstance(p.get('cpu_expected_pass_groups'),dict) or not p['cpu_expected_pass_groups']
    or any(not re.fullmatch('[a-z_]+_passes',k) or type(v) is not int or v<=0 for k,v in p['cpu_expected_pass_groups'].items())
    or sum(p['cpu_expected_pass_groups'].values())!=p['cpu_expected_lowerings']):
    raise SystemExit('bind exact final CPU schema/counts after source freeze')
remote="\nimport json,os,pathlib,shlex,subprocess,sys\np=json.load(sys.stdin); source=p['source']\nassert subprocess.check_output(['git','-C',source,'rev-parse','HEAD'],text=True).strip()==p['revision']\nassert not subprocess.check_output(['git','-C',source,'status','--porcelain'],text=True).strip()\nsys.path.insert(0,source+'/probes')\nfrom glm53_ep_tiled_compile import source_receipt, EXPECTED_CPU_TESTS, CPU_TEST_COUNTS, opt_shared_capacity\ncpu_result_raw=subprocess.check_output(['ssh','-o','BatchMode=yes','choiceoh@'+p['cpu_host'],'cat '+shlex.quote(p['cpu_output']+'/result.json')])\nresult=json.loads(cpu_result_raw)\nassert all(result[k]==v for k,v in source_receipt(pathlib.Path(source)).items())\nkernel_source=pathlib.Path(source+'/overlay/modules/glm53_moe/moe_static_ep_tiled.py').read_text()\nimport ast\nkernel_ast=ast.parse(kernel_source)\nassert any(isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='EP_TILED_DECODE_OPT_CACHE_TAG' for t in n.targets) and ast.literal_eval(n.value)==p['candidate_cache_tag'] for n in kernel_ast.body)\nassert result['verdict']=='PASS' and result['contracts']==dict(tests_run=EXPECTED_CPU_TESTS,failures=0,errors=0,skips=0)\nassert result['cuda_initialized'] is False and result['binding_runtime_rechecked'] is True\nassert result['phase']=='complete' and result['compile_only'] is True and result['contracts_process_isolated'] is True\nassert p['cpu_expected_tests']==EXPECTED_CPU_TESTS and result['selected_test_counts']==CPU_TEST_COUNTS\nassert not any(k in result for k in ('error','cleanup_error','recheck_error'))\nassert {k:len(v) for k,v in result.items() if k.endswith('_passes')}==p['cpu_expected_pass_groups']\ndef verify_native_mapping_result(result):\n    fixed = dict(proven=True, threads=128, raw_stage_bytes=2048, num_k_blocks=4,\n                 word_coverage_bytes=2048, stages=2, stage_stride_bytes=2048,\n                 offset_engine='static_scalar_physical_layout', slot_zero_relative_offsets=True)\n    expected_arms = ['opt-static/M6-local', 'opt-static/M4-map288-i32',\n                     'opt-static/M6-map288-i32', 'opt-static/M8-map288-i32']\n    passes = result['opt_static_passes']\n    assert type(passes) is list and [item['arm'] for item in passes] == expected_arms\n    for item in passes:\n        receipt = item['register_layout']\n        assert type(receipt) is dict and set(receipt) == set(fixed) | {'copy_shape', 'words_per_thread'}\n        assert all(type(receipt[key]) is type(value) and receipt[key] == value for key,value in fixed.items())\n        assert type(receipt['copy_shape']) is str and receipt['copy_shape']\n        words = receipt['words_per_thread']\n        assert type(words) is list and words and all(type(x) is int and x > 0 for x in words)\n        assert words == sorted(set(words))\n        capacity = item['shared_capacity']\n        assert type(capacity) is dict and set(capacity) == {'dynamic_bytes', 'static_bytes', 'total_bytes', 'block_limit_bytes'}\n        assert all(type(value) is int for value in capacity.values())\n        assert capacity['dynamic_bytes'] == 98304 and 0 <= capacity['static_bytes'] <= 1024\n        assert capacity['total_bytes'] == capacity['dynamic_bytes'] + capacity['static_bytes'] <= 101376\n        assert capacity['block_limit_bytes'] == 101376 and capacity == opt_shared_capacity(item)\n        assert item['specialization']['decode_opt'] is True and item['specialization']['storage_bytes'] == 98304\n        assert item['cache_key'][-1] == 'glm53_ep_static_sf6_fc1_register_v2'\n        assert len(item['cache_key']) == (20 if item['arm'] == 'opt-static/M6-local' else 24)\n\nverify_native_mapping_result(result)\njob=pathlib.Path(p['job']); job.mkdir(mode=0o700)\n(job/'submission.json').write_text(json.dumps(p,indent=2)+'\\n')\n(job/'cpu-result.json').write_bytes(cpu_result_raw)\n(job/'cpu-result-binding.json').write_text(json.dumps(dict(revision=p['revision'],cpu_host=p['cpu_host'],cpu_output=p['cpu_output'],sha256=__import__('hashlib').sha256(cpu_result_raw).hexdigest(),source_receipt_exact=True,scope='Original CPU result validated before canonical submission; includes four native mapping receipts'),indent=2)+'\\n')\nenv=dict(os.environ,REPO=source)\nraise SystemExit(subprocess.call(p['command'],env=env))\n"

result=subprocess.run(['ssh','-o','BatchMode=yes','-o','ConnectTimeout=8','choiceoh@srv2','python3 -B -c '+shlex.quote(remote)],input=json.dumps(p),text=True)
raise SystemExit(result.returncode)
