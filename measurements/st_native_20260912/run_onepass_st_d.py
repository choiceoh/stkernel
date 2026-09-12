"""Run unchanged onepass workload/gates; adapt only ST identity and step telemetry."""
import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import urllib.request

ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT/'bench'))
spec=importlib.util.spec_from_file_location('onepass', ROOT/'bench/onepass.py')
op=importlib.util.module_from_spec(spec);spec.loader.exec_module(op)
original_load, original_ask = op._load, op.ask_stream
outputs=[]
container=json.loads(subprocess.check_output(['docker','inspect','st-glm53'],text=True))[0]
assert container['State']['Running']
env=dict(v.split('=',1) for v in container['Config']['Env'] if '=' in v)
assert not any(k.startswith('STK_') for k in env)
assert '--production' in ' '.join(container['Config']['Cmd'])
assert container['Config']['Image']=='st-engine:perf-f4d7-d'
runtime=json.loads(subprocess.check_output(['docker','exec','st-glm53','cat','/opt/st/runtime-manifest.json'],text=True))
assert runtime['engine_source_sha256']=='85f781fb303a9be3cb0e990ebc220a31e0ae10397905164d189231b4c0235df7'
identity=dict(engine='st',boot_id=container['Id']+'|'+container['State']['StartedAt'],
              image=container['Config']['Image'],engine_source_sha256=runtime['engine_source_sha256'],
              step_counter_source='GET / -> steps (ST Runner.step counter)',knobs={k:v for k,v in env.items() if k.startswith('STK_')})
(ROOT/'identity.json').write_text(json.dumps(identity,indent=2)+'\n')
def load(fname,modname):
 mod=original_load(fname,modname)
 if fname=='bracket.py':
  def steps(self):
   from window_metrics import traffic_state
   try:
    with urllib.request.urlopen(self.bd.METRICS,timeout=5) as r: metrics=r.read().decode()
    self.traffic_samples.append(traffic_state(metrics))
    base=self.bd.METRICS.rsplit('/metrics',1)[0]
    with urllib.request.urlopen(base+'/',timeout=5) as r: state=json.load(r)
    if state.get('engine')!='ST': raise RuntimeError('the measured backend changed')
    return float(state['steps'])
   except Exception:
    return None
  mod._StepWindows._steps=steps
 return mod
op._load=load
op._served_build=lambda *a,**kw:dict(identity)
def ask(*args,**kwargs):
 result=original_ask(*args,**kwargs)
 outputs.append(dict(prompt_sha256=hashlib.sha256(args[2].encode()).hexdigest(),
                     text=result[0],ttft_s=result[1],prompt_tokens=result[2],
                     completion_tokens=result[3],finish_reason=result[4]))
 (ROOT/'outputs.json').write_text(json.dumps(outputs,ensure_ascii=False,indent=2)+'\n')
 return result
op.ask_stream=ask
code=op.main()
after=json.loads(subprocess.check_output(['docker','inspect','st-glm53'],text=True))[0]
assert after['Id']==container['Id'] and after['State']['StartedAt']==container['State']['StartedAt'] and after['State']['Running']
raise SystemExit(code)
