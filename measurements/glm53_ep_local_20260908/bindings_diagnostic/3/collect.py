"""Capture cuda-bindings resolver behavior without exposing any CUDA device."""
import hashlib,json,os,subprocess,sys,time,uuid
from pathlib import Path
source=Path('/home/choiceoh/stkernel-ep-local-gpu-0908-4')
sys.path.insert(0,str(source/'probes'))
import glm53_ep_sanitizer as san
job=Path('/tmp/glm53-ep-bindings0908-3')
job.mkdir(exist_ok=False)
image='sha256:a3dd4c0f6cbb053097d65d10cd8ff8f6ae0cb9115cf0ff142e1cafe124c09211'
driver=Path('/lib/aarch64-linux-gnu/libcuda.so.1').resolve(strict=True)
report=dict(started=time.time(),image=image,driver_library=str(driver),driver_sha256=hashlib.sha256(driver.read_bytes()).hexdigest(),kernel_module_version=Path('/proc/driver/nvidia/version').read_text(),scope='cuDriverGetVersion then original hardware-info cuDeviceGetCount; no MoE/CuTe, no device nodes, no GPU kernels',performance_acceptance=False)
inner="""from pathlib import Path
import importlib.metadata,json
assert not list(Path('/dev').glob('nvidia*')), 'CUDA device exposed'
from cuda.bindings import driver
code,version=driver.cuDriverGetVersion()
count_result=driver.cuDeviceGetCount()
print('BINDING_DIAGNOSTIC '+json.dumps(dict(cuda_bindings=importlib.metadata.version('cuda-bindings'),driver_api_version=version,result=int(code),device_count_result=[int(count_result[0]), count_result[1]],device_nodes=[])),flush=True)
raise SystemExit(0 if int(code)==0 else 1)
"""
(job/'inner.py').write_text(inner)
receipt=san.preflight(image,job/'sanitizer-preflight.json')
name='ep-bindings-diagnostic-'+uuid.uuid4().hex
command=['docker','run','--rm','--name',name,'--runtime=runc','--network=none','--memory=512m','--memory-swap=512m','--cpus=1','--pids-limit=64','-e','NVIDIA_VISIBLE_DEVICES=void','-e','CUDA_VISIBLE_DEVICES=','-e','LD_LIBRARY_PATH=/opt/glm-ep-driver','--mount',f'type=bind,source={driver},target=/opt/glm-ep-driver/libcuda.so.1,readonly','--mount',f'type=bind,source={job},target=/evidence,readonly','--entrypoint=/usr/bin/env']+san.mount_args(receipt)+[image]+san.command('memcheck')+['python3','/evidence/inner.py']
report['command']=command
try:
 with (job/'memcheck.log').open('x') as log:
  result=subprocess.run(command,text=True,stdout=log,stderr=subprocess.STDOUT,timeout=45)
 report['sanitizer_exit_code']=result.returncode
except BaseException as exc:
 report['error']=repr(exc)
 subprocess.run(['docker','rm','-f',name],text=True,capture_output=True,timeout=15)
 raise
finally:
 report['ended']=time.time()
 (job/'result.json').write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps(dict(path=str(job),sanitizer_exit_code=report.get('sanitizer_exit_code'),scope=report['scope'])),flush=True)
# This diagnostic records the original tool result; nonzero stays nonzero.
raise SystemExit(report['sanitizer_exit_code'])
