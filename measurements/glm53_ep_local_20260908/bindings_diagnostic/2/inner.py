from pathlib import Path
import importlib.metadata,json
assert not list(Path('/dev').glob('nvidia*')), 'CUDA device exposed'
from cuda.bindings import driver
code,version=driver.cuDriverGetVersion()
count_result=driver.cuDeviceGetCount()
print('BINDING_DIAGNOSTIC '+json.dumps(dict(cuda_bindings=importlib.metadata.version('cuda-bindings'),driver_api_version=version,result=int(code),device_count_result=[int(v) for v in count_result],device_nodes=[])),flush=True)
raise SystemExit(0 if int(code)==0 else 1)
