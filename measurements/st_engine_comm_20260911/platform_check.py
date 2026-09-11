import ctypes,datetime,json,re,subprocess
from pathlib import Path
result={}
headers=[Path('/usr/local/cuda/include/cuda.h'),Path('/usr/local/cuda-13.0/include/cuda.h')]
header=next((p for p in headers if p.exists()),None)
if header:
    text=header.read_text();driver=ctypes.CDLL('libcuda.so.1')
    result['cuInit']=driver.cuInit(0)
    result['attributes']={}
    for name in ['GPU_DIRECT_RDMA_SUPPORTED','DMA_BUF_SUPPORTED']:
        found=re.search(r'CU_DEVICE_ATTRIBUTE_'+name+r'\s*=\s*(\d+)',text)
        if found:
            val=ctypes.c_int();error=driver.cuDeviceGetAttribute(ctypes.byref(val),int(found[1]),0)
            result['attributes'][name]=dict(value=val.value,error=error)
result['collected_utc']=datetime.datetime.now(datetime.timezone.utc).isoformat()
end=datetime.datetime.now(datetime.timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ')
p=subprocess.run(['docker','events','--since','2026-09-11T13:15:00Z','--until',end,'--filter','type=container','--format','{{json .}}'],capture_output=True,text=True,check=True)
events=[]
for line in p.stdout.splitlines():
    row=json.loads(line);a=row.get('Actor',{}).get('Attributes',{});name=a.get('name','')
    if name.startswith(('st-tp4-lat-f4d7','glm53','st-glm53')) and row.get('Action') in ['start','die','kill','destroy','oom']:
        events.append(dict(name=name,action=row.get('Action'),time=row.get('timeNano'),exit=a.get('exitCode'),signal=a.get('signal')))
result['events']=events
lock=Path('/home/choiceoh/st-fleet.lock')
result['lock']=lock.read_text() if lock.exists() else None
result['active_containers']=subprocess.run(['docker','ps','--format','{{.Names}}'],capture_output=True,text=True,check=True).stdout.splitlines()
print(json.dumps(result,indent=2))
