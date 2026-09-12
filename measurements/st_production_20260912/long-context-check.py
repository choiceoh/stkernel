import json,time,urllib.request
from pathlib import Path
base='http://10.10.10.2:8000'
prompt='The secret verification number is 7349. Remember it.\n'+('This is unrelated filler about blue skies and green fields.\n'*12000)+'\nWhat is the secret verification number at the beginning? Answer only the number.'
body=dict(model='glm-5.3-flash',messages=[dict(role='user',content=prompt)],max_tokens=32,temperature=0,chat_template_kwargs=dict(thinking=False))
start=time.monotonic()
req=urllib.request.Request(base+'/v1/chat/completions',data=json.dumps(body).encode(),headers={'Content-Type':'application/json'})
out=dict(base=base,input_chars=len(prompt),started=time.time(),passed=False)
try:
 with urllib.request.urlopen(req,timeout=600) as r: d=json.load(r)
 out.update(seconds=time.monotonic()-start,response=d)
 assert d['usage']['prompt_tokens']>=128000
 assert '7349' in (d['choices'][0]['message'].get('content') or '')
 out['passed']=True
except Exception as e:
 out['error']=type(e).__name__+': '+str(e)
 raise
finally:
 Path('/tmp/st-production-f4d7-long-context.json').write_text(json.dumps(out,ensure_ascii=False,indent=2))
 print(json.dumps(out,ensure_ascii=False),flush=True)
