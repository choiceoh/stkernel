import json,os,pathlib,subprocess,sys
p=json.loads(sys.stdin.readline())
assert subprocess.check_output(['git','rev-parse','HEAD'],cwd=p['source'],text=True).strip()==p['revision']
assert not subprocess.check_output(['git','status','--porcelain'],cwd=p['source'],text=True).strip()
job=pathlib.Path(p['job']);job.mkdir(exist_ok=False)
(job/'submission.json').write_text(json.dumps(p,indent=2)+'\n')
env=os.environ.copy();env['REPO']=p['source']
r=subprocess.run(p['command'],cwd=p['source'],env=env,text=True,stdout=subprocess.PIPE,stderr=subprocess.STDOUT)
(job/'submission-output.txt').write_text(r.stdout)
print(r.stdout)
raise SystemExit(r.returncode)
