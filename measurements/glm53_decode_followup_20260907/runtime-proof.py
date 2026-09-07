import hashlib,json,socket,subprocess
from pathlib import Path
names=subprocess.check_output(['docker','ps','--format','{{.Names}}'],text=True).splitlines()
name=next(n for n in names if n.startswith('glm53'))
item=json.loads(subprocess.check_output(['docker','inspect',name],text=True))[0]
keep={'VLLM_GLM53_MK_FP8_PACK2','VLLM_GLM53_MK_GEMM_TRANSPOSE_M8','VLLM_GLM53_MK_M8_FASTPATH','VLLM_GLM53_MK_MHC_BF16','VLLM_GLM53_FP8_CACHE','VLLM_GLM53_SPEC_K'}
env={k:v for x in item['Config']['Env'] if '='in x for k,v in [x.split('=',1)] if k in keep}
root=Path('/home/choiceoh/overlays/glm53')
sha={n:hashlib.sha256((root/n).read_bytes()).hexdigest() for n in ('glm53_megakernel.py','glm53_megakernel.cu')}
log=Path('/home/choiceoh/glm53-logs/glm53.log').read_text(errors='replace')
markers=[l for l in log.splitlines() if 'selftest mhc-bf16' in l or 'mhc-bf16 CAPTURED T=6 ' in l]
print(json.dumps({'host':socket.gethostname(),'container':name,'image':item['Image'],'boot_id':item['Id']+'|'+item['State']['StartedAt'],'knobs':env,'source_sha256':sha,'manifest':(root/'manifest.tsv').read_text().splitlines()[0],'markers':markers},indent=2))
