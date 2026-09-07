import hashlib,json,struct
from pathlib import Path
root=Path('/home/choiceoh/models/glm53-redhat-nvfp4')
out=Path('/home/choiceoh/glm53-cache/mhc-reuse-probe/mhc-fixtures')
out.mkdir(exist_ok=True)
index=json.loads((root/'model.safetensors.index.json').read_text())['weight_map']
manifest=[]
for layer in (0,1,23,44):
 for kind in ('attn','ffn'):
  name=f'model.language_model.layers.{layer}.hc_{kind}'
  item={'name':name}
  for suffix in ('fn','scale','base'):
   key=name+'_'+suffix
   with (root/index[key]).open('rb') as f:
    size=struct.unpack('<Q',f.read(8))[0]
    meta=json.loads(f.read(size))[key]
    assert meta['dtype']=='BF16', (key,meta)
    start,end=meta['data_offsets']
    f.seek(8+size+start)
    raw=f.read(end-start)
   file=f'layer{layer}-{kind}-{suffix}.bf16'
   (out/file).write_bytes(raw)
   item[suffix]={'file':file,'shape':meta['shape'],'sha256':hashlib.sha256(raw).hexdigest(),'dtype':meta['dtype'],'checkpoint_key':key,'shard':index[key]}
  manifest.append(item)
(out/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
print('Extracted',len(manifest),'trained weight sets; bytes',sum(p.stat().st_size for p in out.glob('*.bf16')))
