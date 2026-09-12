"""Preshard NVIDIA GLM NVFP4 losslessly, retaining separate FP32 multipliers.

Outputs a versioned offline layout, not the current folded-scale serving layout.
The live loader intentionally rejects it until an explicit ModelOpt adapter is
implemented. No dequantized BF16 replacement of the dense NVFP4 layers is made.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys
import time

sys.path.insert(0,str(Path(__file__).resolve().parents[3]))

import torch

from engine.base.checkpoint import Checkpoint
from engine.base.loader import RankLoader
from engine.base.preshard import RankWriter
from engine.profiles.glm53 import modelopt_weights as layout
from engine.profiles.glm53.preshard import parse_layers
from engine.profiles.glm53 import vision
from engine.profiles.glm53.modelopt_coverage import source_coverage


def tensor_hash(tensor):
    return hashlib.sha256(tensor.contiguous().reshape(-1).view(torch.uint8).numpy().tobytes()).hexdigest()


def file_hash(path):
    value=hashlib.sha256()
    with Path(path).open('rb') as stream:
        while block:=stream.read(16<<20):value.update(block)
    return value.hexdigest()


def plan(ckpt,layers):
    F=layout.load_facts(ckpt)
    if not layers or min(layers)<0 or max(layers)>=F.layers:
        raise ValueError('nonempty valid model layer selection required')
    groups=list(layout.groups(F,layers))
    index=json.loads((Path(ckpt)/'model.safetensors.index.json').read_text())['weight_map']
    sources={key for _,group in groups for s in group for key in s.sources}
    if missing:=sources-set(index):raise ValueError(('missing checkpoint tensors',sorted(missing)[:8]))
    quantized={key.removesuffix('.weight_scale_2') for key in index if key.endswith('.weight_scale_2')}
    covered={key.removesuffix('.weight_scale_2') for key in sources if key.endswith('.weight_scale_2')}
    wanted={key for key in quantized if any(key.startswith(f'model.language_model.layers.{layer}.') for layer in layers)}
    if wanted!=covered:raise ValueError('unhandled quantized projection')
    specs=[s for _,group in groups for s in group]
    if len({s.name for s in specs})!=len(specs):raise ValueError('duplicate output tensor')
    vision_specs=vision.specs(vision.load(ckpt)) if layers==list(range(F.layers)) else []
    vision_bytes=sum(s.nbytes() for s in vision_specs)
    coverage=source_coverage(ckpt,sources,{key for s in vision_specs for key in s.sources}) if vision_specs else None
    return F,groups,dict(weight_layout=layout.WEIGHT_LAYOUT,world=4,layers=layers,
        tensors_per_rank=len(specs),payload_bytes_per_rank=sum(s.nbytes() for s in specs),
        vision_payload_bytes=vision_bytes,
        source_tensors=len(sources),quantized_projections=len(covered),
        global_scales='FP32 multipliers preserved separately; no folding or inversion',
        dense_nvfp4_preserved=True,live_serving_compatible=False,
        source_coverage=coverage,
        source_config_sha256=file_hash(Path(ckpt)/'config.json'),
        source_index_sha256=file_hash(Path(ckpt)/'model.safetensors.index.json'))


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--ckpt',type=Path,required=True)
    ap.add_argument('--out',type=Path,required=True)
    ap.add_argument('--layers',default='all')
    ap.add_argument('--plan',action='store_true')
    ap.add_argument('--source-revision',required=True)
    args=ap.parse_args()
    F=layout.load_facts(args.ckpt)
    layers=parse_layers(args.layers,F.layers)
    F,groups,report=plan(args.ckpt,layers)
    report['source_revision']=args.source_revision
    print(json.dumps(report),flush=True)
    if args.plan:return
    out=args.out.resolve();partial=out.with_name(out.name+'.incomplete')
    if out.exists() or partial.exists():raise ValueError('output must be a new immutable directory')
    out.parent.mkdir(parents=True,exist_ok=True)
    required=4*report['payload_bytes_per_rank']+report['vision_payload_bytes']+(8<<30)
    if shutil.disk_usage(out.parent).free<required:raise ValueError('insufficient disk space for four ranks and reserve')
    partial.mkdir()
    torch.set_num_threads(4)
    ck=Checkpoint(str(args.ckpt));all_specs=[s for _,group in groups for s in group]
    writers=[RankWriter(partial/f'rank{rank}of4.safetensors',all_specs,
        dict(model='glm53',world=4,rank=rank,layers=args.layers,weight_layout=layout.WEIGHT_LAYOUT,
             source_revision=args.source_revision,scale_convention='modelopt-multiplier')) for rank in range(4)]
    hashes=[{} for _ in writers];started=time.monotonic()
    for label,group in groups:
        source=ck.load(sorted({key for spec in group for key in spec.sources}))
        for rank,writer in enumerate(writers):
            for spec in group:
                tensor=spec.build(source,rank,4)
                if tensor.is_floating_point() and not torch.isfinite(tensor.float()).all():
                    raise ValueError(f'nonfinite weight: {spec.name}')
                hashes[rank][spec.name]=tensor_hash(tensor)
                writer.put(spec.name,tensor)
                del tensor
        del source
        print(json.dumps(dict(stage='write',group=label,seconds=time.monotonic()-started)),flush=True)
    for writer in writers:writer.close()
    report['rank_files']=[]
    for rank in range(4):
        path=partial/f'rank{rank}of4.safetensors';reader=RankLoader(path)
        if reader.metadata['weight_layout']!=layout.WEIGHT_LAYOUT or reader.metadata['rank']!=str(rank):
            raise AssertionError('rank identity mismatch')
        for label,group in groups:
            loaded=reader.load([spec.name for spec in group],device='cpu',max_run=32<<20)
            for spec in group:
                tensor=loaded[spec.name]
                if tuple(tensor.shape)!=tuple(spec.shape) or tensor.dtype!=spec.dtype or tensor_hash(tensor)!=hashes[rank][spec.name]:
                    raise AssertionError(('rank readback mismatch',rank,spec.name))
            del tensor,loaded
        entry=dict(name=path.name,bytes=path.stat().st_size,sha256=file_hash(path),
                   tensors_verified=len(hashes[rank]),all_tensor_bytes_exact=True)
        report['rank_files'].append(entry)
        print(json.dumps(dict(stage='verify',rank=rank,**entry)),flush=True)
    if layers==list(range(F.layers)):
        vision.write_file(args.ckpt,partial)
        path=partial/vision.FILE
        report['vision']=dict(name=path.name,bytes=path.stat().st_size,sha256=file_hash(path))
    metadata=[]
    for path in sorted(args.ckpt.iterdir()):
        if path.is_file() and (path.suffix in ('.json','.jinja','.md') or path.name.upper().startswith(('LICENSE','NOTICE'))) and path.name!='model.safetensors.index.json':
            shutil.copyfile(path,partial/path.name)
            metadata.append(dict(name=path.name,sha256=file_hash(path)))
    report.update(metadata_files=metadata,seconds=time.monotonic()-started,
                  torch_version=torch.__version__,source_script_sha256=file_hash(__file__))
    (partial/'preshard-manifest.json').write_text(json.dumps(report,indent=2)+'\n')
    checks=report['rank_files']+([report['vision']] if 'vision' in report else [])
    (partial/'SHA256SUMS').write_text(''.join(row['sha256']+'  '+row['name']+'\n' for row in checks))
    os.rename(partial,out)
    print(json.dumps(dict(stage='complete',out=str(out),seconds=report['seconds'])),flush=True)


if __name__=='__main__':main()
