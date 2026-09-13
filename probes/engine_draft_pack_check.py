"""Single-GPU local selection cost: dense baseline, Torch packets, fused packets.

Identity collectives isolate added GPU work; these timings are not TP4 latency.
"""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import statistics
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import torch
from engine.base.comm import Comm
from engine.modules.vocab import topk
from probes.engine_draft_candidate_check import capture


@torch.inference_mode()
def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--unfused', required=True, help='exported 9de1a199 engine/modules/vocab.py')
    ap.add_argument('--output', required=True)
    args = ap.parse_args()
    torch.set_num_threads(2)
    torch.cuda.set_per_process_memory_fraction((512<<20)/torch.cuda.mem_get_info()[1])
    spec = importlib.util.spec_from_file_location('unfused_vocab', args.unfused)
    old = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(old)
    gen = torch.Generator(device='cuda').manual_seed(729)
    report = dict(scope=__doc__, device=torch.cuda.get_device_name(), torch=torch.__version__,
                  torch_git=torch.version.git_version, cases=[],
                  unfused_sha256=hashlib.sha256(Path(args.unfused).read_bytes()).hexdigest(),
                  source_sha256={p:hashlib.sha256(Path(p).read_bytes()).hexdigest() for p in (
                      'engine/modules/vocab.py', 'engine/kernels/common/vocab_candidates.py',
                      'probes/engine_draft_pack_check.py')})
    for rows in (1, 5, 20):
        x = torch.randn(rows, 38720, dtype=torch.bfloat16, device='cuda', generator=gen)
        def dense():
            values = x.float()
            values[:, 38710:] = float('-inf')
            return values.topk(16, dim=-1)
        funcs = dict(dense=dense, torch_packet=lambda:old.topk(x, Comm(), 0, 16, 38710),
                     fused_packet=lambda:topk(x, Comm(), 0, 16, 38710))
        graphs = {name:capture(call) for name,call in funcs.items()}
        for graph,out in graphs.values():
            graph.replay()
            expected = dense()
            assert torch.equal(out.indices, expected.indices)
            assert torch.equal(out.values, expected.values)
        samples = {name:[] for name in graphs}
        for round_id in range(5):
            for name in list(graphs) if round_id%2==0 else list(graphs)[::-1]:
                graph = graphs[name][0]
                for _ in range(10): graph.replay()
                events = [(torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True)) for _ in range(50)]
                for a,b in events:
                    a.record(); graph.replay(); b.record()
                events[-1][1].synchronize()
                samples[name].append([a.elapsed_time(b)*1000 for a,b in events])
        case = dict(rows=rows, exact=True, timing={name:dict(median_us=statistics.median(sum(v,[])), samples_us=v)
                                                 for name,v in samples.items()})
        report['cases'].append(case)
        for graph,out in graphs.values(): graph.reset()
        print(rows, {k:round(v['median_us'],2) for k,v in case['timing'].items()}, flush=True)
    report.update(passed=True, peak_reserved_bytes=torch.cuda.max_memory_reserved())
    Path(args.output).write_text(json.dumps(report, indent=2)+'\n')


if __name__ == '__main__': main()
