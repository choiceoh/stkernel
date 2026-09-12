"""GB10 A/B: mHC shared-memory dtype and TileLang TMA lowering.

Loads isolated copies of the actual ST kernels. No serving switch or vLLM is
involved. All variants use identical tensors and alternating CUDA-graph timing.
"""
import argparse
import ctypes
import gc
import hashlib
import importlib.util
import json
from pathlib import Path
import statistics
import sys


def check_post_oracle(comb, residual, post, x, output):
    """CPU fmaf reference, including input/output-channel orientation.

    Sample across every hidden partition boundary without copying a complete
    prefill activation to the CPU. Each intermediate follows FP32 rounding.
    """
    import torch
    columns = sorted({0, 1, 255, 256, 511, 512, 1023, 1024, 2047, 2048, 4095})
    rows = sorted({0, len(x) - 1})
    index = torch.tensor(columns, device=x.device)
    a = comb[rows].cpu()
    b = residual[rows].index_select(-1, index).float().cpu()
    c = post[rows].cpu()
    d = x[rows].index_select(-1, index).float().cpu()
    observed = output[rows].index_select(-1, index).cpu()
    reference = torch.empty_like(observed, dtype=torch.float32)
    fma = ctypes.CDLL('libm.so.6').fmaf
    fma.argtypes = [ctypes.c_float] * 3
    fma.restype = ctypes.c_float
    for row in range(len(rows)):
        for out_channel in range(4):
            for col in range(len(columns)):
                value = ctypes.c_float(float(c[row,out_channel]) * float(d[row,col])).value
                for in_channel in range(4):
                    value = fma(float(a[row,in_channel,out_channel]),
                                float(b[row,in_channel,col]), value)
                reference[row,out_channel,col] = value
    exact = torch.equal(reference.to(torch.bfloat16).view(torch.int16), observed.view(torch.int16))
    assert exact, 'independent FP32 FMA post reference changed bits'
    return {'cpu_fmaf_bits_exact': True, 'elements': observed.numel()}


def load_module(text, directory, name, passes):
    import engine.kernels
    path = directory / (name + '.py')
    path.write_text(text)
    engine.kernels.MHC_PASSES = passes
    spec = importlib.util.spec_from_file_location('mhc_probe_' + name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module, hashlib.sha256(text.encode()).hexdigest()


def load_variant(source, directory, name, bf16, tma, ws, split=False, h_blk=1024, n_thr=128, force_tma=False, joined=False, pipelined=False):
    text = source.read_text()
    if bf16:
        text = text.replace(
            'xs = T.alloc_shared((hc_mult, hidden_block), T.float32)',
            'xs = T.alloc_shared((hc_mult, hidden_block), T.bfloat16)')
    start = text.index('def mhc_post_tilelang(')
    end = text.index('\n\n@tilelang.jit(', start)
    post = text[start:end]
    post = post.replace('n_thr: int = 128,', f'n_thr: int = {n_thr},')
    post = post.replace('h_blk: int = 1024,', f'h_blk: int = {h_blk},')
    if split:
        post = post.replace('with T.Kernel(n, threads=n_thr) as i_n:',
                            'with T.Kernel(n, T.ceildiv(h, h_blk), threads=n_thr) as (i_n, i0_h):')
        post = post.replace('for i0_h in T.Serial(T.ceildiv(h, h_blk)):',
                            'for _ in T.Serial(1):')
    if pipelined:
        post = post.replace('for i0_h in T.Serial(T.ceildiv(h, h_blk)):',
                            'for i0_h in T.Pipelined(T.ceildiv(h, h_blk), num_stages=2):')
    if force_tma:
        post = post.replace('], b_shared)', '], b_shared, prefer_instruction="tma")')
        post = post.replace('], d_shared)', '], d_shared, prefer_instruction="tma")')
    if joined:
        post = post.replace('        b_shared = T.alloc_shared',
                            '        ready = T.alloc_barrier([1])\n        b_shared = T.alloc_shared',1)
        post = post.replace('T.copy(b[i_n, 0, i0_h * h_blk], b_shared)',
                            'T.tma_copy(b[i_n, 0, i0_h * h_blk], b_shared, barrier=ready[0])')
        post = post.replace('T.copy(d[i_n, i0_h * h_blk], d_shared)',
                            'T.tma_copy(d[i_n, i0_h * h_blk], d_shared, barrier=ready[0])')
        phase = '0' if split else 'i0_h % 2'
        post = post.replace('            T.copy(b_shared, b_local)',
                            '            if T.shuffle_elect(n_thr):\n'
                            '                T.mbarrier_arrive(ready[0])\n'
                            f'            T.mbarrier_wait_parity(ready[0], {phase})\n'
                            '            T.copy(b_shared, b_local)',1)
        post = post.replace('            T.copy(d_shared, d_local)',
                            '            T.copy(d_shared, d_local)\n            T.sync_threads()',1)
    text = text[:start] + post + text[end:]
    if force_tma and bf16:
        text = text.replace('], xs)', '], xs, prefer_instruction="tma")')
        text = text.replace('], w_shared)', '], w_shared, prefer_instruction="tma")')
    return load_module(text, directory, name, (tma, ws))


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--output', type=Path, required=True)
    ap.add_argument('--source', type=Path, required=True, help='mHC source before optimization, for the baseline.')
    ap.add_argument('--adopted-source', type=Path, default=Path('engine/kernels/mhc/tilelang_kernels.py'))
    ap.add_argument('--tokens', default='1,6,32,64,128,512,2048,4096,6912')
    ap.add_argument('--variants', default='stock,adopted')
    ap.add_argument('--kinds', default='post,pre_norm')
    ap.add_argument('--inner-repeats', type=int, default=16,
                    help='Kernel calls per timing graph; amortizes Python graph-launch overhead.')
    args = ap.parse_args()
    if args.inner_repeats < 1:
        ap.error('--inner-repeats must be positive')
    args.output.mkdir(parents=True, exist_ok=True)
    import torch
    assert torch.cuda.get_device_capability() == (12, 1), 'This probe is qualified for GB10 SM121.'
    torch.cuda.set_per_process_memory_fraction(3 * 2**30 / torch.cuda.get_device_properties(0).total_memory)
    torch.manual_seed(9341)
    settings = {'stock': (False,False,False), 'bf16': (True,False,False),
                'tma': (True,True,False), 'tma_ws': (True,True,True),
                'split': (False,False,False,True), 'split_tma': (False,True,False,True),
                'split512': (False,False,False,True,512), 'split512_tma': (False,True,False,True,512),
                'wide256_tma': (False,True,False,False,4096,256),
                'wide512_tma': (False,True,False,False,4096,512),
                'forced_tma': (True,True,False,False,1024,128,True),
                'split_forced_tma': (False,True,False,True,1024,128,True),
                'wide256_forced_tma': (False,True,False,False,4096,256,True),
                'wide512_forced_tma': (False,True,False,False,4096,512,True),
                'joined_tma': (False,True,False,False,1024,128,False,True),
                'joined4096_tma': (False,True,False,False,4096,256,False,True),
                'split_joined_tma': (False,True,False,True,1024,128,False,True),
                'pipeline_tma': (False,True,True,False,1024,128,False,False,True),
                'pipeline512_tma': (False,True,True,False,512,128,False,False,True)}
    variants = {}
    report = {'source_sha256': hashlib.sha256(args.source.read_bytes()).hexdigest(),
              'harness_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              'torch_version':torch.__version__, 'cuda_version':torch.version.cuda,
              'device_name':torch.cuda.get_device_name(),
              'timing_graph_kernel_calls':args.inner_repeats,
              'variants': {}, 'cases': []}
    for name in args.variants.split(','):
        if name == 'adopted':
            variants[name], sha = load_module(args.adopted_source.read_text(),args.output,name,None)
            report['variants'][name] = {'mode':'unmodified engine default','sha256':sha}
            continue
        variants[name], sha = load_variant(args.source, args.output, name, *settings[name])
        report['variants'][name] = {'bf16_shared': settings[name][0], 'tma': settings[name][1],
                                    'warp_specialized': settings[name][2], 'settings':settings[name], 'sha256': sha}
    for tokens in map(int, args.tokens.split(',')):
        residual = torch.randn(tokens,4,4096,device='cuda',dtype=torch.bfloat16) * .1
        x = torch.randn(tokens,4096,device='cuda',dtype=torch.bfloat16) * .1
        comb = torch.rand(tokens,4,4,device='cuda',dtype=torch.float32) * .25
        post = torch.rand(tokens,4,device='cuda',dtype=torch.float32)
        splits = max(1, min(48 // ((tokens+63)//64), 64))
        mul = torch.randn(splits,tokens,24,device='cuda') * .01
        sq = torch.full((splits,tokens), 16384 * .01 / splits,device='cuda')
        scale = torch.tensor([.8,1.2,.6],device='cuda')
        base = torch.randn(24,device='cuda') * .1
        norm = torch.randn(4096,device='cuda',dtype=torch.bfloat16)
        for kind in args.kinds.split(','):
            outputs, graphs, runners, failed = {}, {}, {}, {}
            for name,module in variants.items():
                if kind == 'post':
                    outputs[name] = (torch.empty_like(residual),)
                    def run(module=module,out=outputs[name][0]):
                        module.mhc_post_tilelang(comb,residual,post,x,out,4,4096)
                else:
                    outputs[name] = (torch.empty_like(post),torch.empty(tokens,16,device='cuda'),torch.empty_like(x))
                    def run(module=module,out=outputs[name]):
                        module.mhc_pre_big_fuse_with_norm_tilelang(mul,sq,scale,base,residual,*out,norm,
                            4096,1e-6,1e-6,1e-6,2.,20,1e-6,splits,4)
                print('compile',tokens,kind,name,flush=True)
                try:
                    run(); torch.cuda.synchronize()
                    graph = torch.cuda.CUDAGraph()
                    with torch.cuda.graph(graph): run()
                    graphs[name] = graph
                    runners[name] = run
                except Exception as exc:
                    failed[name] = str(exc)
                    print('failed',tokens,kind,name,str(exc)[-1500:],flush=True)
            if 'stock' not in graphs:
                raise RuntimeError(f'baseline failed: {failed}')
            correctness = {}
            oracle = []
            for repeat in range(4):
                if repeat:
                    residual.mul_(-.75); x.mul_(.9); mul.mul_(-.8)
                for graph in graphs.values(): graph.replay()
                torch.cuda.synchronize()
                for name in graphs:
                    exact = all(torch.equal(a.view(torch.uint8), b.view(torch.uint8))
                                for a,b in zip(outputs['stock'],outputs[name]))
                    errors = [0.0] * len(outputs[name]) if exact else [
                        float((a.float()-b.float()).norm() / a.float().norm().clamp_min(1e-20))
                        for a,b in zip(outputs['stock'],outputs[name])]
                    correctness.setdefault(name,[]).append({'bits_exact':exact,'relative_l2':errors})
                if kind == 'post':
                    oracle.append(check_post_oracle(comb,residual,post,x,outputs['stock'][0]))
            timing_graphs = {}
            for name,run in runners.items():
                if not all(case['bits_exact'] for case in correctness[name]):
                    failed[name] = 'Output storage bits differ; timing is excluded.'
                    continue
                graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph):
                    for _ in range(args.inner_repeats): run()
                timing_graphs[name] = graph
            samples = {name:[] for name in timing_graphs}
            for graph in timing_graphs.values():
                for _ in range(5): graph.replay()
            for iteration in range(7):
                order = list(timing_graphs) if iteration % 2 == 0 else list(reversed(timing_graphs))
                for name in order:
                    start = torch.cuda.Event(enable_timing=True); end = torch.cuda.Event(enable_timing=True)
                    start.record()
                    for _ in range(4): timing_graphs[name].replay()
                    end.record(); end.synchronize()
                    samples[name].append(start.elapsed_time(end)*1000/(4*args.inner_repeats))
            row = {'tokens':tokens,'kind':kind,'splits':splits,'correctness':correctness,
                   'independent_oracle':oracle,
                   'samples_us':samples,'median_us':{n:statistics.median(s) for n,s in samples.items()},'failed':failed}
            report['cases'].append(row)
            (args.output/'results.json').write_text(json.dumps(report,indent=2)+'\n')
            print(json.dumps({k:v for k,v in row.items() if k not in ('samples_us','failed')}),flush=True)
            del outputs, graphs, runners, timing_graphs
            gc.collect(); torch.cuda.empty_cache()
        del residual,x,comb,post,mul,sq,norm
        gc.collect(); torch.cuda.empty_cache()


if __name__ == '__main__':
    main()
