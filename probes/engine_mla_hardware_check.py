"""Compare the exact baseline/candidate MLA bodies on GB10 in a bounded process.

The CUDA bodies are extracted verbatim from the two supplied translation units.
The small C ABI avoids rebuilding unrelated Torch/GEMM bindings for each trial.
Timing covers the kernel only and reports warm and explicitly evicted L2 cases.
"""
import argparse
import ctypes
import hashlib
import json
import os
from pathlib import Path
import statistics
import subprocess


def cuda_source(source):
    barrier = source[source.index("__device__ __forceinline__ void mk_grid_barrier("):
                     source.index("__device__ __forceinline__ float mk_sigmoid(")]
    copies = source[source.index("__device__ __forceinline__ void mk_cp_async16("):
                    source.index("// wait_group takes an immediate")]
    mla = source[source.index("constexpr int MLA_D = 512;"):
                 source.index("// Exact-selection prefill pair reuse")]
    templated = "template <bool CLUSTER = false>" in mla
    ordinary = "mk_mla_kernel<false>" if templated else "mk_mla_kernel"
    cluster = "mk_mla_kernel<true>" if templated else ordinary
    # A packed PV loader must also preserve the scalar-strided conversion's
    # NaN/subnormal bits; normal attention fixtures alone cannot prove that.
    packed_pair = ("mla_e4m3x2_value(*(const uint16_t*)(input + i * 2))"
                   if "mla_e4m3x2_value(" in mla else
                   "mla_e4m3x2_strided(input + 131072 + i, 65536)")
    return """
#include <cuda_runtime.h>
#include <cuda_fp8.h>
#include <cuda_bf16.h>
#include <cooperative_groups.h>
#include <cmath>
#include <cstdint>
constexpr int MK_THREADS = 256;
#define MK_SPIN_WAIT(condition, ns, label) while (condition) { __nanosleep(ns); }
""" + barrier + copies + mla + f"""
extern "C" int setup(int* values) {{
  int sms, blocks;
  cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, 0);
  auto error = cudaFuncSetAttribute({ordinary}, cudaFuncAttributeMaxDynamicSharedMemorySize, MLA_SMEM);
  if (error != cudaSuccess) return error;
  error = cudaOccupancyMaxActiveBlocksPerMultiprocessor(&blocks, {ordinary}, MK_THREADS, MLA_SMEM);
  if (error != cudaSuccess) return error;
  cudaFuncAttributes attr;
  cudaFuncGetAttributes(&attr, {ordinary});
  values[0] = min(192, sms * blocks); values[1] = attr.numRegs;
  cudaFuncGetAttributes(&attr, {cluster});
  values[2] = attr.numRegs; values[3] = MLA_SMEM;
  return cudaFuncSetAttribute({cluster}, cudaFuncAttributeMaxDynamicSharedMemorySize, MLA_SMEM);
}}
extern "C" int launch(const uint64_t* pointers, int T, int W, int splits,
                      float sm, float scale, uint64_t stream, int grid, int clustered) {{
  MKMlaArgs args{{}};
  args.q = (const __nv_bfloat16*)pointers[0]; args.ckv = (const uint8_t*)pointers[1];
  args.slots = (const int*)pointers[2]; args.lens = (const int*)pointers[3];
  args.out = (__nv_bfloat16*)pointers[4]; args.part = (float*)pointers[5];
  args.pml = (float*)pointers[6]; args.barrier_ctr = (unsigned long long*)pointers[7];
  args.T = T; args.W = W; args.splits = splits; args.sm_scale = sm; args.ckv_scale = scale;
  args.grid = clustered ? T * splits : grid;
  cudaLaunchConfig_t cfg{{}};
  cfg.gridDim = dim3(args.grid); cfg.blockDim = dim3(MK_THREADS);
  cfg.dynamicSmemBytes = MLA_SMEM; cfg.stream = (cudaStream_t)stream;
  cudaLaunchAttribute attribute{{}};
  attribute.id = cudaLaunchAttributeClusterDimension;
  attribute.val.clusterDim = {{(unsigned)splits, 1, 1}};
  cfg.attrs = &attribute; cfg.numAttrs = clustered ? 1 : 0;
  if (clustered) return cudaLaunchKernelEx(&cfg, {cluster}, args);
  return cudaLaunchKernelEx(&cfg, {ordinary}, args);
}}
__global__ void convert_pairs(const uint8_t* input, uint32_t* output) {{
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < 65536) {{
    output[i] = mla_e4m3x2(input + i * 2);
    output[65536+i] = mla_e4m3x2_strided(input + 131072 + i, 65536);
    output[131072+i] = {packed_pair};
  }}
}}
extern "C" int convert(uint64_t input, uint64_t output, uint64_t stream) {{
  convert_pairs<<<256, 256, 0, (cudaStream_t)stream>>>((const uint8_t*)input, (uint32_t*)output);
  return cudaGetLastError();
}}
__global__ void max_values(const float* input, float* output, int size) {{
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < size) output[i] = mla_warp_max(input[i]);
}}
extern "C" int warp_max(uint64_t input, uint64_t output, int size, uint64_t stream) {{
  max_values<<<(size+255)/256, 256, 0, (cudaStream_t)stream>>>((const float*)input, (float*)output, size);
  return cudaGetLastError();
}}
"""


def build(path, directory, name):
    text = path.read_text()
    code = cuda_source(text)
    cu = directory / (name + ".cu")
    library = directory / (name + ".so")
    if not (cu.exists() and cu.read_text() == code and library.exists()):
        cu.write_text(code)
        command = ["nvcc", "-O2", "-std=c++17", "-arch=sm_121a", "--shared",
                   "-Xcompiler=-fPIC", "--ptxas-options=-v", str(cu), "-o", str(library)]
        compiled = subprocess.run(command, capture_output=True, text=True, timeout=180)
        (directory / (name + "-compile.log")).write_text(compiled.stdout + compiled.stderr)
        if compiled.returncode:
            raise RuntimeError(compiled.stderr)
    lib = ctypes.CDLL(str(library.resolve()))
    lib.launch.argtypes = [ctypes.POINTER(ctypes.c_uint64)] + [ctypes.c_int] * 3 + [
        ctypes.c_float, ctypes.c_float, ctypes.c_uint64, ctypes.c_int, ctypes.c_int]
    lib.convert.argtypes = [ctypes.c_uint64] * 3
    lib.warp_max.argtypes = [ctypes.c_uint64,ctypes.c_uint64,ctypes.c_int,ctypes.c_uint64]
    info = (ctypes.c_int * 4)()
    assert lib.setup(info) == 0
    return lib, list(info), hashlib.sha256(text.encode()).hexdigest()


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--baseline", type=Path, required=True)
    ap.add_argument("--candidate", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--shapes", help="Explicit T:W pairs, separated by commas, for candidate screening.")
    args = ap.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("CUDA_MODULE_LOADING", "EAGER")
    import torch
    assert torch.cuda.get_device_capability() == (12, 1)
    torch.cuda.set_per_process_memory_fraction((768 * 2**20) / torch.cuda.get_device_properties(0).total_memory)
    torch.manual_seed(512)
    torch.backends.cuda.matmul.allow_tf32 = False
    old, old_info, old_hash = build(args.baseline, args.output, "baseline")
    new, new_info, new_hash = build(args.candidate, args.output, "candidate")
    print("kernel_info", old_info, new_info, flush=True)
    stream = torch.cuda.current_stream().cuda_stream
    output = {"baseline_sha256": old_hash, "candidate_sha256": new_hash,
              "harness_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              "torch_version": torch.__version__, "cuda_version": torch.version.cuda,
              "device_name": torch.cuda.get_device_name(),
              "baseline_info": old_info, "candidate_info": new_info, "cases": []}

    # Exhaust every possible packed byte pair, including subnormals, signed
    # zeros and NaNs. Compare integer storage, not floating point NaN equality.
    pairs = torch.arange(65536, dtype=torch.int32, device="cuda")
    pair_bytes = torch.cat((torch.stack((pairs & 255, pairs >> 8), dim=1).flatten(),
                            pairs & 255, pairs >> 8)).to(torch.uint8)
    converted = [torch.empty(196608, dtype=torch.int32, device="cuda") for _ in range(2)]
    for index, lib in enumerate((old, new)):
        assert lib.convert(pair_bytes.data_ptr(), converted[index].data_ptr(), stream) == 0
    torch.cuda.synchronize()
    assert torch.equal(*converted), "FP8 conversion changed bits"
    output["fp8_pair_bits_exact"] = True
    output["packed_vs_strided_fp8_pair_bits_exact"] = True
    max_input = torch.randint(-(2**31),2**31-1,(4096,32),dtype=torch.int32,device="cuda").view(torch.float32)
    max_input[0].fill_(-float("inf")); max_input[1].fill_(float("inf"))
    max_input[2].fill_(float("nan")); max_input[3].fill_(-0.0)
    max_input[4].zero_(); max_input[4,::2] = -0.0
    max_input[5].fill_(float("nan")); max_input[5,0] = -float("inf")
    max_outputs = [torch.empty_like(max_input) for _ in range(2)]
    for index, lib in enumerate((old,new)):
        assert lib.warp_max(max_input.data_ptr(),max_outputs[index].data_ptr(),max_input.numel(),stream)==0
    torch.cuda.synchronize()
    finite = ~torch.isnan(max_outputs[0])
    assert torch.equal(torch.isnan(max_outputs[0]),torch.isnan(max_outputs[1]))
    assert torch.equal(max_outputs[0][finite],max_outputs[1][finite])
    output["warp_max_4096_vectors_equal"] = True
    cache = torch.empty(131072, 512, dtype=torch.float8_e4m3fn, device="cuda")
    for start in range(0, len(cache), 8192):
        cache[start:start+8192] = (torch.randn(8192, 512, device="cuda") * .5).to(torch.float8_e4m3fn)
    eviction = torch.zeros(16 * 2**20, dtype=torch.float32, device="cuda")
    splits_for = lambda t, grid: next((s for s in range(1, min(64, 192 // t)+1)
                                      if t*s % grid == 0), min(64, 192//t, max(1, round(grid/t)))) if t<=64 else 1
    shapes = [(6,2048),(24,2048),(32,2048),(48,2048),(64,2048),(128,2048)] if args.quick else [
        (1,64),(1,2048),(6,512),(6,2048),(8,2048),(12,64),(12,512),(12,2048),
        (16,2048),(18,2048),(24,512),(24,2048),(32,64),(32,512),(32,2048),
        (32,1),(32,17),(32,2176),(33,2048),(36,2048),(40,2048),(42,2048),
        (48,64),(48,512),(48,2048),(48,2176),(54,2048),(60,2048),(63,2048),
        (64,1),(64,2048),(64,2176),(128,2048),
        (512,2048),(2048,2048),(4096,2048),(6912,2048),(8192,2176)]
    if args.shapes:
        shapes = [tuple(map(int, shape.split(':'))) for shape in args.shapes.split(',')]
        if any(len(shape) != 2 or not 1 <= shape[0] <= 8192 or not 1 <= shape[1] <= 2176
               for shape in shapes):
            ap.error('--shapes requires bounded T:W pairs (T<=8192, W<=2176)')
    for T,W in shapes:
        q = torch.randn(T,16,512,dtype=torch.bfloat16,device="cuda") * .3
        slots = torch.randint(len(cache),(T,W),dtype=torch.int32,device="cuda")
        lens = torch.full((T,),W,dtype=torch.int32,device="cuda")
        splits = splits_for(T,old_info[0])
        # The real unsplit path needs no partials; avoid allocating hundreds
        # of MiB of unused scratch when checking full prefill shapes.
        part = torch.empty(T*splits*16*512 if splits>1 else 1,dtype=torch.float32,device="cuda")
        ml = torch.empty(T*splits*32 if splits>1 else 1,dtype=torch.float32,device="cuda")
        counter = torch.zeros(8,dtype=torch.int32,device="cuda")
        variants = [("baseline",old,0,old_info[0]), ("candidate",new,0,new_info[0])]
        if 2 <= splits <= 8:
            # Isolate a candidate's cluster change from the cluster speedup
            # that already exists in the supplied baseline.
            if "template <bool CLUSTER = false>" in args.baseline.read_text():
                variants.append(("baseline_cluster",old,1,old_info[0]))
            variants.append(("cluster",new,1,new_info[0]))
        values = {name:torch.empty_like(q) for name, *_ in variants}
        functions = {}
        for name,lib,cluster,grid in variants:
            ptr_values = [x.data_ptr() for x in (q,cache,slots,lens,values[name],part,ml,counter)]
            if cluster: ptr_values[5:] = [0,0,0]
            pointers = (ctypes.c_uint64 * 8)(*ptr_values)
            def run(lib=lib,pointers=pointers,cluster=cluster,grid=grid):
                status = lib.launch(pointers,T,W,splits,512**-.5,.7,
                                    torch.cuda.current_stream().cuda_stream,grid,cluster)
                if status: raise RuntimeError(f"CUDA launch error {status}")
            functions[name] = run
        for fixture in ("full", "ragged", "empty", "duplicates"):
            lens.fill_(W)
            if fixture == "ragged":
                lens.copy_(torch.arange(T,device="cuda",dtype=torch.int32) * (W//max(1,T)) % (W+1))
                lens[-1] = W
            if fixture == "empty": lens.zero_()
            if fixture == "duplicates": slots[:,::2] = slots[:,0:1].clone()
            for run in functions.values(): run()
            torch.cuda.synchronize()
            for name in values:
                if not torch.equal(values["baseline"].view(torch.int16),values[name].view(torch.int16)):
                    error = ((values[name].float()-values["baseline"].float()).abs().max()).item()
                    raise AssertionError((T,W,fixture,name,error))
            if fixture != "empty":
                for row in sorted({0,T-1}):
                    n=int(lens[row])
                    if n == 0: continue
                    c=cache.view(torch.uint8)[slots[row,:n].long()].view(torch.float8_e4m3fn).float() * .7
                    ref=torch.softmax((q[row].float() @ c.T) * (512**-.5),dim=-1) @ c
                    err=((values["baseline"][row].float()-ref).norm()/ref.norm().clamp_min(1e-12)).item()
                    assert err < .02,(T,W,fixture,err)
        lens.fill_(W)
        slots.copy_(torch.randint(len(cache),(T,W),dtype=torch.int32,device="cuda"))
        graphs={}
        for name,run in functions.items():
            g=torch.cuda.CUDAGraph()
            with torch.cuda.graph(g): run()
            graphs[name]=g
        for replay_index in range(3):
            lens.copy_(torch.arange(T,device="cuda",dtype=torch.int32) * 17 % (W+1))
            if replay_index == 1: lens.zero_()
            if replay_index == 2: lens.fill_(W)
            q.mul_(.75)
            for graph in graphs.values(): graph.replay()
            torch.cuda.synchronize()
            for name,value in values.items():
                assert torch.equal(values["baseline"].view(torch.int16),value.view(torch.int16)),(T,W,name,"graph replay")
        lens.fill_(W)
        samples={name:{"warm":[],"evicted":[]} for name in graphs}
        for mode in ("warm","evicted"):
            for iteration in range(7):
                order=list(graphs) if iteration%2==0 else list(reversed(graphs))
                for name in order:
                    if mode=="evicted": eviction.add_(1)
                    else: graphs[name].replay()
                    start=torch.cuda.Event(enable_timing=True); end=torch.cuda.Event(enable_timing=True)
                    start.record(); graphs[name].replay(); end.record(); end.synchronize()
                    samples[name][mode].append(start.elapsed_time(end)*1000)
        row={"T":T,"W":W,"splits":splits,"bits_exact":True,"graph_replay_exact":True,"samples_us":samples,
             "median_us":{name:{m:statistics.median(v) for m,v in modes.items()} for name,modes in samples.items()}}
        output["cases"].append(row)
        print(json.dumps({k:v for k,v in row.items() if k!='samples_us'}),flush=True)
        (args.output/"results.json").write_text(json.dumps(output,indent=2)+"\n")
        del q, slots, lens, part, ml, counter, values, value, graphs, functions
    print("PASS",flush=True)


if __name__ == "__main__":
    main()
