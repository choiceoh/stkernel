"""Exercise actual MLA copy scheduling and reduction addresses on the CPU.

The extracted copy code runs against a deliberately delayed async-copy model:
copies complete only when wait_group guarantees them. This checks the schedule
and address contract, not GPU timing, cross-thread visibility, or execution.
"""
import argparse
import hashlib
import json
from pathlib import Path
import re
import resource
import subprocess


def host_source(source):
    body = source[source.index('void mk_mla_kernel('):source.index('// Exact-selection prefill pair reuse')]
    q = body[body.index('    if (j1 > j0) {'):body.index('    float acc[8][4];')]
    q, replacements = re.subn(
        r'asm volatile\("cp\.async\.ca\.shared\.global \[%0\], \[%1\], 16;"\s*'
        r':: "r"\(dst\), "l"\(src\) : "memory"\);', 'host_copy(dst, src);', q)
    assert replacements == 1
    schedule = body[body.index('    const int ntile ='):body.index('      const uint8_t* tile8 =')]
    constants = '\n'.join(re.search(r'constexpr int '+name+r'\s*=.*?;', source).group()
                          for name in ('MLA_D','MLA_H','MLA_WARPS','MLA_TILE','MLA_NSTAGE',
                                       'MLA_KQ','MLA_CP','MLA_PP','MLA_RP','MLA_SMEM_RING',
                                       'MLA_SMEM_Q','MLA_SMEM_S','MLA_SMEM_P','MLA_SMEM_C','MLA_SMEM'))
    head = re.search(r'for \(int h = (.*?); h < MLA_H; h \+= (.*?)\)', body).groups()
    combine = body[body.index('// phase 1 -- log-sum-exp combine'):]
    read_index = re.search(r'o\[e\] = fmaf\(src\[(.*?)\]', combine).group(1)
    write_index = re.search(r'dst\[(.*?)\] = __float2bfloat16', combine).group(1)
    assert ' + lane * MLA_VD' not in combine
    return r'''
#include <algorithm>
#include <cassert>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <deque>
#include <vector>
using std::min;
using __nv_bfloat16 = uint16_t;
constexpr int MK_THREADS = 256;
''' + constants + r'''
struct Copy { uint32_t dst; const void* src; };
struct { int x; } threadIdx;
struct Args { const uint16_t* q; const uint8_t* ckv; const int* slots; int W, splits; };
std::vector<uint8_t> memory(MLA_SMEM);
std::vector<uint16_t> query(3*MLA_H*MLA_D);
std::vector<uint8_t> cache(257*MLA_D);
std::vector<int> slots(3*2304);
std::vector<Copy> uncommitted;
std::deque<std::vector<Copy>> groups;
int copies, barriers;
uint32_t __cvta_generic_to_shared(const void* p) {
  return static_cast<const uint8_t*>(p)-memory.data();
}
bool inside(uintptr_t p, const void* base, size_t size) {
  return p>=reinterpret_cast<uintptr_t>(base) && p+16<=reinterpret_cast<uintptr_t>(base)+size;
}
void host_copy(uint32_t dst, const void* src) {
  auto p=reinterpret_cast<uintptr_t>(src);
  assert(dst+16<=memory.size() && dst%16==0 && p%16==0);
  assert(inside(p,query.data(),query.size()*2) || inside(p,cache.data(),cache.size()));
  uncommitted.push_back({dst,src}); ++copies;
}
void mk_cp_async16(void* dst, const void* src) { host_copy(__cvta_generic_to_shared(dst),src); }
void mk_cp_commit() { groups.push_back(uncommitted); uncommitted.clear(); }
template<int N> void mk_cp_wait() {
  while (groups.size()>N) {
    for (auto copy:groups.front()) std::memcpy(memory.data()+copy.dst,copy.src,16);
    groups.pop_front();
  }
}
void __syncthreads() { ++barriers; }
void check_slice(int t, int j0, int j1) {
  Args a{query.data(),cache.data(),slots.data(),2304,3};
  for (int tid=0;tid<MK_THREADS;++tid) {
    threadIdx.x=tid; int lane=tid%32, warp=tid/32;
    auto* ring=memory.data(); auto* sq=reinterpret_cast<uint16_t*>(ring+MLA_SMEM_RING);
    std::fill(memory.begin(),memory.end(),0xa5);
    uncommitted.clear(); groups.clear(); copies=barriers=0;
''' + q + schedule + r'''
      assert(barriers==ti+1);
      // Independent row-major query and selected-slot references for this thread.
      for (int linear=tid;linear<MLA_H*MLA_D/8;linear+=MK_THREADS) {
        int row=linear/64, col=(linear%64)*8;
        assert(std::memcmp(sq+row*MLA_CP+col,a.q+(t*MLA_H+row)*MLA_D+col,16)==0);
      }
      for (int row=warp;row<MLA_TILE;row+=MLA_WARPS) {
        int position=j0+ti*MLA_TILE+row;
        int selected=slots[t*a.W+(position<j1?position:j0)];
        auto* dst=ring+(ti%MLA_NSTAGE)*MLA_TILE*MLA_RP+row*MLA_RP+lane*16;
        assert(std::memcmp(dst,a.ckv+selected*MLA_D+lane*16,16)==0);
      }
    }
    assert(uncommitted.empty());
    for (auto& group:groups) assert(group.empty());
    assert(copies==(j1>j0 ? 4+2*ntile : 0));
  }
}
int main() {
  for (unsigned i=0;i<query.size();++i) query[i]=uint16_t(i*17+3);
  for (unsigned i=0;i<cache.size();++i) cache[i]=uint8_t((i*29)^(i>>8));
  int cases=0;
  for (int length:{0,1,2,15,16,17,31,32,33,47,48,49,63,64,65,127,128,129,2048,2176}) {
    for (int start:{0,1,17}) {
      int t=cases%3;
      std::fill(slots.begin(),slots.end(),-1);
      for (int j=start;j<start+length;++j) slots[t*2304+j]=(j*13+cases)%257;
      check_slice(t,start,start+length); ++cases;
    }
  }
  for (int splits:{2,3,4,8}) {
    Args a{}; a.splits=splits;
    int coverage[MLA_H]={}, work[8]={};
    for (int rank=0;rank<splits;++rank) for (int warp=0;warp<MLA_WARPS;++warp)
''' + f'      for (int h={head[0]};h<MLA_H;h+={head[1]}) {{ ++coverage[h]; ++work[rank]; }}\n' + r'''
    for (int count:coverage) assert(count==1);
    for (int rank=0;rank<splits;++rank) assert(work[rank]>0);
    if (splits==3) assert(work[0]==6 && work[1]==5 && work[2]==5);
  }
  int seen[MLA_D]={};
  for (int e=0;e<MLA_D/32;++e) for (int lane=0;lane<32;++lane) {
''' + f'    int input={read_index}, output={write_index};\n' + r'''
    assert(input==output && input==e*32+lane); ++seen[output];
  }
  for (int count:seen) assert(count==1);
  std::printf("PASS: %d slices x 256 threads; Q/KV waits, cluster ownership, reduction addresses\n",cases);
}
'''


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, default=Path('engine/kernels/mla/glm53_megakernel.cu'))
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    source = args.source.read_text()
    code = host_source(source)
    cpp = args.output/'stream_check.cpp'
    cpp.write_text(code)
    executable = args.output/'stream_check'
    subprocess.run(['g++','-O2','-std=c++17',str(cpp),'-o',str(executable)],check=True)
    checked = subprocess.run([str(executable.resolve())],capture_output=True,text=True,check=True)
    print(checked.stdout,end='')
    # Prove the delayed-copy model catches a missing first-group dependency.
    mutant = args.output/'missing_wait.cpp'
    assert code.count('mk_cp_wait<MLA_NSTAGE - 2>();') == 1
    mutant.write_text(code.replace('mk_cp_wait<MLA_NSTAGE - 2>();','mk_cp_wait<MLA_NSTAGE - 1>();'))
    bad = args.output/'missing_wait'
    subprocess.run(['g++','-O2','-std=c++17',str(mutant),'-o',str(bad)],check=True)
    resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
    rejected = subprocess.run([str(bad.resolve())],capture_output=True,text=True)
    assert rejected.returncode != 0 and 'memcmp(sq+row*MLA_CP+col' in rejected.stderr, (
        'negative control must fail because Q is not ready: ' + rejected.stderr)
    report = {'source_sha256': hashlib.sha256(source.encode()).hexdigest(),
              'harness_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              'gpu_used': False, 'slice_cases':60, 'threads_per_case':256,
              'cluster_splits':[2,3,4,8], 'three_cta_head_counts':[6,5,5],
              'reduction_dimensions':512, 'missing_wait_rejected':True}
    (args.output/'results.json').write_text(json.dumps(report,indent=2)+'\n')


if __name__ == '__main__':
    main()
