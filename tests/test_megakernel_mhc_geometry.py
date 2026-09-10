#!/usr/bin/env python3
"""MHC geometry control/ABI regressions, without torch, CUDA, or a GPU.

Extracted C++ exercises the real host dispatch, numerical contract branches,
and index/publication expressions; it does not prove device
codegen, concurrent-stream safety, GPU numerics, or performance.
"""
from __future__ import annotations
import pathlib
import re
import shutil
import subprocess
import tempfile
import unittest

ROOT = pathlib.Path(__file__).resolve().parents[1]
SOURCE = ROOT / "overlay/modules/glm53_megakernel/glm53_megakernel.cu"

def construct(source, marker):
    start = source.index(marker)
    opening = source.index("{", start)
    end, depth = opening + 1, 1
    while depth:
        depth += (source[end] == "{") - (source[end] == "}")
        end += 1
    return source[start:end]


def body(source, marker):
    text = construct(source, marker)
    return text[text.index("{") + 1:-1]


def legacy_body(text):
    text = re.sub(r"^  constexpr int (HIDDEN|NCHUNK|MHC_EPT) = MHIDDEN(?: / (?:HCHUNK|MK_THREADS))?;\n", "", text, flags=re.M)
    while "if constexpr (V41) {" in text:
        start = text.index("if constexpr (V41) {")
        line = text.rfind("\n", 0, start) + 1
        indent = text[line:start]
        branch = construct(text, "if constexpr (V41) {")
        after = start + len(branch)
        assert text[after:after + 7] == " else {"
        other = construct(text[after:], " else {")
        selected = other[other.index("{") + 1:-1]
        selected = "\n".join(v[2:] if v.startswith("  ") else v for v in selected.split("\n"))
        # Replace the complete if/else statement, preserving baseline indentation.
        selected = selected.removeprefix("\n").rstrip() + "\n"
        end = after + len(other)
        if text[end:end + 1] == "\n": end += 1
        text = text[:line] + selected + text[end:]
    text = text.replace("        float v;\n        v = pm[j] * xv;", "        float v = pm[j] * xv;")
    text = re.sub(r"  // Preserve the established 4096 schedule\.[\s\S]*?max\(1, a.grid / NCHUNK\)\);", "  const int groups = max(1, a.grid / NCHUNK);  // token groups per chunk", text, count=1)
    return text.replace("MhcTailRegs<MHIDDEN> tr;", "MhcTailRegs tr;")


def compile_run(source):
    compiler = shutil.which("clang++") or shutil.which("g++")
    if not compiler:
        raise RuntimeError("C++17 compiler required for source-extracted checks")
    with tempfile.TemporaryDirectory(prefix="mhc-geometry-") as td:
        cpp, exe = pathlib.Path(td) / "test.cpp", pathlib.Path(td) / "test"
        cpp.write_text(source)
        subprocess.run([compiler, "-std=c++17", "-O1", str(cpp), "-o", str(exe)], check=True, capture_output=True, text=True)
        return subprocess.run([str(exe)], check=True, capture_output=True, text=True).stdout


class MhcGeometryTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.source = SOURCE.read_text()

    def test_v41_post_collapse_rounding_and_external_pre_contract(self):
        p1 = body(self.source, "__device__ void mk_mhc_p1_impl(")
        post = construct(p1[p1.index("float r[HC], sqr"):], "for (int j = 0; j < HC; ++j)")
        p34 = body(self.source, "__device__ void mk_mhc_p34_compute(")
        collapse = construct(p34, "for (int i = 0; i < MHC_EPT; ++i)")
        p2 = body(self.source, "__device__ void mk_mhc_p2_token(")
        pre_select = construct(p2, "if (lane >= 2 * HC && lane < 3 * HC)")
        code = r'''#include <cmath>
#include <cstring>
#include <cstdint>
#include <stdexcept>
#include <iostream>
constexpr int HC=4, HIDDEN=5120, MHC_EPT=20;
using __nv_bfloat16=uint16_t;
uint16_t __float2bfloat16(float x){uint32_t b;std::memcpy(&b,&x,4);b+=0x7fff+((b>>16)&1);return uint16_t(b>>16);}
float __bfloat162float(uint16_t x){uint32_t b=uint32_t(x)<<16;float f;std::memcpy(&f,&b,4);return f;}
void need(bool x){if(!x)throw std::runtime_error("V4.1 rounding/pre seam");}
float mk_sigmoid(float x){return 1.f/(1.f+std::exp(-x));}
struct {int x=0;} threadIdx;
struct Args{uint16_t* residual_out;float* current_pre_output;const float* collapse_pre_input;float pre_eps;} a;
template<bool V41> float post_case(){
 uint16_t output[HC*HIDDEN]{};a.residual_out=output;
 const int t=0,h=0;float pm[HC]={.333f,.27f,.713f,.117f},xv=1.5f;
 float res[HC]={1.0078125f,1.25f,.5f,1.75f};
 float cm[HC][HC]={{.1f,.2f,.3f,.4f},{.125f,.375f,.2f,.5f},{.333f,.75f,.1f,.25f},{.6f,.05f,.125f,.333f}};
 float r[HC],sqr=0;
''' + post + r'''
 float expected_sq=0;bool differs=false;
 for(int j=0;j<HC;++j){float v=0;for(int k=0;k<HC;++k)v+=cm[k][j]*res[k];v=pm[j]*xv+v;
 float rounded=__bfloat162float(__float2bfloat16(v));differs|=(v!=rounded);
 if constexpr(V41){need(r[j]==rounded);expected_sq+=rounded*rounded;}
 need(output[j*HIDDEN]==__float2bfloat16(r[j]));}
 need(differs);if constexpr(V41)need(sqr==expected_sq);return sqr;
}
template<bool V41> float collapse_case(){
 struct {float res[HC][MHC_EPT];} r;float pre[HC]={.17f,.71f,.31f,.11f};
 for(int j=0;j<HC;++j)for(int i=0;i<MHC_EPT;++i)r.res[j][i]=__bfloat162float(__float2bfloat16(.3f*(i+1)+.17f*j));
 float vals[MHC_EPT],sq=0;
''' + collapse + r'''
 float expected_sq=0;
 for(int i=0;i<MHC_EPT;++i){float v=0;for(int j=0;j<HC;++j)v+=pre[j]*r.res[j][i];
 float rounded=__bfloat162float(__float2bfloat16(v));need(vals[i]==rounded);
 expected_sq+=(V41?rounded:v)*(V41?rounded:v);}
 need(sq==expected_sq);return sq;
}
template<bool V41> void pre_case(){
 float current[HC]={},external[HC]={.13f,.31f,.57f,.79f},s_pmix[HC]={};
 a.current_pre_output=current;a.collapse_pre_input=external;a.pre_eps=1e-6f;
 const int t=0;const float hs0=.7f,hb_pre=.2f,pre_in=.31f;
 for(int lane=8;lane<12;++lane){
''' + pre_select + r'''
 }
 for(int j=0;j<HC;++j){float computed=mk_sigmoid(pre_in*hs0+hb_pre)+a.pre_eps;
 if constexpr(V41){need(s_pmix[j]==external[j]);need(current[j]==computed);need(current[j]!=s_pmix[j]);}
 else {need(s_pmix[j]==computed);need(current[j]==0);}}
}
int main(){need(post_case<true>()!=post_case<false>());need(collapse_case<true>()!=collapse_case<false>());
pre_case<true>();pre_case<false>();std::cout<<"V41 seam PASS\n";}
'''
        self.assertIn("V41 seam PASS", compile_run(code))

    def test_all_projection_chunks_publish_once_and_tail_covers_hidden(self):
        p1 = body(self.source, "__device__ void mk_mhc_p1_impl(")
        declaration = re.search(r"  const int groups = [\s\S]*?;", p1).group(0)
        loop = re.search(r"  for \(int cg = bid;.*?\{", p1).group(0)
        coordinates = re.search(r"    const int c = cg % NCHUNK, g = cg / NCHUNK;", p1).group(0)
        token_loop = re.search(r"    for \(int t = g;.*?\{", p1).group(0)
        publication = p1[p1.index("if (pend >= 0)"):p1.index("      xv = nxv;")]
        tail = body(self.source, "__device__ __forceinline__ void mk_mhc_p34_load(")
        tail_loop = re.search(r"  for \(int i = 0; i < MHC_EPT; \+\+i\) \{", tail).group(0)
        tail_h = re.search(r"    const int h = .*?;", tail).group(0)
        code = r'''#include <algorithm>
#include <vector>
#include <stdexcept>
#include <iostream>
using std::max; using std::min;
void need(bool x) { if (!x) throw std::runtime_error("mapping/publication"); }
constexpr int MK_THREADS=256, HCHUNK=256, MAX_TOK=32, NOUT=24;
struct { int x=0; } threadIdx;
unsigned g_mk_mhc_tok_arrive[MAX_TOK];
void __threadfence() {} void __syncthreads() {}
unsigned atomicAdd(unsigned* p,unsigned n){auto old=*p;*p+=n;return old;}
int main() {
for(int HIDDEN: {4096,5120}) for(bool V41:{false,true}) {
 int MHIDDEN=HIDDEN;
 int NCHUNK=HIDDEN/HCHUNK,MHC_EPT=HIDDEN/MK_THREADS;
 std::vector<int> coverage(HIDDEN);
 for(int tid=0;tid<MK_THREADS;++tid){threadIdx.x=tid;
''' + tail_loop + tail_h + r'''
 need(h>=0 && h<HIDDEN); ++coverage[h]; }}
 for(auto n:coverage)need(n==1);
 for(int grid=1;grid<=144;++grid)for(int tokens=1;tokens<=MAX_TOK;++tokens) {
 struct { int grid,num_tokens; } a{grid,tokens};
 for(int replay=0;replay<2;++replay) {
 std::fill_n(g_mk_mhc_tok_arrive,MAX_TOK,0u);
 std::vector<int> pairs(NCHUNK*tokens);
 threadIdx.x=0;
 for(int bid=0;bid<grid;++bid){
''' + declaration + loop + coordinates + r'''
 if(V41 || HIDDEN!=4096)need(g<tokens);
 int pend=-1;
''' + token_loop + r'''
 need(c>=0 && c<NCHUNK && t>=0 && t<tokens);
 ++pairs[c*tokens+t];
 const size_t yp=((size_t)c*MAX_TOK+t)*NOUT+(NOUT-1);
 need(yp<size_t(NCHUNK*MAX_TOK*NOUT));
''' + publication + r'''
 } need(pend==-1); }}
 for(auto n:pairs)need(n==1);
 for(int t=0;t<tokens;++t)need(g_mk_mhc_tok_arrive[t]==unsigned(NCHUNK));
 }}} std::cout<<"mapping PASS\n";}
'''
        self.assertIn("mapping PASS", compile_run(code))

    def test_host_dispatch_admission_and_independent_occupancy(self):
        args = "template <int MHIDDEN, bool V41=false>\n" + construct(self.source, "struct MKMhcArgsT {") + ";\n"
        args += "template <int MHIDDEN>\n" + construct(self.source, "struct MKMhcArgsT<MHIDDEN, true>") + ";\nusing MKMhcArgs=MKMhcArgsT<4096>;\n"
        host = "template <int MHIDDEN, bool V41=false>\n" + construct(self.source, "void mk_run_mhc_geometry(")
        entry = construct(self.source, "void mk_run_mhc(") + "\n" + construct(self.source, "void mk_run_mhc_v41(")
        code = r'''#include <algorithm>
#include <cmath>
#include <limits>
#include <cstdint>
#include <vector>
#include <stdexcept>
#include <iostream>
using __nv_bfloat16=uint16_t;
constexpr int HIDDEN=4096, HCHUNK=256, MK_THREADS=256, MAX_TOK=32, MK_MHC_GRID_CAP=144;
void need(bool x){if(!x)throw std::runtime_error("host ABI");}
#define TORCH_CHECK(x,...) need(bool(x))
#define MK_CHECK_CUDA(x) need((x)==0)
''' + args + r'''
void mk_mhc_kernel(MKMhcArgs){} void mk_mhc_bf16_kernel(MKMhcArgs){}
template<bool BF> void mk_mhc_ar_kernel(MKMhcArgs){}
template<int H,bool BF,bool AR,bool V41=false> void mk_mhc_geometry_kernel(MKMhcArgsT<H,V41>){}
int queries=0,launches=0,current_device=0; bool pdl=true;
void set_kernel_attrs(){} bool mk_pdl_enabled(){return pdl;}
namespace c10 { namespace cuda {int getCurrentCUDAStream(){return 0;}} }
constexpr int cudaDevAttrMultiProcessorCount=1;
int cudaGetDevice(int* n){*n=current_device;return 0;}
int cudaDeviceGetAttribute(int* n,int,int device){need(device==current_device);*n=(device==0?48:24);return 0;}
template<int H,bool V41> int cudaOccupancyMaxActiveBlocksPerMultiprocessor(int* n,void(*)(MKMhcArgsT<H,V41>),int,int){++queries;*n=(H==4096?3:2);return 0;}
template<int H,bool V41> void mk_launch(void(*kernel)(MKMhcArgsT<H,V41>),int grid,int smem,int,const MKMhcArgsT<H,V41>& a){
 bool early;
 if constexpr(V41) early=false;
 else if constexpr(H==4096) early=kernel==mk_mhc_ar_kernel<false> || kernel==mk_mhc_ar_kernel<true>;
 else early=kernel==mk_mhc_geometry_kernel<H,false,true> || kernel==mk_mhc_geometry_kernel<H,true,true>;
 int sms=(current_device==0?48:24);
 need(grid==(early?48:(H==4096?3:2)*sms)); need(a.grid==grid&&smem==0); ++launches;
}
''' + host + "\n" + entry + r'''
int main(){std::vector<int64_t> p(18,0),i{6,20};std::vector<double>s(5,1e-6);
for(int repeat=0;repeat<2;++repeat){for(bool bf:{false,true})for(bool ar:{false,true})mk_run_mhc(p,s,i,bf,ar,4096);mk_run_mhc(p,s,i,false,false,5120);}
need(queries==5 && launches==10);
mk_run_mhc(p,s,i); need(queries==5&&launches==11);
std::vector<int64_t> p41(20,0);
for(int repeat=0;repeat<2;++repeat)for(int h:{4096,5120})mk_run_mhc_v41(p41,s,i,h);
need(queries==7&&launches==15);
current_device=1;
for(int repeat=0;repeat<2;++repeat){mk_run_mhc(p,s,i,false,false,5120);for(int h:{4096,5120})mk_run_mhc_v41(p41,s,i,h);}
need(queries==10&&launches==21);
current_device=0;mk_run_mhc(p,s,i,false,false,5120);need(queries==10&&launches==22);
auto bad=[&](auto fn){bool rejected=false;try{fn();}catch(const std::runtime_error&){rejected=true;}need(rejected);};
bad([&]{mk_run_mhc(p,s,i,false,false,8192);});
for(int t:{0,33})bad([&]{mk_run_mhc(p,s,{t,20},false,false,5120);});
bad([&]{mk_run_mhc(p,s,{},false,false,5120);});
bad([&]{mk_run_mhc({},s,i,false,false,5120);});
bad([&]{mk_run_mhc(p,{},i,false,false,5120);});
bad([&]{mk_run_mhc(p,s,{9,20},false,true,5120);});
bad([&]{mk_run_mhc(p,s,i,true,false,5120);});
bad([&]{mk_run_mhc_v41(p,s,i);});
bad([&]{mk_run_mhc_v41(p41,s,{0,20});});
bad([&]{mk_run_mhc_v41(p41,s,i,1234);});
for(int repeat:{0,65})bad([&]{mk_run_mhc_v41(p41,s,{6,repeat});});
for(int index=0;index<5;++index)for(double value:{-1.,std::numeric_limits<double>::infinity(),std::numeric_limits<double>::quiet_NaN(),1e100}){
 auto invalid=s;invalid[index]=value;bad([&]{mk_run_mhc_v41(p41,invalid,i);});bad([&]{mk_run_mhc(p,invalid,i,false,false,5120);});}
for(int index:{0,2,4}){auto invalid=s;invalid[index]=0;bad([&]{mk_run_mhc_v41(p41,invalid,i);});invalid[index]=1e-100;bad([&]{mk_run_mhc_v41(p41,invalid,i);});}
pdl=false;bad([&]{mk_run_mhc(p,s,i,false,true,4096);});need(launches==22);
static_assert(sizeof(MKMhcArgsT<4096>)==sizeof(MKMhcArgsT<5120>));
std::cout<<"host PASS\n";}
'''
        self.assertIn("host PASS", compile_run(code))

    def test_geometry_is_mhc_only_and_complete(self):
        self.assertIn("constexpr int HIDDEN = 4096;", self.source)
        self.assertIn('pybind11::arg("hidden") = HIDDEN', self.source)
        for marker in ("__device__ void mk_mhc_p1_impl(", "__device__ void mk_mhc_p2_token("):
            text = body(self.source, marker)
            self.assertIn("constexpr int NCHUNK = MHIDDEN / HCHUNK;", text)
            self.assertIn("constexpr int HIDDEN = MHIDDEN;", text)
        tail = construct(self.source, "struct MhcTailRegs {")
        self.assertIn("static constexpr int MHC_EPT = MHIDDEN / MK_THREADS;", tail)
        self.assertEqual(5120//256,20)
        self.assertEqual(20*32*24*4,61440)
        self.assertEqual(20*32*4,2560)

    def test_new_entry_retains_dependency_and_shared_ticket_lifecycle(self):
        entry = body(self.source, "__global__ void mk_mhc_geometry_kernel(")
        self.assertIn('if constexpr (!AR_CONSUMER)\n    asm volatile("griddepcontrol.wait;" ::: "memory");', entry)
        self.assertIn("mk_mhc_p1_impl<BF16_FN, AR_CONSUMER>(a, blockIdx.x);", entry)
        p1 = body(self.source, "__device__ void mk_mhc_p1_impl(")
        self.assertIn('if (bid >= NCHUNK * groups)\n      asm volatile("griddepcontrol.wait;" ::: "memory");', p1)
        self.assertIn('MK_SPIN_WAIT(*v < (unsigned int)NCHUNK, 128, "mhc token arrive");', p1)
        self.assertIn("if (e + 1u == (unsigned int)a.grid)", p1)
        self.assertIn("g_mk_mhc_tail_next = 0u;", p1)
        # Tickets remain shared: passing this check does not imply multi-stream safety.
        self.assertEqual(self.source.count("__device__ unsigned int g_mk_mhc_tail_next = 0u;"),1)

    def test_legacy_rounding_and_scalar_separation_are_explicit(self):
        p1 = legacy_body(body(self.source, "__device__ void mk_mhc_p1_impl("))
        self.assertLess(p1.index("r[j] = v;"),p1.index("sqr += v * v;"))
        self.assertIn("v += fnr[m][j] * r[j];",p1)
        p34 = legacy_body(body(self.source,"__device__ void mk_mhc_p34_compute("))
        self.assertLess(p34.index("sq += v * v;"),p34.index("vals[i] = __bfloat162float(__float2bfloat16(v));"))
        self.assertIn("a.rms_eps",body(self.source,"__device__ void mk_mhc_p2_token("))
        self.assertIn("a.norm_eps",p34)
        self.assertIn("a.sinkhorn_eps",body(self.source,"__device__ void mk_mhc_p2_token("))


if __name__ == "__main__":
    unittest.main()
