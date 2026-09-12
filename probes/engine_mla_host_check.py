"""Check the adopted MLA conversion helpers and fragment addresses on the CPU.

Uses the CUDA headers' host implementations. This does not execute GPU
instructions or replace the GPU numerical/integration checks.
"""
import argparse
import hashlib
import json
from pathlib import Path
import re
import subprocess


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, default=Path('engine/kernels/mla/glm53_megakernel.cu'))
    parser.add_argument('--cuda-root', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    source = args.source.read_text()
    helpers = source[source.index('// two adjacent e4m3 bytes'):source.index('__device__ __forceinline__ void mla_mma_bf16(')]
    helpers = helpers.replace('__device__ __forceinline__', 'inline')
    row = re.search(r'const int row = lane & 15;.*?const int krow = .*?;', source, re.S).group()
    qaddr = re.search(r'const uint32_t qaddr = .*?sq \+ (.*?)\)\);', source, re.S).group(1)
    paddr = re.search(r'const uint32_t paddr = .*?sp \+ (.*?)\)\);', source, re.S).group(1)
    constants = '\n'.join(re.search(r'constexpr int '+name+r'\s*=.*?;',source).group()
                          for name in ('MLA_D','MLA_H','MLA_TILE','MLA_KQ','MLA_CP','MLA_PP'))
    code = '''#include <cuda_fp8.h>
#include <cuda_bf16.h>
#include <cstdint>
#include <cstdio>
''' + constants + '\n' + helpers + f'''
int fp8_row(int lane) {{ {row} return krow; }}
int q_address(int lane,int kq,int ks) {{ return {qaddr}; }}
int p_address(int lane) {{ return {paddr}; }}
int main() {{
  for (unsigned pair=0;pair<65536;++pair) {{
    alignas(2) uint8_t bytes[2]={{uint8_t(pair),uint8_t(pair>>8)}};
    uint8_t strided[17]={{}}; strided[0]=bytes[0];strided[16]=bytes[1];
    auto old_value=mla_e4m3x2_strided(strided,16);
    if (old_value!=mla_e4m3x2(bytes) || old_value!=mla_e4m3x2_value(pair)) {{
      std::printf("conversion mismatch at %u\\n",pair); return 1;
    }}
  }}
  for (int lane=0;lane<32;++lane) {{
    int g=lane/4,q=lane%4;
    for (int byte=0;byte<4;++byte)
      if (fp8_row(4*q+byte)!=2*q+(byte%2)+8*(byte/2)) return 2;
    for (int reg=0;reg<4;++reg) {{
      int address_lane=reg*8+g;
      int row=g+(reg%2)*8,col=q*2+(reg/2)*8;
      if (p_address(address_lane)+q*2!=row*MLA_PP+col) return 3;
      for (int kq=0;kq<4;++kq) for (int ks=0;ks<8;++ks)
        if (q_address(address_lane,kq,ks)+q*2!=row*MLA_CP+kq*128+ks*16+col) return 4;
    }}
  }}
  std::puts("PASS: 65536 FP8 pairs and adopted FP8/Q/P fragment addresses (CPU)");
}}
'''
    cpp = args.output/'host_check.cpp'
    cpp.write_text(code)
    executable = args.output/'host_check'
    subprocess.run(['g++','-O2','-std=c++17','-I'+str(args.cuda_root/'include'),str(cpp),'-o',str(executable)],check=True)
    result = subprocess.run([str(executable.resolve())],capture_output=True,text=True,check=True)
    report = {'source_sha256':hashlib.sha256(source.encode()).hexdigest(),
              'gpu_used':False,'host_fp8_pairs_exact':65536,'fragment_addresses_pass':True}
    (args.output/'result.json').write_text(json.dumps(report,indent=2)+'\n')
    print(result.stdout, end='')


if __name__ == '__main__':
    main()
