"""Generate isolated SM121 MLA ldmatrix candidates for hardware A/B checks.

No serving files or defaults are modified. The baseline must contain the exact
MMA fragment loads this experiment replaces. Generation alone provides no GPU
execution or timing evidence; record those separately from adoption decisions.
"""
import argparse
from pathlib import Path

old = '''        const int krow = q4 * 2;                     // this lane's two k (slot) rows
        const uint8_t* cb = tile8 + (size_t)krow * MLA_RP + warp * 64;
#pragma unroll
        for (int nt = 0; nt < 8; ++nt) {
          const int n = nt * 8 + g;                  // this lane's output column
          mla_mma_bf16(acc[nt][0], acc[nt][1], acc[nt][2], acc[nt][3], a0, a1, a2, a3,
                       mla_e4m3x2_strided(cb + n, MLA_RP),
                       mla_e4m3x2_strided(cb + 8 * MLA_RP + n, MLA_RP));
        }'''

new = '''        // ldmatrix.trans produces four consecutive K bytes for column g
        // and four for g+8. Permute its row addresses so each register contains
        // the two MMA K pairs (2q,2q+1) and (8+2q,9+2q), in that order.
        const int row = lane & 15;
        const int krow = (row >> 2) * 2 + (row & 1) + ((row >> 1) & 1) * 8;
        const uint32_t cb = static_cast<uint32_t>(__cvta_generic_to_shared(
            tile8 + (size_t)krow * MLA_RP + warp * 64));
#pragma unroll
        for (int nt = 0; nt < 8; nt += 2) {
          uint32_t v0, v1;
          asm volatile("ldmatrix.sync.aligned.m16n16.x1.trans.shared.b8 {%0, %1}, [%2];"
                       : "=r"(v0), "=r"(v1) : "r"(cb + nt * 8));
          mla_mma_bf16(acc[nt][0], acc[nt][1], acc[nt][2], acc[nt][3], a0, a1, a2, a3,
                       mla_e4m3x2_value(v0), mla_e4m3x2_value(v0 >> 16));
          mla_mma_bf16(acc[nt+1][0], acc[nt+1][1], acc[nt+1][2], acc[nt+1][3], a0, a1, a2, a3,
                       mla_e4m3x2_value(v1), mla_e4m3x2_value(v1 >> 16));
        }'''

helper = '''// Convert a packed pair without scalar strided shared loads.
__device__ __forceinline__ uint32_t mla_e4m3x2_value(uint32_t packed) {
  const __half2 h = __nv_cvt_fp8x2_to_halfraw2(static_cast<__nv_fp8x2_storage_t>(packed), __NV_E4M3);
  const __nv_bfloat162 b = __float22bfloat162_rn(__half22float2(h));
  return *(const uint32_t*)&b;
}
'''

before = '''          mla_mma_bf16(c0, c1, c2, c3,
                       *(const uint32_t*)(qa + k0),
                       *(const uint32_t*)(qa + 8 * MLA_CP + k0),
                       *(const uint32_t*)(qa + k0 + 8),
                       *(const uint32_t*)(qa + 8 * MLA_CP + k0 + 8),
                       mla_e4m3x2(cb + k0), mla_e4m3x2(cb + k0 + 8));'''

after = '''          uint32_t a0, a1, a2, a3;
          const uint32_t qaddr = static_cast<uint32_t>(__cvta_generic_to_shared(
              sq + (lane & 15) * MLA_CP + kq * (MLA_D / MLA_KQ) + ks * 16 + (lane >> 4) * 8));
          asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];"
                       : "=r"(a0), "=r"(a1), "=r"(a2), "=r"(a3) : "r"(qaddr));
          mla_mma_bf16(c0, c1, c2, c3, a0, a1, a2, a3,
                       mla_e4m3x2(cb + k0), mla_e4m3x2(cb + k0 + 8));'''

p_before = '''        const uint32_t a0 = *(const uint32_t*)(pa + q4 * 2);
        const uint32_t a1 = *(const uint32_t*)(pa + 8 * MLA_PP + q4 * 2);
        const uint32_t a2 = *(const uint32_t*)(pa + q4 * 2 + 8);
        const uint32_t a3 = *(const uint32_t*)(pa + 8 * MLA_PP + q4 * 2 + 8);'''

p_after = '''        uint32_t a0, a1, a2, a3;
        const uint32_t paddr = static_cast<uint32_t>(__cvta_generic_to_shared(
            sp + (lane & 15) * MLA_PP + (lane >> 4) * 8));
        asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];"
                     : "=r"(a0), "=r"(a1), "=r"(a2), "=r"(a3) : "r"(paddr));'''

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--baseline', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    source = args.baseline.read_text()
    marker = '__device__ __forceinline__ void mla_mma_bf16('
    assert source.count(old) == source.count(marker) == 1
    b8 = source.replace(marker, helper + marker).replace(old, new)
    (args.output / 'ldmatrix_b8.cu').write_text(b8)
    for name, original in [('baseline', source), ('ldmatrix_b8', b8)]:
        assert original.count(before) == original.count(p_before) == 1
        q = original.replace(before, after)
        qp = q.replace(p_before, p_after)
        (args.output / (name + '_q.cu')).write_text(q)
        (args.output / (name + '_qp.cu')).write_text(qp)

    # PTX Figure 105: transposed rows map to groups of four lanes, with four
    # adjacent bytes in each register; r1 has the column eight positions later.
    # Enumerate the logical-to-physical address mapping independently of CUDA.
    for lane in range(32):
        g, q = divmod(lane, 4)
        for nt in range(0, 8, 2):
            for reg in range(2):
                loaded = []
                for byte in range(4):
                    logical_row = q * 4 + byte
                    physical_row = (logical_row // 4) * 2 + logical_row % 2 + (logical_row // 2 % 2) * 8
                    loaded.append((physical_row, nt * 8 + g + reg * 8))
                expected = [(2*q+k, (nt+reg)*8+g) for k in (0,1,8,9)]
                assert loaded == expected, (lane,nt,reg,loaded,expected)
    print('ldmatrix_b8.cu: all 1,024 PV fragment byte coordinates match the baseline')

    # Validate the A operand register placement for all 32 lanes and four matrices.
    for lane in range(32):
        group, thread = divmod(lane, 4)
        for reg in range(4):
            row = group + (reg % 2) * 8
            k = thread * 2 + (reg // 2) * 8
            address_lane = reg * 8 + group
            physical_row = address_lane & 15
            physical_col = (address_lane >> 4) * 8 + thread * 2
            assert (row, k) == (physical_row, physical_col)
    print('Q/P ldmatrix x4: all 256 BF16 coordinates match baseline MMA A')


if __name__ == '__main__':
    main()
