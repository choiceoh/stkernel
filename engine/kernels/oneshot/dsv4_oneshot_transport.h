// Shared by the CUDA transport and device-free C++ protocol oracles.
#pragma once
#include <cstdint>
#include <cstring>

#ifdef __CUDACC__
#define OSAR_HD __host__ __device__
#else
#define OSAR_HD
#endif

constexpr unsigned OSAR_COMPACT_GRID = 12;
constexpr unsigned OSAR_PUBLICATION_TICKETS = 48;
constexpr unsigned OSAR_COMPACT_MAXEL = 8 * 4096;
constexpr unsigned OSAR_COMPACT_VECITER =
    (OSAR_COMPACT_MAXEL / 8 + OSAR_COMPACT_GRID * 256 - 1) /
    (OSAR_COMPACT_GRID * 256);
static_assert(OSAR_PUBLICATION_TICKETS % OSAR_COMPACT_GRID == 0);

OSAR_HD constexpr bool osar_compact_eligible(uint64_t n) {
  return n > 0 && n <= OSAR_COMPACT_MAXEL;
}

// Peers are stored in ascending rank order with this rank omitted. Always
// fold rank 0, 1, 2, 3: local-first addition lets replicated hidden states
// disagree after cancellation, despite receiving exactly the same bytes.
OSAR_HD constexpr float osar_sum_rank_order(float mine, float p0, float p1,
                                            float p2, int rank) {
  return rank == 0 ? ((mine + p0) + p1) + p2
       : rank == 1 ? ((p0 + mine) + p1) + p2
       : rank == 2 ? ((p0 + p1) + mine) + p2
                   : ((p0 + p1) + p2) + mine;
}

// Every launch contributes 48 tickets, whether 12 CTAs contribute four each
// or 48 CTAs contribute one each. Sequence multiplication and addition wrap
// in uint64_t together. Unlike old % 48 this remains correct across 2^64:
// 48 does not divide 2^64. The enabled mode uses this for BOTH geometries.
OSAR_HD constexpr bool osar_publication_last(uint64_t old, unsigned weight,
                                           uint64_t sequence) {
  return old + uint64_t(weight) == sequence * uint64_t(OSAR_PUBLICATION_TICKETS);
}

#undef OSAR_HD

// Include verbs.h before this header (the CPU oracle supplies mock verbs).
// Only flag is inline. Payload and flag remain two writes on the same RC QP,
// in that order, with only the flag signaled. Thus a flag CQE still retires
// the associated payload and the existing all-peer ACK/ring reuse rule.
struct OsarProxyInlineWrs {
  OsarProxyInlineWrs() = default;
  OsarProxyInlineWrs(const OsarProxyInlineWrs&) = delete;
  OsarProxyInlineWrs& operator=(const OsarProxyInlineWrs&) = delete;
  OsarProxyInlineWrs(OsarProxyInlineWrs&&) = delete;
  OsarProxyInlineWrs& operator=(OsarProxyInlineWrs&&) = delete;
  ibv_sge sge[2]{};
  ibv_send_wr wr[2]{};
  uint64_t flag = 0;

  bool init(uintptr_t tx, uint32_t lkey, uint64_t remote_tx,
            uint64_t remote_flag, uint32_t rkey, unsigned inline_cap) {
    if (inline_cap < sizeof(flag)) return false;
    sge[0].addr = tx;
    sge[0].lkey = lkey;
    sge[1].addr = reinterpret_cast<uintptr_t>(&flag);
    sge[1].length = sizeof(flag);
    // Inline bytes are copied by post_send, so this host-only source neither
    // needs registration nor remains in flight after post_send returns.
    sge[1].lkey = 0;
    for (unsigned i = 0; i != 2; ++i) {
      wr[i].sg_list = &sge[i];
      wr[i].num_sge = 1;
      wr[i].opcode = IBV_WR_RDMA_WRITE;
      wr[i].wr.rdma.rkey = rkey;
    }
    wr[0].wr.rdma.remote_addr = remote_tx;
    wr[0].next = &wr[1];
    wr[1].wr.rdma.remote_addr = remote_flag;
    wr[1].send_flags = IBV_SEND_SIGNALED | IBV_SEND_INLINE;
    return true;
  }

  int post(ibv_qp* qp, uint64_t sequence, unsigned peer, uint32_t bytes) {
    flag = sequence;
    sge[0].length = bytes;
    wr[0].wr_id = (sequence << 4) | peer;
    wr[1].wr_id = (sequence << 4) | 0x8u | peer;
    ibv_send_wr* bad = nullptr;
    return ibv_post_send(qp, wr, &bad);
  }
};
