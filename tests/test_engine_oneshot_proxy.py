"""Run the production proxy and WR helper with CPU verbs, delayed rails and failures.

This checks descriptor/ring ownership and CQ retirement, not RDMA or GPU timing.
"""
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
DIRECTORY = ROOT / 'engine/kernels/oneshot'

MOCKS = r'''
#include <array>
#include <atomic>
#include <cassert>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <ctime>
#include <deque>
#include <type_traits>
#define NPEER 3
#define RING 4
#define MAXEL 128
#define PROXY_CORE 18
#define STALL_GUARD 1
struct FakeCpuSet {};
#define cpu_set_t FakeCpuSet
#define CPU_ZERO(x) ((void)(x))
#define CPU_SET(x,y) ((void)(x),(void)(y))
#define sched_setaffinity(a,b,c) ((void)(a),(void)(b),(void)(c))
struct ibv_sge { uint64_t addr; uint32_t length, lkey; };
struct ibv_send_wr {
  uint64_t wr_id = 0;
  ibv_send_wr* next = nullptr;
  ibv_sge* sg_list = nullptr;
  int num_sge = 0, opcode = 0;
  unsigned send_flags = 0;
  struct { struct { uint64_t remote_addr; uint32_t rkey; } rdma; } wr{};
};
struct ibv_qp { int peer; };
struct ibv_cq { int rail; };
struct ibv_mr { uint32_t lkey; };
struct ibv_wc { uint64_t wr_id; int status; };
constexpr int IBV_WR_RDMA_WRITE = 1, IBV_WC_SUCCESS = 0;
constexpr unsigned IBV_SEND_SIGNALED = 2, IBV_SEND_INLINE = 4;
int ibv_post_send(ibv_qp*, ibv_send_wr*, ibv_send_wr**);
int ibv_poll_cq(ibv_cq*, int, ibv_wc*);
#include "dsv4_oneshot_transport.h"

static constexpr unsigned limit = 512;
static unsigned ticks, polls, idle_ticks, pending_ticks, posts, mode, finished_tick, ack_writes;
static unsigned burst_width, completion_width;
static uint64_t published;
static std::array<unsigned, limit + 1> delivered;
struct Ack {
  uint64_t value = 0;
  operator uint64_t() const { return value; }
  Ack& operator=(uint64_t seq) {
    assert(seq > value && seq <= published);
    for (uint64_t i = value + 1; i <= seq; ++i) assert(delivered[i] == 7);
    ++ack_writes;
    value = seq;
    return *this;
  }
};
struct Publication { operator uint64_t(); };
struct Ctrl {
  uint64_t stop = 0;
  Publication tx_seq;
  Ack ack_seq;
  uint64_t flag_src[NPEER]{}, nbytes[RING]{}, pad[1]{}, rxf[RING][NPEER]{};
  uint16_t tx[RING][MAXEL]{};
};
static Ctrl control;
static Ctrl* g_ctrl = &control;
static ibv_qp qps[NPEER], *g_qp[NPEER];
static ibv_cq cqs[OSAR_RAILS], *g_cq[OSAR_RAILS];
static ibv_mr mrs[OSAR_RAILS], *g_mr[OSAR_RAILS];
static int g_peer_rail[NPEER], g_rank;
static unsigned g_inline_cap[NPEER];
static std::atomic<bool> g_proxy_running{false};
static std::atomic<uint64_t> g_proxy_heartbeat_ns{0};
static uint64_t proxy_now_ns() { return 1000 + ticks; }
struct Info { uint64_t rx_base, rxf_base; uint32_t rkey; };
static Info g_remote[NPEER];
static std::deque<ibv_wc> completion[OSAR_RAILS];
static const ibv_send_wr* stable[RING][NPEER];

Publication::operator uint64_t() {
  ++ticks;
  assert(ticks < 10000);
  if (ticks < 3) {
    assert(polls == 0 && posts == 0);
    ++idle_ticks;
    return 0;
  }
  if (control.ack_seq.value == published && published < limit) published += burst_width;
  else if (published < limit) ++pending_ticks;
  if (control.ack_seq.value == limit) {
    if (!finished_tick) finished_tick = ticks;
    ++idle_ticks;
    if (ticks == finished_tick + 3) control.stop = 1;
  }
  return published;
}

int ibv_post_send(ibv_qp* qp, ibv_send_wr* first, ibv_send_wr** bad) {
  if (mode == 3) { *bad = first; return 7; }
  assert(first && first->next && !first->next->next);
  const auto& a = *first;
  const auto& b = *first->next;
  const unsigned p = qp->peer;
  const int rail = g_peer_rail[p];
  const uint64_t seq = b.wr_id >> 4;
  const unsigned slot = seq % RING;
  assert(seq > control.ack_seq.value && seq <= control.ack_seq.value + RING);
  if (stable[slot][p]) assert(stable[slot][p] == first);
  else stable[slot][p] = first;
  assert(a.opcode == IBV_WR_RDMA_WRITE && b.opcode == IBV_WR_RDMA_WRITE);
  assert(a.num_sge == 1 && b.num_sge == 1 && a.send_flags == 0);
  assert(a.wr_id == ((seq << 4) | p) && b.wr_id == ((seq << 4) | 8 | p));
  assert(b.send_flags == (IBV_SEND_SIGNALED | (OSAR_PROXY_INLINE ? IBV_SEND_INLINE : 0)));
  assert(a.sg_list->addr == (uintptr_t)control.tx[slot]);
  assert(a.sg_list->length == control.nbytes[slot] && a.sg_list->lkey == mrs[rail].lkey);
  assert(b.sg_list->length == 8 && b.sg_list->lkey == (OSAR_PROXY_INLINE ? 0 : mrs[rail].lkey));
  if (!OSAR_PROXY_INLINE) assert(b.sg_list->addr == (uintptr_t)&control.flag_src[p]);
  assert(*reinterpret_cast<const uint64_t*>(b.sg_list->addr) == seq);
  assert(a.wr.rdma.rkey == g_remote[p].rkey && b.wr.rdma.rkey == g_remote[p].rkey);
  assert(a.wr.rdma.remote_addr == g_remote[p].rx_base + slot * NPEER * MAXEL * 2);
  assert(b.wr.rdma.remote_addr == g_remote[p].rxf_base + slot * NPEER * 8);
  completion[rail].push_back({b.wr_id, 0});
  ++posts;
  return 0;
}

int ibv_poll_cq(ibv_cq* cq, int capacity, ibv_wc* output) {
  assert(control.ack_seq.value < published);  // no empty-CQ calls while idle
  assert(!completion[cq->rail].empty());  // or after only this rail retires
  ++polls;
  if (mode == 1) return -7;
  // One function lags for two passes, including passes with no new sends.
  if (cq->rail == OSAR_RAILS - 1 && ticks % 3 != 2) return 0;
  if (capacity > int(completion_width)) capacity = int(completion_width);
  int n = 0;
  while (n < capacity && !completion[cq->rail].empty()) {
    auto wc = completion[cq->rail].front();
    completion[cq->rail].pop_front();
    if (mode == 2) wc.status = 7;
    else delivered[wc.wr_id >> 4] |= 1u << (wc.wr_id & 7);
    output[n++] = wc;
  }
  return n;
}
'''

DRIVER = r'''
int main() {
  static_assert(!std::is_copy_constructible_v<OsarProxyWrs<false>>);
  static_assert(!std::is_move_constructible_v<OsarProxyInlineWrs>);
  OsarProxyWrs<false> invalid;
  assert(!invalid.init(1, 2, 3, 4, 5, 0));
  OsarProxyInlineWrs unsupported;
  assert(!unsupported.init(1, 2, 3, 4, 5, 7));
  for (unsigned width : {1u, 2u, 4u}) for (unsigned cq_width : {1u, 16u})
  for (int rank = 0; rank < 4; ++rank) for (unsigned failure = 0; failure < 4; ++failure) {
    burst_width = width; completion_width = cq_width;
    mode = failure;
    ticks = polls = idle_ticks = pending_ticks = posts = finished_tick = ack_writes = 0;
    published = 0;
    control = Ctrl{};
    delivered.fill(0);
    std::memset(stable, 0, sizeof(stable));
    g_rank = rank;
    for (int rail = 0; rail < OSAR_RAILS; ++rail) {
      completion[rail].clear();
      cqs[rail] = {rail}; g_cq[rail] = &cqs[rail];
      mrs[rail] = {uint32_t(17 + rail)}; g_mr[rail] = &mrs[rail];
    }
    for (int peer = 0, p = 0; peer < 4; ++peer) if (peer != rank) {
      qps[p] = {p}; g_qp[p] = &qps[p];
      g_peer_rail[p] = osar_pair_rail(rank, peer, OSAR_RAILS);
      g_inline_cap[p] = OSAR_PROXY_INLINE ? 8 : 0;
      g_remote[p] = {uint64_t(0x100000 + p * 0x10000), uint64_t(0x200000 + p * 0x10000), uint32_t(23 + g_peer_rail[p])};
      ++p;
    }
    for (unsigned slot = 0; slot < RING; ++slot) control.nbytes[slot] = 8 + 16 * slot;
    g_proxy_running.store(true);
    g_proxy_heartbeat_ns.store(0);
    proxy_fn(nullptr);
    assert(!g_proxy_running.load());
    assert(g_proxy_heartbeat_ns.load() != 0);
    if (!failure) {
      assert(control.ack_seq.value == limit && posts == limit * NPEER);
      assert(ack_writes > 0 && ack_writes <= limit);
      if (cq_width == 16) assert(ack_writes == limit / burst_width);
      assert(idle_ticks >= 6 && pending_ticks > 0);
    } else {
      assert(control.ack_seq.value == 0 && ticks < 10);
      if (failure == 3) assert(polls == 0 && posts == 0);
    }
  }
}
'''


class ProxyTests(unittest.TestCase):
    @unittest.skipUnless(shutil.which('c++'), 'requires a host C++ compiler')
    def test_actual_proxy_reuses_ring_descriptors_and_retires_delayed_rails(self):
        source = (DIRECTORY / 'dsv4_oneshot_ar.cu').read_text()
        proxy = source[source.index('static void *proxy_fn(void *)'):source.index('// ---------------- setup')]
        with tempfile.TemporaryDirectory() as directory:
            cpp, binary = Path(directory) / 'proxy.cc', Path(directory) / 'proxy'
            cpp.write_text(MOCKS + proxy + DRIVER)
            for rails in (1, 2):
                for inline in (0, 1):
                    with self.subTest(rails=rails, inline=inline):
                        command = ['c++', '-std=c++17', '-O2', '-fsanitize=undefined',
                                   f'-DOSAR_RAILS={rails}', f'-DOSAR_PROXY_INLINE={inline}',
                                   '-I', str(DIRECTORY), str(cpp), '-o', str(binary)]
                        built = subprocess.run(command, capture_output=True, text=True, timeout=60)
                        self.assertEqual(built.returncode, 0, built.stderr)
                        run = subprocess.run([str(binary)], capture_output=True, text=True, timeout=20)
                        self.assertEqual(run.returncode, 0, run.stderr)


if __name__ == '__main__':
    unittest.main()
