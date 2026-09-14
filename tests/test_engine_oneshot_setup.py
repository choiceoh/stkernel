"""Fault-inject the actual native preparation/cleanup code without opening devices."""
from pathlib import Path
import re
import shutil
import subprocess
import tempfile
import unittest

DIRECTORY = Path(__file__).resolve().parents[1] / 'engine/kernels/oneshot'

MOCKS = r'''
#include <atomic>
#include <cassert>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <ctime>
#include <memory>
#include <pthread.h>
#include <stdexcept>
#include <string>
#include <vector>
#define CHK(x) do { if (!(x)) throw std::runtime_error(#x); } while (0)
#define NPEER 3
#define RING 4
#define MAXEL 8
using bf16 = uint16_t;
struct ibv_device { int rail; };
struct ibv_context { int rail, children = 0; };
struct ibv_pd { ibv_context* ctx; int children = 0; };
struct ibv_cq { ibv_context* ctx; int cqe, children = 0; };
struct ibv_mr { ibv_pd* pd; uint32_t lkey = 7, rkey = 8; };
struct ibv_qp { ibv_pd* pd; ibv_cq* cq; uint32_t qp_num = 9; };
union ibv_gid { uint8_t raw[16]; };
struct ibv_port_attr { int state, gid_tbl_len, active_mtu; };
struct ibv_qp_init_attr {
  ibv_cq *send_cq, *recv_cq;
  struct { unsigned max_send_wr, max_recv_wr, max_send_sge, max_inline_data; } cap;
  int qp_type;
};
constexpr int IBV_PORT_ACTIVE = 4, IBV_ACCESS_LOCAL_WRITE = 1, IBV_ACCESS_REMOTE_WRITE = 2;
constexpr int IBV_QPT_RC = 1, cudaSuccess = 0, cudaHostRegisterDefault = 0;
static int acquire_step, fail_acquire, release_step, fail_release;
static int contexts, lists, allocations, registrations, released;
static int gid_count = 40, gid_match = 31, queries, missing_rail = -1, inactive_rail = -1;
static unsigned inline_cap = 16;
static bool join_fails, short_cq, short_sq;
static int peer_count(int rail);
static bool acquire_fails() { return ++acquire_step == fail_acquire; }
static bool release_fails() { return ++release_step == fail_release; }
static ibv_device devices[2] = {{0},{1}};
ibv_device** ibv_get_device_list(int* count) {
  if (acquire_fails()) return nullptr;
  ++lists;
  auto list = new ibv_device*[2]; *count = 0;
  for (int i = 0; i < 2; ++i) if (i != missing_rail) list[(*count)++] = &devices[i];
  return list;
}
void ibv_free_device_list(ibv_device** list) { assert(lists == 1); --lists; delete[] list; }
const char* ibv_get_device_name(ibv_device* d) { return d->rail ? "roceP2p1s0f0" : "rocep1s0f0"; }
ibv_context* ibv_open_device(ibv_device* d) {
  if (acquire_fails()) return nullptr;
  ++contexts; return new ibv_context{d->rail};
}
int ibv_query_port(ibv_context* c, int, ibv_port_attr* a) {
  if (acquire_fails()) return 1;
  *a = {c->rail == inactive_rail ? 0 : IBV_PORT_ACTIVE, gid_count, 3}; return 0;
}
int ibv_query_gid(ibv_context* c, int, int index, ibv_gid* g) {
  assert(index >= 0 && index < gid_count); ++queries;
  memset(g, 0, sizeof(*g));
  if (index == 0) return 1;  // an unreadable entry must not stop discovery
  if (index == 1 || index == 2 || index == gid_match) {
    g->raw[10] = g->raw[11] = 255;
    g->raw[12] = 10; g->raw[13] = 10; g->raw[14] = 10 + c->rail; g->raw[15] = 1;
  }
  return 0;
}
FILE* mock_fopen(const char* path, const char*) {
  const int index = atoi(strrchr(path, '/') + 1);
  if (index == 2) return nullptr;
  FILE* file = tmpfile(); assert(file);
  fputs(index == 1 ? "RoCE v1\n" : "RoCE v2\n", file); rewind(file); return file;
}
void* mock_aligned_alloc(size_t alignment, size_t bytes) {
  assert(alignment == 4096 && bytes % alignment == 0);
  if (acquire_fails()) return nullptr;
  void* result = std::aligned_alloc(alignment, bytes); assert(result); ++allocations; return result;
}
void mock_free(void* p) {
  if (!p) return;
  assert(!registrations && !contexts && allocations == 1);
  --allocations; std::free(p);
}
int cudaHostRegister(void*, size_t, int) {
  assert(contexts == OSAR_RAILS);
  if (acquire_fails()) return 1;
  assert(!registrations); ++registrations; return 0;
}
ibv_pd* ibv_alloc_pd(ibv_context* c) {
  if (acquire_fails()) return nullptr;
  ++c->children; return new ibv_pd{c};
}
ibv_cq* ibv_create_cq(ibv_context* c, int entries, void*, void*, int) {
  assert(entries == 2 * RING * peer_count(c->rail));
  if (acquire_fails()) return nullptr;
  ++c->children; return new ibv_cq{c, entries + (short_cq ? -1 : 3)};
}
ibv_mr* ibv_reg_mr(ibv_pd* pd, void*, size_t, int) {
  if (acquire_fails()) return nullptr;
  assert(registrations); ++pd->children; return new ibv_mr{pd};
}
ibv_qp* ibv_create_qp(ibv_pd* pd, ibv_qp_init_attr* a) {
  assert(a->cap.max_send_wr == 2 * RING && a->cap.max_recv_wr == 0);
  assert(a->cap.max_send_sge == 1 && a->qp_type == IBV_QPT_RC);
  if (acquire_fails()) return nullptr;
  a->cap.max_send_wr += short_sq ? -1 : 8;
  assert(a->send_cq == a->recv_cq && a->send_cq->ctx == pd->ctx);
  a->cap.max_inline_data = inline_cap;
  ++pd->children; ++a->send_cq->children; return new ibv_qp{pd, a->send_cq};
}
int ibv_destroy_qp(ibv_qp* q) {
  if (release_fails()) return 1;
  --q->pd->children; --q->cq->children; delete q; ++released; return 0;
}
int ibv_dereg_mr(ibv_mr* mr) {
  if (release_fails()) return 1;
  assert(mr->pd->children == 1); --mr->pd->children; delete mr; ++released; return 0;
}
int ibv_destroy_cq(ibv_cq* cq) {
  if (release_fails()) return 1;
  assert(!cq->children); --cq->ctx->children; delete cq; ++released; return 0;
}
int ibv_dealloc_pd(ibv_pd* pd) {
  if (release_fails()) return 1;
  assert(!pd->children); --pd->ctx->children; delete pd; ++released; return 0;
}
int ibv_close_device(ibv_context* c) {
  if (release_fails()) return 1;
  assert(!c->children); delete c; --contexts; ++released; return 0;
}
int cudaHostUnregister(void*) {
  if (release_fails()) return 1;
  assert(!contexts && registrations == 1); --registrations; ++released; return 0;
}
int mock_join(pthread_t, void**) { return join_fails ? 1 : 0; }
#define fopen mock_fopen
#define aligned_alloc mock_aligned_alloc
#define free mock_free
#define pthread_join mock_join
'''

CHECKS = r'''
static int peer_count(int rail) {
  int peers = 0;
  for (int rank = 0; rank < 4; ++rank)
    if (rank != g_rank && osar_pair_rail(rank, g_rank, OSAR_RAILS) == rail) ++peers;
  return peers;
}
static const std::vector<std::string> ips = OSAR_RAILS == 2
    ? std::vector<std::string>{"10.10.10.1", "10.10.11.1"}
    : std::vector<std::string>{"10.10.10.1"};
static void empty() {
  assert(!contexts && !lists && !allocations && !registrations && !g_ctrl);
  assert(g_rank == -1 && g_world == 0 && !g_started && !g_host_registered);
  for (int p = 0; p < NPEER; ++p) assert(!g_qp[p] && !g_inline_cap[p]);
  for (int r = 0; r < OSAR_RAILS; ++r) assert(!g_ctx[r] && !g_pd[r] && !g_cq[r] && !g_mr[r]);
}
static void rejects(int rank = 0, int world = 4) {
  bool threw = false;
  try { init_ctx(rank, world, ips); } catch (const std::runtime_error&) { threw = true; }
  assert(threw);
}
int main() {
  init_ctx(0, 4, ips);
  assert(!lists && g_sgid[0] == 31 && contexts == OSAR_RAILS);
  const int acquisitions = acquire_step;
  auto original = g_ctrl;
  rejects(); assert(g_ctrl == original && contexts == OSAR_RAILS); // no overwrite/cleanup of live owner
  py_shutdown(); empty();
  const int releases = release_step;
  py_shutdown(); empty();
  for (int fault = 1; fault <= acquisitions; ++fault) {
    acquire_step = release_step = 0; fail_acquire = fault;
    rejects(); empty(); // every partial acquisition unwinds
  }
  fail_acquire = 0;
  short_cq = true; rejects(); empty(); short_cq = false;
  short_sq = true; rejects(); empty(); short_sq = false;
  for (int rank = 1; rank < 4; ++rank) { init_ctx(rank, 4, ips); py_shutdown(); empty(); }
  for (int fault = 1; fault <= releases; ++fault) {
    acquire_step = release_step = 0; fail_release = fault;
    init_ctx(0, 4, ips); py_shutdown();
    assert(g_ctrl && allocations == 1); // failed releases keep dependent memory alive
    fail_release = 0; py_shutdown(); empty(); py_shutdown(); empty();
  }
  missing_rail = OSAR_RAILS - 1; rejects(); empty(); missing_rail = -1;
  inactive_rail = OSAR_RAILS - 1; rejects(); empty(); inactive_rail = -1;
  gid_count = 16; queries = 0; rejects(); empty(); assert(queries == 16);
  gid_count = 0; queries = 0; rejects(); empty(); assert(queries == 0);
  gid_count = 40;
  rejects(-1); empty(); rejects(4); empty(); rejects(0, 5); empty();
  inline_cap = 4;
#if OSAR_PROXY_INLINE
  rejects(); empty();
#else
  init_ctx(0, 4, ips); py_shutdown(); empty();
#endif
  inline_cap = 8; init_ctx(0, 4, ips); // exact minimum is sufficient
  g_started = true; join_fails = true;
  const int before_join = released;
  py_shutdown(); assert(g_started && released == before_join && g_ctrl);
  join_fails = false; py_shutdown(); empty();
  init_ctx(0, 4, ips); g_started = true;
  assert(!cleanup_prepared()); // never tear resources out from under a running proxy
  g_device_used = true;
  const int before_device = released;
  py_shutdown(); assert(!g_started && released == before_device && g_ctrl);
  py_shutdown(); assert(released == before_device); rejects();
  g_device_used = false; // test has no GPU; production never resets this lifetime guard
  py_shutdown(); empty();
  printf("PASS rails=%d inline=%d acquisitions=%d release_failures=%d\n",
         OSAR_RAILS, OSAR_PROXY_INLINE, acquisitions, releases);
}
'''


class SetupTests(unittest.TestCase):
    @unittest.skipUnless(shutil.which('g++'), 'requires a host C++ compiler')
    def test_partial_preparation_release_failures_gid_bounds_and_device_lifetime(self):
        source = (DIRECTORY / 'dsv4_oneshot_ar.cu').read_text()
        header = (DIRECTORY / 'dsv4_oneshot_transport.h').read_text()
        globals_ = source[source.index('struct Ctrl {'):source.index('static uint64_t proxy_now_ns')]
        setup = source[source.index('template <typename T, typename Destroy>'):source.index('static void to_rts')]
        shutdown = source[source.index('static void py_shutdown()'):source.index('#ifndef ST_ONESHOT_LOCAL_TEST')]
        program = MOCKS + header[:header.index('// Include verbs.h')] + globals_ + setup + shutdown + CHECKS
        with tempfile.TemporaryDirectory() as d:
            path, binary = Path(d) / 'setup.cc', Path(d) / 'setup'
            path.write_text(program)
            for rails in (1, 2):
                for inline in (0, 1):
                    with self.subTest(rails=rails, inline=inline):
                        result = subprocess.run(['g++', '-std=c++17', '-O1', '-fsanitize=undefined',
                                                 '-fno-sanitize-recover=all', f'-DOSAR_RAILS={rails}',
                                                 f'-DOSAR_PROXY_INLINE={inline}', str(path), '-o', str(binary)],
                                                capture_output=True, text=True)
                        self.assertEqual(result.returncode, 0, result.stderr)
                        result = subprocess.run([str(binary)], capture_output=True, text=True, timeout=15)
                        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
                        self.assertIn('PASS', result.stdout)

    def test_every_native_launch_protects_the_ctrl_lifetime_before_enqueue(self):
        source = (DIRECTORY / 'dsv4_oneshot_ar.cu').read_text()
        functions = re.split(r'(?=static at::Tensor py_)', source)[1:]
        launchers = 0
        for function in functions:
            function = function.split('\n}', 1)[0]
            launches = list(re.finditer(r'<<<|cudaLaunchKernelEx\(', function))
            if not launches:
                continue
            launchers += 1
            self.assertIn('g_device_used = true;', function)
            self.assertLess(function.index('g_device_used = true;'), launches[0].start())
        self.assertEqual(launchers, 7)


if __name__ == '__main__':
    unittest.main()
