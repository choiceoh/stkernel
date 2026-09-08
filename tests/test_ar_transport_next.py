"""Device-free transport proofs against production helpers and mock verbs.

These are protocol/dispatch tests, not GPU numerics, racecheck or timings.
"""
import importlib.util
import math
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

ROOT = Path(__file__).resolve().parents[1]
OSAR = ROOT / "overlay/modules/tp_oneshot_ar"


def load_shim():
    spec = importlib.util.spec_from_file_location(
        "ar_transport_next_shim", OSAR / "dsv4_oneshot_shim.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


MOCK_VERBS = r'''
#include <cstdint>
#include <cstring>
#include <vector>
#include <cassert>
#include <algorithm>
#include <array>
#include <random>
#include <type_traits>
struct ibv_sge { uint64_t addr; uint32_t length, lkey; };
struct ibv_send_wr {
  uint64_t wr_id = 0;
  ibv_send_wr* next = nullptr;
  ibv_sge* sg_list = nullptr;
  int num_sge = 0, opcode = 0;
  unsigned send_flags = 0;
  struct { struct { uint64_t remote_addr; uint32_t rkey; } rdma; } wr{};
};
struct ibv_qp { unsigned peer; };
constexpr int IBV_WR_RDMA_WRITE = 1;
constexpr unsigned IBV_SEND_SIGNALED = 2, IBV_SEND_INLINE = 4;
struct Submission {
  unsigned peer;
  uint64_t payload_id, flag_id, payload_remote, flag_remote, flag;
  uintptr_t payload_source;
  uint32_t bytes, lkey, rkey;
};
static std::vector<Submission> submitted;
static bool fail_post = false;
int ibv_post_send(ibv_qp* qp, ibv_send_wr* first, ibv_send_wr** bad) {
  if (fail_post) { *bad = first; return 7; }
  assert(first && first->next && !first->next->next);
  const auto& a = *first;
  const auto& b = *first->next;
  assert(a.opcode == IBV_WR_RDMA_WRITE && b.opcode == IBV_WR_RDMA_WRITE);
  assert(a.num_sge == 1 && b.num_sge == 1);
  assert(a.send_flags == 0);  // payload remains registered/non-inline
  assert(b.send_flags == (IBV_SEND_SIGNALED | IBV_SEND_INLINE));
  assert(b.sg_list->length == 8);
  uint64_t copied_flag = 0;
  std::memcpy(&copied_flag, (const void*)b.sg_list->addr, sizeof(copied_flag));
  submitted.push_back({qp->peer, a.wr_id, b.wr_id,
      a.wr.rdma.remote_addr, b.wr.rdma.remote_addr, copied_flag,
      (uintptr_t)a.sg_list->addr, a.sg_list->length,
      a.sg_list->lkey, a.wr.rdma.rkey});
  return 0;
}
'''


GEOMETRY_ORACLE = r'''
int main() {
  static_assert(OSAR_COMPACT_VECITER == 2);
  assert(!osar_compact_eligible(0));
  assert(osar_compact_eligible(32768));
  assert(!osar_compact_eligible(32769));
  // Visit actual vector positions incrementally, through every payload size.
  // This independently enumerates owner blocks and per-thread stash slots.
  for (unsigned grid : {12u, 48u}) {
    const int limit = grid == 12 ? 32768 : 131072;
    std::vector<unsigned> trips(grid * 256);
    std::vector<bool> owner(grid);
    for (int n = 0; n <= limit; ++n) {
      if (n && n % 8 == 0) {
        const unsigned lane = (n / 8 - 1) % (grid * 256);
        owner[lane / 256] = true;
        assert(++trips[lane] <= 2);
      }
      for (unsigned b = 0; b < grid; ++b) {
        const bool scalar_tail = b == 0 && n % 8 != 0;
        assert(osar_block_owns<true>(b, 256, n) ==
               (b == 0 || owner[b] || scalar_tail));
      }
    }
  }
  // Explicit byte coverage at vector, CTA, two-trip and fallback boundaries.
  for (unsigned n : {0u,1u,7u,8u,9u,2047u,2048u,2049u,24575u,24576u,
                     24577u,32767u,32768u,32769u,65536u,131072u}) {
    unsigned grid = osar_compact_eligible(n) ? OSAR_COMPACT_GRID : 48;
    std::vector<unsigned> visited(n);
    for (unsigned b = 0; b < grid; ++b) {
      for (unsigned t = 0; t < 256; ++t) {
        unsigned stash = 0;
        for (unsigned v = b * 256 + t; v < n / 8; v += grid * 256) {
          assert(stash++ < 2);
          for (unsigned q = 0; q < 8; ++q) ++visited[v * 8 + q];
        }
        for (unsigned i = n / 8 * 8 + b * 256 + t; i < n; i += grid * 256)
          ++visited[i];
      }
    }
    assert(std::all_of(visited.begin(), visited.end(), [](unsigned n) {return n == 1;}));
  }
  // Replay ordinary/compact in mixed order, including the actual uint64
  // done-counter wrap. Sequence itself stays far below its separate limit.
  const uint64_t wrap_sequence = UINT64_MAX / 48;
  std::mt19937 random(9173);
  for (uint64_t initial : {uint64_t(0), wrap_sequence - 9, 2 * wrap_sequence - 9}) {
    uint64_t sequence = initial, counter = initial * uint64_t(48);
    for (unsigned replay = 0; replay < 10000; ++replay) {
      ++sequence;
      const unsigned grid = (random() & 1) ? 12 : 48;
      const unsigned weight = 48 / grid;
      std::vector<unsigned> order(grid);
      for (unsigned b = 0; b < grid; ++b) order[b] = b;
      std::shuffle(order.begin(), order.end(), random);
      unsigned publishers = 0, finished_writers = 0;
      for (unsigned b : order) {
        (void)b;
        // A ticket is submitted only after this CTA's writer fences/barrier.
        ++finished_writers;
        bool last = osar_publication_last(counter, weight, sequence);
        counter += weight;
        if (last) { ++publishers; assert(finished_writers == grid); }
      }
      assert(publishers == 1);
      assert(counter == (uint64_t)((unsigned __int128)sequence * 48));
    }
  }
}
'''


PROXY_ORACLE = r'''
int main() {
  static_assert(!std::is_copy_constructible_v<OsarProxyInlineWrs>);
  static_assert(!std::is_move_constructible_v<OsarProxyInlineWrs>);
  OsarProxyInlineWrs unsupported;
  assert(!unsupported.init(1, 2, 3, 4, 5, 7));
  constexpr unsigned ring = 4, peers = 3, maxel = 131072;
  std::array<std::array<OsarProxyInlineWrs, peers>, ring> descriptor;
  ibv_qp qp[peers] = {{0},{1},{2}};
  for (unsigned s = 0; s < ring; ++s) for (unsigned p = 0; p < peers; ++p) {
    assert(descriptor[s][p].init(0x10000000ull + s * maxel * 2, 17,
        0x20000000ull + p * 0x1000000ull + s * peers * maxel * 2,
        0x30000000ull + p * 0x1000ull + s * peers * 8, 23 + p, 8));
  }
  // Four outstanding ring slots. Acknowledgment requires all three CQEs,
  // irrespective of peer completion order. The fifth send must wait.
  uint64_t ack = 0;
  uint64_t done[64] = {};
  std::vector<Submission> pending;
  for (uint64_t seq = 1; seq <= 512; ++seq) {
    if (seq > ack + ring) {
      const uint64_t retiring = ack + 1;
      for (int p : {2,0,1}) {
        auto it = std::find_if(pending.begin(), pending.end(), [&](const auto& x) {
          return x.peer == (unsigned)p && x.flag == retiring;
        });
        assert(it != pending.end());
        const uint64_t cs = it->flag_id >> 4;
        if (++done[cs % 64] == peers) { done[cs % 64] = 0; if (cs > ack) ack = cs; }
        if (p != 1) assert(ack == retiring - 1);
        pending.erase(it);
      }
    }
    assert(seq <= ack + ring);
    const unsigned slot = seq % ring;
    const unsigned bytes = seq % 3 == 0 ? 65536 : seq % 3 == 1 ? 49152 : 2;
    for (unsigned peer = 0; peer < peers; ++peer) {
      auto& item = descriptor[slot][peer];
      assert(item.post(&qp[peer], seq, peer, bytes) == 0);
      const auto captured = submitted.back();
      assert(captured.flag == seq);
      assert(captured.payload_id == ((seq << 4) | peer));
      assert(captured.flag_id == ((seq << 4) | 8 | peer));
      assert(captured.bytes == bytes && captured.lkey == 17 && captured.rkey == 23 + peer);
      assert(captured.payload_source == 0x10000000ull + slot * maxel * 2);
      assert(captured.payload_remote == 0x20000000ull + peer * 0x1000000ull + slot * peers * maxel * 2);
      assert(captured.flag_remote == 0x30000000ull + peer * 0x1000ull + slot * peers * 8);
      // Provider has consumed inline data. Later host source changes cannot
      // turn an earlier flag into a newer sequence, even before its CQE.
      item.flag = UINT64_MAX;
      assert(submitted.back().flag == seq);
      pending.push_back(captured);
    }
  }
  fail_post = true;
  const auto count = submitted.size();
  assert(descriptor[0][0].post(&qp[0], 513, 0, 49152) == 7);
  assert(submitted.size() == count);
}
'''


class TransportTests(unittest.TestCase):
    def compile_run(self, body):
        compiler = shutil.which("c++")
        self.assertIsNotNone(compiler, "host C++ compiler required")
        source = (OSAR / "dsv4_oneshot_ar.cu").read_text()
        start = source.index("template <bool VECTOR_EXACT>")
        end = source.index("\n}", start) + 2
        ownership = source[start:end].replace("__host__ __device__ ", "")
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "oracle.cc"
            binary = Path(temporary) / "oracle"
            path.write_text(MOCK_VERBS + '\n#include "dsv4_oneshot_transport.h"\n' + ownership + body)
            subprocess.run([compiler, "-std=c++17", "-O2", "-fsanitize=undefined",
                            "-I", str(OSAR), str(path), "-o", str(binary)],
                           check=True, capture_output=True, text=True, timeout=60)
            subprocess.run([str(binary)], check=True, capture_output=True,
                           text=True, timeout=60)

    def test_actual_geometry_and_publication_helpers(self):
        self.compile_run(GEOMETRY_ORACLE)

    def test_actual_inline_descriptors_with_mock_verbs_and_ring_reuse(self):
        self.compile_run(PROXY_ORACLE)

    def test_modes_are_exact_opt_in(self):
        for value, expected in [("0", False), ("1", True), ("true", False), ("", False)]:
            with patch.dict(os.environ, {"VLLM_GLM53_AR_COMPACT_CTA": value,
                                         "VLLM_GLM53_AR_PROXY_INLINE": value}):
                shim = load_shim()
                self.assertEqual(shim._COMPACT_CTA, expected)
                self.assertEqual(shim._PROXY_INLINE, expected)

    def test_header_and_mode_flags_bind_the_actual_build(self):
        shim = load_shim()
        shim._CONSUMER_PDL = True
        load = Mock(return_value=object())
        fake = SimpleNamespace(load=load)
        with tempfile.TemporaryDirectory() as temporary:
            header = Path(temporary) / "dsv4_oneshot_transport.h"
            header.write_bytes(Path(shim._TRANSPORT_HEADER).read_bytes())
            shim._TRANSPORT_HEADER = str(header)
            shim._build_dir = Mock(return_value=temporary)
            keys = set()
            with patch.dict(sys.modules, {"torch.utils.cpp_extension": fake}):
                for compact, inline in [(False, False), (True, False), (False, True), (True, True)]:
                    shim._COMPACT_CTA, shim._PROXY_INLINE = compact, inline
                    shim._build()
                    flags = load.call_args.kwargs["extra_cuda_cflags"]
                    self.assertEqual("-DOSAR_COMPACT_CTA=1" in flags, compact)
                    self.assertEqual("-DOSAR_PROXY_INLINE=1" in flags, inline)
                    keys.add(tuple(shim._build_dir.call_args.args[1]))
                self.assertEqual(len(keys), 4)
                old_digest = shim._build_dir.call_args.args[0]
                header.write_text(header.read_text() + "\n// changed helper\n")
                shim._build()
                self.assertNotEqual(shim._build_dir.call_args.args[0], old_digest)
                shim._CONSUMER_PDL = False
                with self.assertRaisesRegex(RuntimeError, "requires"):
                    shim._build()

    def test_compact_capture_and_poison_failure_dispatch(self):
        shim = load_shim()
        shim._COMPACT_CTA = shim._CONSUMER_PDL = True
        shim._disabled = shim._SHADOW = False
        shim._connected = shim._selftest_ok = True
        shim._eligible = lambda _: True
        shim._ext = SimpleNamespace(healthy=lambda: True,
                                    oneshot_ar=lambda _: "ordinary",
                                    oneshot_ar_consumer=lambda _: "consumer")
        torch = SimpleNamespace(cuda=SimpleNamespace(is_current_stream_capturing=lambda: True))
        with patch.dict(sys.modules, {"torch": torch}):
            for n in (1, 24576, 32768):
                self.assertEqual(shim.maybe_all_reduce(None, SimpleNamespace(numel=lambda: n), None), "consumer")
            self.assertTrue(shim._COMPACT_CAPTURED)
            self.assertEqual(shim.maybe_all_reduce(None, SimpleNamespace(numel=lambda: 32769), None), "ordinary")
            shim._ext.healthy = lambda: False
            with self.assertRaises(shim.OneShotFatal):
                shim.maybe_all_reduce(None, SimpleNamespace(numel=lambda: 24576), None)

    def test_rank_mode_mismatch_declines_before_connect(self):
        class Tensor:
            def __init__(self, values):
                self.values = list(values)

            def item(self):
                return self.values[0]

            def __getitem__(self, index):
                return Tensor([self.values[index]])

        for compact, inline, mismatch, actual in (
                (True, True, False, [1, 1, 8, 8, 8]),
                (True, True, True, [1, 1, 8, 8, 8]),
                (True, True, False, [0, 1, 8, 8, 8]),
                (True, True, False, [1, 1, 8, 7, 8]),
                (False, False, False, [1, 1, 8, 8, 8]),
                (False, False, False, [0, 0, 0, 0, 0])):
            setup_bad = actual[:2] != [int(compact), int(inline)] or (inline and min(actual[2:]) < 8)
            declined = mismatch or setup_bad
            shim = load_shim()
            shim._disabled = False
            shim._CONSUMER_PDL = True
            shim._COMPACT_CTA, shim._PROXY_INLINE = compact, inline
            extension = SimpleNamespace(init=Mock(), local_infos=lambda: b"qp", connect=Mock(),
                                        transport_modes=lambda: actual)
            shim._build = lambda: extension
            shim._self_test = Mock()
            max_calls = []

            def reduce(tensor, *, group, op=None):
                if op is None:
                    tensor.values[0] = 4 if tensor.values[0] else 3
                else:
                    max_calls.append(tensor.values[:])
                    if len(max_calls) == 2 and mismatch:
                        tensor.values[:] = [7, -1]

            gather = Mock(side_effect=lambda target, value, **kw: target.__setitem__(slice(None), [value] * 4))
            dist = SimpleNamespace(all_reduce=reduce, all_gather_object=gather,
                                   ReduceOp=SimpleNamespace(MAX=object()))
            torch = SimpleNamespace(tensor=lambda values, **kw: Tensor(values),
                                    int32="int32", int64="int64", distributed=dist)
            comm = SimpleNamespace(unique_name="tp", rank_in_group=0,
                                   world_size=4, cpu_group=object())
            with patch.dict(sys.modules, {"torch": torch, "torch.distributed": dist}), \
                    patch.dict(os.environ, {"VLLM_HOST_IP": "192.0.2.1"}):
                shim._bootstrap(comm)
            if setup_bad:
                self.assertEqual(max_calls, [])
            else:
                mode = 1 | int(compact) << 1 | int(inline) << 2
                self.assertEqual(max_calls[1], [mode, -mode])
            self.assertEqual(shim._disabled, declined)
            self.assertEqual(extension.connect.call_count, 0 if declined else 1)
            self.assertEqual(gather.call_count, 0 if declined else 1)
            self.assertEqual(shim._self_test.call_count, 0 if declined else 1)

    def test_nonfinite_error_cannot_hide_behind_exact_baseline(self):
        error = load_shim()._transport_max_error
        self.assertEqual(error(0.0, 0.0), 0.0)
        self.assertEqual(error(0.0, 0.5, 0.25), 0.5)
        for bad in (float("nan"), float("inf"), -float("inf")):
            self.assertTrue(math.isinf(error(0.0, bad)))
            self.assertTrue(math.isinf(error(bad, 0.0)))

    def test_publication_and_proxy_integration_keep_order(self):
        source = (OSAR / "dsv4_oneshot_ar.cu").read_text()
        body = source.split("__device__ __forceinline__ void k_oneshot_impl(", 1)[1]
        body = body.split("// ---------------- proxy", 1)[0]
        self.assertLess(body.index("__threadfence_system();"), body.index("osar_publication_last("))
        fence = body.index("__threadfence_system();")
        self.assertLess(body.index("__syncthreads();", fence), body.index("osar_publication_last("))
        self.assertLess(body.index("osar_publication_last("), body.index("c->tx_seq = nxt;"))
        self.assertLess(body.index("c->tx_seq = nxt;"), body.index("griddepcontrol.launch_dependents;"))
        self.assertIn("k_oneshot_compact_mode<true, false>", source)
        self.assertIn("k_oneshot_compact_mode<false, false>", source)
        self.assertIn("last = atomicAdd((unsigned long long *)&c->done_ctr, 1ULL) %", source)
        post = source.index("prepared[slot][p].post(")
        self.assertLess(post, source.index("inline proxy serving"))
        self.assertIn("CHK(g_inline_cap[s] >= sizeof(uint64_t));", source)
        self.assertIn("if (wc[i].status != IBV_WC_SUCCESS)", source)
        self.assertIn("if (++done[cs % 64] == NPEER)", source)


if __name__ == "__main__":
    unittest.main()
