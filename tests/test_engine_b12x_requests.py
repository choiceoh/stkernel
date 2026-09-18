"""engine/kernels/b12x_requests: a boot's kernel requests are recorded as the getter saw them, once, and never stop the
boot; the prebuild replays them on the CPU. The replay against the real dispatcher (the same kernel names, then hits) is
checked in the ST image by `ImageRoundTripTests`."""
import importlib.util
import json
import os
import subprocess
import sys
import tempfile
import types
import unittest
import unittest.mock
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
torch = None
if importlib.util.find_spec("torch") is not None:
    import torch


def fake_dispatcher():
    """A module shaped like moe_dispatch where it matters: getters that fill their caches, the three settings, the
    device queries."""
    md = types.ModuleType("fake_md")
    md._STATIC_KERNEL_CACHE, md._STATIC_V2_KERNEL_CACHE, md._MICRO_KERNEL_CACHE, md._DYNAMIC_KERNEL_CACHE = {}, {}, {}, {}
    md._GLM53_B12X_STATIC_V2, md._TP_SF6_Q0_ENABLED, md._EP_ZERO_WEIGHT_MICRO_CELL = None, False, None
    md.get_num_sm = lambda device: 48
    md.get_max_active_clusters = lambda size: 47
    md.calls = []

    def getter(cache_name):
        def get(*args, **kwargs):
            md.calls.append((cache_name, args, kwargs))
            key = (args, tuple(sorted(kwargs.items(), key=lambda kv: kv[0])))
            cache = getattr(md, cache_name)
            cache.setdefault(repr(key), object())
            return cache[repr(key)]
        return get

    md._get_static_kernel = getter("_STATIC_KERNEL_CACHE")
    md._get_static_kernel_v2 = getter("_STATIC_V2_KERNEL_CACHE")
    md._get_micro_kernel = getter("_MICRO_KERNEL_CACHE")
    md._get_dynamic_kernel = getter("_DYNAMIC_KERNEL_CACHE")
    return md


@unittest.skipUnless(torch is not None, "requires torch")
class EncodingTests(unittest.TestCase):
    def test_every_argument_kind_comes_back_as_itself(self):
        from engine.kernels import b12x_requests as br
        value = [128, 2560, 1.702, None, True, "silu", torch.int32, torch.device("cuda"), (1, (2, 3)),
                 {"tile": 128, ("a", 1): (4, 5), "flag": False}]
        back = br.decode(json.loads(json.dumps(br.encode(value))))
        self.assertEqual(back, value)
        self.assertIsInstance(back[8], tuple)
        self.assertIs(back[6], torch.int32)

    def test_an_unknown_kind_is_refused_not_guessed(self):
        from engine.kernels import b12x_requests as br
        with self.assertRaises(TypeError):
            br.encode(object())
        with self.assertRaises(ValueError):
            br.decode({"__t": "dtype", "v": "not_a_dtype"})


@unittest.skipUnless(torch is not None, "requires torch")
class RecordTests(unittest.TestCase):
    def setUp(self):
        from engine.kernels import b12x_requests as br
        self.br = br
        self.dir = tempfile.TemporaryDirectory()
        self.path = Path(self.dir.name) / "req" / "qwen38.jsonl"

    def tearDown(self):
        self.dir.cleanup()

    def lines(self):
        return [json.loads(line) for line in self.path.read_text().splitlines()]

    def test_a_new_kernel_is_one_line_and_a_repeat_is_none(self):
        md = fake_dispatcher()
        recorder = self.br.record(md, self.path, "qwen38")
        md._EP_ZERO_WEIGHT_MICRO_CELL = (128, 2560)
        md._get_micro_kernel(128, 129, 4, 2560, 640, 10, 40, topk_ids_dtype=torch.int32, skip_zero_weight_expert_id=128)
        md._get_micro_kernel(128, 129, 4, 2560, 640, 10, 40, topk_ids_dtype=torch.int32, skip_zero_weight_expert_id=128)
        md._get_static_kernel(128, 128, 16, 2560, 640, 10, 160)
        lines = self.lines()
        self.assertEqual([line["getter"] for line in lines], ["_get_micro_kernel", "_get_static_kernel"])
        self.assertEqual(recorder.written, 2)
        first = lines[0]
        self.assertEqual((first["device"]["sm"], first["device"]["clusters"]), (48, {"1": 47}))
        self.assertEqual(self.br.decode(first["config"]["_EP_ZERO_WEIGHT_MICRO_CELL"]), (128, 2560))
        self.assertIs(self.br.decode(first["kwargs"]["topk_ids_dtype"]), torch.int32)
        self.assertEqual(first["profile"], "qwen38")

    def test_the_getters_still_answer_as_before(self):
        md = fake_dispatcher()
        plain = fake_dispatcher()
        self.br.record(md, self.path, "qwen38")
        self.assertIs(md._get_dynamic_kernel(128, 640, 2560, 640, 1, 640), md._DYNAMIC_KERNEL_CACHE[
            next(iter(md._DYNAMIC_KERNEL_CACHE))])
        plain._get_dynamic_kernel(128, 640, 2560, 640, 1, 640)
        self.assertEqual(md.calls, plain.calls)

    def test_a_second_boot_does_not_write_what_the_file_already_has(self):
        md = fake_dispatcher()
        self.br.record(md, self.path, "qwen38")
        md._get_static_kernel(128, 128, 16, 2560, 640, 10, 160)
        again = fake_dispatcher()                                  # the next boot: a fresh process, the same file
        recorder = self.br.record(again, self.path, "qwen38")
        again._get_static_kernel(128, 128, 16, 2560, 640, 10, 160)
        self.assertEqual(recorder.written, 0)
        self.assertEqual(len(self.lines()), 1)

    def test_installing_twice_keeps_one_recorder(self):
        md = fake_dispatcher()
        first = self.br.record(md, self.path, "qwen38")
        self.assertIs(self.br.record(md, self.path, "qwen38"), first)
        md._get_static_kernel(1, 1, 1, 1, 1, 1, 1)
        self.assertEqual(len(self.lines()), 1)

    def test_no_path_records_nothing(self):
        md = fake_dispatcher()
        getter = md._get_static_kernel
        self.assertIsNone(self.br.record(md, None, "qwen38"))
        self.assertIs(md._get_static_kernel, getter)
        self.assertIsNone(self.br.path_under(None, "qwen38"))
        self.assertEqual(self.br.path_under("/cache/cu132", "qwen38"), Path("/cache/cu132/st-b12x-requests/qwen38.jsonl"))

    def test_a_write_that_fails_stops_the_recording_not_the_boot(self):
        md = fake_dispatcher()
        blocker = Path(self.dir.name) / "file"
        blocker.write_text("")
        recorder = self.br.record(md, blocker / "qwen38.jsonl", "qwen38")         # a file where the directory would be
        md._get_static_kernel(1, 1, 1, 1, 1, 1, 1)
        md._get_static_kernel(2, 1, 1, 1, 1, 1, 1)
        self.assertIsNotNone(recorder.failed)
        self.assertEqual(len(md._STATIC_KERNEL_CACHE), 2)

    def test_read_keeps_distinct_requests_and_skips_torn_lines(self):
        md = fake_dispatcher()
        self.br.record(md, self.path, "qwen38")
        md._get_static_kernel(1, 1, 1, 1, 1, 1, 1)
        with open(self.path, "a") as fh:
            fh.write(self.path.read_text())                      # the same request again (another node's copy)
            fh.write('{"getter": "_get_static_kernel", "args": [1\n')   # a torn last line
        self.assertEqual(len(self.br.read([self.path, Path(self.dir.name) / "missing.jsonl"])), 1)

    def test_a_process_whose_lanes_never_loaded_b12x_records_nothing(self):
        md = fake_dispatcher()
        path = self.br.path_under(self.dir.name, "glm53")
        with unittest.mock.patch.dict(sys.modules, {self.br.DISPATCHER: None}):
            sys.modules.pop(self.br.DISPATCHER)
            self.assertIsNone(self.br.record_loaded("glm53", path))
        with unittest.mock.patch.dict(sys.modules, {self.br.DISPATCHER: md}):
            recorder = self.br.record_loaded("glm53", path)
        md._get_static_kernel(1, 1, 1, 1, 1, 1, 1)
        self.assertEqual(recorder.path, Path(self.dir.name) / "st-b12x-requests" / "glm53.jsonl")
        self.assertEqual(recorder.written, 1)

    def test_prebuild_refuses_a_visible_gpu(self):
        from engine.runtime import b12x_prebuild
        for visible in ("0", None):
            env = {k: v for k, v in os.environ.items() if k != "CUDA_VISIBLE_DEVICES"}
            if visible is not None:
                env["CUDA_VISIBLE_DEVICES"] = visible
            with self.subTest(visible=visible), unittest.mock.patch.dict(os.environ, env, clear=True):
                with self.assertRaises(SystemExit):
                    b12x_prebuild.prebuild([self.path])


class WiringTests(unittest.TestCase):
    def test_every_fleet_boot_records_before_its_capture(self):
        for profile, path, before in (("qwen38", "engine/profiles/qwen38/fleet.py", "= build("),
                                      ("glm53", "engine/profiles/glm53/boot.py", "build(comm, None, lanes")):
            with self.subTest(profile=profile):
                source = (ROOT / path).read_text(encoding="utf-8")
                call = f'b12x_requests.record_loaded("{profile}", b12x_requests.path_under('
                self.assertIn(call, source)
                self.assertLess(source.index(call), source.index(before))

    def test_the_node_prebuild_runs_where_production_serves_without_touching_it(self):
        """No GPU, no network, the tree at /repo (the path the objects' hash includes), the tree's pinned seed, the
        boot's arch directory, and only with the cap plus earlyoom's floor free."""
        script = (ROOT / "launchers/b12x-prebuild.sh").read_text(encoding="utf-8")
        for needle in ("-e CUDA_VISIBLE_DEVICES= -e NVIDIA_VISIBLE_DEVICES=void", "--network none", "--pull never",
                       "-v $STAGE:/repo:ro", "-e FLASHINFER_WORKSPACE_BASE=/cache/cu132", "FLASHINFER_CUDA_ARCH_LIST=12.1",
                       '["seed_image_id"]', "NEED_GIB=$((MEMORY_GIB + FLOOR_GIB + MARGIN_GIB))",
                       '"$avail" ', "--entrypoint nice", "STAGE=/home/choiceoh/st-prebuild/$PROFILE",
                       "python3 -m engine.runtime.b12x_prebuild prebuild"):
            with self.subTest(needle=needle):
                self.assertIn(needle.replace('"$avail" ', '"${avail:-0}" -lt "$NEED_GIB"'), script)
        self.assertNotIn("--gpus", script)
        self.assertIn("exit 0", script.splitlines()[-1])
        launcher = (ROOT / "launchers/start-st-qwen38.sh").read_text(encoding="utf-8")
        self.assertIn('launchers/b12x-prebuild.sh" --tree "$REPO" --profile qwen38', launcher)
        self.assertLess(launcher.index("  prebuild)"), launcher.index("for ip in \"${NODES[@]}\"; do\n  busy="))

    def test_the_getters_named_are_the_dispatchers_disk_cached_ones(self):
        source = (ROOT / "engine/kernels/b12x/moe_dispatch.py").read_text(encoding="utf-8")
        from engine.kernels import b12x_requests as br
        for getter, cache in br.GETTERS.items():
            with self.subTest(getter=getter):
                self.assertIn(f"\ndef {getter}(", source)
                self.assertIn(f"\n{cache}: Dict[Tuple, Tuple] = {{}}", source)
                self.assertIn(f"{cache}[cache_key] = result", source)
        for name in br.CONFIG:
            self.assertIn(f"global {name}", source)


@unittest.skipUnless(all(importlib.util.find_spec(n) for n in ("torch", "cutlass", "flashinfer")),
                     "requires the ST image (torch, CuTe DSL, flashinfer)")
class ImageRoundTripTests(unittest.TestCase):
    """Record in one process against the real dispatcher (compiling into one cache), prebuild in another into an empty
    cache: the same kernel names are built, and a second prebuild finds them all on disk."""

    BOOT = r'''
import os, sys, json
from unittest.mock import patch
os.environ["CUTE_DSL_ARCH"] = "sm_121a"
import torch
with patch.object(torch.cuda, "is_available", return_value=True), \
        patch.object(torch.cuda, "get_device_capability", return_value=(12, 1)):
    from engine.kernels.b12x import moe_dispatch as md
    from engine.kernels import b12x_requests as br
    names = []
    real = md.build_and_load_cute_dsl_kernel
    def build(module, name, fn, **kw):
        names.append(f"{module}/{name}")
        return real(module, name, fn, **kw)
    with patch.object(md, "get_num_sm", return_value=48), patch.object(md, "get_max_active_clusters", return_value=48), \
            patch.object(md, "build_and_load_cute_dsl_kernel", build):
        br.record(md, sys.argv[1], "test")
        md._get_static_kernel(128, 128, 16, 2560, 640, 10, 160)
        md._get_micro_kernel(129, 128, 4, 2560, 640, 10, 40, skip_zero_weight_expert_id=128)
    print("NAMES " + json.dumps(names))
    print("ARCH " + json.dumps([json.loads(l)["device"]["jit_arch"] for l in open(sys.argv[1])]))
'''

    def run_python(self, code, *args, base):
        # the boot's side runs on the CPU here too, so it is told the arch a GB10 boot's flashinfer derives (121a)
        env = dict(os.environ, CUDA_VISIBLE_DEVICES="", FLASHINFER_WORKSPACE_BASE=base, PYTHONPATH=str(ROOT),
                   PYTHONDONTWRITEBYTECODE="1", FLASHINFER_CUDA_ARCH_LIST="12.1")
        return subprocess.run([sys.executable, "-c", code, *args], env=env, capture_output=True, text=True, check=True,
                              timeout=900).stdout

    def test_the_prebuild_makes_the_boots_kernels_and_then_hits_them(self):
        with tempfile.TemporaryDirectory() as root:
            requests = Path(root) / "requests.jsonl"
            boot = self.run_python(self.BOOT, str(requests), base=str(Path(root) / "boot"))
            names = json.loads(next(line for line in boot.splitlines() if line.startswith("NAMES "))[6:])
            self.assertEqual(len(names), 2)
            self.assertEqual(json.loads(next(line for line in boot.splitlines() if line.startswith("ARCH "))[5:]),
                             ["121a", "121a"])
            prebuild = ("import sys, json; from engine.runtime import b12x_prebuild; "
                        "r = b12x_prebuild.prebuild([sys.argv[1]], emit=lambda s: None); print('REPORT ' + json.dumps(r))")
            first = json.loads(next(line for line in self.run_python(prebuild, str(requests), base=str(Path(root) / "new"))
                                    .splitlines() if line.startswith("REPORT "))[7:])
            self.assertEqual(first["summary"]["built"], 2, first)
            self.assertEqual(sorted(k for r in first["requests"] for k in r["kernels"]), sorted(names))
            into = Path(first["summary"]["into"])
            self.assertEqual((into.name, into.parent.name), ("cached_ops", "121a"))
            self.assertEqual(len(list(into.glob("st_b12x_moe_*_sm121a_cute_dsl/*.o"))), 2)
            second = json.loads(next(line for line in self.run_python(prebuild, str(requests), base=str(Path(root) / "new"))
                                     .splitlines() if line.startswith("REPORT "))[7:])
            self.assertEqual(second["summary"]["hit"], 2, second)


if __name__ == "__main__":
    unittest.main()
