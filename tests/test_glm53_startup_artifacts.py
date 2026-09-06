"""CPU tests of exact FP8 artifacts and pre-finalization rank checkpoints."""
import ast
from concurrent.futures import ThreadPoolExecutor
import importlib.util
import json
import multiprocessing
import time
from datetime import timedelta
import os
from pathlib import Path
import sys
import tempfile
import types
import unittest
from unittest.mock import Mock, patch

try:
    import torch
except ImportError:
    torch = None

ROOT = Path(__file__).resolve().parents[1]
MODULES = ROOT / "overlay/modules/glm53_model"


def import_file(name, filename):
    spec = importlib.util.spec_from_file_location(name, MODULES / filename)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _vote_worker(rank, rendezvous, output):
    import torch.distributed as dist
    dist.init_process_group("gloo", init_method=rendezvous, rank=rank,
                            world_size=4, timeout=timedelta(seconds=20))
    try:
        common = import_file("startup_cache_vote", "glm53_startup_cache.py")
        world = types.SimpleNamespace(world_size=4, device_group=dist.group.WORLD)
        with patch.dict(sys.modules, {
            "vllm.model_executor.layers.glm53_startup_cache": common,
            "vllm.distributed": types.SimpleNamespace(get_world_group=lambda: world),
        }):
            cache = import_file("rank_cache_vote", "glm53_rank_cache.py")
            votes = [cache._all_ranks_ready(True), cache._all_ranks_ready(rank != 2),
                     cache._all_ranks_ready(False)]
            Path(output, str(rank)).write_text(json.dumps(votes))
    finally:
        dist.destroy_process_group()


class LoaderTimingTests(unittest.TestCase):
    def test_cache_hit_starts_timer_without_reading_source_iterator(self):
        path = ROOT / "overlay/modules/glm53_runtime/deneb_boot_stamps.py"
        spec = importlib.util.spec_from_file_location("stamps_test", path)
        stamps = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(stamps)
        events = []
        class Loader:
            counter_before_loading_weights = 0.0
            def get_all_weights(self):
                events.append("read")
                yield ("w", object())
        stamps._wrap_weights_iter(Loader, "get_all_weights")
        loader = Loader()
        iterator = loader.get_all_weights()
        self.assertGreater(loader.counter_before_loading_weights, 0)
        self.assertEqual(events, [])
        self.assertEqual(len(list(iterator)), 1)
        self.assertEqual(events, ["read"])


@unittest.skipIf(torch is None, "CPU torch required")
class RankConsensusTests(unittest.TestCase):
    def test_four_real_gloo_ranks_agree_on_all_hit_and_partial_miss(self):
        if not torch.distributed.is_gloo_available():
            self.skipTest("Gloo unavailable")
        with tempfile.TemporaryDirectory() as root:
            rendezvous = "file://" + str(Path(root) / "store")
            ctx = multiprocessing.get_context("spawn")
            workers = [ctx.Process(target=_vote_worker, args=(rank, rendezvous, root))
                       for rank in range(4)]
            try:
                for worker in workers:
                    worker.start()
                deadline = time.monotonic() + 30
                for worker in workers:
                    worker.join(max(0, deadline - time.monotonic()))
                    self.assertEqual(worker.exitcode, 0, "readiness vote failed or hung")
                for rank in range(4):
                    self.assertEqual(json.loads(Path(root, str(rank)).read_text()),
                                     [True, False, False])
            finally:
                for worker in workers:
                    if worker.is_alive():
                        worker.terminate()
                    worker.join(2)


@unittest.skipIf(torch is None, "CPU torch required")
class Fp8ArtifactTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.cache = import_file("startup_cache_test", "glm53_startup_cache.py")
        self.weight = torch.randn(129, 133).to(torch.bfloat16)
        self.quantizer = Mock(side_effect=self.quantize)

    @staticmethod
    def quantize(weight):
        rows, cols = weight.shape
        padded = torch.zeros((rows + 127) // 128 * 128, (cols + 127) // 128 * 128)
        padded[:rows, :cols] = weight.float()
        q = padded.to(torch.float8_e4m3fn)
        # Packed scale view with column-major stride, nonzero storage offset,
        # and trailing allocation padding that vector loads may access.
        backing = torch.arange(96, dtype=torch.int32)
        ws = backing.as_strided((8, 2), (1, 16), 3)
        return q, ws, rows, cols

    def make_cache(self, identity="runtime-1"):
        return self.cache.Fp8Cache(self.tmp.name, identity)

    def assert_exact(self, left, right):
        for a, b in zip(left[:2], right[:2]):
            self.assertEqual(self.cache.tensor_spec(a), self.cache.tensor_spec(b))
            staging = self.cache.HostStaging()
            ar, br = self.cache._storage_record(a, staging), self.cache._storage_record(b, staging)
            self.assertEqual(ar["offset"], br["offset"])
            self.assertTrue(torch.equal(ar["raw"], br["raw"]))
        self.assertEqual(left[2:], right[2:])

    def test_cache_hit_preserves_bytes_strides_offset_and_padding(self):
        first = self.make_cache().quantize(self.weight, self.quantizer)
        cache = self.make_cache()
        second = cache.quantize(self.weight, self.quantizer)
        self.assert_exact(first, second)
        self.assertEqual(self.quantizer.call_count, 1)
        self.assertEqual(cache.hits, 1)
        second[1][0, 0] = -999
        self.assert_exact(first, self.make_cache().quantize(self.weight, self.quantizer))

    def test_float_scales_and_noncontiguous_weight(self):
        def quantize(w):
            q, ws, rows, cols = self.quantize(w)
            return q, ws.float().t(), rows, cols
        quantizer = Mock(side_effect=quantize)
        weight = self.weight.t()
        self.assert_exact(self.make_cache().quantize(weight, quantizer),
                          self.make_cache().quantize(weight, quantizer))
        self.assertEqual(quantizer.call_count, 1)

    def test_source_bytes_shape_and_runtime_invalidate(self):
        self.make_cache().quantize(self.weight, self.quantizer)
        self.weight[0, 0] += 1
        self.make_cache().quantize(self.weight, self.quantizer)
        self.make_cache().quantize(self.weight.t(), self.quantizer)
        self.make_cache("runtime-2").quantize(self.weight, self.quantizer)
        self.assertEqual(self.quantizer.call_count, 4)

    def test_disabled_cache_does_no_hashing_or_disk_io(self):
        with patch.object(self.cache, "tensor_digest", side_effect=AssertionError("unexpected hash")):
            for value in ("", "0", "off", "false"):
                result = self.cache.Fp8Cache(value).quantize(self.weight, self.quantizer)
        self.assertEqual(result[2:], self.weight.shape)
        self.assertEqual(list(Path(self.tmp.name).iterdir()), [])

    def test_corrupt_storage_metadata_and_truncation_rebuild(self):
        for corruption in ("bytes", "stride", "truncated"):
            with self.subTest(corruption=corruption):
                cache = self.make_cache()
                expected = cache.quantize(self.weight, self.quantizer)
                path = cache.last_path
                if corruption == "truncated":
                    path.write_bytes(b"truncated")
                else:
                    payload = torch.load(path, weights_only=True)
                    if corruption == "bytes":
                        payload["q"]["raw"][0] ^= 1
                    else:
                        payload["ws"]["stride"][1] += 1
                    torch.save(payload, path)
                before = self.quantizer.call_count
                self.assert_exact(expected, self.make_cache().quantize(self.weight, self.quantizer))
                self.assertEqual(self.quantizer.call_count, before + 1)

    def test_write_failure_preserves_result_and_cleans_temporary(self):
        with patch.object(self.cache.os, "replace", side_effect=OSError("disk error")):
            result = self.make_cache().quantize(self.weight, self.quantizer)
        self.assert_exact(result, self.quantize(self.weight))
        self.assertEqual(list(Path(self.tmp.name).iterdir()), [])

    def test_source_check_can_evict_artifact(self):
        cache = self.make_cache()
        cache.quantize(self.weight, self.quantizer)
        cache.reject_last()
        self.assertFalse(cache.last_path.exists())
        cache.quantize(self.weight, self.quantizer)
        self.assertEqual(self.quantizer.call_count, 2)

    def test_concurrent_writers_publish_complete_artifacts(self):
        with ThreadPoolExecutor(max_workers=3) as pool:
            values = list(pool.map(lambda _: self.make_cache().quantize(self.weight, self.quantize), range(3)))
        self.assert_exact(values[0], self.make_cache().quantize(self.weight, Mock(side_effect=AssertionError("miss"))))
        self.assertEqual(len(list(Path(self.tmp.name).iterdir())), 1)


@unittest.skipIf(torch is None, "CPU torch required")
class RankArtifactTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.common = import_file("startup_cache_test", "glm53_startup_cache.py")
        self.modules = patch.dict(sys.modules, {
            "vllm.model_executor.layers.glm53_startup_cache": self.common,
            "vllm.distributed": types.SimpleNamespace(get_tensor_model_parallel_rank=lambda: 0),
        })
        self.modules.start()
        self.addCleanup(self.modules.stop)
        self.rank = import_file("rank_cache_test", "glm53_rank_cache.py")
        self.rank.CHUNK_BYTES = 16
        self.rank.runtime_identity = lambda: {"sources": {}, "version": "test"}
        self.rank._all_ranks_ready = Mock(side_effect=lambda ready: ready)
        self.env = patch.dict(os.environ, {"VLLM_GLM53_RANK_CACHE": str(self.root / "cache")})
        self.env.start()
        self.addCleanup(self.env.stop)
        self.source = self.root / "source"
        self.source.mkdir()
        (self.source / "model.safetensors").write_bytes(b"source checkpoint")
        (self.source / "config.json").write_text('{"hidden_size":4}')
        self.loader = Mock(side_effect=self.source_load)

    def model(self, fill=0):
        model = torch.nn.Module()
        model.weight = torch.nn.Parameter(torch.full((7, 8), fill, dtype=torch.bfloat16), requires_grad=False)
        model.register_buffer("scale", torch.full((), fill, dtype=torch.float32))
        model.register_buffer("empty", torch.empty(0))
        model.register_buffer("packed", torch.arange(17, dtype=torch.uint8))
        model.model_config = types.SimpleNamespace(
            model=str(self.source), model_weights=None, hf_config=types.SimpleNamespace(to_dict=lambda: {"hidden_size": 4}),
            dtype=torch.bfloat16, quantization="modelopt", revision=None)
        model.vllm_config = types.SimpleNamespace(
            parallel_config=types.SimpleNamespace(tensor_parallel_size=4, enable_expert_parallel=False),
            load_config=types.SimpleNamespace(load_format="instanttensor"), lora_config=None)
        return model

    def source_load(self, model, weights):
        values = dict(weights)
        with torch.no_grad():
            model.weight.fill_(values["value"])
            model.scale.fill_(values["value"] + 1)
        return {"weight", "scale", "packed"}

    def load(self, model, value=7):
        return self.rank.load_rank_cached(model, iter([("value", value)]), lambda w: self.loader(model, w))

    def artifact(self):
        return next((self.root / "cache").glob("*/manifest.json")).parent

    def rewrite_manifest(self, update):
        path = self.artifact() / "manifest.json"
        envelope = json.loads(path.read_text())
        update(envelope["manifest"])
        envelope["sha256"] = self.common.digest_json(envelope["manifest"])
        path.write_text(json.dumps(envelope))

    def test_rank_hit_skips_iterator_and_preserves_parameters_buffers(self):
        first = self.model()
        loaded = self.load(first)
        second = self.model(-4)
        def never_iterate():
            raise AssertionError("source checkpoint was read on cache hit")
            yield
        result = self.rank.load_rank_cached(second, never_iterate(), Mock(side_effect=AssertionError("source loader called")))
        self.assertEqual(result, loaded)
        for name, tensor in first.state_dict().items():
            self.assertTrue(torch.equal(tensor, second.state_dict()[name]), name)
        self.assertEqual(self.loader.call_count, 1)

    def test_pre_finalization_snapshot_and_reload_bypass(self):
        first = self.model()
        self.load(first)
        first.weight.data.mul_(2)  # stand-in for post-load kernel repacking
        second = self.model()
        self.load(second)
        self.assertTrue(torch.equal(first.weight, second.weight * 2))
        self.load(second, value=13)  # cache cannot override a later weight reload
        self.assertTrue(torch.all(second.weight == 13))
        self.assertEqual(self.loader.call_count, 2)

    def test_registered_layer_aliases_are_saved_once_and_restored_in_place(self):
        def aliased(fill):
            model = self.model(fill)
            model.layers = torch.nn.ModuleList([torch.nn.Linear(8, 7, bias=False)])
            model._active_layers = model.layers[:]
            return model
        first = aliased(0)
        self.load(first)
        envelope = json.loads((self.artifact() / "manifest.json").read_text())
        manifest = envelope["manifest"]
        self.assertEqual(manifest["aliases"], {"_active_layers.0.weight": "layers.0.weight"})
        expected = sum(t.numel() * t.element_size() for n, t in first.state_dict().items()
                       if not n.startswith("_active_layers."))
        self.assertEqual(manifest["size"], expected)
        second = aliased(-1)
        self.load(second)
        self.assertEqual(self.loader.call_count, 1)
        self.assertEqual(second.layers[0].weight.data_ptr(), second._active_layers[0].weight.data_ptr())
        for name, tensor in first.state_dict().items():
            self.assertTrue(torch.equal(tensor, second.state_dict()[name]), name)
        # Same shapes with independent registrations cannot consume an alias.
        third = aliased(-2)
        third._active_layers = torch.nn.ModuleList([torch.nn.Linear(8, 7, bias=False)])
        self.load(third)
        self.assertEqual(self.loader.call_count, 2)

    def test_peer_disk_shortfall_prevents_every_rank_from_writing(self):
        self.rank._all_ranks_ready = Mock(side_effect=[False, False])
        with patch.object(self.rank, "_write", side_effect=AssertionError("peer disk is full")):
            self.load(self.model())
        self.assertEqual([c.args for c in self.rank._all_ranks_ready.call_args_list], [(False,), (True,)])
        self.assertEqual(list((self.root / "cache").iterdir()), [])

    def test_partially_overlapping_views_keep_the_source_loader(self):
        model = self.model()
        shared = torch.arange(12, dtype=torch.float32)
        model.register_buffer("first_view", shared[:8])
        model.register_buffer("shifted_view", shared[4:])
        self.load(model)
        self.assertTrue(torch.all(model.weight == 7))
        self.assertFalse((self.root / "cache").exists())

    def test_source_change_even_with_restored_mtime_invalidates(self):
        self.load(self.model())
        path = self.source / "model.safetensors"
        before = path.stat()
        path.write_bytes(b"SOURCE checkpoint")
        os.utime(path, ns=(before.st_atime_ns, before.st_mtime_ns))
        self.load(self.model(), value=9)
        self.assertEqual(self.loader.call_count, 2)

    def test_config_tp_rank_runtime_and_env_invalidate(self):
        self.load(self.model())
        changed = self.model()
        changed.vllm_config.parallel_config.tensor_parallel_size = 2
        self.load(changed)
        self.rank.runtime_identity = lambda: {"sources": {}, "version": "changed"}
        self.load(self.model())
        with patch.dict(os.environ, {"VLLM_GLM53_FP8_DENSE": "new"}):
            self.load(self.model())
        self.assertEqual(self.loader.call_count, 4)

    def test_transformers_label_keys_and_tuples_survive_json_identity(self):
        def configured():
            model = self.model()
            model.model_config.hf_config.to_dict = lambda: {
                "id2label": {i: f"LABEL_{i}" for i in range(12)},
                "vision_config": {"id2label": {0: "LABEL_0", 1: "LABEL_1"}},
                "shape_hint": (2, 4),
            }
            return model
        first = configured()
        self.load(first)
        second = configured()
        self.load(second)
        self.assertEqual(self.loader.call_count, 1)
        self.assertTrue(torch.equal(first.weight, second.weight))

    def test_missing_truncated_or_incomplete_metadata_falls_back_before_copy(self):
        self.load(self.model())
        for kind in ("truncated", "missing", "extent"):
            with self.subTest(kind=kind):
                if kind == "truncated":
                    (self.artifact() / "weights.bin").write_bytes(b"short")
                else:
                    self.rewrite_manifest(lambda m: m["chunks"].pop() if kind == "missing" else m["chunks"][0].update(start=1))
                target = self.model()
                self.load(target, value=19)
                self.assertTrue(torch.all(target.weight == 19))
        self.assertEqual(self.loader.call_count, 4)

    def test_payload_corruption_aborts_without_source_fallback(self):
        self.load(self.model())
        path = self.artifact() / "weights.bin"
        raw = bytearray(path.read_bytes())
        raw[20] ^= 1  # second chunk, after a first chunk may have been restored
        path.write_bytes(raw)
        with self.assertRaisesRegex(RuntimeError, "checksum mismatch"):
            self.load(self.model(), value=19)
        self.assertEqual(self.loader.call_count, 1)

    def test_disk_failure_and_unpublished_directory_keep_source_path(self):
        with patch.object(self.rank.os, "rename", side_effect=OSError("disk failure")):
            model = self.model()
            self.load(model)
        self.assertTrue(torch.all(model.weight == 7))
        self.assertEqual(list((self.root / "cache").iterdir()), [])
        (self.root / "cache" / ".rank-incomplete").mkdir()
        self.load(self.model())
        self.assertEqual(self.loader.call_count, 2)

    def test_checkpoint_change_during_load_is_not_published(self):
        def change(model, weights):
            result = self.source_load(model, weights)
            (self.source / "config.json").write_text('{"hidden_size":8}')
            return result
        self.loader.side_effect = change
        self.load(self.model())
        self.assertFalse((self.root / "cache").exists())

    def test_ep_meta_noncontiguous_and_secondary_sources_decline(self):
        models = [self.model() for _ in range(3)]
        models[0].vllm_config.parallel_config.enable_expert_parallel = True
        models[1].weight.data = models[1].weight.t()
        models[2].secondary_weights = ["other"]
        for model in models:
            self.load(model)
        self.assertEqual(self.loader.call_count, 3)
        self.assertFalse((self.root / "cache").exists())

    def test_disabled_path_does_not_read_source_metadata(self):
        for value in ("", "0", "off", "false"):
            with patch.dict(os.environ, {"VLLM_GLM53_RANK_CACHE": value}), \
                 patch.object(self.rank, "_context", side_effect=AssertionError("unexpected context")):
                self.load(self.model())
        self.assertFalse((self.root / "cache").exists())

    def test_peer_miss_forces_source_load_even_with_local_hit(self):
        self.load(self.model())
        self.rank._all_ranks_ready = Mock(return_value=False)
        target = self.model()
        self.load(target, value=23)
        self.assertEqual([c.args for c in self.rank._all_ranks_ready.call_args_list], [(True,), (True,)])
        self.assertTrue(torch.all(target.weight == 23))
        self.assertEqual(self.loader.call_count, 2)

    def test_ineligible_rank_still_joins_readiness_vote(self):
        model = self.model()
        model.vllm_config.parallel_config.enable_expert_parallel = True
        self.load(model)
        self.assertEqual([c.args for c in self.rank._all_ranks_ready.call_args_list], [(False,), (False,)])

    def test_outer_wrapper_runs_post_load_once_after_cache_restore(self):
        # Execute the real wrapper load_weights body with a tiny parent loader.
        tree = ast.parse((MODULES / "glm5next_model.py").read_text())
        cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "Glm5NextForConditionalGeneration")
        method = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == "load_weights")
        parent = type("Parent", (torch.nn.Module,), {"load_weights": lambda m, w: self.loader(m, w)})
        wrapper = ast.ClassDef(name="Wrapper", bases=[ast.Name(id="Parent", ctx=ast.Load())],
                               keywords=[], body=[method], decorator_list=[])
        module = ast.fix_missing_locations(ast.Module(body=[wrapper], type_ignores=[]))
        namespace = {"Parent": parent, "torch": torch, "Iterable": list}
        rank_module = patch.dict(sys.modules, {"vllm.model_executor.layers.glm53_rank_cache": self.rank})
        with rank_module:
            exec(compile(module, "wrapper", "exec"), namespace)
            for _ in range(2):
                model = self.model()
                model.__class__ = namespace["Wrapper"]
                hook = Mock(side_effect=lambda: model.weight.data.mul_(2))
                model.language_model = types.SimpleNamespace(run_post_load=hook)
                model.load_weights(iter([("value", 7)]))
                hook.assert_called_once_with()
                self.assertTrue(torch.all(model.weight == 14))
        self.assertEqual(self.loader.call_count, 1)


if __name__ == "__main__":
    unittest.main()
