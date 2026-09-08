"""Exact CPU checks of the serving sf6 packer and both stage layouts."""
from __future__ import annotations

import ast
import importlib
from pathlib import Path
import random
import sys
import types
import unittest
import weakref
import logging
import gc

ROOT = Path(__file__).resolve().parents[1]
MODULE = ROOT / "overlay/modules/glm53_moe"
PACKAGE = "_test_reform_sf6"
package = types.ModuleType(PACKAGE)
package.__path__ = [str(MODULE)]
sys.modules.setdefault(PACKAGE, package)
sf6 = importlib.import_module(PACKAGE + ".moe_reform_sf_pack")


class ByteContract(unittest.TestCase):
    def test_all_byte_codes_and_packing_boundaries(self):
        rng = random.Random(991)
        for base, span in [(0, 1), (0, 64), (64, 64), (128, 64), (192, 64),
                           (255, 1), (31, 17), (112, 63)]:
            raw = bytes(base + i % span for i in range(2048))
            raw = bytearray(raw)
            rng.shuffle(raw)
            packed = sf6.pack_stage_bytes(bytes(raw))
            self.assertEqual(len(packed), 1552)
            self.assertEqual(sf6.unpack_stage_bytes(packed), bytes(raw))
            self.assertEqual(packed[1537:], bytes(15))

    def test_full_code_range_falls_back_without_clipping(self):
        for raw in (bytes(range(256))*8, bytes([0, 64])*1024,
                    bytes([191, 255])*1024):
            before = bytes(raw)
            self.assertIsNone(sf6.pack_stage_bytes(raw))
            self.assertEqual(raw, before)

    def test_rejects_invalid_encodings_and_geometry(self):
        for raw in (b"", bytes(2047), bytes(2049)):
            with self.assertRaises(ValueError):
                sf6.pack_stage_bytes(raw)
        packed = bytearray(sf6.pack_stage_bytes(bytes(2048)))
        packed[1537] = 1
        with self.assertRaises(ValueError):
            sf6.unpack_stage_bytes(bytes(packed))
        packed = bytearray(sf6.pack_stage_bytes(bytes([63])*2048))
        packed[1536] = 255
        packed[0] = 15
        with self.assertRaises(ValueError):
            sf6.unpack_stage_bytes(bytes(packed))
        for args in ((127, 256, "fc1"), (256, 127, "fc2"), (256, 256, "other")):
            with self.assertRaises(ValueError):
                sf6.stage_shape(*args)

    def test_byte_map_matches_nvfp4_row_and_k_coordinates(self):
        # Derive the original and shared offsets from NVFP4's independent
        # ((32,4), row_blocks), ((16,4), k_blocks) scale layout.
        def offset(row, col, rows, k):
            return ((row // 128) * (128 * k // 16) + (col // 4) * 512
                    + (row % 32) * 16 + ((row // 32) % 4) * 4 + col % 4)
        for kind, rows, k, rn, kn in [("fc1", 1024, 4096, 128, 256),
                                     ("fc2", 4096, 512, 256, 128)]:
            nr, nk = sf6.stage_shape(rows, k, kind)
            for expert in (0, 1, 2):
                for rt in (0, nr-1):
                    for kt in (0, nk-1):
                        seen = set()
                        for row in range(rn):
                            for col in range(kn // 16):
                                stage = offset(row, col, rn, kn)
                                seen.add(stage)
                                source = expert*rows*k//16 + offset(
                                    rt*rn+row, kt*(kn//16)+col, rows, k)
                                self.assertEqual(sf6.stage_source_offset(
                                    rows, k, kind, expert, rt, kt, stage), source)
                        self.assertEqual(seen, set(range(2048)))


class TensorContract(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # No skip: CPU release evidence must actually exercise torch packing.
        cls.torch = importlib.import_module("torch")

    def plane(self, experts, rows, k):
        torch = self.torch
        # Each 128-row SF block has its own distinct base; the FC2 gather
        # therefore cannot pass by accidentally grouping adjacent K blocks.
        size = experts * rows * k // 16
        x = torch.arange(size, dtype=torch.int64)
        return ((x % 16) + ((x // 1024) % 4)*16 + (x // (rows*k//16))*64).to(torch.uint8)

    def test_actual_torch_packer_both_planes_and_source_retention(self):
        torch = self.torch
        for kind, rows, k in (("fc1", 256, 512), ("fc2", 512, 256)):
            raw = self.plane(2, rows, k)
            before = raw.clone()
            packed, reason = sf6.pack_plane(raw, experts=2, rows=rows, k=k, kind=kind)
            self.assertIsNone(reason)
            nr, nk = sf6.stage_shape(rows, k, kind)
            self.assertEqual(tuple(packed.shape), (2, nr*nk, 1552))
            self.assertTrue(torch.equal(raw, before))
            for expert in range(2):
                for rt in range(nr):
                    for kt in range(nk):
                        expected = bytes(int(raw[sf6.stage_source_offset(
                            rows, k, kind, expert, rt, kt, byte)]) for byte in range(2048))
                        actual = bytes(packed[expert, rt*nk+kt].tolist())
                        self.assertEqual(actual, sf6.pack_stage_bytes(expected))
                        self.assertEqual(sf6.unpack_stage_bytes(actual), expected)

    def test_chunk_edges_and_both_plane_admission(self):
        torch = self.torch
        original = sf6.REFORM_SF_CHUNK
        sf6.REFORM_SF_CHUNK = 3
        try:
            fc1, fc2 = self.plane(2, 512, 512), self.plane(2, 512, 256)
            owner = sf6.prepare_reform_scales(fc1, fc2, experts=2, n=256, k=512)
            self.assertTrue(owner.enabled)
            first = owner.fc1.clone()
            fc2[0], fc2[1] = 0, 255
            fallback = sf6.prepare_reform_scales(fc1, fc2, experts=2, n=256, k=512)
            self.assertFalse(fallback.enabled)
            self.assertIsNone(fallback.fc1)
            self.assertIsNone(fallback.fc2)
            self.assertIn("fc2", fallback.reason)
            self.assertTrue(torch.equal(owner.fc1, first))
        finally:
            sf6.REFORM_SF_CHUNK = original

    def test_actual_packer_rejects_bad_storage(self):
        torch = self.torch
        with self.assertRaises(ValueError):
            sf6.pack_plane(torch.zeros(0, dtype=torch.uint8), experts=0,
                           rows=128, k=256, kind="fc1")
        for raw in (torch.zeros(2048, dtype=torch.int16),
                    torch.zeros(4096, dtype=torch.uint8)[::2],
                    torch.zeros(2047, dtype=torch.uint8)):
            with self.assertRaises(ValueError):
                sf6.pack_plane(raw, experts=1, rows=128, k=256, kind="fc1")

    def test_serving_cache_generation_and_owner_lifetime(self):
        tree = ast.parse((MODULE / "moe_dispatch.py").read_text())
        names = {"_prepared_reform_scales", "_sf6_tensor_version", "_register_cache_eviction"}
        nodes = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name in names]
        ns = {"torch": self.torch, "weakref": weakref, "logging": logging,
              "Dict": dict, "Tuple": tuple, "_REFORM_SF_CACHE": {},
              "prepare_reform_scales": sf6.prepare_reform_scales}
        exec(compile(ast.Module(body=nodes, type_ignores=[]), "serving-sf6-cache", "exec"), ns)
        prepare = ns["_prepared_reform_scales"]
        first, second = self.plane(1, 512, 512), self.plane(1, 512, 256)
        owner = prepare(first, second, first, second, experts=1, n=256, k=512)
        self.assertIs(owner, prepare(first, second, first.clone(), second.clone(),
                                     experts=1, n=256, k=512))
        original = owner.fc1.clone()
        first.add_(1)  # Same pointer, new version: old graphs retain old owner.
        fresh = prepare(first, second, first, second, experts=1, n=256, k=512)
        self.assertIsNot(fresh, owner)
        self.assertTrue(self.torch.equal(owner.fc1, original))
        self.assertEqual(len(ns["_REFORM_SF_CACHE"]), 2)
        del first, second
        gc.collect()
        self.assertEqual(ns["_REFORM_SF_CACHE"], {})
        self.assertTrue(self.torch.equal(owner.fc1, original))


class DispatchContract(unittest.TestCase):
    def namespace(self):
        tree = ast.parse((MODULE / "moe_dispatch.py").read_text())
        wanted = {"_parse_glm53_static_v2", "_static_v2_decode_config"}
        nodes = [node for node in tree.body if isinstance(node, ast.FunctionDef)
                 and node.name in wanted]
        ns = {"_GLM53_B12X_STATIC_V2_ENV": "VLLM_GLM53_B12X_STATIC_V2"}
        for node in tree.body:
            if isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and
                    t.id in ("_STATIC_V2_DEFAULT", "_STATIC_SUNSET_TOKENS") for t in node.targets):
                exec(compile(ast.Module(body=[node], type_ignores=[]), "dispatch", "exec"), ns)
        exec(compile(ast.Module(body=nodes, type_ignores=[]), "dispatch", "exec"), ns)
        return ns

    def test_defaults_stay_off_and_q_stays_probe_only(self):
        ns = self.namespace()
        parse = ns["_parse_glm53_static_v2"]
        self.assertFalse(parse("t,r")["reform_sf_pack"])
        self.assertTrue(parse("t,r,sf6")["reform_sf_pack"])
        for bad in ("sf6", "t,sf6", "t,r,q", "t,r,sf6,q", "t,r,sf6,v"):
            with self.assertRaises(ValueError):
                parse(bad)
        self.assertTrue(parse("t,q", probe=True)["sf_pack"])
        for m in (0, 9, 16, 8192):
            config = ns["_static_v2_decode_config"](parse("t,r,sf6"), m)
            self.assertFalse(config["reform_sf_pack"])
            self.assertFalse(config["decode_reform"])
        for m in (1, 2, 6, 8):
            self.assertTrue(ns["_static_v2_decode_config"](
                parse("t,r,sf6"), m)["reform_sf_pack"])


if __name__ == "__main__":
    unittest.main()
