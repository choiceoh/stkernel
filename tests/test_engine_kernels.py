"""CPU checks for the standalone kernel package's dependency and source boundary."""
import ast
import builtins
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import symtable
import struct
import tempfile
from types import ModuleType
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
KERNELS = ROOT / "engine/kernels"


class KernelPackageTests(unittest.TestCase):
    def test_glm_rank_rejects_ambiguous_expert_layout_before_loading(self):
        from engine.profiles.glm53.weights import WEIGHT_LAYOUT, rank_loader
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "rank.safetensors"
            for marker in (None, "gate-up-v0", WEIGHT_LAYOUT):
                header = {"L3.moe.w13": {"dtype": "U8", "shape": [0], "data_offsets": [0, 0]}}
                if marker:
                    header["__metadata__"] = {"weight_layout": marker}
                raw = json.dumps(header).encode()
                path.write_bytes(struct.pack("<Q", len(raw)) + raw)
                if marker == WEIGHT_LAYOUT:
                    self.assertEqual(rank_loader(path).keys(), ["L3.moe.w13"])
                else:
                    with self.assertRaisesRegex(ValueError, "regenerate rank files"):
                        rank_loader(path)

    def test_ported_functions_have_their_module_globals(self):
        # Catch globals lost when extracting MLA/mHC from larger overlay files,
        # even in optional kernel branches that import-only checks cannot run.
        for path in KERNELS.rglob("*.py"):
            table = symtable.symtable(path.read_text(), str(path), "exec")
            known = {s.get_name() for s in table.get_symbols()
                     if s.is_assigned() or s.is_imported() or s.is_namespace()}
            known |= set(dir(builtins)) | {"__file__", "__name__", "__package__", "__annotations__"}
            pending = list(table.get_children())
            while pending:
                scope = pending.pop()
                for symbol in scope.get_symbols():
                    if symbol.is_global() and symbol.is_referenced():
                        self.assertIn(symbol.get_name(), known, (path, scope.get_name(), symbol.get_name()))
                pending.extend(scope.get_children())

    def test_deep_gemm_initializes_on_execution_and_forwards_the_library_contract(self):
        calls = []
        library = ModuleType("deep_gemm")
        library.set_pdl = lambda value: calls.append(("pdl", value))
        sentinel = object()
        def logits(*args, **kwargs):
            calls.append(("logits", args, kwargs))
            return sentinel
        library.fp8_fp4_mqa_logits = logits
        library.tf32_hc_prenorm_gemm = lambda *args: calls.append(("prenorm", args))
        spec = importlib.util.spec_from_file_location("st_test_deep_gemm", KERNELS / "deep_gemm.py")
        module = importlib.util.module_from_spec(spec)
        with patch.dict(sys.modules, {"deep_gemm": library}):
            spec.loader.exec_module(module)
        self.assertEqual(calls, [])  # CPU package inspection must not create a CUDA context.
        values = tuple(object() for _ in range(5))
        self.assertIs(module.fp8_fp4_mqa_logits(*values, clean_logits=False), sentinel)
        module.tf32_hc_prenorm_gemm(*values)
        self.assertEqual(calls, [("pdl", False), ("logits", values, {"clean_logits": False}), ("prenorm", values)])

    def test_engine_never_imports_vllm_or_mounted_moe_kernels(self):
        for path in (ROOT / "engine").rglob("*.py"):
            for node in ast.walk(ast.parse(path.read_text())):
                names = []
                if isinstance(node, ast.Import):
                    names = [a.name for a in node.names]
                elif isinstance(node, ast.ImportFrom):
                    names = [node.module or ""]
                elif isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and node.func.attr == "import_module":
                    names = [a.value for a in node.args if isinstance(a, ast.Constant) and isinstance(a.value, str)]
                for name in names:
                    with self.subTest(path=path, name=name):
                        self.assertFalse(name == "vllm" or name.startswith(("vllm.", "flashinfer.fused_moe")))

    def test_relative_imports_resolve_inside_the_vendored_package(self):
        for path in KERNELS.rglob("*.py"):
            package = ".".join(path.relative_to(ROOT).parent.parts)
            for node in ast.walk(ast.parse(path.read_text())):
                if not isinstance(node, ast.ImportFrom) or not node.level:
                    continue
                base = importlib.util.resolve_name("." * node.level + (node.module or ""), package)
                module = ROOT.joinpath(*base.split("."))
                if node.module:
                    self.assertTrue(module.with_suffix(".py").is_file() or (module / "__init__.py").is_file(), (path, base))
                else:
                    for alias in node.names:
                        child = module / alias.name
                        self.assertTrue(child.with_suffix(".py").is_file() or (child / "__init__.py").is_file(), (path, base, alias.name))

    def test_cuda_translation_unit_and_pinned_dynamic_helpers_are_unchanged(self):
        manifest = json.loads((KERNELS / "SOURCES.json").read_text())["files"]
        for name in ("mla/glm53_megakernel.cu", "b12x/_moe_dynamic/gated.py"):
            self.assertEqual(hashlib.sha256((KERNELS / name).read_bytes()).hexdigest(), manifest[name]["sha256"])

    def test_strided_kda_guard_tracks_the_ported_norm_source(self):
        tree = ast.parse((KERNELS / "kda/kda.py").read_text())
        assignment = next(n for n in tree.body if isinstance(n, ast.Assign)
                          and any(isinstance(t, ast.Name) and t.id == "_GLM53_L2NORM_SHA256" for t in n.targets))
        self.assertEqual(ast.literal_eval(assignment.value), hashlib.sha256((KERNELS / "kda/l2norm.py").read_bytes()).hexdigest())


if __name__ == "__main__":
    unittest.main()
