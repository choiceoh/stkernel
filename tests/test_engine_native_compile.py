"""The compiler comparison must not mask changes to GPU code or metadata."""
from pathlib import Path
import struct
import tempfile
import unittest

from probes.engine_native_compile_check import canonical_cubin


class CubinComparisonTests(unittest.TestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.root = Path(directory.name)
        self.serial = 0

    def cubin(self, *, file_id=b'1234abcd', code=b'\x17\x33', constant=b'\x00\x10',
              metadata=b'\x01\x00', public=b'public_kernel'):
        # A small valid ELF64 section table keeps the test independent of
        # Torch, CUDA tooling and checked-in executable binaries.
        names = b'\x00.shstrtab\x00.strtab\x00.text.kernel\x00.nv.constant0.kernel\x00.nv.info.kernel\x00'
        symbols = (b'\x00_INTERNAL_' + file_id + b'_18_dsv4_oneshot_ar_cu_suffix\x00_GLOBAL__N__'
                   + file_id + b'_18_dsv4_oneshot_ar_cu_suffix\x00' + public + b'\x00')
        sections = [(b'.shstrtab', names), (b'.strtab', symbols), (b'.text.kernel', code),
                    (b'.nv.constant0.kernel', constant), (b'.nv.info.kernel', metadata)]
        data = bytearray(64)
        rows = [(0,) * 10]
        for name, content in sections:
            rows.append((names.index(name + b'\x00'), 3 if name.endswith(b'strtab') else 1,
                         0, 0, len(data), len(content), 0, 0, 1, 0))
            data.extend(content)
        table_offset = len(data)
        for row in rows:
            data.extend(struct.pack('<IIQQQQIIQQ', *row))
        ident = b'\x7fELF\x02\x01\x01' + bytes(9)
        struct.pack_into('<16sHHIQQQIHHHHHH', data, 0, ident, 1, 190, 1, 0, 0, table_offset,
                         0, 64, 0, 0, 64, len(rows), 1)
        self.serial += 1
        path = self.root / f'{self.serial}.cubin'
        path.write_bytes(data)
        return path

    def test_only_observed_internal_file_ids_are_normalized(self):
        first = canonical_cubin(self.cubin())
        self.assertEqual(first, canonical_cubin(self.cubin(file_id=b'5678beef')))
        self.assertEqual(first[1:], (2, 1))
        self.assertNotEqual(first[0], canonical_cubin(self.cubin(public=b'public_5678beef'))[0])

    def test_instruction_constant_and_launch_metadata_changes_fail_equality(self):
        first = canonical_cubin(self.cubin())[0]
        for change in (dict(code=b'\x16\x33'), dict(constant=b'\x01\x10'), dict(metadata=b'\x02\x00')):
            with self.subTest(change=change):
                self.assertNotEqual(first, canonical_cubin(self.cubin(**change))[0])

    def test_unknown_binary_format_is_refused(self):
        path = self.root / 'wrong.cubin'
        path.write_bytes(b'not ELF64')
        with self.assertRaisesRegex(ValueError, 'ELF64'):
            canonical_cubin(path)


if __name__ == '__main__':
    unittest.main()
