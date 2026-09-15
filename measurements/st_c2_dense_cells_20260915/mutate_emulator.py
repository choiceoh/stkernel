"""Mutation check: the emulator must report mismatches for plausible offset bugs in the kernel formulas."""
import importlib.util
import pathlib

src = pathlib.Path(__file__).with_name('emulate_rows16.py').read_text()
MUTATIONS = {
    'W row offset in tile': ('(nt % 16) * 512', '(nt % 16) * 1024'),
    'scale chunk destination': ('NIB + (warp * 4 + lane) * 16', 'NIB + (warp * 8 + lane) * 16'),
    'swizzle key': ('((q ^ ((r >> 1) & 3)) << 4)\n                    ring', '((q ^ (r & 3)) << 4)\n                    ring'),
    'second activation row': ('xr0 + 8 * KSTEP + koff:', 'xr0 + 16 * KSTEP + koff:'),
    'partial row/col layout': ('partial[kb * 128 + (g + 8 * (i >> 1)) * 8 + 2 * q + (i & 1)]', 'partial[kb * 128 + (2 * q + (i & 1)) * 16 + g + 8 * (i >> 1)]'),
    'slice boundary': ('range(kblks * s // slices, kblks * (s + 1) // slices):\n                        o =', 'range(kblks * s // slices + 1, kblks * (s + 1) // slices + 1 if s < slices - 1 else kblks):\n                        o ='),
    'exponent halfword': ('ring[NIB + r * 8 + 2 * q], ring[NIB + r * 8 + 2 * q + 1]', 'ring[NIB + r * 8 + q], ring[NIB + r * 8 + q + 1]'),
}
for name, (old, new) in MUTATIONS.items():
    assert src.count(old) == 1, name
    code = src.replace(old, new).replace("if __name__ == '__main__':", "if False:")
    spec = importlib.util.spec_from_loader('mutant', loader=None)
    module = importlib.util.module_from_spec(spec)
    exec(compile(code, 'mutant', 'exec'), module.__dict__)
    for n, k, slices in ((4096, 2048, 3), (6416, 4096, 8)):
        blocks = n // 8
        try:
            checked, bad = module.run(n, k, slices, [0, 17, blocks - 1])
        except Exception as exc:
            print(f"{name}: n={n}: detected as {type(exc).__name__}"); continue
        print(f'{name}: n={n} k={k} slices={slices}: {bad} mismatches of {checked * 16}')
