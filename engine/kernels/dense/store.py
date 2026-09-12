"""Weight-addressed W4 packs with explicit calibration provenance.

Legacy v4 packs require weight, shape, namespace and layout checks. A GPTQ
pack also needs an exact calibration digest; historical caches without it
are rebuilt because the old builder could silently fall back to RTN.
"""
from collections import Counter
import hashlib
import inspect
import os
from pathlib import Path

import torch


class PackStore:
    def __init__(self, root, rank):
        self.root, self.rank = Path(root), rank
        self.stats = Counter()
        self.read_files = set()
        from engine.kernels.dense import pack_w4
        self.algorithm = hashlib.sha256(
            Path(__file__).with_name('packing.py').read_bytes()
            + inspect.getsource(pack_w4).encode()).hexdigest()

    TILE = 4096                                  # DenseLinear packs K in tiles of this width, one calibration blob each

    @staticmethod
    def tiles(name, cols):
        """(blob key, first column, width) of every K tile a dense weight of `cols` columns is packed and calibrated in."""
        if cols <= PackStore.TILE:
            return [(name, 0, cols)]
        return [(f"{name}.k{i}", i * PackStore.TILE, min(PackStore.TILE, cols - i * PackStore.TILE))
                for i in range((cols + PackStore.TILE - 1) // PackStore.TILE)]

    def calibration_path(self, key, rank=None):
        return self.root/'mkcalib'/f'rank{self.rank if rank is None else rank}'/(key+'.pt')

    def missing_calibration(self, name, cols):
        """The tiles of `name` this store has no calibration blob for: what a calibrating boot must sum. A blob that
        exists but does not fit the tile is reported by name -- `pack` refuses it, so the boot says which file to remove."""
        missing, foreign = [], []
        for key, start, width in self.tiles(name, cols):
            path = self.calibration_path(key)
            if not path.is_file():
                missing.append((key, start, width))
                continue
            blob = torch.load(path, map_location='cpu', mmap=True, weights_only=True)
            hessian = blob.get('H')
            if (hessian is None or tuple(hessian.shape) != (width, width) or int(blob.get('ntok', 0)) <= 0
                    or blob.get('name', key) != key or not hessian.is_floating_point()):
                foreign.append(str(path))
        if foreign:
            raise ValueError(f"calibration blobs that do not fit {name} [{cols} columns]: remove them and reboot -- {foreign}")
        return missing

    def pack(self, weight, name, *, rank=None):
        from engine.kernels.dense import pack_w4
        rank = self.rank if rank is None else rank
        per_row = not name.startswith('DFlash2Qwen3ForCausalLM/')
        n, k = weight.shape
        path = self.root/'mkcalib'/f'rank{rank}'/(name+'.pt')
        hessian = None
        if path.is_file():
            self.read_files.add(path)
            blob = torch.load(path, map_location='cpu', mmap=True, weights_only=True)
            hessian = blob['H']
            if (tuple(hessian.shape) != (k,k) or int(blob['ntok']) <= 0
                    or blob.get('name', name) != name or not hessian.is_floating_point()
                    or not torch.isfinite(hessian).all()):
                raise ValueError(f'incompatible calibration: {path} for {tuple(weight.shape)}')
        raw = weight.detach().contiguous().view(torch.uint8).cpu().numpy()
        weight_sha = hashlib.sha256(raw).hexdigest()
        calibration_sha = (hashlib.sha256(hessian.contiguous().numpy()).hexdigest()
                           if hessian is not None else 'rtn')
        identity = dict(version=2, weight=weight_sha, shape=(n,k), name=name,
                        calibration=calibration_sha, per_row=per_row, algorithm=self.algorithm)
        key = hashlib.sha256(repr(identity).encode()).hexdigest()
        cache = self.root/'st-dense-packs'/(key+'.pt')
        pack = None
        if cache.is_file():
            self.read_files.add(cache)
            blob = torch.load(cache, map_location='cpu', mmap=True, weights_only=True)
            if blob['identity'] != identity:
                raise ValueError(f'dense pack identity mismatch: {cache}')
            pack = self.decode(blob, weight.device, n, k)
            self.stats['cache'] += 1
        else:
            kind = 'gptq' if hessian is not None else 'rtn'
            # The original cache used both MD5 and SHA256 aliases.
            for digest in ('sha256-'+weight_sha, hashlib.md5(raw).hexdigest()):
                mode = 'row' if per_row else 'ten'
                legacy = self.root/'mkpacks'/f'rank{rank}'/(digest+f'-{n}x{k}-bfloat16-v4-{mode}-{kind}-lr0.pt')
                if not legacy.is_file():
                    continue
                self.read_files.add(legacy)
                blob = torch.load(legacy, map_location='cpu', mmap=True, weights_only=True)
                if hessian is not None and blob.get('calibration_sha256') != calibration_sha:
                    self.stats['legacy_unverified_skipped'] += 1
                    continue
                if (blob.get('version') != 4 or tuple(blob.get('shape', ())) != (n,k)
                        or blob.get('name') != name or blob.get('lr_a') is not None
                        or blob.get('lr_b') is not None or (per_row and float(blob['wgs']) != 1.)):
                    raise ValueError(f'incompatible legacy pack: {legacy}')
                rowscale = blob['rgs'] if per_row else torch.full(((n+127)//128*128,),float(blob['wgs']))
                pack = self.decode(dict(data=blob['wq4'], scale=blob['ws4'], rowscale=rowscale), weight.device,n,k)
                self.stats['legacy'] += 1
                break
            if pack is None:
                pack = pack_w4(weight, hessian=hessian, per_row=per_row)
                self.stats['built'] += 1
            cache.parent.mkdir(parents=True, exist_ok=True)
            temporary = cache.with_suffix(f'.{os.getpid()}.tmp')
            try:
                torch.save(dict(identity=identity, data=pack.data.cpu(), scale=pack.scale.cpu(),
                                rowscale=pack.rowscale.cpu()), temporary)
                os.replace(temporary,cache)
            finally:
                temporary.unlink(missing_ok=True)
        self.stats['gptq' if hessian is not None else 'rtn'] += 1
        return pack

    def release_pages(self):
        """All mmap readers have returned; return their clean UMA file cache."""
        for path in self.read_files:
            with path.open('rb') as handle:
                os.posix_fadvise(handle.fileno(),0,0,os.POSIX_FADV_DONTNEED)
        self.read_files.clear()

    @staticmethod
    def decode(blob, device, n, k):
        from engine.kernels.dense import W4Pack
        padded = (n+127)//128*128
        values = [blob[key] for key in ('data','scale','rowscale')]
        for t, shape, dtype in zip(values, ((padded//128,k//128,128,64),
                                           (padded//128,k//128,128,8),(padded,)),
                                  (torch.uint8,torch.int8,torch.float32)):
            if t is None or tuple(t.shape) != shape or t.dtype != dtype or not t.is_contiguous():
                raise ValueError('invalid native W4 pack layout')
        if not torch.isfinite(values[2]).all() or not (values[2]>0).all():
            raise ValueError('invalid W4 row scales')
        return W4Pack(*(t.to(device) for t in values),n,k)
