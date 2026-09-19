"""Weight-addressed W4 packs with explicit calibration provenance.

Legacy v4 packs require weight, shape, namespace and layout checks. A GPTQ
pack also needs an exact calibration digest; historical caches without it
are rebuilt because the old builder could silently fall back to RTN.
"""
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
import hashlib
import inspect
import os
from pathlib import Path
from typing import NamedTuple

import torch


def _bytes_key(tensor):
    """What names a tensor's bytes while it lives: its storage and place in it, its layout and its write count."""
    return (str(tensor.device), tensor.dtype, tuple(tensor.shape), tuple(tensor.stride()),
            tensor.untyped_storage().data_ptr(), tensor.storage_offset(), tensor._version)


class WeightDigest:
    """sha256 of one weight's bytes, hashed once for every lane that packs it (`PackStore.weight_digest`).

    A weight's W4 pack and its FP8 pack are looked up by the same bytes, and each lane used to copy them to
    the host and hash them again. The digest answers only for the bytes it hashed -- the same storage, offset,
    layout and write count -- because a pack must never be read under another tensor's hash.
    """

    def __init__(self, weight, future):
        self.key = _bytes_key(weight)
        self._future = future
        # (blob path, device, inode, size, mtime, k, smoothing hash) -> sha256 of that validated, smoothed Hessian: the
        # W4 and FP8 lanes of this weight read one blob under one smoothing. Scoped to this weight's lanes, so a store
        # that is asked again later (a rewritten blob, a new boot's calibration) reads the file again.
        self.calibrations = {}

    def covers(self, weight):
        return _bytes_key(weight) == self.key

    def of(self, weight):
        if not self.covers(weight):
            raise ValueError("a weight digest was offered for different bytes")
        return self._future.result()


class Need(NamedTuple):
    """A calibration blob a boot has to sum, and how much of it. `hessian` False is a blob whose Gram sums already
    fit the served weight but that predates the channel peaks the smoothing reads (kernels/dense/smoothing): the
    peaks are [K] floats beside a [K, K] Hessian that stays on disk, so summing them costs kilobytes instead of the
    gigabytes a whole Hessian would take from the boot's budget."""
    key: str
    start: int
    width: int
    hessian: bool = True


class PackStore:
    def __init__(self, root, rank, weights_id=None, *, require_identity=False):
        if require_identity and weights_id is None:
            raise ValueError("strict calibration provenance requires a weights_id")
        self.root, self.rank = Path(root), rank
        self.weights_id = weights_id             # the weights this boot serves; a blob summed under others is refused
        self.require_identity = require_identity
        self.foreign = set()                     # names whose blob was summed under other weights (re-sum these)
        self._claims = {}                        # calibration path -> the weights_id it claims, cached per boot
        self.stats = Counter()
        self.read_files = set()
        self._factor_entry = None                # (identity, factor) of the last weight: its two lanes share one factorisation
        self._hasher = None                       # one worker: a weight's bytes hash while its calibration loads and hashes
        self.gptq_damping = {}                    # explicit per-reader preparation tuning; default identities stay intact
        from engine.kernels.dense import pack_w4
        self.algorithm = hashlib.sha256(
            Path(__file__).with_name('packing.py').read_bytes()
            + inspect.getsource(pack_w4).encode()).hexdigest()

    TILE = 4096                                  # DenseLinear packs K in tiles of this width, one calibration blob each
    FACTOR_BYTES = 256 << 20                     # the largest inverse factor kept between a weight's two lanes (K <= 8192)

    @staticmethod
    def tiles(name, cols):
        """(blob key, first column, width) a dense weight of `cols` columns is calibrated in: one blob over the whole K
        -- a weight wider than the decode kernel's tile is packed by one GPTQ over its full Hessian (pack_wide), so
        the calibration covers the columns' correlations across the tiles."""
        return [Need(name, 0, cols)]

    def calibration_path(self, key, rank=None):
        return self.root/'mkcalib'/f'rank{self.rank if rank is None else rank}'/(key+'.pt')

    def fits_weights(self, path):
        """Whether that blob's Hessian describes the inputs THIS boot's weights make.

        A Hessian is a sum over one boot's activations, so it belongs to the checkpoint that produced them --
        but `calibration_path` keys a blob by the weight's NAME alone, so a boot that changed checkpoints would
        otherwise GPTQ every weight against a distribution it no longer has. Measured 2026-09-16 on the layer-1
        KDA projections: a blob summed on another arm cost +43..114%, and on o_proj the mismatched GPTQ came out
        WORSE than RTN (8.95e-2 vs 7.48e-2) at every damping and shrinkage tried -- a stale Hessian is worth less
        than no Hessian, so refusing it is the point. The three 2026-09-14 eval arms shared one blob set and their
        head NLL ordered by how far their experts sat from the arm that summed it.

        A blob written before this field makes no claim and is taken unless require_identity is set. A new profile
        can require a stamp without changing the legacy fleet's cache policy.
        """
        if self.weights_id is None:
            return True
        if path not in self._claims:
            try:
                blob = torch.load(path, map_location='cpu', mmap=True, weights_only=True)
                self._claims[path] = blob.get('weights_id')
            except Exception:
                return True                              # unreadable here: the normal blob validator reports it
        claimed = self._claims[path]
        return (claimed is None and not self.require_identity) or claimed == self.weights_id

    def calibrated(self, name):
        """Whether this store holds a calibration blob for `name` that was summed under this boot's weights
        (any shape: `pack` checks the fit). A foreign blob reads as uncalibrated, so the pack rounds to nearest."""
        path = self.calibration_path(name)
        if not path.is_file():
            return False
        if not self.fits_weights(path):
            self.foreign.add(name)
            return False
        return True

    def missing_calibration(self, name, cols):
        """The blobs of `name` this store lacks: what a calibrating boot must sum. A blob that exists but does not fit
        the weight is reported by name -- `pack` refuses it, so the boot says which file to remove."""
        missing, foreign = [], []
        for tile in self.tiles(name, cols):
            key, start, width = tile.key, tile.start, tile.width
            path = self.calibration_path(key)
            if not path.is_file():
                missing.append(Need(key, start, width))
                continue
            if not self.fits_weights(path):           # summed under other weights: this boot must sum its own
                self.foreign.add(name)
                missing.append(Need(key, start, width))
                continue
            blob = torch.load(path, map_location='cpu', mmap=True, weights_only=True)
            hessian = blob.get('H')
            if (hessian is None or tuple(hessian.shape) != (width, width) or int(blob.get('ntok', 0)) <= 0
                    or blob.get('name', key) != key or not hessian.is_floating_point()):
                foreign.append(str(path))
            elif blob.get('amax') is None:                  # a blob from before the peaks: only they are summed, its Hessian packs GPTQ meanwhile
                missing.append(Need(key, start, width, hessian=False))
        if foreign:
            raise ValueError(f"calibration blobs that do not fit {name} [{cols} columns]: remove them and reboot -- {foreign}")
        return missing

    def _hessian(self, name, k, smooth=None):
        """The calibration blob of `name` as a finite [k, k] float Hessian, or None when the store has none; with
        `smooth` [k] the Hessian of the input divided by it (the weight was multiplied by it: kernels/dense/smoothing)."""
        path = self.calibration_path(name)
        if not path.is_file():
            return None
        self.read_files.add(path)
        blob = torch.load(path, map_location='cpu', mmap=True, weights_only=True)
        hessian = blob['H']
        if (tuple(hessian.shape) != (k,k) or int(blob['ntok']) <= 0
                or blob.get('name', name) != name or not hessian.is_floating_point()
                or not torch.isfinite(hessian).all()):
            raise ValueError(f'incompatible calibration: {path} for a [.., {k}] weight')
        if smooth is not None:
            from engine.kernels.dense.smoothing import smooth_hessian
            hessian = smooth_hessian(hessian.float(), smooth.cpu())
        return hessian

    def weight_digest(self, weight):
        """Start hashing `weight`'s bytes on the store's worker; every lane of the weight reads the returned digest.

        The copy to the host is taken here, on the caller's thread, and it is always a copy: a later write to the
        source cannot reach the hash. Only sha256 runs beside the caller's next work, the calibration's load and hash.
        """
        raw = weight.detach().contiguous().view(torch.uint8).to('cpu', copy=True).numpy()
        if self._hasher is None:
            self._hasher = ThreadPoolExecutor(max_workers=1, thread_name_prefix='st-pack-digest')
        return WeightDigest(weight, self._hasher.submit(lambda: hashlib.sha256(raw).hexdigest()))

    def _calibration_sha(self, name, k, smooth, digest):
        """sha256 of `name`'s validated, smoothed [k, k] Hessian, or None when the store has no blob. Loaded, checked
        and hashed once for all the lanes of `digest`'s weight: a GLM-5.3 rank used to push 7.2 GiB of Hessians
        through the load, the finite check, the smoothing and sha256 twice, once per lane."""
        path = self.calibration_path(name)
        try:
            stat = path.stat()
        except FileNotFoundError:
            return None
        if not self.fits_weights(path):     # summed under other weights: round to nearest rather than compensate
            self.foreign.add(name)          # for a distribution this boot does not have (`fits_weights`)
            return None
        key = (str(path), stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns, int(k), self._smooth_sha(smooth))
        sha = digest.calibrations.get(key)
        if sha is not None:
            self.read_files.add(path)
            self.stats['calibration_digest_reused'] += 1
            return sha
        hessian = self._hessian(name, k, smooth)
        if hessian is None:
            return None
        sha = hashlib.sha256(hessian.contiguous().numpy()).hexdigest()
        digest.calibrations[key] = sha
        return sha

    def _calibrated_hessian(self, name, k, smooth, sha):
        """The Hessian a build packs from, read again: it must still be the bytes its identity named."""
        hessian = self._hessian(name, k, smooth)
        if hessian is None or hashlib.sha256(hessian.contiguous().numpy()).hexdigest() != sha:
            raise ValueError(f'calibration changed while packing: {self.calibration_path(name)}')
        return hessian

    def amax(self, name, width=None):
        """The channel peaks [k] of `name`'s calibrated input (unsmoothed domain), or None when this boot may not
        fold them.

        The peaks are a sum over one boot's activations exactly like the Hessian beside them, so the blob answers
        to the same provenance rule (`fits_weights`): `calibration_path` keys a blob by the weight's NAME alone,
        and a boot that changed checkpoints would otherwise fold that other checkpoint's channel distribution
        into its norms and into every reader the fold rescales -- the silent class behind the 2026-09-17
        indexer-gate corruption, where a fold the served weights never measured reshaped the indexer's gates.
        A blob written before the field makes no claim and is taken, like `fits_weights`; so is one recorded
        with no peaks at all (a Hessian-only blob: the peaks were never summed). A sum that never happened
        (`ntok` 0), a non-floating or non-finite vector, or a width that does not match the weight about to
        read it are refused too: the fold divides a norm channel by channel, so a half-wrong peaks vector
        would scale only half the channels and look like a working boot.
        """
        path = self.calibration_path(name)
        if not path.is_file():
            return None
        if not self.fits_weights(path):
            self.foreign.add(name)
            self.stats['amax_foreign'] += 1
            return None
        self.read_files.add(path)
        blob = torch.load(path, map_location='cpu', mmap=True, weights_only=True)
        amax = blob.get('amax')
        if (amax is None or int(blob.get('ntok', 0)) <= 0 or not amax.is_floating_point()
                or not torch.isfinite(amax).all() or (width is not None and amax.numel() != width)):
            self.stats['amax_refused'] += 1
            return None
        return amax.float()

    @staticmethod
    def _smooth_sha(smooth):
        return 'none' if smooth is None else hashlib.sha256(smooth.detach().float().cpu().contiguous().numpy()).hexdigest()

    def _factor(self, name, hessian, smooth_sha, device):
        """The column order and inverse factor of `name`'s Hessian (packing.gptq_factor), computed once for the
        weight and reused by its other lane: the W4 pack and the FP8 pack walk the same columns of the same H, and
        the factorisation is a fifth of a tile-wide pack. One entry -- the next weight replaces it, and a factor
        above FACTOR_BYTES (the drafter's fc is 1.6 GiB) is used and dropped, because a boot packs beside a full
        arena. `device`: where the fp64 work runs -- the weight's device for a tile-wide K, the CPU above it."""
        from engine.kernels.dense import GPTQ_ACT_ORDER
        from engine.kernels.dense.packing import gptq_factor
        damping = self.gptq_damping.get(name, 0.01)
        identity = (name, smooth_sha, tuple(hessian.shape), str(device), damping)
        if self._factor_entry is not None and self._factor_entry[0] == identity:
            self.stats['factor_reused'] += 1
            return self._factor_entry[1]
        self._factor_entry = None                                   # the previous weight's, freed before this one's
        factor = gptq_factor(hessian, percdamp=damping, act_order=GPTQ_ACT_ORDER, factor_device=device)
        self.stats['factor_built'] += 1
        if factor[1].numel() * factor[1].element_size() <= self.FACTOR_BYTES:
            self._factor_entry = (identity, factor)
        return factor

    def _tuning_identity(self, name):
        damping = self.gptq_damping.get(name, 0.01)
        return {} if damping == 0.01 else {'gptq_damping': damping}

    def pack_wide(self, weight, name, *, rank=None, smooth=None, digest=None):
        """The tiles of a weight wider than the decode kernel's K, from one GPTQ over the whole weight and its full
        calibration Hessian (kernels/dense.pack_w4_wide); cached as one blob under the wide identity. `digest`: this
        weight's `WeightDigest` when another lane already hashed it."""
        from engine.kernels.dense import W4Pack, pack_w4_wide
        rank = self.rank if rank is None else rank
        n, k = weight.shape
        if not self.calibration_path(name).is_file():
            raise ValueError(f"pack_wide needs the calibration of {name}")
        if digest is None:
            digest = self.weight_digest(weight)
        calibration = self._calibration_sha(name, k, smooth, digest)
        if calibration is None:
            raise ValueError(f"pack_wide needs the calibration of {name}")
        identity = dict(version=2, weight=digest.of(weight), shape=(n,k), name=name, wide=True,
                        calibration=calibration,
                        per_row=not name.startswith('DFlash2Qwen3ForCausalLM/'), algorithm=self.algorithm,
                        smooth=self._smooth_sha(smooth), **self._tuning_identity(name))
        key = hashlib.sha256(repr(identity).encode()).hexdigest()
        cache = self.root/'st-dense-packs'/(key+'.pt')
        if cache.is_file():
            self.read_files.add(cache)
            blob = torch.load(cache, map_location='cpu', mmap=True, weights_only=True)
            if blob['identity'] != identity:
                raise ValueError(f'dense pack identity mismatch: {cache}')
            packs = []
            for d, s in zip(blob['data'], blob['scale']):
                tile = self.decode(dict(data=d, scale=s, rowscale=blob['rowscale']), weight.device, n, self.TILE)
                packs.append(W4Pack(tile.data, tile.scale, tile.rowscale, n, self.TILE, True))
            self.stats['cache'] += 1
        else:
            hessian = self._calibrated_hessian(name, k, smooth, calibration)
            packs = pack_w4_wide(weight, hessian, per_row=identity['per_row'],
                                 factor=self._factor(name, hessian, identity['smooth'], 'cpu'))
            self.stats['built'] += 1
            cache.parent.mkdir(parents=True, exist_ok=True)
            temporary = cache.with_suffix(f'.{os.getpid()}.tmp')
            try:
                torch.save(dict(identity=identity, data=[p.data.cpu() for p in packs], scale=[p.scale.cpu() for p in packs],
                                rowscale=packs[0].rowscale.cpu()), temporary)
                os.replace(temporary,cache)
            finally:
                temporary.unlink(missing_ok=True)
        self.stats['gptq'] += 1
        return packs

    def pack(self, weight, name, *, rank=None, smooth=None, digest=None):
        from engine.kernels.dense import pack_w4
        rank = self.rank if rank is None else rank
        per_row = not name.startswith('DFlash2Qwen3ForCausalLM/')
        n, k = weight.shape
        if digest is None:
            digest = self.weight_digest(weight)
        calibration = self._calibration_sha(name, k, smooth, digest)      # None: no blob, the pack rounds to nearest
        weight_sha = digest.of(weight)
        calibration_sha = calibration if calibration is not None else 'rtn'
        identity = dict(version=2, weight=weight_sha, shape=(n,k), name=name,
                        calibration=calibration_sha, per_row=per_row, algorithm=self.algorithm,
                        smooth=self._smooth_sha(smooth), **self._tuning_identity(name))
        key = hashlib.sha256(repr(identity).encode()).hexdigest()
        cache = self.root/'st-dense-packs'/(key+'.pt')
        pack = None
        if cache.is_file():
            self.read_files.add(cache)
            blob = torch.load(cache, map_location='cpu', mmap=True, weights_only=True)
            if blob['identity'] != identity:
                raise ValueError(f'dense pack identity mismatch: {cache}')
            pack = self.decode(blob, weight.device, n, k)
            if calibration is not None:
                from dataclasses import replace
                pack = replace(pack, calibrated=True)
            self.stats['cache'] += 1
        else:
            hessian = None if calibration is None else self._calibrated_hessian(name, k, smooth, calibration)
            kind = 'gptq' if hessian is not None else 'rtn'

            def legacy_digests():
                yield 'sha256-'+weight_sha
                yield hashlib.md5(weight.detach().contiguous().view(torch.uint8).cpu().numpy()).hexdigest()
            # The original cache used both MD5 and SHA256 aliases.
            for alias in legacy_digests():
                if self._tuning_identity(name):
                    break            # legacy blobs do not attest a non-default inverse damping
                mode = 'row' if per_row else 'ten'
                legacy = self.root/'mkpacks'/f'rank{rank}'/(alias+f'-{n}x{k}-bfloat16-v4-{mode}-{kind}-lr0.pt')
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
                factor = (None if hessian is None else
                          self._factor(name, hessian, identity['smooth'], weight.device))
                pack = pack_w4(weight, hessian=hessian, per_row=per_row, factor=factor)
                self.stats['built'] += 1
            elif hessian is not None:
                from dataclasses import replace
                pack = replace(pack, calibrated=True)
            cache.parent.mkdir(parents=True, exist_ok=True)
            temporary = cache.with_suffix(f'.{os.getpid()}.tmp')
            try:
                torch.save(dict(identity=identity, data=pack.data.cpu(), scale=pack.scale.cpu(),
                                rowscale=pack.rowscale.cpu()), temporary)
                os.replace(temporary,cache)
            finally:
                temporary.unlink(missing_ok=True)
        self.stats['gptq' if calibration is not None else 'rtn'] += 1
        return pack

    def pack_fp8(self, weight, name, *, rank=None, smooth=None, digest=None):
        """The FP8 lane's (q e4m3, UE8M0 block scales) of a calibrated weight: GPTQ on the fp8 grid (packing.fp8_gptq),
        cached under the weight's, the Hessian's and the packer's identity; None when the store has no calibration."""
        from engine.kernels.dense.packing import fp8_gptq
        n, k = weight.shape
        if not self.calibration_path(name).is_file():
            return None
        if digest is None:
            digest = self.weight_digest(weight)
        calibration = self._calibration_sha(name, k, smooth, digest)
        if calibration is None:
            return None
        identity = dict(version=2, weight=digest.of(weight), shape=(n,k), name=name, kind='fp8',
                        calibration=calibration, algorithm=self.algorithm,
                        smooth=self._smooth_sha(smooth), **self._tuning_identity(name))
        key = hashlib.sha256(repr(identity).encode()).hexdigest()
        cache = self.root/'st-dense-packs'/(key+'.pt')
        if cache.is_file():
            self.read_files.add(cache)
            blob = torch.load(cache, map_location='cpu', mmap=True, weights_only=True)
            if blob['identity'] != identity:
                raise ValueError(f'dense pack identity mismatch: {cache}')
            q, scale = blob['q'].to(weight.device), blob['scale'].to(weight.device)
            self.stats['fp8_cache'] += 1
        else:
            hessian = self._calibrated_hessian(name, k, smooth, calibration)
            device = 'cpu' if k > self.TILE else weight.device
            q, scale = fp8_gptq(weight, hessian.to(weight.device),
                                factor=self._factor(name, hessian, identity['smooth'], device))
            self.stats['fp8_built'] += 1
            cache.parent.mkdir(parents=True, exist_ok=True)
            temporary = cache.with_suffix(f'.{os.getpid()}.tmp')
            try:
                torch.save(dict(identity=identity, q=q.cpu(), scale=scale.cpu()), temporary)
                os.replace(temporary,cache)
            finally:
                temporary.unlink(missing_ok=True)
        self.stats['fp8_gptq'] += 1
        return q, scale

    def release_pages(self):
        """All mmap readers have returned; return their clean UMA file cache, and the digest worker."""
        if self._hasher is not None:
            self._hasher.shutdown(wait=True)
            self._hasher = None
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
