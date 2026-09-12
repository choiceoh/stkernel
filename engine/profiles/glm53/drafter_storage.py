"""Resident DFlash weights, sized before boot and finalized before capture.

Checkpoint tensors are preparation inputs. The serving arena owns only the
selected packs, context projection and remaining BF16 readers. Each region is
bounded independently so a pack format change cannot overwrite its neighbour.
"""
from engine.kernels.dense import packed_nbytes

ALIGN = 256


def block_rows(F, max_seqs):
    if type(max_seqs) is not int or max_seqs <= 0:
        raise ValueError('drafter storage requires a positive sequence capacity')
    return max_seqs * (F.k + 1)


def needs_fp8(F, max_seqs, name):
    # Context ingestion includes long prefill; block proposals only see n*(K+1).
    return name == 'fc.weight' or max_seqs is None or block_rows(F, max_seqs) > 32


def retained_specs(F):
    from .drafter import specs
    retired = {'fc.weight'}
    for L in range(F.layers):
        retired.update(f'layers.{L}.{suffix}' for suffix in (
            'self_attn.q_proj.weight', 'self_attn.k_proj.weight', 'self_attn.v_proj.weight',
            'self_attn.o_proj.weight', 'mlp.gate_proj.weight', 'mlp.up_proj.weight',
            'mlp.down_proj.weight', 'attention_conv.kernel_projection.weight',
            'mlp_conv.kernel_projection.weight'))
    return [spec for spec in specs(F) if spec.name not in retired]


def layout(F, world, max_seqs):
    from .drafter import dense_shapes
    block_rows(F, max_seqs)
    if world <= 0 or any(n % world for n in (F.heads, F.kv_heads, F.inter)):
        raise ValueError('drafter storage requires evenly sharded TP dimensions')
    regions, end = {}, 0
    def add(name, size):
        nonlocal end
        start = (end + ALIGN - 1) // ALIGN * ALIGN
        regions[name] = (start, size)
        end = start + size
    for spec in retained_specs(F):
        add('source/' + spec.name, spec.nbytes())
    for name, (rows, cols) in dense_shapes(F, world).items():
        add('pack/' + name, packed_nbytes(rows, cols, prefill=needs_fp8(F, max_seqs, name)))
        # Smoothing factors are optional FP32 vectors; reserve their maximum
        # independent of which calibration files happened to exist at boot.
        add('smooth/' + name, cols * 4)
    add('context_kv', F.layers * 2 * (F.kv_heads // world) * F.head_dim * F.hidden * 2)
    return regions, (end + ALIGN - 1) // ALIGN * ALIGN


def nbytes(F, world, max_seqs):
    return layout(F, world, max_seqs)[1]


def compact(drafter, arena, max_seqs):
    """Copy the live readers into the declared region; retire all raw sources."""
    regions, size = layout(drafter.F, drafter.target.comm.world_size, max_seqs)
    storage = arena.carve(size, 'drafter resident weights')
    def region(name):
        start, size = regions[name]
        return storage[start:start + size]
    def copy(name, tensor):
        if tensor.device != storage.device or not tensor.is_contiguous():
            raise ValueError(f'drafter resident tensor must be contiguous on the arena device: {name}')
        count = tensor.numel() * tensor.element_size()
        target = region(name)
        if count > target.numel():
            raise ValueError(f'drafter resident tensor exceeds its declared region: {name}')
        result = target[:count].view(tensor.dtype).view(tensor.shape)
        result.copy_(tensor)
        return result
    kept = {spec.name: copy('source/' + spec.name, drafter.p[spec.name])
            for spec in retained_specs(drafter.F)}
    for name, layer in drafter.dense.items():
        if (layer.fp8 is not None) != needs_fp8(drafter.F, max_seqs, name):
            raise ValueError(f'drafter precision does not match its resident declaration: {name}')
        layer.consume_weight(region('pack/' + name))
        if layer.smooth is not None:
            layer.smooth = copy('smooth/' + name, layer.smooth)
    drafter.context_kv = copy('context_kv', drafter.context_kv)
    drafter.p = kept
    drafter.resident_bytes = size
