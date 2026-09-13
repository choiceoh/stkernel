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


def needs_w4(name, policy=None):
    return not (name == 'fc.weight' and policy and policy.fc_precision == 'fp8')


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


def layout(F, world, max_seqs, *, policy=None):
    from .drafter import dense_shapes
    if policy is not None and policy.fc_calibration == 'auto':
        raise ValueError('resolve automatic draft calibration before sizing its resident packs')
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
        add('pack/' + name, packed_nbytes(rows, cols, prefill=needs_fp8(F, max_seqs, name),
            decode_fp8=bool(policy and policy.separate_decode_fp8 and name == 'fc.weight'),
            decode_w4=needs_w4(name, policy)))
        # Smoothing factors are optional FP32 vectors; reserve their maximum
        # independent of which calibration files happened to exist at boot.
        add('smooth/' + name, cols * 4)
    add('context_kv', F.layers * 2 * (F.kv_heads // world) * F.head_dim * F.hidden * 2)
    add('context_norm', F.layers * F.head_dim * 2)
    add('fc_bias', F.hidden * 4)  # optional decode correction; one fixed FP32 vector
    return regions, (end + ALIGN - 1) // ALIGN * ALIGN


def nbytes(F, world, max_seqs, *, policy=None):
    return layout(F, world, max_seqs, policy=policy)[1]


def compact(drafter, arena, max_seqs, *, policy=None):
    """Copy the live readers into the declared region; retire all raw sources."""
    regions, size = layout(drafter.F, drafter.target.comm.world_size, max_seqs, policy=policy)
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
        if bool(layer.packs) != needs_w4(name, policy):
            raise ValueError(f'drafter W4 does not match its resident declaration: {name}')
        if (getattr(layer, 'decode_fp8', None) is not None) != bool(
                policy and policy.separate_decode_fp8 and name == 'fc.weight'):
            raise ValueError(f'drafter decode FP8 does not match its resident declaration: {name}')
        layer.consume_weight(region('pack/' + name))
        if layer.smooth is not None:
            layer.smooth = copy('smooth/' + name, layer.smooth)
    drafter.context_kv = copy('context_kv', drafter.context_kv)
    drafter.context_norm = copy('context_norm', drafter.context_norm)
    if getattr(drafter, 'fc_bias', None) is not None:
        drafter.fc_bias = copy('fc_bias', drafter.fc_bias)
    drafter.p = kept
    drafter.resident_bytes = size
