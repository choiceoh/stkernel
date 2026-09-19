"""Operator-selected cuBLAS head/FC defaults, with declared resident ownership."""


def resident_bytes(F, D=None, policy=None):
    head_rows = (F.vocab_local+127)//128*128
    size = head_rows * (F.hidden//32)
    if D is not None:
        n, k = D.hidden, D.hidden*len(D.target_layers)
        scale = n*(k//32)
        size += scale  # prefill FC's direct MX scales
        if policy is not None and policy.fc_precision == 'fp8':
            size += n*(k+5*384) + scale  # five-way decode weight, padded physical pitch, and scales
            if policy.separate_decode_fp8:
                size += scale  # independent calibrated decode reader's direct scales
    return size


def prepare(net, drafter, arena, policy):
    """Called after both original packs reach their final arena addresses."""
    readers = {'head': (net.dense['head'], False)}
    if drafter is not None:
        fc = drafter.dense['fc.weight']
        separate = fc.decode_fp8 is not None
        split = fc.decode_precision == 'fp8'
        readers['fc_prefill'] = (fc.fp8, split and not separate)
        if separate:
            readers['fc_decode'] = (fc.decode_fp8, True)
    start = arena.used
    for name, (layer, split) in readers.items():
        if layer is None:
            raise RuntimeError(f'missing FP8 reader for cuBLAS replacement: {name}')
        from engine.kernels.dense.cublaslt_serving import weight_nbytes
        count = weight_nbytes(*layer.weight[0].shape, split_decode=split)
        layer.prepare_cublas(split_decode=split, storage=arena.carve(count, 'cublas/'+name))
    net.cublas_readers = {name: layer for name, (layer, _) in readers.items()}
    net.cublas_head_producer_required = drafter is not None
    expected = resident_bytes(net.F, None if drafter is None else drafter.F, policy)
    actual = sum(layer.cublas.resident_bytes for layer in net.cublas_readers.values())
    if actual != expected:
        raise RuntimeError(f'cuBLAS resident declaration mismatch: {actual} != {expected}')
    # The head's decode rows take the W8A16 lane (net.py); the reader keeps larger batches. Held here, on the head's
    # own weight, before anything is captured -- and at MAX_SEQS 2 this is the only call the reader's paths get.
    head_rows = net.dense['head'].qualify_decode_rows(producer=drafter is not None)
    return dict(resident_bytes=actual, arena_growth=arena.used-start, readers=sorted(readers), head_rows=head_rows)


def execution_report(net):
    if not net.cublas_readers or 'head' not in net.cublas_readers:
        raise RuntimeError('default cuBLAS readers were not prepared')
    result = {}
    for name, layer in net.cublas_readers.items():
        reader = layer.cublas
        required = {'split_decode'} if name == 'fc_decode' else {'direct'}
        if name == 'head' and getattr(net, 'cublas_head_producer_required', False):
            required.add('producer_mx')
        if reader is not None and reader.split_decode:
            required.add('split_decode')
            required.add('split_decode_norm')
        if reader is None or not required.issubset(reader.executed):
            raise RuntimeError(f'default cuBLAS reader was not executed: {name}, expected {sorted(required)}')
        result[name] = reader.report()
        if getattr(layer, 'decode_rows', False) == 'w8a16':
            # the lane the head declares for decode rows must have served them, not only been qualified
            if not layer.decode_rows_executed:
                raise RuntimeError(f'declared W8A16 decode-row lane was not executed: {name}')
            result[name] = dict(result[name], decode_rows='w8a16')
    return result
