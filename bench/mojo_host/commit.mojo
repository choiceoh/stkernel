# SPDX-License-Identifier: Apache-2.0
# CPU experiment only. Nothing in the serving engine imports this module.
from std.os import abort
from std.python import PythonObject
from std.python.bindings import PythonModuleBuilder


@export
def PyInit_st_mojo_host() abi("C") -> PythonObject:
    try:
        var module = PythonModuleBuilder("st_mojo_host")
        module.def_function[apply_outcome]("apply_outcome")
        return module.finalize()
    except error:
        abort(String(error))


def apply_outcome(
    e: PythonObject,
    seqs: PythonObject,
    result: PythonObject,
    finished: PythonObject,
    step_tokens: PythonObject,
) raises:
    """Commit one C=1..4 host outcome; all Python/native conversions are real.

    Inputs are the built-in dicts/lists and bounded integers from decode readback.
    Counts and liveness are cached in native storage, but Python owns token history.
    Rank agreement and publishing the outcome remain in the Python caller.
    """
    var rows = len(seqs)
    if rows < 1 or rows > 4:
        raise Error("Mojo host experiment requires 1..4 rows")
    var tokens = e.tokens
    var contexts = e.ctx
    var staged = e.staged
    var histogram = e.accepted_per_step
    var k = Int(py=e.drafter.k)
    var limit = Int(py=step_tokens)
    var block = Int(py=e.F.block)
    var counts = result["count"]
    var dones = result["done"]
    var before = result["before"]
    var accepted = result["accepted"]
    var additions = result["tokens"]
    var count_cache = Array[Int, 4](fill=0)
    var before_cache = Array[Int, 4](fill=0)
    var live_cache = Array[Bool, 4](fill=False)
    var live = 0
    var progress = False
    for i in range(rows):
        var seq = seqs[i]
        if seq in tokens:
            live += 1
            live_cache[i] = True
            var count = Int(py=counts[i])
            count_cache[i] = count
            progress = progress or count > 0 or Bool(py=dones[i])
    if live > 0 and not progress:
        raise Error("bounded decode made no progress")
    # As in BurstDecode, reject the whole row set before any host state changes.
    for i in range(rows):
        if live_cache[i]:
            var count = count_cache[i]
            if count < 0 or count > limit:
                raise Error("bounded decode readback lost row/context order")
            var old_context = Int(py=before[i])
            before_cache[i] = old_context
            if Int(py=contexts[seqs[i]]) != old_context:
                raise Error("bounded decode readback lost row/context order")
    for i in range(rows):
        var seq = seqs[i]
        finished[i] = Bool(py=finished[i]) or Bool(py=dones[i])
        if not live_cache[i]:
            continue
        var count = count_cache[i]
        if count > 0:
            _ = tokens[seq].extend(additions[i][:count])
            var context = Int(py=contexts[seq]) + count
            contexts[seq] = context
            var accept = Int(py=accepted[i])
            # Keep per-row counter updates and Python's duplicate-row behavior.
            e.accepted_total = Int(py=e.accepted_total) + accept
            e.drafted_total = Int(py=e.drafted_total) + k
            histogram[accept] = Int(py=histogram[accept]) + 1
            var boundary = context // block * block
            if boundary > before_cache[i]:
                staged[seq] = boundary
