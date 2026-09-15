"""Which GPU architecture a native lane is compiled for -- read from the bound shape, not a knob.

D5 scopes this engine to four Sparks and forbids, by name, "다른 GPU 도 되게". That still
stands and this does not weaken it: **no other card becomes a place to measure**. What the
operator allowed on 2026-09-15 is narrower -- a card that is not a GB10 may CHECK. It may
compile the lanes and run them so a compile error, a shape bug or a correctness regression
is caught before a ticket takes the fleet, and every verdict it gives is `unmeasured`
(cells.admission), never `admitted`. The price D5 was written against -- generic
quant_config plumbing, dtype fallbacks, a dispatch that costs the Sparks memory or speed --
is not paid here: the fallbacks that make sm_120 compile are `#if __CUDA_ARCH__` branches
below the fleet's target, and the fleet's PTX is byte-identical with and without them
(probes/sm120_b_fragment_check.cu carries the numerical half of that claim).

There is no environment knob, and that is D11, not taste: inputs are facts, and which card
this is is already a fact the bound kernel shape declares (engine/base/kernel_shape.Device).
A knob would be a second source of truth for it, and a knob without an expiry is what D11
kills a boot over. So a check declares its device the way everything else here declares a
shape, and the lanes follow it.

Only the two lanes that have been compiled and checked for another target are listed here.
The rest stay pinned to the fleet's, because nothing has demonstrated them anywhere else --
`oneshot` cannot even be built off the fleet (it needs `<infiniband/verbs.h>` for the RoCE
rails), and a lane that merely compiles is not a lane that ran.
"""
from __future__ import annotations

FLEET = (12, 1)                  # GB10, sm_121a: the only architecture any measurement here was taken on

# capability -> (code target, virtual target). A capability absent from this table is
# refused outright: an untried card is not a check lane, it is an unknown.
TARGETS = {
    (12, 1): ("sm_121a", "compute_121a"),
    (12, 0): ("sm_120", "compute_120"),      # consumer Blackwell (RTX 5050): checks only
}


def target(capability) -> "tuple[str, str] | None":
    """(code, virtual) for that capability, or None when no lane has been built for it."""
    return TARGETS.get(tuple(capability))


def is_fleet(capability) -> bool:
    return tuple(capability) == FLEET


def name(capability) -> str:
    found = target(capability)
    return found[0] if found else f"sm_{capability[0]}{capability[1]}"


def gencode(capability) -> "list[str]":
    """The nvcc flags for that capability, as the lanes pass them to cpp_extension.load.

    They go into the native cache key at every call site, so two architectures never share a
    built module; nothing else is needed to keep them apart.
    """
    found = target(capability)
    if found is None:
        raise RuntimeError(
            f"no ST native lane is built for SM{capability[0]}{capability[1]}: the lanes are compiled for "
            f"{', '.join(code for code, _ in TARGETS.values())} and D5 keeps this engine to the hardware it "
            f"was measured on. Another card is its own build, its own measurements and its own cells.")
    code, virtual = found
    return ["-gencode", f"arch={virtual},code={code}"]
