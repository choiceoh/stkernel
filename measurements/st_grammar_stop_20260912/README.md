# Speculative stop-token grammar repair

A synthetic `get_weather` request crashed both the Red Hat A/B runtime
(`dfd28cb03257`) and the configured Red Hat release (`abceb6a0`). DFlash2
proposed a grammar stop token. `Matcher.fill` accepted it, then called
`fill_next_token_bitmask` for the following speculative position. xgrammar
raises after termination, aborting the TP4 engine.

The draft walk now ends immediately after accepting a stop token. Its
position retains the previously filled mask, later positions are dead, and
the existing `finally` rolls back every accepted draft, including the stop
token. Committed tokens still advance the matcher in `advance`.

The new tests run against real xgrammar in the pinned ST runtime without
a GPU. They cover an EOS draft both with and without trailing drafts, and
a dormant reasoning matcher whose drafts cross the answer boundary and EOS.
Both new tests fail on the original source with the production exception
([before](before.log)); all 28 grammar tests pass with the fix
([after](after.log)). The tested baseline grammar file is identical to
`origin/main` at `fb6aecdc`.

This is a shared speculative grammar bug. It does not establish a Red Hat
versus NVIDIA checkpoint quality difference.

Live recovery retained Red Hat, KV7, DFlash2 and the qualified tile32 prefill
in a separate `prod-abceb6a0-grammar-9391` release. All four containers were
running. The [HTTP receipt](live-recovery.json) records a successful
`get_weather(city="Seoul")` tool response followed by a normal `서울` answer.
The deployment environment was updated to this repaired release only after
these requests passed. The engine source manifest is
`b627103b8c8e546fd9cfefebf89a82c2a4b1f29705254e06f030ec21ad558785`.
