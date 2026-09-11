# qwen38_qsa — the QSA split-K cap for GB10

`ops/qsa.py` carries upstream's split-K profile for the model's attention,
tuned on GB300. GB10 has 48 SMs against GB300's far larger count, so a split-K
chosen for that machine oversubscribes this one.

`DENEB_QSA_MAX_SPLITS` caps it. The overlay mounts unconditionally: with the
env unset its added branch is dead code identical to upstream, so the mount
costs nothing and the knob is available without a redeploy.

Override; carries the image's preimage SHA.
