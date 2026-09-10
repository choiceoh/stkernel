# Full GPU v5 preparation

The reviewed preparer binds the actual successful CPU16 job and committed
receipt to a new immutable GPU source. It requires current approved fleet
content, exact source/runtime identity and exclusive source/bundle/job paths.
The command is normal `fleet.sh run --gpu eplocal0908v5 45`, with full
MoE/remap numerical, stream and sanitizer cells through the existing offline
runner. CPU16's 134 tests, 13 mounted/27 contract sources and all 24 remap
variants are checked against the actual completed job; no fixed historical
test count or partial CPU15 receipt can substitute.

Root and independent read-only review passed. Preparation itself is not a
submission or GPU result. Subsequent admission receipts record the actual
queue state. The existing runner retains GO-time disk/memory, incoming
identity, pause and exact restore guards; no queue bypass or serving reclaim
is added.
