# CPU23: completed head no-device compile receipt

This archive was collected after the normal `fleet --cpu` command returned0. Frozen source `cbf1c7916247f946167c15ff74d92261588e8cea`, the exact bounded runc/network-none/4GiB/2CPU launcher, immutable image, capsule manifest, actual contracts/mounted sources and original evidence file hashes were checked before and after transfer. The head source has full independent history and no alternates. The result must contain 165 CPU tests with zero failures/errors/skips and CUDA uninitialized.

The original tar retains every PTX, cubin, resource log and result byte. Its safe regular-file inventory must exactly equal both head snapshots, and the frozen pure artifact validator checks all descriptor hashes, resource strings and the complete compiled artifact set. Source bytes also match local immutable git objects. This is compilation/CPU proof only; it does not establish serving canary, sanitizer, throughput or default adoption. The CPU uses the isolated13.0.3 bindings capsule; production13.3.1 binary identity is not asserted.

No compilation, GPU execution, HTTP request, queue change or service change is performed by collection. Logs and frozen Python snapshots use deterministic gzip; original/stored hashes are preserved.

Against CPU21b, all30 PTX files, all6 CuTe cubins (including TP stock and Q0 candidate), and all6 resource logs are byte-identical. The24 Triton preparation cubins have different full hashes; the cause is not determined. Thus all30 cubins are not claimed identical. The five listed device-kernel module hashes agree; the dispatcher source hash differs. The actual compiled CuTe binaries remain byte-identical. Exact file hashes are in `comparison-to-cpu21b.json`.
