# CPU22: mock contract failure after lowering

Frozen source `2d6b5d6168a4992645fb4daa60cbf27a2e445262` ran through the normal head `fleet --cpu` command and returned1. Its165 focused CPU tests had0 failures,1 error and0 skips: the old fake owner test raised `NameError: _TP_SF6_Q0_ENABLED is not defined`. The original result remains FAIL at phase `cpu-contracts`. The final CUDA initialization assertion and final binding-runtime recheck were not reached; neither status is claimed as passed. GPU22 was not submitted.

Original30 PTX and30 cubin files plus resource logs are preserved in the tar. The complete artifact descriptors/hashes/resource strings were checked with the frozen pure validator, and contracts/mounted-source hashes agree with both the completed head source and immutable local git objects. The original no-device launcher, image and capsule identity and all source/file hashes were checked around transfer. These observations do not convert the failed job to accepted CPU proof, GPU numerical proof or throughput evidence.

Only read-only collection and this private /tmp archive were performed. Logs and frozen Python snapshots use deterministic gzip. All original/stored hashes remain in the manifests; no recompile, test rerun, service change, HTTP request or queue submission occurred.
