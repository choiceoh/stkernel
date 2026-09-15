measurements/st_c1_input_pack_grid_20260915/gpu.jsonl: 18 exact groups; GPU component intervals only.

| Component | Scope | Cache | Warps/CTA | Control us | Candidate us | Change |
|---|---|---|---:|---:|---:|---:|
| pack K=1536 | single x1 | warm | 4 | 1.125 | 1.047 | -6.91% |
| pack K=1536 | single x1 | warm | 2 | 1.099 | 1.047 | -4.72% |
| pack K=1536 | single x1 | warm | 1 | 1.100 | 1.052 | -4.37% |
| pack K=1536 | single x1 | evicted | 4 | 4.417 | 4.421 | +0.09% |
| pack K=1536 | single x1 | evicted | 2 | 4.588 | 4.338 | -5.45% |
| pack K=1536 | single x1 | evicted | 1 | 4.387 | 4.373 | -0.31% |
| pack K=2048 | single x1 | warm | 4 | 1.106 | 1.057 | -4.40% |
| pack K=2048 | single x1 | warm | 2 | 1.094 | 1.050 | -4.03% |
| pack K=2048 | single x1 | warm | 1 | 1.100 | 1.081 | -1.75% |
| pack K=2048 | single x1 | evicted | 4 | 5.677 | 5.698 | +0.37% |
| pack K=2048 | single x1 | evicted | 2 | 5.735 | 5.726 | -0.16% |
| pack K=2048 | single x1 | evicted | 1 | 5.761 | 5.758 | -0.05% |
| pack K=3072 | single x1 | warm | 4 | 1.098 | 1.047 | -4.60% |
| pack K=3072 | single x1 | warm | 2 | 1.087 | 1.079 | -0.69% |
| pack K=3072 | single x1 | warm | 1 | 1.091 | 1.120 | +2.67% |
| pack K=3072 | single x1 | evicted | 4 | 5.748 | 5.764 | +0.29% |
| pack K=3072 | single x1 | evicted | 2 | 5.758 | 6.076 | +5.52% |
| pack K=3072 | single x1 | evicted | 1 | 5.770 | 5.774 | +0.07% |
| pack K=4096 | single x1 | warm | 4 | 1.108 | 1.104 | -0.36% |
| pack K=4096 | single x1 | warm | 2 | 1.098 | 1.112 | +1.23% |
| pack K=4096 | single x1 | warm | 1 | 1.095 | 1.186 | +8.26% |
| pack K=4096 | single x1 | evicted | 4 | 5.599 | 5.422 | -3.15% |
| pack K=4096 | single x1 | evicted | 2 | 5.559 | 5.365 | -3.49% |
| pack K=4096 | single x1 | evicted | 1 | 5.619 | 5.518 | -1.80% |
| kda.in_proj | single x1 | warm | 4 | 25.862 | 24.158 | -6.59% |
| kda.in_proj | single x1 | evicted | 4 | 127.694 | 126.160 | -1.20% |
| kda.in_proj | single x1 | warm | 2 | 25.789 | 23.754 | -7.89% |
| kda.in_proj | single x1 | evicted | 2 | 126.734 | 127.575 | +0.66% |
| kda.in_proj | single x1 | warm | 1 | 23.537 | 26.002 | +10.47% |
| kda.in_proj | single x1 | evicted | 1 | 128.760 | 127.071 | -1.31% |
| kda.in_proj | chain x8 | warm | 4 | 510.921 | 512.285 | +0.27% |
| kda.in_proj | chain x8 | evicted | 4 | 579.772 | 579.202 | -0.10% |
| kda.in_proj | chain x8 | warm | 2 | 517.039 | 513.492 | -0.69% |
| kda.in_proj | chain x8 | evicted | 2 | 580.172 | 580.049 | -0.02% |
| kda.in_proj | chain x8 | warm | 1 | 511.836 | 512.535 | +0.14% |
| kda.in_proj | chain x8 | evicted | 1 | 579.313 | 578.730 | -0.10% |
| kda.o_proj | single x1 | warm | 4 | 12.967 | 12.996 | +0.23% |
| kda.o_proj | single x1 | evicted | 4 | 51.547 | 51.876 | +0.64% |
| kda.o_proj | single x1 | warm | 2 | 12.511 | 13.008 | +3.97% |
| kda.o_proj | single x1 | evicted | 2 | 53.955 | 52.342 | -2.99% |
| kda.o_proj | single x1 | warm | 1 | 12.911 | 12.969 | +0.45% |
| kda.o_proj | single x1 | evicted | 1 | 50.866 | 51.498 | +1.24% |
| kda.o_proj | chain x8 | warm | 4 | 191.219 | 188.378 | -1.49% |
| kda.o_proj | chain x8 | evicted | 4 | 257.294 | 258.490 | +0.46% |
| kda.o_proj | chain x8 | warm | 2 | 187.288 | 187.503 | +0.11% |
| kda.o_proj | chain x8 | evicted | 2 | 257.983 | 257.681 | -0.12% |
| kda.o_proj | chain x8 | warm | 1 | 187.326 | 188.008 | +0.36% |
| kda.o_proj | chain x8 | evicted | 1 | 258.121 | 258.867 | +0.29% |
| mla.o_proj | single x1 | warm | 4 | 22.017 | 22.089 | +0.33% |
| mla.o_proj | single x1 | evicted | 4 | 88.433 | 89.097 | +0.75% |
| mla.o_proj | single x1 | warm | 2 | 22.100 | 22.530 | +1.94% |
| mla.o_proj | single x1 | evicted | 2 | 88.771 | 87.537 | -1.39% |
| mla.o_proj | single x1 | warm | 1 | 22.511 | 21.628 | -3.92% |
| mla.o_proj | single x1 | evicted | 1 | 87.003 | 91.115 | +4.73% |
| mla.o_proj | chain x4 | warm | 4 | 177.583 | 176.549 | -0.58% |
| mla.o_proj | chain x4 | evicted | 4 | 248.812 | 247.931 | -0.35% |
| mla.o_proj | chain x4 | warm | 2 | 178.755 | 179.101 | +0.19% |
| mla.o_proj | chain x4 | evicted | 2 | 249.796 | 248.776 | -0.41% |
| mla.o_proj | chain x4 | warm | 1 | 179.604 | 181.574 | +1.10% |
| mla.o_proj | chain x4 | evicted | 1 | 249.995 | 248.612 | -0.55% |
| mla.query | single x1 | warm | 4 | 13.999 | 14.099 | +0.71% |
| mla.query | single x1 | evicted | 4 | 70.888 | 69.172 | -2.42% |
| mla.query | single x1 | warm | 2 | 13.929 | 14.066 | +0.98% |
| mla.query | single x1 | evicted | 2 | 69.213 | 70.711 | +2.16% |
| mla.query | single x1 | warm | 1 | 13.877 | 14.047 | +1.23% |
| mla.query | single x1 | evicted | 1 | 69.370 | 69.729 | +0.52% |
| mla.query | chain x4 | warm | 4 | 103.203 | 110.079 | +6.66% |
| mla.query | chain x4 | evicted | 4 | 200.174 | 204.038 | +1.93% |
| mla.query | chain x4 | warm | 2 | 106.489 | 103.726 | -2.59% |
| mla.query | chain x4 | evicted | 2 | 198.736 | 204.853 | +3.08% |
| mla.query | chain x4 | warm | 1 | 106.627 | 103.326 | -3.10% |
| mla.query | chain x4 | evicted | 1 | 202.647 | 199.659 | -1.47% |
| mlp.gate_up | single x1 | warm | 4 | 26.385 | 26.512 | +0.48% |
| mlp.gate_up | single x1 | evicted | 4 | 125.173 | 123.933 | -0.99% |
| mlp.gate_up | single x1 | warm | 2 | 26.732 | 26.491 | -0.90% |
| mlp.gate_up | single x1 | evicted | 2 | 125.111 | 126.184 | +0.86% |
| mlp.gate_up | single x1 | warm | 1 | 26.275 | 26.519 | +0.93% |
| mlp.gate_up | single x1 | evicted | 1 | 123.443 | 124.389 | +0.77% |
| mlp.gate_up | chain x3 | warm | 4 | 184.503 | 184.082 | -0.23% |
| mlp.gate_up | chain x3 | evicted | 4 | 257.689 | 258.150 | +0.18% |
| mlp.gate_up | chain x3 | warm | 2 | 187.216 | 185.065 | -1.15% |
| mlp.gate_up | chain x3 | evicted | 2 | 259.046 | 258.785 | -0.10% |
| mlp.gate_up | chain x3 | warm | 1 | 183.762 | 183.967 | +0.11% |
| mlp.gate_up | chain x3 | evicted | 1 | 257.993 | 261.079 | +1.20% |
| mlp.down | single x1 | warm | 4 | 16.001 | 16.461 | +2.88% |
| mlp.down | single x1 | evicted | 4 | 69.676 | 71.022 | +1.93% |
| mlp.down | single x1 | warm | 2 | 15.991 | 16.420 | +2.68% |
| mlp.down | single x1 | evicted | 2 | 69.523 | 68.199 | -1.90% |
| mlp.down | single x1 | warm | 1 | 16.279 | 16.371 | +0.56% |
| mlp.down | single x1 | evicted | 1 | 71.239 | 69.274 | -2.76% |
| mlp.down | chain x3 | warm | 4 | 54.339 | 54.602 | +0.49% |
| mlp.down | chain x3 | evicted | 4 | 177.302 | 179.157 | +1.05% |
| mlp.down | chain x3 | warm | 2 | 53.631 | 55.991 | +4.40% |
| mlp.down | chain x3 | evicted | 2 | 175.479 | 172.414 | -1.75% |
| mlp.down | chain x3 | warm | 1 | 54.441 | 55.475 | +1.90% |
| mlp.down | chain x3 | evicted | 1 | 179.049 | 181.894 | +1.59% |
