# Native C=1 MHC timing

Negative time change means faster. These are local native intervals; TP4 transport and serving are outside their scope.

| Input | Interval | Cache | Dynamic us | Static us | Time change | Capture range |
| --- | --- | --- | ---: | ---: | ---: | ---: |
| ordinary AR | chain (89) | evicted | 1324.897 | 1213.152 | -8.43% | -9.23% to -7.63% |
| ordinary AR | chain (89) | warm | 1294.155 | 1189.160 | -8.11% | -8.53% to -7.69% |
| ordinary AR | single (1) | evicted | 31.543 | 29.891 | -5.24% | -5.36% to -5.12% |
| ordinary AR | single (1) | warm | 15.082 | 13.449 | -10.83% | -10.99% to -10.67% |
| local packets | chain (89) | evicted | 1389.606 | 1267.212 | -8.81% | -8.90% to -8.72% |
| local packets | chain (89) | warm | 1368.257 | 1248.200 | -8.77% | -8.87% to -8.68% |
| local packets | single (1) | evicted | 33.278 | 32.195 | -3.25% | -3.51% to -3.00% |
| local packets | single (1) | warm | 15.810 | 14.127 | -10.65% | -10.74% to -10.55% |
