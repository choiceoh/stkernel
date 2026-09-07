# GLM unused CUDA graph memory profiling

Runtime source: `b945723bcedde7621773ac78f81e2e6b638323f7`.

PRIME is excluded. Timed order: BASE1, FAST1, FAST2, BASE2. Same source and image; only VLLM_GLM53_SKIP_UNUSED_GRAPH_PROFILE changes. Both arms retain the model/MM memory profile, actual graph capture and kernel warmup; both use early CPU MM warmup. First text/image/video requests precede the canonical 2K/32K Korean onepass.

| Arm | Health s | Model s | Encoder s | Dry capture s | Memory profile s | Real capture s | Compile/warmup s |
|---|---:|---:|---:|---:|---:|---:|---:|
| GRAPHMEMPRIME | 493 | 249.8 | 78.7 | 0 | 103.2 | 2.0 | 52.9 |
| GRAPHMEMBASE1 | 205 | 74.7 | 21.9 | 2.6 | 36.5 | 1.7 | 10.1 |
| GRAPHMEMFAST1 | 212 | 75.8 | 21.8 | 0 | 34.0 | 2.0 | 10.6 |
| GRAPHMEMFAST2 | 202 | 74.2 | 22.1 | 0 | 33.7 | 2.1 | 10.2 |
| GRAPHMEMBASE2 | 205 | 75.9 | 22.7 | 2.4 | 37.0 | 1.7 | 10.1 |

| Metric | BASE mean | FAST mean | Difference |
|---|---:|---:|---:|
| health_wall_s | 205.000 | 207.000 | +2.000 |
| load-model | 75.300 | 75.000 | -0.300 |
| encoder-profile | 22.300 | 21.950 | -0.350 |
| profile-run | 33.450 | 33.000 | -0.450 |
| cudagraph-memory-profile | 2.500 | 0.000 | -2.500 |
| profile/determine-memory | 36.750 | 33.850 | -2.900 |
| cudagraph-capture | 1.700 | 2.050 | +0.350 |
| compile+warmup | 10.100 | 10.400 | +0.300 |
| first_text_ttft_s | 1.227 | 0.780 | -0.447 |
| first_image_ttft_s | 0.545 | 0.498 | -0.048 |
| first_video_ttft_s | 0.430 | 0.434 | +0.004 |

Phase timers are nested and include host work and existing synchronizations. First-request samples do not establish general throughput/quality. POST completions and response evidence are retained to detect interference.

- GRAPHMEMPRIME: quality={'ok': 6, 'total': 6}; corruption={'dirty': 0, 'n': 4, 'kinds': {'replacement': 0, 'lone_jamo': 0, 'cjk_mixed': 0, 'control': 0}, 'hits': []}; posts={'loopback': 7, 'non_loopback': 0, 'before_health': 0}
- GRAPHMEMBASE1: quality={'ok': 6, 'total': 6}; corruption={'dirty': 0, 'n': 4, 'kinds': {'replacement': 0, 'lone_jamo': 0, 'cjk_mixed': 0, 'control': 0}, 'hits': []}; posts={'loopback': 7, 'non_loopback': 0, 'before_health': 0}
- GRAPHMEMFAST1: quality={'ok': 6, 'total': 6}; corruption={'dirty': 0, 'n': 4, 'kinds': {'replacement': 0, 'lone_jamo': 0, 'cjk_mixed': 0, 'control': 0}, 'hits': []}; posts={'loopback': 7, 'non_loopback': 0, 'before_health': 0}
- GRAPHMEMFAST2: quality={'ok': 6, 'total': 6}; corruption={'dirty': 0, 'n': 4, 'kinds': {'replacement': 0, 'lone_jamo': 0, 'cjk_mixed': 0, 'control': 0}, 'hits': []}; posts={'loopback': 7, 'non_loopback': 2, 'before_health': 0}
- GRAPHMEMBASE2: quality={'ok': 6, 'total': 6}; corruption={'dirty': 0, 'n': 4, 'kinds': {'replacement': 0, 'lone_jamo': 0, 'cjk_mixed': 0, 'control': 0}, 'hits': []}; posts={'loopback': 7, 'non_loopback': 0, 'before_health': 0}
