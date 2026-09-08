# GLM unchanged overlay deployment

Runtime source: `f1814b2e676d12d0ceac8cd6843934e8da8b7fdb`.

PRIME is excluded. Timed order: BASE1, FAST1, FAST2, BASE2. Same source and image; only DEPLOY_PRESERVE_IDENTICAL changes during each identical-source publication replay. PRIME uses the official deployer with current-main admission; timed arms require identical manifest/source hashes on all four nodes before reproducing install/scp versus the production rsync helper. No revision is deployed by the timed replay. PREFILL_WARMUP=0 disables the separate background prefill benchmark in every arm. The graph-profile optimization remains disabled in every arm. Both arms retain all model/MM profiles, actual graph capture and kernel warmup; both use early CPU MM warmup. Health wall starts after deployment; source publication duration is reported separately (timed arms exclude repeated CPU admission checks from the official deployer). First text/image/video requests precede the canonical 2K/32K Korean onepass.

| Arm | Health s | Model s | Encoder s | Dry capture s | Memory profile s | Real capture s | Compile/warmup s |
|---|---:|---:|---:|---:|---:|---:|---:|
| DEPLOYCACHEPRIME | 387 | 252.5 | 22.2 | 2.8 | 42.8 | 1.7 | 10.8 |
| DEPLOYCACHEBASE1 | 341 | 132.1 | 97.1 | 2.7 | 112.5 | 1.7 | 10.4 |
| DEPLOYCACHEFAST1 | 211 | 79.2 | 21.5 | 2.9 | 36.1 | 1.7 | 9.8 |
| DEPLOYCACHEFAST2 | 215 | 79.4 | 21.4 | 2.6 | 36.2 | 1.8 | 10.3 |
| DEPLOYCACHEBASE2 | 327 | 137.4 | 80.8 | 2.7 | 95.2 | 1.7 | 10.2 |

| Metric | BASE mean | FAST mean | Difference |
|---|---:|---:|---:|
| health_wall_s | 334.000 | 213.000 | -121.000 |
| load-model | 134.750 | 79.300 | -55.450 |
| encoder-profile | 88.950 | 21.450 | -67.500 |
| profile-run | 100.450 | 32.650 | -67.800 |
| cudagraph-memory-profile | 2.700 | 2.750 | +0.050 |
| profile/determine-memory | 103.850 | 36.150 | -67.700 |
| cudagraph-capture | 1.700 | 1.750 | +0.050 |
| compile+warmup | 10.300 | 10.050 | -0.250 |
| first_text_ttft_s | 1.184 | 1.199 | +0.015 |
| first_image_ttft_s | 0.590 | 0.585 | -0.005 |
| first_video_ttft_s | 0.427 | 0.425 | -0.003 |

Phase timers are nested and include host work and existing synchronizations. First-request samples do not establish general throughput/quality. POST completions and response evidence are retained to detect interference.

- DEPLOYCACHEPRIME: quality={'ok': 6, 'total': 6}; corruption={'dirty': 0, 'n': 4, 'kinds': {'replacement': 0, 'lone_jamo': 0, 'cjk_mixed': 0, 'control': 0}, 'hits': []}; posts={'loopback': 7, 'non_loopback': 0, 'before_health': 0}
- DEPLOYCACHEBASE1: quality={'ok': 6, 'total': 6}; corruption={'dirty': 0, 'n': 4, 'kinds': {'replacement': 0, 'lone_jamo': 0, 'cjk_mixed': 0, 'control': 0}, 'hits': []}; posts={'loopback': 7, 'non_loopback': 0, 'before_health': 0}
- DEPLOYCACHEFAST1: quality={'ok': 6, 'total': 6}; corruption={'dirty': 0, 'n': 4, 'kinds': {'replacement': 0, 'lone_jamo': 0, 'cjk_mixed': 0, 'control': 0}, 'hits': []}; posts={'loopback': 7, 'non_loopback': 0, 'before_health': 0}
- DEPLOYCACHEFAST2: quality={'ok': 6, 'total': 6}; corruption={'dirty': 0, 'n': 4, 'kinds': {'replacement': 0, 'lone_jamo': 0, 'cjk_mixed': 0, 'control': 0}, 'hits': []}; posts={'loopback': 7, 'non_loopback': 0, 'before_health': 0}
- DEPLOYCACHEBASE2: quality={'ok': 6, 'total': 6}; corruption={'dirty': 0, 'n': 4, 'kinds': {'replacement': 0, 'lone_jamo': 0, 'cjk_mixed': 0, 'control': 0}, 'hits': []}; posts={'loopback': 7, 'non_loopback': 0, 'before_health': 0}

| Arm | Publish s | Node | Identical source files rewritten | Ninja build logs changed |
|---|---:|---|---:|---:|
| DEPLOYCACHEPRIME | 65 | srv1 | 0 | 0 |
| DEPLOYCACHEPRIME | 65 | srv2 | 0 | 0 |
| DEPLOYCACHEPRIME | 65 | srv3 | 0 | 0 |
| DEPLOYCACHEPRIME | 65 | srv4 | 0 | 0 |
| DEPLOYCACHEBASE1 | 4 | srv1 | 60 | 2 |
| DEPLOYCACHEBASE1 | 4 | srv2 | 60 | 2 |
| DEPLOYCACHEBASE1 | 4 | srv3 | 60 | 2 |
| DEPLOYCACHEBASE1 | 4 | srv4 | 60 | 2 |
| DEPLOYCACHEFAST1 | 4 | srv1 | 0 | 0 |
| DEPLOYCACHEFAST1 | 4 | srv2 | 0 | 0 |
| DEPLOYCACHEFAST1 | 4 | srv3 | 0 | 0 |
| DEPLOYCACHEFAST1 | 4 | srv4 | 0 | 0 |
| DEPLOYCACHEFAST2 | 3 | srv1 | 0 | 0 |
| DEPLOYCACHEFAST2 | 3 | srv2 | 0 | 0 |
| DEPLOYCACHEFAST2 | 3 | srv3 | 0 | 0 |
| DEPLOYCACHEFAST2 | 3 | srv4 | 0 | 0 |
| DEPLOYCACHEBASE2 | 4 | srv1 | 60 | 2 |
| DEPLOYCACHEBASE2 | 4 | srv2 | 60 | 2 |
| DEPLOYCACHEBASE2 | 4 | srv3 | 60 | 2 |
| DEPLOYCACHEBASE2 | 4 | srv4 | 60 | 2 |
