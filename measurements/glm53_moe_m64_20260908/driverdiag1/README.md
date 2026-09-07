# CUDA binding initialization diagnostic, 2026-09-08

The 34 API lookup reports reproduce without MoE or INT8. Under the same pinned image and unchanged Compute Sanitizer settings, torch-only and driver-only processes each report zero errors; initializing PyTorch CUDA before the first CUDA Python driver call produces all 34 cuGetProcAddress_v2 invalid-value reports. All six requested driver calls still return success and report driver API version 13000 and one device.

This isolates an initialization-order interaction. It does not yet justify reordering initialization in the acceptance probe: the follow-up checks whether deliberate invalid API calls and deliberately bad CuTe device kernels remain detectable after driver-first initialization. No API-error suppression or exception is used. The diagnostic never admits serving.

Normal session moem64driver10908 received GO at 05:29:35 KST. The minimal probe started at 05:30:47 and exact incoming four-container/configuration/source recovery completed before outer exit 0 at 05:33:43. Source ac2c4b99db0735b6a6859172d0d4901d219e2f10, raw /tmp/glm53-driver-lookup.h3RkZ0. Raw logs, per-mode JSON, source request, lifecycle and checksums are included. Summary exit 0 means collection and recovery completed, not that every child process passed.
