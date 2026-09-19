# ost-97x: the 5050 on the tailnet

> 살아 있는 참조 — **이 상자가 단일 GPU 레인으로 무엇을 할 수 있는지. 설정이 바뀌면 여기부터 고친다.** 여기가 틀리면 그건 버그다.

`bench/fleet_single.py` and `probes/run_engine_probe.sh` both name this box as the
single-GPU lane's alternative to a Spark: *"ost-97x, the operator's Windows PC on the
tailnet, once it has sshd in WSL2, docker with the NVIDIA runtime and an x86_64 image."*
As of 2026-09-15 it has the first two. This is how, what it costs the box, and which two
numbers do not carry over from a Spark.

The distro is on the tailnet as its own node, **`ost-97x`**, beside the Windows node
`office-topsolar`. `tailscale ip -4` on the box prints its address; it is not written down
here, and the controller's `~/.ssh/config` is where it belongs. From srv4, end to end:

```
$ ssh ost-97x 'hostname; uname -m; nvidia-smi --query-gpu=name,compute_cap,memory.total --format=csv,noheader'
OST-97X
x86_64
NVIDIA GeForce RTX 5050, 12.0, 8151 MiB

$ ssh ost-97x 'docker run --rm --gpus all ubuntu:24.04 nvidia-smi --query-gpu=name,compute_cap --format=csv,noheader'
NVIDIA GeForce RTX 5050, 12.0
```

The box, as of 2026-09-15: Windows 11 (26220), 31.1 GiB of RAM, 16 logical CPUs, one
**RTX 5050, 8 GiB, sm_120**, driver 595.79 / CUDA 13.2. Windows is on the tailnet as
`office-topsolar`; WSL2 runs Ubuntu 24.04 with systemd, and the GPU
answers there -- `/usr/lib/wsl/lib/libcuda.so` is mounted in and `nvidia-smi` sees the
card.

## Why the node is inside WSL2

The lane's evidence is `cat /proc/meminfo; docker ps`, its payload is a container, and it
rsyncs a tree over ssh. Windows has none of that, so the lane's host has to be the WSL2
distro. WSL2 sits behind a NAT (`eth0 172.26.x/20`) that Windows' own tailnet address does
not cross, and the two usual bridges are both Windows-wide changes: a `netsh portproxy`
chasing an address WSL2 redraws every boot, or `networkingMode=mirrored`, which rewrites
networking for every session on the box.

Neither is needed here. Tailscale is already installed *inside* the distro and running
with a kernel TUN (`/dev/net/tun`, the `tun` module loaded, no `--tun=userspace-networking`
in its flags) -- only logged out. Logging that daemon in gives the distro a node and a
100.x address of its own; Windows forwards nothing, keeps its own node, and the Windows
firewall never enters into it.

## Running it

In WSL2 on this box, as `choiceoh`, not under sudo:

```bash
bash tools/ost-97x-lane-setup.sh            # keys, sshd, the tailnet node
bash tools/ost-97x-lane-setup.sh --docker   # and docker with the NVIDIA runtime
```

It is idempotent, and it needs two things only the operator can give: the sudo password,
and a browser sign-in for `tailscale up` (a new node has to be authorized against the
tailnet, which is an account action, not a shell one).

The controllers' keys come from srv4 rather than `ssh-copy-id`, because `ssh-copy-id`
needs an sshd here to copy into and a password to open it with, and there is neither.
srv4's own `authorized_keys` already lists both controllers' keys, so the script takes just
those two lines, matched by their key comments, and copies nothing else out of that file.
Which two comments is `OST97X_CONTROLLER_KEYS`, because that is this tailnet's business and
not the repository's.

Two things the first run turned up here, both now handled. Ubuntu 24.04 serves ssh through
**socket activation**: `ssh.socket` owns :22 and starts `ssh.service` on the first
connection, so an idle box reads `is-active ssh` = inactive while the door is in fact open
and enabled at boot. `enable --now ssh.service` on top of that live socket cannot bind the
port, and under `set -e` it took the whole script down with it -- the tailnet login never
ran, which looked exactly like a finished setup that had not worked. And `/usr/bin/docker`
was a dangling symlink into `/mnt/wsl/docker-desktop/`, left by a Docker Desktop that is no
longer installed, while `nvidia-container-toolkit 1.19.0` and the `nvidia` runtime in
`/etc/docker/daemon.json` survived from that era: the apt install replaces the symlink and
the toolkit step reports it is already there.

Afterwards, on the Mac and srv4 (`~/.ssh/config` owns address, user and port -- nothing in
the lane overrides it):

```
Host ost-97x
    HostName <what `tailscale ip -4` prints on the box>
    User choiceoh
    IdentityFile ~/.ssh/id_ed25519
    IdentitiesOnly yes
```

The controller needs that alias too -- srv2, where the queue lives: its key was added to this
box's `authorized_keys` on 2026-09-19.

Since 2026-09-19 (operator: two one-GPU lanes at once) this box is **the check lane**, its own
lane beside the single one on srv4 and the fleet -- nothing to export. `bench/fleet.sh` defaults
`FLEET_CHECK_GPU_HOST=ost-97x` and `FLEET_CHECK_GPU_NAME=RTX5050` (empty host turns the lane off),
the lane has its own holder (`holder-check`), and what this box owes itself and runs are facts
in `bench/fleet_single.py` `HOSTS`: floor 4 GiB, a check's budget 4 GiB, the image
`st-engine:glm53-sm120-x86`, and the Sparks' flashinfer under `~/st-x86-flashinfer/vendored`,
which the runner mounts over the image's site-packages. A check goes there by saying so:

```bash
bash bench/fleet.sh run --gpu --check <session> [est] [note] -- bash probes/run_engine_check.sh ...
```

`on_fleet()` reads false for this name, which is what we want: no fleet boot contends for
this box, and the lane's holder is the only reservation. A probe that asks more than a kernel
check's budget (a full-model one asks 64 GiB) is refused for this lane at submission.

## The Windows side

Two things the distro cannot do for itself.

**Memory.** WSL2 defaults to half of host RAM -- about 15.5 GiB here, of which 13.6 GiB
was available on an idle box. That is under the floor by itself, so with a Spark's floor
the lane refused *every* check, including one with a zero budget:

```
$ python3 bench/fleet_single.py evidence --host ost-97x --gib 4
ost-97x: no room beside production -- MemAvailable 13.5 GiB, this check's budget 4.0 GiB,
floor 16.0: 9.5 GiB would be left
```

`FLEET_SINGLE_GPU_FLOOR_GIB=4` is the answer to that, not a bigger VM. With it the same
command returns silent at rc 0 and `reclaim` faults and releases its 4 GiB on the box.

So `%USERPROFILE%\.wslconfig` exists for the opposite reason -- to keep the box **small**,
because it is also somebody's desktop:

```ini
[wsl2]
memory=12GB
processors=6
autoMemoryReclaim=gradual
```

12 of 31 GiB and 6 of 16 CPUs leaves Windows 19 GiB and 10 cores at the worst moment, and
`autoMemoryReclaim` hands page cache back instead of sitting on it. That is not a
theoretical tidiness: an unconstrained image build here on 2026-09-15 left Windows with
**0.6 GiB free** and the desktop crawling, because WSL2 holds cache by default and a 16 GB
image build generates a lot of it. Under the cap the same build holds at its 12 GiB ceiling
and stops there. `memory=` is a ceiling, not a reservation. It applies only after
`wsl.exe --shutdown`, **which kills everything in the distro**, so change it deliberately.

**Staying up.** WSL2 does not start with Windows, so after a reboot the node is simply gone
until someone opens the distro -- which for a box other people run checks against is not a
state to leave to chance. `schtasks /create` is the obvious answer and it is **denied
without elevation** on this account (`ERROR: Access is denied`), so the logon hook is a
script in the operator's own Startup folder instead, which needs no elevation:

```
%APPDATA%\Microsoft\Windows\Start Menu\Programs\Startup\wsl-ost-97x.vbs

  CreateObject("WScript.Shell").Run "wsl.exe -d Ubuntu -u root -e /bin/true", 0, False
```

Hidden window, no wait, so the logon is not held up. systemd is PID 1 in the distro, so the
command exits while `ssh.socket` and `tailscaled` keep running -- all three of those units
are `enabled`, which is why the distro comes back complete. Verified by doing it: after a
`wsl.exe --shutdown` the node returned at the same address with ssh, docker and the GPU
answering from srv4. If the box must answer *before* anyone logs in, that needs a
`/sc onstart /ru SYSTEM` task and an elevated prompt.

## The two numbers that do not carry over

**The floor is a Spark's.** 16 GiB is what a `--test` boot leaves a GB10 so earlyoom does
not pick the engine, on a box with 128 GiB and one pool for host and device. This box has
31 GiB and a discrete 8 GiB card, so a check's device memory is not drawn from the number
the floor guards, and the floor is simply the wrong size here -- it refused a zero-budget
check on a box that had 13.6 GiB free and 8 GiB of idle VRAM.

So the floor is now the box's to state: its entry in `bench/fleet_single.py` `HOSTS`, which
**`FLEET_SINGLE_GPU_FLOOR_GIB`** (or `--floor`) still overrides, and `FLOOR_GIB` for a box
with no entry -- unchanged for every Spark. It is a floor, not a licence -- a budget that eats
past it still refuses -- and it is part of the evidence cache's key, so lowering it cannot read
back an answer computed under the old one. This box's entry says 4 GiB, which leaves the distro
its own working set and still admits a check the Spark's floor rejected outright.

**The card is a different card.** An RTX 5050 is sm_120; the Sparks are sm_121a. A verdict
from this lane is that card's verdict. It is a fine place to catch a compile error, a
correctness regression or a shape bug early, and it is not a number for MEASUREMENTS.md.

## The x86_64 image

`st-engine:glm53` cannot be ported to this box, and the reason is not the wheels. Four
things, each checked rather than assumed:

1. Its parent, `glm53:v13-b12x-it`, is a **locally built ARM64 vLLM dev image**:
   `docker image inspect` on srv4 gives `Architecture: arm64`, `RepoDigests: []` and
   `ai.vllm.build.commit=unknown`. It was never pulled from a registry and records no
   source revision, so it can be neither re-fetched nor rebuilt for another architecture.
2. `promote_deep_gemm.py` lifts DeepGEMM's **compiled** `_C*.so` and its JIT headers
   byte-for-byte out of that image, and `verify.py` checks every one of those SHA256s. An
   x86_64 build cannot be those bytes at any version.
3. `install_cuda132.py` refuses `platform.machine() != 'aarch64'` by design.
4. Even built, it would not run: every native lane is `-gencode arch=compute_121a`, and
   `engine/kernels/cells.py` refuses a device that is not a GB10 with the measured SM count.

What *is* portable is the layer this card can actually run. `engine/kernels/b12x` declares
`@supported_compute_capability([120, 121])` -- 120 is an RTX 5050 -- and imports torch and
flashinfer only, no DeepGEMM. So this box gets a **second, separate image**, never the
production tag:

| | `st-engine:glm53` | `st-engine:glm53-sm120-x86` |
|---|---|---|
| arch | ARM64, sm_121a | x86_64, sm_120 |
| parent | `glm53:v13-b12x-it` (vLLM) | `python:3.12-slim-bookworm` |
| DeepGEMM | promoted from the parent | absent |
| native lanes | mla, dense, oneshot, … | absent (compute_121a only) |
| good for | measurements | compile and correctness checks on `b12x` |

Its chain is the ARM one's shape with no seed to inherit from, so every package is named
from an index:

```bash
python3 engine/runtime/make_x86_64_lock.py engine/runtime/cuda132.x86_64.lock.json
bash engine/runtime/build-x86_64.sh        # fetch (network), then build --network none
```

42 wheels resolve, verified by sha256 -- the cu132 index publishes no digest, so
`make_x86_64_lock.py` fetches those three and hashes them itself. Four entries cannot
match the fleet's and each is recorded in the lock's `deviations`: `nvidia-cudla` is
dropped (Tegra-only, no x86_64 build and no such hardware), `flashinfer-python` moves from
the unpublished `0.6.18.dev20260819` to `0.6.18.post1` (a `py3-none-any` wheel that JITs
its kernels, so this is a version difference and not an arch one), `tilelang` moves
`0.1.12 -> 0.1.14`, the nearest version publishing an x86_64 wheel, and `deep_gemm` is
absent because no x86_64 build can reproduce it.

The lock is not the whole dependency set, and this is the part that bites: the fleet's lock
names 34 wheels because its vLLM parent already supplied everything else. A base image
supplies nothing, so `fetch_x86_64.py` reads `Requires-Dist` out of every locked wheel and
verifies the lock's pinned closure -- 43 wheels -- into `closure.json` at fetch time
(only `--resolve-closure` resolves afresh), which keeps the build itself offline. Reading
only *torch's* requirements is
not enough: that builds an image where `import flashinfer` raises
`ModuleNotFoundError: No module named 'tvm_ffi'`, which is how the first build here failed.

To point the lane at it: `ST_IMAGE=st-engine:glm53-sm120-x86`. Without that,
`run_engine_probe.sh` looks for the production image on the host and stops with
`ABORT: no ST image on ost-97x`, which is the right refusal -- the production image is not
here and what is here is not it.

### What of this is verified, as of 2026-09-15

The **lock** is: all 42 wheels resolve and every digest is checked, and `torch`'s was
confirmed against an independent `curl | sha256sum` (530,327,928 bytes). The **chain runs**:
a build completed and reported `installed 42 locked wheels + 9 closure wheels` (the closure
was resolved then; it is now pinned in the lock at 43).

The **image now is, for compiling** (2026-09-15). `st-engine:glm53-sm120-x86` was built
(18.4 GB) and it imports `flashinfer`. What it cannot do out of the box is import ST's
**b12x** path -- see the next section; bridged, it compiles ST's CuTe kernels for sm_121a
with no device. No GPU check has run against this image, and none can give a number: the
card is sm_120. The
build is also not free: it moves ~3 GB of wheels and writes a ~16 GB image, and running it
unconstrained on this box left Windows with 0.6 GiB of free RAM. Give it a quiet moment, or
a `--cpu-quota`.

### The native lanes: three instructions, not an architecture

Changing only the arch flag does not build them. `ptxas`, on this box:

| source | features `sm_120` does not have |
|---|---|
| `mla/glm53_megakernel.cu` | `add.u8x4` ×288, `ldmatrix.b8` ×16, `.m16n16` ×16 |
| `dense/kernels.cu` | `add.u8x4` ×936 |

Those counts are PTX occurrences, and they are the reason this looked like a wall. In the
**sources** it is three sites:

- `mk_add_u8x4` -- one helper in each file, and the comment above it names its own
  replacement: *"CUDA 13.2: `__vadd4` still lowers to carry-isolation arithmetic on SM121."*
  The fleet rejects `__vadd4` for **speed**; the byte semantics are the same modulo-256 add.
  A check build has no speed to protect.
- `ldmatrix.sync.aligned.m16n16.x1.trans.shared.b8` -- one site, in the `O += P C` inner
  loop. It also needs nothing written for it: the same B fragment is what
  `mla_e4m3x2_strided` already builds for the `ROWS` path further down the same file, which
  is production code the fleet runs today.

So both files now carry an `#else` for `__CUDA_ARCH__ < 1210`, and both compile for sm_120.

**A fallback that merely compiles would be worse than none** -- wrong fragment layout
returns plausible, wrong numbers, and a check lane that lies is not a check lane. So the
layout claim is measured rather than asserted, on the card, against a CPU reference:
`probes/sm120_b_fragment_check.cu`.

```
$ nvcc -O2 -gencode arch=compute_120,code=sm_120 -std=c++17 probes/sm120_b_fragment_check.cu -o /tmp/c && /tmp/c
check 1: mma m16n8k16 fragment layouts
  LAYOUT CONFIRMED: all 128 elements match CPU matmul (worst 0.0)
check 2: the strided B form the sm_120 fallback reuses
  STRIDED FORM CONFIRMED: all 256 elements match the CPU reference
```

And the other half of the claim -- that none of this reaches the fleet -- is checked the
only way worth checking it, by compiling the *same source* for the Sparks' target and
comparing the output to a baseline taken before the edit:

| source | sm_121a PTX vs pre-edit | sm_120 |
|---|---|---|
| `mla/glm53_megakernel.cu` | **identical** | compiles, 2,317,000 bytes |
| `dense/kernels.cu` | **identical** | compiles, 3,775,944 bytes |

Compiling was not running, and running was an operator decision, taken 2026-09-15 and
recorded under D5 in `engine/CHARTER.md`: a card that is not a GB10 may **check**, never
measure. Three things carry that, and the shape of them matters more than the size:

- `engine/kernels/arch.py` maps a capability to its build target. `mla` and `dense` take
  their `-gencode` from `bound().device.capability` instead of a literal -- the bound shape
  already declares the card, so there is **no knob** (D11: inputs are facts, and a knob
  would be a second source of truth for one). Only those two lanes are listed, because only
  those two have been compiled and checked anywhere else; the other five stay pinned, and
  `oneshot` cannot leave the fleet at all.
- `cells.py` gains a middle verdict for the device lane. A GB10 is `admitted`, a card no
  lane was ever built for is `refused` exactly as before, and a **built but unmeasured**
  card is `unmeasured` with a `measure` recipe -- the repository's own word for "the
  wrapper serves it, but it runs by declaration".
- The runtime gates in `mla` and `dense` needed **no change at all**: they compare the
  probed device against the bound one, which is already the right question on either card.

```
GB10       SM121/48   -> admitted    recipe=None
RTX 5050   SM120/20   -> unmeasured  recipe=measure
H100       SM90/132   -> refused     recipe=rewrite
```

The fleet reads exactly as before, and that is checked rather than asserted:
`gencode(MEASURED.device.capability)` returns `['-gencode', 'arch=compute_121a,code=sm_121a']`
-- the literal that used to be there -- and the sm_121a PTX of both kernels is byte-identical
to its pre-edit baseline.

What is still a real difference, and no verdict can talk it away: **20 SMs against 48**, on a
persistent megakernel built around 48 at TP=4. That is why the verdict is `unmeasured` and
why a number from this card is this card's.

`oneshot` is out regardless: it needs `<infiniband/verbs.h>` for the fleet's RoCE rails.

The image carries a host compiler (the full Python base, not `-slim`) because without one a
JIT extension dies at `gcc: No such file or directory` before ptxas can say anything about
the kernel at all.

## Compiling ST's CuTe kernels here — and the one thing that blocks it

The image's flashinfer is pip-resolved **0.6.18.post1**. The Sparks run the vendored
**0.6.18.dev20260819**. Same version line, different build, and only the dev one carries
`Sm120B12xBlockScaledDenseGemmKernel._collapse_to_vmk` --

    flashinfer.gemm.kernels.dense_blockscaled_gemm_sm120_b12x

-- which `engine/kernels/b12x/_moe_dynamic/generic.py` borrows onto `MoEDynamicKernel` **at
import time**. So on the stock image the whole b12x path dies before any kernel is named:

    AttributeError: type object 'DenseGemmKernel' has no attribute '_collapse_to_vmk'

The package is **pure Python** (0 `.so` in 137 MB), so the Sparks' copy carries to x86_64.
`bench/compile_sm121a.sh` bind-mounts it over the image's site-packages and runs a script
with `CUDA_VISIBLE_DEVICES=` and `CUTE_DSL_ARCH=sm_121a`. Its header holds the one-time export.

**The trap**: this image installs into `site-packages`, which SHADOWS `dist-packages`.
Mounting over `dist-packages` changes nothing and the import silently keeps resolving to
post1 -- the dist-info reads `dev20260819` while `flashinfer.__version__` reads `post1`.

Measured on 2026-09-15, `probes/engine_moe_m64_compile.py` through that script:

| arm | compile | resources (sm_121a cubin) |
|---|---|---|
| pinned M128 (`sf6_q0_words`) | ~5 s | `REG:168 STACK:112 SHARED:1024` |
| private M64 (`prefill_m64`) | ~3 s | `REG:128 STACK:80 SHARED:1024` |

`cuda_initialized: false` on both. This is what the section above hoped for: the lane widens
to **kernel edits**, which used to cost a twenty-minute GPU round trip each. It does not
widen to numbers -- those stay on a Spark.

## Verify

From a controller, one command for the whole chain:

```bash
bash tools/ost-97x-selftest.sh ost-97x
```

The lane fails in layers -- ssh, docker, the NVIDIA runtime, the image, what imports inside
it, the queue's admission, staging -- and a failure in one reads like a failure in another:
`fleet.sh run` says `ABORT: no ST image` whether the image is missing, the daemon is down,
or the box never answered at all. The self-test walks them in order and names the first
link that does not hold. Fix that one; the rest usually follow from it.

The pieces, if you want them by hand:

```bash
ssh ost-97x 'hostname; nvidia-smi --query-gpu=name,compute_cap,memory.free --format=csv,noheader'
python3 bench/fleet_single.py evidence --host ost-97x --gib 4   # silent, rc 0 = it has room
python3 bench/fleet_single.py reclaim  --host ost-97x --gib 4
```

## What this box can hold

Measured 2026-09-15, desktop running: **6407 MiB of 8151 MiB VRAM free** (the card also
drives the display, so this moves -- it was 5199 MiB *used* earlier the same day with more
windows open), 6 CPUs and ~10 GiB to the distro under the cap above, 869 GB of disk. A
check that needs more VRAM than the desktop happens to be leaving free will OOM on the
card, and the lane's `MemAvailable` evidence cannot see that: it guards host memory, and
this card's memory is its own. For anything close to the line, read `memory.free` first --
the self-test prints it.

**Leave `MAX_JOBS` alone.** The images set it to 2 (`engine/runtime/Dockerfile`,
`Dockerfile.x86_64`) and `probes/run_engine_probe.sh` defaults it to 2, and on a box this
size that is not conservatism, it is arithmetic: one nvcc on
`mla/glm53_megakernel.cu` peaks at **3,340 MiB** resident, measured here. Two fit in the
12 GiB the distro is capped to with room to spare; four need about 13 GiB and do not.

Raising it to 4 on 2026-09-15 is what wedged the VM. Not a crash -- there is no error in the
Windows event log, because nothing crashed: the VM sat at 11.26 GiB against its 12 GiB
ceiling, stopped answering, and `wsl.exe` timed out against it with
`Wsl/Service/0x8007274c`. `wsl.exe --shutdown` and a restart recovered it with every service
and the tailnet address intact, which is the one good thing to come out of it.

The general shape of that mistake is worth more than the number: the memory cap above and a
build's parallelism are one budget, not two. Tightening the first while raising the second
reads as two sensible changes and is one way to hang the box.
