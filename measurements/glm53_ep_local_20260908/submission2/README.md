Normal fleet GPU admission: eplocal0908v2, 2026-09-08 15:02:51 KST.
Preflight passed. At the archived 15:03:59 KST snapshot this session was third
in the queue, after arconsumer0908v4 and vllmprs0908, while pstep0908v1 held
the fleet. Approximate start ETA was 18:14 KST; it is not a promised time.
No GPU payload or component cell had started.

Frozen execution source is 36d4f006bdb0850011dccdbe2a5b8de64789e0b3 at
/home/choiceoh/stkernel-ep-local-gpu-0908-2 on the head. Its tested source
hashes match cpu5, including every mounted MoE file and probe/test contract.
Driver PID 1036953, job /tmp/glm53-ep-local-gpu-0908-2. Read exit.json and
capture/completion.json before any retry. Do not modify the frozen copy.
Subsequent PR commits only record this admission and evidence.

The prior eplocal0908v1 was cancelled through the normal fleet tool at
15:02:03 KST while it was still waiting. Its driver exited 143, with no
capture directory and no GPU payload started. The new source was prepared
and preflighted before cancellation, then registered at the normal queue
tail. The previous final log/exit and cancellation output are retained here;
this was a source replacement, not a failed GPU numerical test.

The runner first verifies cpu5 evidence before service inventory or pause,
then uses the existing normal-holder/exact-original-restore lifecycle. The
eight isolated fixtures compare actual compact and full-token wrappers with
bounded stock-repeat noise; failures retain partial JSON. Memcheck/racecheck
follow only passed prerequisites. Timing remains component-only. Production
TP4 transport, complete EP serving, quality/decode/capacity and matched direct
TTFT still require their own proof. Both experimental flags remain default 0.

sha256.json records original byte lengths/hashes for every raw file. The
fleet and preflight logs are compressed; hashes cover original bytes.
state.json records collection time and whether either payload had started.
