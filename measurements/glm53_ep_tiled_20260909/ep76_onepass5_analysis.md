# v5 completed B prefill anomaly — head-only audit

Frozen source `ca076d35e64a6a19e90dffe54054269d1a5e1887`.
The completed named B boot log is bound to head `5017137c…04f8c` / start `2026-09-10T04:23:13.693004263Z`. Its first 277,670 bytes exactly equal the earlier strict B-ready head log. Raw log SHA256 `d61634daff8413786ce50cf000255f9c401b6da79e72ff490f47f533bf37d255`; unchanged remote inode/size/mtime before and after capture, mtime `04:36:09.427787 UTC`. No current A or worker reads.

32K TTFT increased from historical v4 B 9.9659s to11.7055s (+1.7396s); 128K from39.3263s to66.9275s (+27.6011s). Prompt tokens and all eight request hashes match v4. The first2K and fixed decode do not show that slowdown; current B fixed pooled76.8057tok/s,19.9935step/s, facts18/18 and Korean0/8. Both originals retain cold_compile=true.

During32K/128K there is no newly logged JIT warning, graph capture or PREP plan, and no ERROR/DISARM/DRIFT. The same202 NVFP4 static-scale freezes appear in both logs: v5 at04:34:26–29, v4 at03:54:28–30. They do not independently explain this difference.

The concrete new symptom is high head OSAR wait samples during128K:1354.5–2629.1µs/collective (22/33 samples per report) versus v4's29.2–63.3µs (44/55 samples). OSAR lines are not timestamped: the v5 cluster is bounded by04:34:28–04:35:15, not an exact request timeline. Its weighted wait sum is0.279671s, far smaller than27.6011s. These samples locate a peer-wait symptom, not its cause or the entire delay. Last scale-freeze→T4143 spans about46s versus28s historically.

Conclusion: a retained, real long-context TTFT anomaly with evidence of increased head collective wait; root cause remains unproved by the available head-only logs. No assertion of external interference, no sample exclusion, and no metric reclassification. Request epoch boundaries are not stored, so exact attribution cannot be reconstructed merely by summing request elapsed times.

Evidence: [canonical archive](ep76_onepass5/README.md), [CPU7](ep76_cpu7/README.md).

# CPU7 register-max 실제 PTX 감사

Source `ca076d35e64a6a19e90dffe54054269d1a5e1887` 고정. 원본 `ep76_cpu7/originals/`의 baseline/opt `M6-map288-i32`만 비교했다. 새 컴파일·GPU·원격·서비스·테스트 없음.

- Candidate PTX SHA256 `3efb8d89aa81c0f9b738e4389bb5532002ccb7349628ceeda8e0870cdde81369`.
- Candidate cubin SHA256 `17707c12595e4736e5b8c221242453a58aeaa8deb5b865f7babcfa0cacae4dde`.
- PTX와 resource 로그 원문은 result.json의 해당 hash/text와 재대조했다.
- B `REG123 STACK0 SHARED1024 LOCAL0` → candidate `REG128 STACK8 SHARED1024 LOCAL0`. Dynamic shared98,304B, total99,328B unchanged.

## Max 교환 수명

후보 PTX에서 max `st.shared.f32`는1853/1887/1921/1955, publication `bar.sync 1,128`은2026이다. 새 `ld.volatile.shared.f32`는2052/2164/2270/2376으로 모두 그 barrier 뒤에 있다. 유효 row와 group leader 조건 아래 읽는다. 재사용 barrier2885, final A2 fence2905/barrier2906도 유지된다. 이 구간 전체는 persistent loop label831와 backedge4702 사이에 남아 있다. 읽기가 loop 밖으로 hoist됐다는 흔적은 없다.

R>8은2030–2032에서 scalar fallback2480–2883으로 분기한다. fallback에는 기존16×BF16 shared load,16×max,8×FP4 x2 conversion이 있고 새 volatile max/shuffle는 없다. 활성화 결과의 기존 full-sC1 store도 R>8 분기1958–2018에 남아 있다.

## 선택 경로의 정적 명령 수

다음은 각 코드 구간에 나타나는 명령 수다. leader/valid row predicate와 inactive lane을 무시한 동적 총량으로 해석하지 않는다.

| 구간 | 주요 실제 명령 |
|---|---|
| max1803–1957 | BF16x2 round4, BF16→F32 8, max16, shuffle8, F32 max store4 |
| 공통2020–2023 | BF16x2 round4 |
| quant2033–2479 | volatile max load4, max4, rcp12, E4M3x2 conversion4, BF16→F32 8, FP4x2 conversion4, shuffle8 |
| scalar fallback2480–2883 | BF16 shared load16, BF16→F32 16, max16, rcp3, E4M3x2 conversion1, FP4x2 conversion8 |

새 Q1 shuffle16개는 전부 실제 `shfl.sync.idx.b32 ...,31,-1`이다. 전체 PTX의19개 중 나머지3개는 기존 startup shuffle다.

중복 변환은 실제로 관찰됐다.1803–1806의 F32 operand pairs `(%r61,%r60),(%r57,%r56),(%r53,%r52),(%r49,%r48)`를2020–2023에서 동일하게 다시 `cvt.rn.bf16x2.f32`로 변환한다. 최대값 경로와 quant 경로의 BF16→F32도 각각8개로 별도 존재한다. 동일 값을 다시 계산해 register 보유기간을 줄이는 compiler 선택일 수 있으며, 이것이 실행시간 저하 원인이라고 단정하지 않는다. 반올림 방식과 operand 값 자체의 차이는 없다.

## STACK8 해석의 한계

후보 전체 PTX에는 `.local` allocation, `ld.local`, `st.local`이 없다. 하지만 cubin을 만든 ptxas 단계에서 발생하는 spill은 PTX에 나타나지 않을 수 있으므로 `LOCAL0`/PTXlocal0만으로 no-spill이라고 하지 않는다. 보존된24개 compiler/resource/fleet 로그에서 `bytes spill stores`, `bytes spill loads`, `bytes stack frame` verbose 진단도 찾지 못했다. 따라서 compiler-reported zero-spill 주장 근거도 없다.

현재 archive에 SASS 원문이 없고 로컬 cuobjdump/nvdisasm도 없다. STACK8의 실제 LDL/STL 발생 여부와 frame 사용처는 미확정이다. 이 감사에서는 기존 원본만 확인했으며 별도 binary audit를 위해 remote/CPU 작업을 추가하지 않았다.
