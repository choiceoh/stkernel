<!-- 속도가 근거인 변경이면 아래를 지우지 말 것 (engine/CHARTER.md D17). 그 외에는 지워도 된다. -->

## 이 변경이 속도를 주장하나?

- [ ] 아니다 — 이 절은 지운다.
- [ ] 그렇다 — **같은 플릿 부팅에서 `bench/onepass.py` 로 C=1 두 판·C=4 한 판 측정한 숫자**를 여기 붙인다.

```
          ctx     tok  cold tok/s  warm tok/s   quality        (1회차 = 부팅 직후, JIT 포함)
                                                               (2회차 = 정상값)
decode:   windows med __ step/s, tokens/step __, raw acc __%
```

- 비교 대상(직전 main 의 같은 두 판):
- 레코드의 `engine_shape`(45차 §72 — 형상이 다르면 비교가 아니다):
- 접두사 캐시를 맞은 브래킷은 프리필이 아니다 — 해당하면 그 칸을 비교에서 뺀다.

프로브·마이크로벤치는 **커널에 대한 증거**지 엔진에 대한 증거가 아니다. 커널이 −39% 인 것과
이 엔진이 요청을 더 빨리 내는 것은 다른 문장이고, 둘째는 네 노드에서만 말할 수 있다.

예약: `launchers/start-st-glm53.sh` (임대를 잡는다) → 문 열리면 onepass 두 판(C=4는 1회차만) → `start-st-glm53.sh stop`.
