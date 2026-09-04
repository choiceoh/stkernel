# glm53_dev_lab — 부팅 없는 커널 반복 루프 (29차 item 5)

`VLLM_GLM53_DEV_LAB=1` 로 부팅한 **개발용** 플릿에서:

```
curl -s -X POST http://10.10.10.2:8000/glm53/lab -H 'content-type: application/json' -d '{"op":"info"}'
curl ... -d '{"op":"replay","args":{"n":50}}'      # 서빙 디코드 그래프 50회 재생, 랭크별 us/step
curl ... -d '{"op":"reload","args":{"src":"/overlays/glm53/glm53_megakernel.cu"}}'   # 새 .cu 로 확장 재빌드 + 셀프테스트
curl ... -d '{"op":"recapture"}'                    # 그래프 재캡처(새 커널이 박힘)
```

루프: `.cu` 수정 → 4 노드의 오버레이 경로로 scp → reload → recapture → replay. 25 분 브래킷 대신 1~2 분.
`replay` 는 마지막 배치의 KV/상태 슬롯을 덮어쓴다 — 트래픽 없는 개발 부팅에서만. 프로덕션 기본값 0.
