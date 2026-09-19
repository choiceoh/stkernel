#!/usr/bin/env bash
# Synthetic self-distribution data v3: new seeds from the taxonomy, conversations answered by the served model at the
# served settings (datagen2.py), filtered, names swapped -- the synthetic half of the next data window.
cd ~/q38mtp-gen2
D=v3syn
echo "[$(date +%H:%M:%S)] seeds"
bash gen2.sh seeds $D "${CALLS:-320}" 2>&1 | grep -E '"final"|error' | tail -5
echo "[$(date +%H:%M:%S)] seeds: $(wc -l < $D/seeds.jsonl)"
bash gen2.sh convos $D "${LIMIT:-1300}" 2>&1 | grep -E '"final"|error' | tail -5
echo "[$(date +%H:%M:%S)] convos: $(wc -l < $D/convos.jsonl)"
bash gen2.sh filter $D
bash gen2.sh stats $D | head -3
echo "[$(date +%H:%M:%S)] done"
