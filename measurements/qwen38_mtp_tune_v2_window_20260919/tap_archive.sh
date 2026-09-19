#!/usr/bin/env bash
# After window 5 (operator 2026-09-19: "탭 디렉터리는 너가 알아서 판단해"): the MTP input tap's directory holds its 64 GiB
# cap, so the default-on tap records nothing. The shards are the only copy of the day's collected text -- move them (same
# filesystem, instant) to an archive beside the directory, which frees the cap without deleting anything; drop the three
# rsync copies of the processed runs on srv1/srv3/srv4 (srv2 keeps its copy, and the archive can rebuild it).
set -uo pipefail
D=/home/choiceoh/glm53-logs/st-qwen38-dumps
A=$D/mtp-inputs-archive-20260919
W=/home/choiceoh/q38mtp-train-0919
stamp() { echo "[$(date +%H:%M:%S)] $*"; }

if docker ps --format '{{.Names}}' | grep -q '^st-qwen38$'; then
  stamp "a Qwen fleet is up on srv2 (rank 0 writes the tap): nothing moved"; exit 1
fi
mkdir -p "$A"
n=$(find "$D/mtp-inputs" -maxdepth 1 -name 'mtp-inputs-*.npz' | wc -l)
stamp "moving $n shards ($(du -sh "$D/mtp-inputs" | cut -f1)) to $A"
find "$D/mtp-inputs" -maxdepth 1 -name 'mtp-inputs-*.npz' -exec mv -n {} "$A"/ \;
left=$(find "$D/mtp-inputs" -maxdepth 1 -type f | wc -l)
stamp "left in mtp-inputs: $left files; archive: $(find "$A" -name '*.npz' | wc -l) shards, $(du -sh "$A" | cut -f1)"
cat > "$A/README.txt" <<'EOF'
MTP input tap shards, 2026-09-19 (Qwen3.8 MTP head tuning, windows 3-5), moved out of ../mtp-inputs when it held the tap's
64 GiB cap so the default-on tap records again. Each shard: streams [R, hc*H] BF16 as int16 bits, meta [R, 4] int64
(sequence, position, next token, decoded). Shards are grouped by their boot prefix (UTC):

  mtp-inputs-20260919-063621-*  window 3: OpenRouter same-model text prefilled (v1), 15:34 KST
  mtp-inputs-20260919-073806-*  window 3: the target's own decoding of the train prompts, 16:38 KST
  mtp-inputs-20260919-090839-*  window 4: data3a, v2 synthetic conversations answered greedy + synthetic documents
  mtp-inputs-20260919-093244-*  window 5: data3b, v2 synthetic conversations answered at T=1
  mtp-inputs-20260919-093746-*  window 5: data3c, Deneb transcript windows + wiki/code/files + rendered tool
                                conversations (the cap stopped it at 18:48:25 KST; ~65 of Deneb's file documents missing)

The token ids spell out the text, including Deneb's own conversations and records: approved for prefill on the fleet
only -- never copy these off the fleet, never commit them. Processed runs: srv2 ~/q38mtp-train-0919/data
(mtp_tune data --taps <this directory>). Record: measurements/qwen38_mtp_tune_window_20260919 and the v2 window record.
EOF
for ip in 10.10.10.1 10.10.10.3 10.10.10.4; do
  before=$(ssh -o BatchMode=yes -o ConnectTimeout=5 "choiceoh@$ip" "df -h /home/choiceoh | tail -1 | awk '{print \$4}'")
  ssh -o BatchMode=yes -o ConnectTimeout=5 "choiceoh@$ip" "docker ps --format '{{.Names}}' | grep -q '^q38tune$' && exit 3; rm -rf $W/data" \
    && stamp "$ip: dropped the runs copy (free $before -> $(ssh -o BatchMode=yes -o ConnectTimeout=5 "choiceoh@$ip" "df -h /home/choiceoh | tail -1 | awk '{print \$4}'"))" \
    || stamp "$ip: kept (a training container is up, or ssh failed)"
done
stamp "srv2 free: $(df -h /home/choiceoh | tail -1 | awk '{print $4}')"
