#!/usr/bin/env bash
# Stage an immutable template on every node before any serving teardown.
ct_prepare_glm53_chat() {
  local requested=${CHAT_TEMPLATE:-chat_template_mm_v2.jinja}
  local repo_dir=$1 source digest stage_cmd got ip
  case "$requested" in
    ''|.*|*[!A-Za-z0-9._-]*) echo "ABORT: CHAT_TEMPLATE must be a filename" >&2; return 1 ;;
  esac
  source="$repo_dir/$requested"
  # The repo owns bundled templates even when a stale model-dir copy exists.
  [ -f "$source" ] || source="$MODEL_HOST_PATH/$requested"
  [ -f "$source" ] || { echo "ABORT: template missing: $requested" >&2; return 1; }
  if command -v sha256sum >/dev/null 2>&1; then
    digest=$(sha256sum "$source") || return 1
  else
    digest=$(shasum -a 256 "$source") || return 1
  fi
  digest=${digest%% *}
  CHAT_TEMPLATE="chat_template.$digest.jinja"
  echo "chat: $requested -> $CHAT_TEMPLATE (reasoning=$REASONING_PARSER, tools=glm47)"
  [ "${DRY_RUN:-0}" != 1 ] || return 0

  stage_cmd=$(printf '%q ' bash -c '
    set -euo pipefail
    dir=$1; name=$2; expected=$3
    temp=$(mktemp "$dir/.glm53-template.XXXXXX")
    trap '\''rm -f -- "$temp"'\'' EXIT
    cat > "$temp"
    got=$(sha256sum "$temp"); got=${got%% *}
    [ "$got" = "$expected" ] || { echo "template transfer hash mismatch" >&2; exit 1; }
    chmod 0644 "$temp"
    mv -f -- "$temp" "$dir/$name"
    got=$(sha256sum "$dir/$name"); got=${got%% *}
    [ "$got" = "$expected" ] || exit 1
  ' glm53-chat "$MODEL_HOST_PATH" "$CHAT_TEMPLATE" "$digest")
  for ip in "$HEAD_IP" "${WORKER_IPS[@]}"; do
    if [ "$ip" = "$HEAD_IP" ]; then
      bash -c "$stage_cmd" < "$source" || return 1
    else
      ssh $SSHOPT "choiceoh@$ip" "$stage_cmd" < "$source" || return 1
    fi
  done
  # This is the exact model mount and template path used by vLLM below.
  got=$(docker run --rm -v "$MODEL_HOST_PATH:$MODEL_PATH:ro" \
      --entrypoint sha256sum "$IMAGE" "$MODEL_PATH/$CHAT_TEMPLATE") || return 1
  got=${got%% *}
  [ "$got" = "$digest" ] || { echo "ABORT: container template hash mismatch" >&2; return 1; }
}
