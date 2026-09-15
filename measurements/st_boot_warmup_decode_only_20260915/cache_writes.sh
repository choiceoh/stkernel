#!/usr/bin/env bash
# Files written anywhere under this node's /cache (host /home/choiceoh/glm53-cache) inside UTC windows.
# Usage: cache_writes.sh START END [START END ...]   (read-only: a throwaway container lists them)
set -euo pipefail
args=("$@")
script='host=$1; shift; while [ $# -ge 2 ]; do echo "== $host $1 .. $2"; find /c -xdev -newermt "$1" ! -newermt "$2" 2>/dev/null | head -20; echo "count $(find /c -xdev -newermt "$1" ! -newermt "$2" 2>/dev/null | wc -l)"; shift 2; done'
docker run --rm --network none -v /home/choiceoh/glm53-cache:/c:ro --entrypoint sh st-engine:glm53 -c "$script" _ "$(hostname)" "${args[@]}"
