"""Parse configured GLM53 container launches without executing shell commands.

This records configuration, not a live process, kernel launch, or all-rank
proof. Source/manifest attestation and fresh rank logs are still required.
"""
import base64
import binascii
import hashlib
import json
import re
import shlex


# The current launcher's only permitted shell prelude. Keep this literal: a
# permissive search for a later `vllm serve` line could hide another command.
_GID_PRELUDE = r'''# Auto-detect the RoCE-v2 IPv4 GID index (per node, re-numbers across reboots).
for HCA in $(echo "${NCCL_IB_HCA}" | tr ',' ' '); do
  for i in $(seq 0 15); do
    t=$(cat /sys/class/infiniband/$HCA/ports/1/gid_attrs/types/$i 2>/dev/null || true)
    g=$(cat /sys/class/infiniband/$HCA/ports/1/gids/$i 2>/dev/null || true)
    case "$t" in *"RoCE v2"*) case "$g" in *0000:0000:0000:0000:0000:ffff:*) export NCCL_IB_GID_INDEX=$i; break 2;; esac;; esac
  done
done
'''
_WRAPPER = re.compile(
    r"echo ([A-Za-z0-9+/]+={0,2}) \| base64 -d > /tmp/serve\.sh; bash /tmp/serve\.sh"
)
_REDIRECTION = " > /glmlogs/glm53.log 2>&1"
_EP = "--enable-expert-parallel"
_TOPOLOGY = {
    "--tensor-parallel-size": "tensor_parallel_size",
    "--nnodes": "nnodes",
    "--node-rank": "node_rank",
}
# Arity from the pinned launcher's SERVE_ARGS and its optional flags. Unknown
# options fail closed: otherwise an EP-looking token could be another option's
# missing value rather than a boolean flag. This is not a vLLM CLI validator.
_VALUE_OPTIONS = frozenset((
    "--served-model-name", "--host", "--port", "--gpu-memory-utilization",
    "--profiler-config", "--attention-backend", "--max-model-len",
    "--max-num-seqs", "--max-num-batched-tokens", "--block-size", "--moe-backend",
    "--load-format", "--speculative-config", "--kv-cache-dtype",
    "--num-gpu-blocks-override", "--kv-cache-memory-bytes", "--mamba-cache-dtype",
    "--scheduler-cls", "--max-cudagraph-capture-size", "--compilation-config",
    "--limit-mm-per-prompt", "--mm-encoder-tp-mode", "--mm-encoder-attn-backend",
    "--tool-call-parser", "--reasoning-parser", "--chat-template", "--middleware",
    "--distributed-executor-backend", "--master-addr", "--master-port",
))
_BOOLEAN_OPTIONS = frozenset((
    "--trust-remote-code", "--enable-prefix-caching", "--no-enable-prefix-caching",
    "--no-async-scheduling", "--enforce-eager", "--enable-flashinfer-autotune",
    "--skip-mm-profiling", "--enable-auto-tool-choice", "--disable-custom-all-reduce",
    "--headless",
))


def _digest(value):
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _argv_digest(argv):
    return _digest(json.dumps(argv, ensure_ascii=False, separators=(",", ":")))


def _literal_argv(line):
    """Reject shell evaluation while allowing literal single-quoted JSON."""
    quote = None
    escaped = False
    for char in line:
        if ord(char) < 32 or ord(char) == 127:
            raise ValueError("control character in serve command")
        if escaped:
            escaped = False
            continue
        if quote == "'":
            if char == "'":
                quote = None
            continue
        if char == "\\":
            escaped = True
            continue
        if char in ("$", "`"):
            raise ValueError("shell expansion in serve command")
        if quote == '"':
            if char == '"':
                quote = None
            continue
        if char in ("'", '"'):
            quote = char
        elif char in ";|&<>()#*?[]{}~":
            raise ValueError("shell syntax in serve command")
    if quote or escaped:
        raise ValueError("incomplete quoting in serve command")
    return shlex.split(line, comments=False, posix=True)


def launch_parallelism(cmd):
    """Return schema-1 configured-launch metadata from Docker Config.Cmd.

    Only the current base64 wrapper and exact known GID prelude are accepted.
    No command is executed. Unsupported or ambiguous inputs raise ValueError;
    absence of valid evidence never defaults to EP off. Returned hashes keep
    model paths, literal JSON and other argv values private. command_sha256
    hashes the decoded script; argv hashes use compact UTF-8 JSON arrays.
    This is not evidence that a process or kernel ran, or that ranks agree.
    """
    if not isinstance(cmd, list) or len(cmd) != 2 or cmd[0] != "-c" or not isinstance(cmd[1], str):
        raise ValueError("unsupported container command wrapper")
    match = _WRAPPER.fullmatch(cmd[1])
    if match is None:
        raise ValueError("unsupported container command wrapper")
    try:
        raw = base64.b64decode(match[1], validate=True)
        script = raw.decode("utf-8")
    except (binascii.Error, UnicodeDecodeError) as exc:
        raise ValueError("invalid launch script encoding") from exc
    if base64.b64encode(raw).decode("ascii") != match[1]:
        raise ValueError("noncanonical launch script encoding")
    if not script.startswith(_GID_PRELUDE):
        raise ValueError("unsupported launch script prelude")
    line = script[len(_GID_PRELUDE):]
    if line.endswith("\n"):
        line = line[:-1]
    if "\n" in line or not line.endswith(_REDIRECTION):
        raise ValueError("expected one serve command with the pinned log redirection")
    argv = _literal_argv(line[:-len(_REDIRECTION)])
    if len(argv) < 3 or argv[:2] != ["vllm", "serve"] or not argv[2] or argv[2].startswith("-"):
        raise ValueError("expected vllm serve and a literal model argument")
    topology = {}
    ep_index = None
    index = 3
    while index < len(argv):
        arg = argv[index]
        option, equals, value = arg.partition("=")
        if option == _EP:
            if equals or ep_index is not None:
                raise ValueError("ambiguous or duplicate expert-parallel flag")
            ep_index = index
        elif option in _TOPOLOGY or option in _VALUE_OPTIONS:
            if option in _TOPOLOGY and _TOPOLOGY[option] in topology:
                raise ValueError("duplicate topology argument")
            if not equals:
                index += 1
                value = argv[index] if index < len(argv) else ""
            if not value or value.startswith("--"):
                raise ValueError("missing or ambiguous option value")
            if option in _TOPOLOGY:
                if re.fullmatch(r"0|[1-9][0-9]*", value) is None:
                    raise ValueError("invalid topology value")
                topology[_TOPOLOGY[option]] = int(value)
        elif option not in _BOOLEAN_OPTIONS or equals:
            raise ValueError("unsupported or ambiguous serve argument")
        index += 1
    if set(topology) != set(_TOPOLOGY.values()):
        raise ValueError("incomplete launch topology")
    if topology["tensor_parallel_size"] < 1 or topology["nnodes"] < 1 or not 0 <= topology["node_rank"] < topology["nnodes"]:
        raise ValueError("invalid launch topology")
    return dict(schema=1, source="container-launch-script", enabled=ep_index is not None,
                **topology, command_sha256=_digest(script),
                prelude_sha256=_digest(_GID_PRELUDE),
                serve_argv_sha256=_argv_digest(argv),
                serve_argv_without_ep_sha256=_argv_digest(
                    argv if ep_index is None else argv[:ep_index] + argv[ep_index + 1:]))
