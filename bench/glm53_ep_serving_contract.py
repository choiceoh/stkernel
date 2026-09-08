"""Pure configuration checks for a future actual-capacity EP B1/A/B2 run.

No shell, process, container or GPU is started here. These records describe
configured launches, not live capacity, kernel execution or performance. A
runner must additionally bind them to inspected container IDs and fresh logs,
attest image/source/model identity, preserve public recovery, and collect the
existing fresh-cache request and quality evidence.
"""
import base64
from decimal import Decimal, InvalidOperation
import hashlib
import json
import re

import glm53_launch_metadata as launch


NODES = ("10.10.10.2", "10.10.10.1", "10.10.10.3", "10.10.10.4")
EP_LOCAL = "VLLM_GLM53_EP_PREFILL_LOCAL"
EP_COMPACT = "VLLM_B12X_EP_COMPACT"
ARM_KNOBS = frozenset((EP_LOCAL, EP_COMPACT))
COMMON_KNOBS = {
    "VLLM_GLM53_PREFILL_SP": "1",
    "VLLM_GLM53_PREFILL_SP_FP8": "3",
    "VLLM_GLM53_PREFILL_SP_FP8_MIN_TOKENS": "4096",
}
_CAPACITY_INTS = ("max-model-len", "max-num-seqs", "max-num-batched-tokens",
                  "block-size", "num-gpu-blocks-override")


def _integer(value, name, *, minimum=1):
    if (not isinstance(value, str) or re.fullmatch(r"0|[1-9][0-9]*", value) is None
            or not minimum <= int(value) <= 2**31 - 1):
        raise ValueError("invalid " + name)
    return int(value)


def launch_capacity(cmd, *, public=False):
    """Inspect only a script already accepted by the strict launch parser.

    Require explicit block-pinned capacity; auto-sized and byte-sized caches
    need a different replay contract. Keep numeric spelling for the launcher's
    overrides, and never return the model path or the full command.
    """
    if type(public) is not bool:
        raise ValueError("explicit endpoint mode required")
    parallelism = launch.launch_parallelism(cmd)
    # This is extraction after full validation, not another shell parser.
    encoded = launch._WRAPPER.fullmatch(cmd[1])[1]
    script = base64.b64decode(encoded, validate=True).decode("utf-8")
    line = script[len(launch._GID_PRELUDE):].removesuffix("\n")
    argv = launch._literal_argv(line[:-len(launch._REDIRECTION)])
    options = {}
    index = 3
    while index < len(argv):
        option, equals, value = argv[index].partition("=")
        if option in launch._VALUE_OPTIONS or option in launch._TOPOLOGY:
            if not equals:
                index += 1
                value = argv[index]
        else:
            value = True
        # The pinned launcher legitimately appends a second middleware when
        # the development lab is enabled; all other options are singleton.
        if option in options and option != "--middleware":
            raise ValueError("duplicate serve option: " + option)
        options[option] = value
        index += 1
    capacity = {}
    for name in _CAPACITY_INTS:
        value = options.get("--" + name)
        _integer(value, name)
        capacity[name] = value
    if capacity["block-size"] != "2304" or "--kv-cache-memory-bytes" in options:
        raise ValueError("requires the launcher's 2304-token block-pinned cache")
    gmu = options.get("--gpu-memory-utilization")
    try:
        valid_gmu = isinstance(gmu, str) and Decimal(gmu).is_finite() and 0 < Decimal(gmu) < 1
    except InvalidOperation:
        valid_gmu = False
    if not valid_gmu:
        raise ValueError("invalid gpu-memory-utilization")
    capacity["gpu-memory-utilization"] = gmu
    eager = options.get("--enforce-eager", False)
    graph = options.get("--max-cudagraph-capture-size")
    if eager:
        if graph is not None or "--compilation-config" in options:
            raise ValueError("conflicting eager and graph controls")
    else:
        _integer(graph, "max-cudagraph-capture-size")
        if "--compilation-config" not in options:
            raise ValueError("missing graph compilation configuration")
    capacity.update(eager=eager, graph_cap=graph, compilation_config=options.get("--compilation-config"))
    prefix_flags = [key for key in ("--enable-prefix-caching", "--no-enable-prefix-caching") if key in options]
    if len(prefix_flags) != 1:
        raise ValueError("missing or conflicting prefix-cache flags")
    capacity["prefix_cache"] = prefix_flags[0] == "--enable-prefix-caching"
    endpoint = {name: options.get("--" + name) for name in ("host", "port")}
    expected_endpoint = {"host": "0.0.0.0", "port": "8000"} if public else {"host": "127.0.0.1", "port": "18000"}
    if endpoint != expected_endpoint:
        raise ValueError("unexpected configured endpoint")
    return dict(parallelism=parallelism, capacity=capacity, endpoint=endpoint)


def replay_controls(configured, kv_witness):
    """Preserve actual argv capacity using resolved launcher KV inputs.

    KV_TOKENS and KV_HYBRID_BLOCKS are not recoverable uniquely from argv.
    The caller must capture both resolved inputs from the original launch;
    this helper only checks their consistency with its observed block count.
    A KV_BLOCKS environment override would be overwritten by the launcher.
    """
    if not isinstance(kv_witness, dict) or set(kv_witness) != {"KV_TOKENS", "KV_HYBRID_BLOCKS"}:
        raise ValueError("require exactly both resolved launcher KV inputs")
    tokens = _integer(kv_witness["KV_TOKENS"], "KV_TOKENS")
    hybrid = _integer(kv_witness["KV_HYBRID_BLOCKS"], "KV_HYBRID_BLOCKS", minimum=0)
    capacity = configured["capacity"]
    blocks = (tokens + 2303) // 2304 + hybrid
    if blocks != int(capacity["num-gpu-blocks-override"]):
        raise ValueError("resolved KV inputs do not reproduce observed block count")
    controls = dict(kv_witness, KV_BYTES="auto", MAX_LEN=capacity["max-model-len"],
                    MAX_SEQS=capacity["max-num-seqs"], MAX_BATCHED=capacity["max-num-batched-tokens"],
                    GMU=capacity["gpu-memory-utilization"], CG_UTIL_DELTA="0",
                    EAGER=str(int(capacity["eager"])), PREFIX_CACHE=str(int(capacity["prefix_cache"])))
    if capacity["graph_cap"] is not None:
        controls["GRAPH_CAP"] = capacity["graph_cap"]
        compilation = capacity["compilation_config"]
        # ct_load_profile restores the profile's CUSTOM_OPS_AXIS=all even if
        # the caller unsets it. Empty is an active axis, not a neutral value.
        # Its existing substitution is byte-preserving only for this literal
        # token. Reject other layouts rather than rewriting a captured config.
        try:
            def unique(pairs):
                result = {}
                for key, value in pairs:
                    if key in result:
                        raise ValueError("duplicate compilation JSON key")
                    result[key] = value
                return result
            parsed = json.loads(compilation, object_pairs_hook=unique)
        except (TypeError, json.JSONDecodeError) as exc:
            raise ValueError("invalid compilation configuration") from exc
        if (not isinstance(parsed, dict) or parsed.get("custom_ops") != ["all"]
                or '"custom_ops":["all"]' not in compilation or "'" in compilation):
            raise ValueError("captured compilation configuration cannot be replayed unchanged by the launcher axis")
        controls.update(COMPILE_CFG=compilation, CUSTOM_OPS_AXIS="all", PIECEWISE="0")
    return controls


def configured_arm(nodes, *, enabled, expected_knobs, kv_witness):
    """Validate four supplied Docker Cmd/Env pairs; never query Docker.

    ENABLE_EP is proved by the actual boolean argv flag, not an environment
    value. Only the two explicit EP knobs can vary; every other environment
    value is hashed together for same-node comparison, including secrets.
    expected_knobs must state both values, using None for an absent variable.
    """
    if type(enabled) is not bool or not isinstance(nodes, dict) or set(nodes) != set(NODES):
        raise ValueError("require explicit EP state and exactly four nodes")
    if (not isinstance(expected_knobs, dict) or set(expected_knobs) != ARM_KNOBS
            or expected_knobs[EP_LOCAL] != str(int(enabled))
            or expected_knobs[EP_COMPACT] not in ("0", "1", None)
            or enabled and expected_knobs[EP_COMPACT] is None):
        raise ValueError("require explicitly allowlisted EP-local and compact settings")
    result = dict(schema=1, source="configured-ep-prefill-arm", enabled=enabled,
                  knobs=dict(expected_knobs), nodes={})
    for rank, node in enumerate(NODES):
        config = nodes[node]
        if not isinstance(config, dict):
            raise ValueError("invalid Docker configuration at " + node)
        record = launch_capacity(config.get("cmd"))
        parallelism = record["parallelism"]
        if (parallelism["enabled"] is not enabled or parallelism["tensor_parallel_size"] != 4
                or parallelism["nnodes"] != 4 or parallelism["node_rank"] != rank):
            raise ValueError("configured EP/TP/rank mismatch at " + node)
        values = config.get("env")
        if not isinstance(values, list):
            raise ValueError("missing Docker environment at " + node)
        env = {}
        for entry in values:
            if not isinstance(entry, str) or "=" not in entry:
                raise ValueError("invalid Docker environment entry")
            key, value = entry.split("=", 1)
            if not key or key in env:
                raise ValueError("empty or duplicate Docker environment key")
            env[key] = value
        for key, value in {**COMMON_KNOBS, **expected_knobs}.items():
            if env.get(key) != value:
                raise ValueError("configured setting mismatch: " + key + " at " + node)
        other_env = {key: value for key, value in env.items() if key not in ARM_KNOBS}
        record["other_env_sha256"] = hashlib.sha256(
            json.dumps(other_env, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
        controls = replay_controls(record, kv_witness)
        if rank and controls != result["controls"]:
            raise ValueError("ranks disagree on effective capacity")
        result["controls"] = controls
        result["nodes"][node] = record
    return result


def assert_matched_arms(arms, *, incoming):
    """Require B1/A/B2 configuration equality except the explicit EP changes.

    incoming contains the original public launch_capacity records for all four
    nodes. Check original capacity too: three newly reduced boots must not pass
    just because they agree with one another. This intentionally returns no
    timing or pass marker for live acceptance.
    Container identity, source, actual launches and fresh requests are outside
    this helper's scope and must not be inferred from its successful return.
    """
    try:
        if len(arms) != 3 or [arm["enabled"] for arm in arms] != [False, True, False]:
            raise ValueError("require B1/A/B2 in order")
        if set(incoming) != set(NODES):
            raise ValueError("require original public capacity from exactly four nodes")
        baseline = arms[0]
        for arm in arms:
            if (type(arm["enabled"]) is not bool or arm["schema"] != 1 or arm["source"] != "configured-ep-prefill-arm"
                    or set(arm["nodes"]) != set(NODES)):
                raise ValueError("invalid configured arm record")
            knobs = arm["knobs"]
            if (set(knobs) != ARM_KNOBS or knobs[EP_LOCAL] != str(int(arm["enabled"]))
                    or knobs[EP_COMPACT] not in ("0", "1", None)
                    or arm["enabled"] and knobs[EP_COMPACT] is None):
                raise ValueError("invalid configured arm settings")
            if arm["controls"] != baseline["controls"]:
                raise ValueError("capacity replay controls changed across arms")
            for rank, node in enumerate(NODES):
                record, reference = arm["nodes"][node], baseline["nodes"][node]
                original = incoming[node]
                if (original["endpoint"] != {"host": "0.0.0.0", "port": "8000"}
                        or original["parallelism"]["tensor_parallel_size"] != 4
                        or original["parallelism"]["nnodes"] != 4
                        or original["parallelism"]["node_rank"] != rank):
                    raise ValueError("invalid original public capacity evidence")
                if record["capacity"] != original["capacity"]:
                    raise ValueError("capacity changed from original public launch at " + node)
                parallelism = record["parallelism"]
                if (parallelism["schema"] != 1 or parallelism["source"] != "container-launch-script"
                        or parallelism["enabled"] is not arm["enabled"]
                        or parallelism["tensor_parallel_size"] != 4 or parallelism["nnodes"] != 4
                        or parallelism["node_rank"] != rank):
                    raise ValueError("inconsistent configured EP/TP/rank evidence")
                if record["endpoint"] != {"host": "127.0.0.1", "port": "18000"}:
                    raise ValueError("configured measurement endpoint is not private")
                for digest in (record["other_env_sha256"], *(parallelism[key] for key in (
                        "command_sha256", "prelude_sha256", "serve_argv_sha256", "serve_argv_without_ep_sha256"))):
                    if not isinstance(digest, str) or re.fullmatch("[a-f0-9]{64}", digest) is None:
                        raise ValueError("missing configured identity digest")
                for key in ("capacity", "endpoint", "other_env_sha256"):
                    if record[key] != reference[key]:
                        raise ValueError(key + " changed at " + node)
                for key in ("prelude_sha256", "serve_argv_without_ep_sha256"):
                    if record["parallelism"][key] != reference["parallelism"][key]:
                        raise ValueError("serve command changed beyond EP flag at " + node)
        if arms[2]["knobs"] != baseline["knobs"]:
            raise ValueError("B2 does not restore baseline EP settings")
    except (KeyError, TypeError, AttributeError) as exc:
        raise ValueError("incomplete configured arm evidence") from exc
