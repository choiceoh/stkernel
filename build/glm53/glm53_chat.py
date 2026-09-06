"""GLM Chat Completions option validation, before vLLM renders or parses.

ASGI middleware leaves response chunks and disconnects untouched. In particular,
it never promotes an unfinished reasoning response to final answer content.
"""
from __future__ import annotations

import json
from copy import deepcopy


class ChatContractError(ValueError):
    def __init__(self, param: str, message: str):
        super().__init__(message)
        self.param = param


def _required_first_schema(schema):
    """Reorder schema objects only; never traverse default/enum/example data."""
    if not isinstance(schema, dict):
        return
    props, required = schema.get("properties"), schema.get("required")
    if isinstance(props, dict) and isinstance(required, list):
        names = [name for name in required if isinstance(name, str) and name in props]
        schema["properties"] = {name: props[name] for name in [*names, *props]}
    for key in ("properties", "patternProperties", "$defs", "definitions", "dependentSchemas", "dependencies"):
        children = schema.get(key)
        if isinstance(children, dict):
            for child in children.values():
                _required_first_schema(child)
    for key in ("items", "additionalItems", "additionalProperties", "contains", "not",
                "if", "then", "else", "propertyNames", "unevaluatedProperties", "unevaluatedItems"):
        child = schema.get(key)
        if isinstance(child, list):
            for item in child:
                _required_first_schema(item)
        else:
            _required_first_schema(child)
    for key in ("allOf", "anyOf", "oneOf", "prefixItems"):
        children = schema.get(key)
        if isinstance(children, list):
            for child in children:
                _required_first_schema(child)


def normalize_chat_options(body: dict) -> dict:
    """Normalize explicit options only; preserve vLLM's server-side defaults."""
    raw = body.get("chat_template_kwargs")
    if raw is not None and not isinstance(raw, dict):
        raise ChatContractError("chat_template_kwargs", "Must be an object or null.")
    kwargs = dict(raw or {})
    for key in ("thinking", "enable_thinking", "clear_thinking", "legacy_reasoning_content"):
        if key in kwargs and type(kwargs[key]) is not bool:
            raise ChatContractError(f"chat_template_kwargs.{key}", "Must be a boolean.")
    if "thinking" in kwargs and "enable_thinking" in kwargs:
        if kwargs["thinking"] != kwargs["enable_thinking"]:
            raise ChatContractError("chat_template_kwargs", "thinking and enable_thinking must agree.")
    if "thinking" in kwargs or "enable_thinking" in kwargs:
        enabled = kwargs.get("thinking", kwargs.get("enable_thinking"))
        kwargs.update(thinking=enabled, enable_thinking=enabled)

    efforts = [(key, value) for key, value in (
        ("reasoning_effort", body.get("reasoning_effort")),
        ("chat_template_kwargs.reasoning_effort", kwargs.get("reasoning_effort")),
    ) if value is not None]
    for key, value in efforts:
        if value not in ("low", "high", "max"):
            raise ChatContractError(key, "GLM reasoning_effort must be low, high, or max.")
    if len(efforts) == 2 and efforts[0][1] != efforts[1][1]:
        raise ChatContractError("reasoning_effort", "Top-level and template reasoning_effort must agree.")
    result = dict(body)
    # Experimental prompt ordering, before the renderer sees tools. Default off;
    # remove the extension before vLLM request validation. Schema semantics and
    # the caller's payload stay unchanged; tool_choice is never downgraded.
    required_first = result.pop("glm53_required_first", False)
    if type(required_first) is not bool:
        raise ChatContractError("glm53_required_first", "Must be a boolean.")
    if required_first and isinstance(body.get("tools"), list):
        result["tools"] = deepcopy(body["tools"])
        for tool in result["tools"]:
            if isinstance(tool, dict) and isinstance(tool.get("function"), dict):
                _required_first_schema(tool["function"].get("parameters"))
    if efforts:
        result["reasoning_effort"] = kwargs["reasoning_effort"] = efforts[0][1]
    if raw is not None or kwargs:
        result["chat_template_kwargs"] = kwargs
    return result


class ChatContractMiddleware:
    def __init__(self, app):
        self.app = app

    async def __call__(self, scope, receive, send):
        if (scope["type"] != "http" or scope.get("method") != "POST"
                or scope.get("path", "").rstrip("/") != "/v1/chat/completions"):
            return await self.app(scope, receive, send)

        chunks = []
        while True:
            message = await receive()
            if message["type"] == "http.disconnect":
                return
            chunks.append(message.get("body", b""))
            if not message.get("more_body", False):
                break
        try:
            body = json.loads(b"".join(chunks))
            if not isinstance(body, dict):
                raise ChatContractError("body", "Must be a JSON object.")
            encoded = json.dumps(normalize_chat_options(body), ensure_ascii=False,
                                 allow_nan=False, separators=(",", ":")).encode()
        except (ValueError, UnicodeError) as exc:
            payload = json.dumps({"error": {
                "message": str(exc), "type": "invalid_request_error",
                "param": getattr(exc, "param", "body"), "code": 400,
            }}).encode()
            await send({"type": "http.response.start", "status": 400,
                        "headers": [(b"content-type", b"application/json"),
                                    (b"content-length", str(len(payload)).encode())]})
            await send({"type": "http.response.body", "body": payload})
            return

        scope = dict(scope)
        scope["headers"] = [(k, v) for k, v in scope.get("headers", [])
                            if k.lower() != b"content-length"]
        scope["headers"].append((b"content-length", str(len(encoded)).encode()))
        delivered = False

        async def normalized_receive():
            nonlocal delivered
            if not delivered:
                delivered = True
                return {"type": "http.request", "body": encoded, "more_body": False}
            return await receive()

        await self.app(scope, normalized_receive, send)
