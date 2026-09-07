"""32차 item 5 -- the dev lab's API side: POST /glm53/lab {op, args, timeout}
-> engine_client.collective_rpc("glm53_lab", ...) on every rank. Added to
the server with `--middleware vllm.glm53_lab_middleware.lab` (the launcher
does that when VLLM_GLM53_DEV_LAB=1). Everything else passes through."""
from __future__ import annotations


async def lab(request, call_next):
    if request.url.path != "/glm53/lab":
        return await call_next(request)
    from fastapi.responses import JSONResponse

    try:
        body = await request.json()
        op = str(body.get("op", "info"))
        kw = body.get("args") or {}
        timeout = float(body.get("timeout", 900))
        client = request.app.state.engine_client
        res = await client.collective_rpc("glm53_lab", timeout=timeout, args=(op,), kwargs=kw)
        return JSONResponse({"op": op, "ranks": res})
    except Exception as e:  # the lab is a dev tool: say it, do not hide it
        return JSONResponse({"error": repr(e)}, status_code=500)
