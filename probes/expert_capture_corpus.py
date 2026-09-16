"""The expert capture's calibration corpus (engine/profiles/glm53/capture.py): chat conversations from text already on
the fleet, mixed by source, each tagged fit or held-out by document, with its token count under the served template.

Sources, none downloaded for this:
  repo_md    this repository's Markdown (Korean and English technical prose), a document a conversation
  repo_code  its Python, shell and CUDA sources, a file a conversation
  onepass    the onepass workload documents and the model's own distinct responses to them (reasoning and answers)
  gsm8k      openai/gsm8k train, already in the host's datasets cache: multi-turn worked problems
  kmmlu      HAERAE-HUB/KMMLU (Law) train, already cached: multi-turn Korean exam questions
Each source is filled to its token budget; every fifth document of a source is held out. Rows are interleaved by
source so a capture cut short still holds every source.

    python3 probes/expert_capture_corpus.py --repo DIR --onepass onepass.jsonl --gsm8k gsm8k-train.arrow \
        --kmmlu kmmlu-train.arrow --ckpt /home/choiceoh/models/st-glm53-nvidia-tp4-9391 --out corpus.jsonl
"""
from __future__ import annotations

import argparse
import json
import random
import struct
from pathlib import Path

BUDGET = dict(repo_md=90_000, repo_code=90_000, onepass=110_000, gsm8k=60_000, kmmlu=50_000)
MAX_TOKENS = 30_000              # a conversation stays inside one 32,256-token prefill chunk and the capture's KV horizon
HELD_OUT_EVERY = 5


# -- Arrow IPC stream, the subset a datasets cache file uses (flat columns, no compression) ---------------------------
class _Table:
    def __init__(self, buf: bytes, pos: int):
        self.buf, self.pos = buf, pos
        vt = pos - struct.unpack_from("<i", buf, pos)[0]
        self.vsize = struct.unpack_from("<H", buf, vt)[0]
        self.vt = vt

    def _field(self, i):
        o = 4 + 2 * i
        return struct.unpack_from("<H", self.buf, self.vt + o)[0] if o < self.vsize else 0

    def scalar(self, i, fmt, default=0):
        off = self._field(i)
        return struct.unpack_from(fmt, self.buf, self.pos + off)[0] if off else default

    def _ref(self, i):
        off = self._field(i)
        if not off:
            return None
        at = self.pos + off
        return at + struct.unpack_from("<I", self.buf, at)[0]

    def table(self, i):
        at = self._ref(i)
        return None if at is None else _Table(self.buf, at)

    def string(self, i):
        at = self._ref(i)
        if at is None:
            return None
        n = struct.unpack_from("<I", self.buf, at)[0]
        return self.buf[at + 4: at + 4 + n].decode()

    def tables(self, i):
        at = self._ref(i)
        if at is None:
            return []
        n = struct.unpack_from("<I", self.buf, at)[0]
        return [_Table(self.buf, at + 4 + 4 * k + struct.unpack_from("<I", self.buf, at + 4 + 4 * k)[0]) for k in range(n)]

    def structs(self, i, size):
        at = self._ref(i)
        if at is None:
            return []
        n = struct.unpack_from("<I", self.buf, at)[0]
        return [self.buf[at + 4 + size * k: at + 4 + size * (k + 1)] for k in range(n)]


UTF8, LARGE_UTF8, INT, FLOAT, BOOL = 5, 20, 2, 3, 6


def read_arrow(path) -> "list[dict]":
    """Rows of a flat Arrow IPC stream file (utf8, int, float and bool columns)."""
    data = Path(path).read_bytes()
    pos, fields, rows = 0, None, []
    while pos + 8 <= len(data):
        n = struct.unpack_from("<i", data, pos)[0]
        pos += 4
        if n == -1:                                   # the continuation token, then the metadata length
            n = struct.unpack_from("<i", data, pos)[0]
            pos += 4
        if n <= 0:
            break
        meta = data[pos: pos + n]
        pos += n
        pos += -pos % 8
        root = _Table(meta, struct.unpack_from("<I", meta, 0)[0])
        kind, body_len = root.scalar(1, "<B"), root.scalar(3, "<q")
        body = data[pos: pos + body_len]
        pos += body_len
        header = root.table(2)
        if kind == 1:                                 # Schema
            fields = []
            for f in header.tables(1):
                t = f.table(3)
                bits = t.scalar(0, "<i") if f.scalar(2, "<B") == INT else t.scalar(0, "<h") if f.scalar(2, "<B") == FLOAT else 0
                fields.append((f.string(0), f.scalar(2, "<B"), bits))
        elif kind == 3:                               # RecordBatch
            length = header.scalar(0, "<q")
            if header.table(3) is not None:
                raise ValueError(f"{path}: compressed record batches are not read here")
            buffers = [struct.unpack("<qq", b) for b in header.structs(2, 16)]
            columns, b = {}, 0
            for name, kind_, bits in fields:
                if kind_ in (UTF8, LARGE_UTF8):
                    (_, _), (oo, ol), (do, dl) = buffers[b], buffers[b + 1], buffers[b + 2]
                    b += 3
                    width, fmt = (4, "<i") if kind_ == UTF8 else (8, "<q")
                    offsets = struct.unpack_from(f"<{length + 1}{fmt[1]}", body, oo)
                    blob = body[do: do + dl]
                    columns[name] = [blob[offsets[k]: offsets[k + 1]].decode() for k in range(length)]
                elif kind_ in (INT, FLOAT):
                    (_, _), (vo, vl) = buffers[b], buffers[b + 1]
                    b += 2
                    code = {(INT, 8): "b", (INT, 16): "h", (INT, 32): "i", (INT, 64): "q",
                            (FLOAT, 0): "e", (FLOAT, 1): "f", (FLOAT, 2): "d"}[(kind_, bits)]
                    columns[name] = list(struct.unpack_from(f"<{length}{code}", body, vo))
                elif kind_ == BOOL:
                    (_, _), (vo, vl) = buffers[b], buffers[b + 1]
                    b += 2
                    columns[name] = [bool(body[vo + k // 8] >> (k % 8) & 1) for k in range(length)]
                else:
                    raise ValueError(f"{path}: column {name} has Arrow type {kind_}, not read here")
            rows.extend(dict(zip(columns, values)) for values in zip(*columns.values()))
    return rows


# -- conversations -----------------------------------------------------------------------------------------------------
def repo_documents(repo: Path, suffixes, rng, max_chars):
    files = sorted(p for p in repo.rglob("*") if p.is_file() and p.suffix in suffixes
                   and not any(part.startswith(".") or part in ("__pycache__", "fixtures") for part in p.relative_to(repo).parts))
    rng.shuffle(files)
    for path in files:
        try:
            text = path.read_text()
        except (UnicodeDecodeError, OSError):
            continue
        if len(text) < 400:
            continue
        rel = str(path.relative_to(repo))
        for start in range(0, len(text), max_chars):
            yield rel, text[start: start + max_chars]


def conversations(args, rng):
    repo = Path(args.repo)
    md = (dict(source="repo_md", messages=[dict(role="user", content=f"다음 문서({rel})를 읽고 핵심 내용과 근거를 정리해줘.\n\n{text}")])
          for rel, text in repo_documents(repo, {".md"}, rng, 24_000))
    code = (dict(source="repo_code", messages=[dict(role="user", content=f"Review this file ({rel}). What does it do, and what could go wrong?\n\n```\n{text}\n```")])
            for rel, text in repo_documents(repo, {".py", ".sh", ".cu", ".c", ".h", ".cpp", ".jinja"}, rng, 40_000))

    def onepass():
        rows = [json.loads(line) for line in open(args.onepass)]
        workloads = [r for r in rows if r["kind"] == "workload" and len(r["content"] or "") < 60_000]
        responses = [r for r in rows if r["kind"] == "response"]
        rng.shuffle(workloads)
        rng.shuffle(responses)
        by_question = {(w["ctx"], str(w["question"])): w for w in workloads}
        for i, r in enumerate(responses):
            if i < len(workloads):                    # the documents themselves, once each, between responses
                yield dict(source="onepass", messages=[dict(role="user", content=workloads[i]["content"])])
            w = by_question.get((r["ctx"], str(r["question"])))
            ask = w["content"] if w is not None and len(w["content"]) < 8_000 else "앞의 문서에 대한 질문에 답해줘."
            answer = (r["reasoning"] + "\n\n" + r["answer"]).strip()[:48_000]
            yield dict(source="onepass", messages=[dict(role="user", content=ask), dict(role="assistant", content=answer),
                                                   dict(role="user", content="결론만 한 문단으로 다시 정리해줘.")])

    def gsm8k():
        rows = read_arrow(args.gsm8k)
        rng.shuffle(rows)
        for k in range(0, len(rows) - 11, 11):
            group = rows[k: k + 11]
            messages = []
            for r in group[:-1]:
                messages += [dict(role="user", content=r["question"]), dict(role="assistant", content=r["answer"])]
            messages.append(dict(role="user", content=group[-1]["question"]))
            yield dict(source="gsm8k", messages=messages)

    def kmmlu():
        rows = [r for path in args.kmmlu for r in read_arrow(path)]
        rng.shuffle(rows)
        for k in range(0, len(rows) - 9, 9):
            group = rows[k: k + 9]
            messages = []
            for r in group:
                q = f"{r['question']}\nA. {r['A']}\nB. {r['B']}\nC. {r['C']}\nD. {r['D']}\n정답을 고르고 이유를 설명해줘."
                messages.append(dict(role="user", content=q))
                if r is not group[-1]:
                    messages.append(dict(role="assistant", content=f"정답은 {'ABCD'[int(r['answer']) - 1]}입니다."))
            yield dict(source="kmmlu", messages=messages)

    return dict(repo_md=md, repo_code=code, onepass=onepass(), gsm8k=gsm8k(), kmmlu=kmmlu())


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--repo", required=True)
    ap.add_argument("--onepass", required=True)
    ap.add_argument("--gsm8k", required=True)
    ap.add_argument("--kmmlu", required=True, nargs="+", help="KMMLU train splits only: the test splits are what a quality run reads")
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--template", default=None, help="the served chat template (default: <repo>/launchers/chat_template_mm_v2.jinja)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--seed", type=int, default=20260913)
    ap.add_argument("--scale", type=float, default=1.0, help="multiply every source's token budget")
    args = ap.parse_args()
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(args.ckpt)
    template = Path(args.template or Path(args.repo) / "launchers/chat_template_mm_v2.jinja").read_text()
    rng = random.Random(args.seed)
    streams = conversations(args, rng)
    filled = {k: 0 for k in streams}
    chosen = {k: [] for k in streams}
    for source, stream in streams.items():
        budget = BUDGET[source] * args.scale
        for conv in stream:
            if filled[source] >= budget:
                break
            ids = tok.apply_chat_template(conv["messages"], chat_template=template, add_generation_prompt=True, tokenize=True)
            if hasattr(ids, "keys"):
                ids = ids["input_ids"]
            n = len(ids)
            if n > MAX_TOKENS or n < 64:
                continue
            conv["tokens"] = n
            conv["split"] = "heldout" if len(chosen[source]) % HELD_OUT_EVERY == HELD_OUT_EVERY - 1 else "fit"
            chosen[source].append(conv)
            filled[source] += n
    order = []
    while any(chosen.values()):
        for source in list(chosen):
            if chosen[source]:
                order.append(chosen[source].pop(0))
    with open(args.out, "w") as f:
        for i, conv in enumerate(order):
            f.write(json.dumps(dict(i=i, **conv), ensure_ascii=False) + "\n")
    summary = {}
    for conv in order:
        row = summary.setdefault(conv["source"], dict(documents=0, tokens=0, heldout_tokens=0))
        row["documents"] += 1
        row["tokens"] += conv["tokens"]
        row["heldout_tokens"] += conv["tokens"] if conv["split"] == "heldout" else 0
    print(json.dumps(dict(out=args.out, documents=len(order), tokens=sum(c["tokens"] for c in order), sources=summary), indent=1))


if __name__ == "__main__":
    main()
