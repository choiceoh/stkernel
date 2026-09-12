#!/usr/bin/env python3
"""Can a masked generation write Korean? Runs inside the ST image; no server, no GPU (45차 §41).

xgrammar compiles a regex character class into UTF-8 byte branches. A range that ENDS inside
the 0xED branch -- U+D000-U+D7FF, the lead byte whose continuations are cut short by the
surrogate hole -- loses everything in that branch but its exact endpoint. Hangul is the only
common script that straddles it, so `[가-힣]` admits 이 and 힣 and refuses 타 파 하 한 해 호 후.

  A  a plain string value: every script may be written
  B  maxLength / minLength count characters, not bytes
  C  an enum of Korean values
  D  `[가-힣]` before the door repairs it -- this is the one that fails on stock xgrammar
  E  the repaired form: the same set of characters, compiled correctly

  usage: docker exec -i <container> python3 - < probes/st_grammar_korean.py [--ckpt PATH]
         (ST_REPO points at the engine tree to gate; /repo, the released one, by default)
"""
from __future__ import annotations

import argparse
import json
import os
import sys

CKPT = "/home/choiceoh/models/glm53-redhat-nvfp4"
PREFIX = '{"v": "'


def gate(compiler, xgr, tok, vocab, pattern_or_schema, samples, *, schema=False):
    """Which of `samples` may open the string value: 'Y' allowed, '.' refused."""
    body = pattern_or_schema if schema else {"type": "object", "required": ["v"],
                                             "properties": {"v": {"type": "string", "pattern": pattern_or_schema}}}
    grammar = compiler.compile_json_schema(json.dumps(body))
    out = []
    for text in samples:
        matcher = xgr.GrammarMatcher(grammar)
        mask = xgr.allocate_token_bitmask(1, vocab)
        ok = True
        for tid in tok(PREFIX + text, add_special_tokens=False)["input_ids"]:
            matcher.fill_next_token_bitmask(mask)
            if not (mask[0, tid >> 5].item() >> (tid & 31)) & 1:
                ok = False
                break
            matcher.accept_token(tid)
        out.append("Y" if ok else ".")
    return "".join(out)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default=os.environ.get("ST_CKPT", CKPT))
    a = ap.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    import xgrammar as xgr
    from transformers import AutoTokenizer

    sys.path.insert(0, os.environ.get("ST_REPO", "/repo"))   # ST_REPO=/tmp/... to gate a worktree
    from engine.base.serve import split_surrogate_branch

    tok = AutoTokenizer.from_pretrained(a.ckpt, local_files_only=True)
    vocab = len(tok)
    info = xgr.TokenizerInfo.from_huggingface(tok, vocab_size=vocab)
    compiler = xgr.GrammarCompiler(info)
    print(f"vocab_type={info.vocab_type} vocab={vocab}")

    korean = ["이", "하", "한", "타", "힣"]
    rows, failed = [], []

    def row(name, got, want):
        rows.append((name, got, want))
        if got != want:
            failed.append(name)

    plain = {"type": "object", "required": ["v"], "properties": {"v": {"type": "string"}}}
    row("A plain string", gate(compiler, xgr, tok, vocab, plain, korean + ["漢", "a", "🙂"], schema=True), "YYYYYYYY")
    long5 = {"type": "object", "required": ["v"], "properties": {"v": {"type": "string", "maxLength": 5}}}
    row("B maxLength 5 counts characters",
        gate(compiler, xgr, tok, vocab, long5, ["안녕하세요", "안녕하세요요"], schema=True), "Y.")
    enum = {"type": "object", "required": ["v"], "properties": {"v": {"enum": ["예", "아니오"]}}}
    row("C enum of Korean", gate(compiler, xgr, tok, vocab, enum, ["예", "아니오", "네"], schema=True), "YY.")
    row("D [가-힣] as written (stock xgrammar drops the 0xED branch)",
        gate(compiler, xgr, tok, vocab, "^[가-힣]+$", korean), "Y...Y")
    row("E [가-힣] as the door sends it",
        gate(compiler, xgr, tok, vocab, split_surrogate_branch("^[가-힣]+$"), korean), "YYYYY")

    width = max(len(n) for n, _, _ in rows)
    for name, got, want in rows:
        print(f"  {name:<{width}}  {got}  {'ok' if got == want else f'MISMATCH (wanted {want})'}")
    print(("FAIL: " + ", ".join(failed)) if failed else "PASS")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
