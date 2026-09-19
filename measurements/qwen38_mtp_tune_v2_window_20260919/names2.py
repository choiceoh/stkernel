#!/usr/bin/env python3
"""The writer model's names replaced with the people the served model will meet (operator 2026-09-19: "사용자 본명으로
김민준을 바꾸면 어떨까", then thirteen names of the operator's -- real people, so this copy reads them from MTP_NAMES and the repo holds none of them).

A writer model reuses a few names everywhere ("김민준" chairs meeting after meeting). Every Korean full name in a
conversation (and in each raw document) becomes one of the operator's thirteen, distinct within the conversation and the
same everywhere inside it (turns, thinking, tool arguments and results); past thirteen, a fresh name. English full names
that recur across the set get fresh ones the same way.

    names2.py count DIR                  the recurring names and how many conversations hold each
    names2.py apply DIR [MIN_CONVOS=5]   conversations.named.jsonl and raw.named.jsonl
"""
import json
import os
import random
import re
import sys
from collections import Counter

PEOPLE = os.environ.get("MTP_NAMES", "").split()      # the operator's thirteen, space separated; kept off the repo
if len(PEOPLE) < 2:
    sys.exit("names2.py: set MTP_NAMES to the names to use (the operator's list lives on the generating host)")
SURNAMES = "김이박최정강조윤장임한오서신권황안송류전홍고문양손배백허유남심노하곽성차주우구민진나지엄채원천방공현함변염여추도소석선설마길연위표명기반왕금옥육인맹제모"
DETECT_SURNAMES = "김이박최정강조윤장임한오서신권황안송류전홍"    # the twenty commonest: rarer surname syllables start common
#                                                               words ("인허가 담당", "제시된 기준") more often than names
TITLES = ("팀장|과장|대리|차장|부장|상무|전무|이사|사장|대표|연구원|매니저|주임|사원|선임|책임|수석|교수|박사|님|씨|변호사|세무사|회계사|"
          "기사|소장|센터장|본부장|실장|위원|의원|선생|원장|기자|엔지니어|담당")
KO_NAME_RAW = re.compile(rf"(?<![가-힣])([{DETECT_SURNAMES}][가-힣]{{2}})(?=\s?(?:{TITLES}))")
# words that start with a surname's syllable and sit before a title word, but are not names (the first pass renamed
# "마케팅 팀장" and "서비스 담당" -- the operator's data quality rule: no damage to the text for the sake of the names)
NOT_NAMES = {"마케팅", "반드시", "백엔드", "지자체", "서비스", "추가로", "문의를", "사용자", "관리자", "담당자", "개발자", "운영자",
             "전문가", "기술팀", "영업팀", "개발팀", "인사팀", "총무팀", "재무팀", "법무팀", "고객님", "대표님", "선생님", "교수님",
             "이사회", "위원회", "사업팀", "기획팀", "정부의", "한국의", "우리의", "이번에", "오늘의", "지금은", "모든분", "고객센",
             "안전관", "전기기", "시공사", "발전소", "주민분", "현장소", "구매팀", "품질팀", "생산팀", "연구소", "장비팀", "설계팀", "오늘이"}
PARTICLE_END = set("를을로의에도와과는은")


class _KoNames:
    """KO_NAME's findall: the raw pattern's matches that are names -- not a common word, not a word ending in a particle."""

    def findall(self, text):
        return [m for m in KO_NAME_RAW.findall(text) if m not in NOT_NAMES and m[-1] not in PARTICLE_END]


KO_NAME = _KoNames()
EN_FIRST = ("Sarah|John|Michael|David|Emily|Jennifer|James|Robert|Maria|Anna|Chris|Christopher|Daniel|Laura|Kevin|Rachel|Tom|"
            "Alex|Jessica|Mark|Lisa|Mike|Emma|Olivia|Liam|Noah|Sophia|Ethan|Mia|Lucas|Jane|Peter|Paul|Linda|Susan|Karen|Brian|"
            "Jason|Ryan|Megan|Nicole|Andrew|Steven|Amanda|Rebecca|Hannah|Grace")
EN_NAME = re.compile(rf"\b((?:{EN_FIRST})\s[A-Z][a-z]{{2,}})\b")
KO_GIVEN = ["민수", "지영", "서연", "도현", "하은", "준호", "수빈", "예린", "현우", "지민", "태윤", "은지", "성훈", "다은", "재원", "유나",
            "승민", "혜진", "동욱", "가영", "진우", "소희", "영호", "미경", "상철", "정아", "우진", "나래", "기현", "수정", "한별", "보람",
            "경민", "채원", "시우", "윤서", "지훈", "민재", "하준", "서윤", "건우", "아린", "정훈", "미진", "용석", "선영", "대성", "은비",
            "현아", "태호", "주원", "예준", "지호", "세희", "도윤", "하람", "창민", "소연", "형준", "유진"]
KO_SURNAME_POOL = list("김이박최정강조윤장임한오서신권황안송류전홍고문양손배백허유남심노")
EN_FIRST_POOL = ["Olivia", "Ethan", "Priya", "Marcus", "Chloe", "Daniel", "Aisha", "Tomás", "Hannah", "Kenji", "Sofia", "Liam",
                 "Grace", "Omar", "Elena", "Nathan", "Maya", "Victor", "Julia", "Samuel", "Leah", "Ivan", "Nora", "Felix"]
EN_LAST_POOL = ["Park", "Nguyen", "Patel", "Garcia", "Müller", "Rossi", "Kowalski", "Okafor", "Tanaka", "Silva", "Johansson",
                "Brooks", "Chen", "Dubois", "Haddad", "Ivanova", "Reyes", "Novak", "Walsh", "Fischer", "Kim", "Lee", "Choi", "Moreno"]


def texts_of(row):
    """Every string a conversation carries that a name can sit in."""
    for m in row["messages"]:
        for key in ("content", "reasoning_content"):
            if m.get(key):
                yield m[key]
        for call in m.get("tool_calls") or []:
            yield call["function"].get("arguments") or ""


def names_in(texts):
    ko, en = set(), set()
    for t in texts:
        ko.update(KO_NAME.findall(t))
        en.update(EN_NAME.findall(t))
    return ko, en


def load(path):
    return [json.loads(line) for line in open(path)]


def count(d):
    convos = load(f"{d}/conversations.jsonl")
    raw = load(f"{d}/raw.jsonl")
    ko, en = Counter(), Counter()
    for r in convos:
        k, e = names_in(texts_of(r))
        ko.update(k)
        en.update(e)
    for doc in raw:
        k, e = names_in([doc["text"]])
        ko.update(k)
        en.update(e)
    total = len(convos) + len(raw)
    print(json.dumps({"units": total, "units_with_a_korean_name": sum(1 for r in convos if names_in(texts_of(r))[0]),
                      "top_korean": ko.most_common(25), "top_english": en.most_common(15)}, ensure_ascii=False))
    return ko, en


def fresh_ko(rng, taken):
    while True:
        name = rng.choice(KO_SURNAME_POOL) + rng.choice(KO_GIVEN)
        if name not in taken:
            return name


def fresh_en(rng, taken):
    while True:
        name = rng.choice(EN_FIRST_POOL) + " " + rng.choice(EN_LAST_POOL)
        if name not in taken:
            return name


JOSA = {"이": ("이", "가"), "가": ("이", "가"), "은": ("은", "는"), "는": ("은", "는"), "을": ("을", "를"), "를": ("을", "를"),
        "과": ("과", "와"), "와": ("과", "와"), "으로": ("으로", "로"), "로": ("으로", "로"), "아": ("아", "야"), "야": ("아", "야")}


def batchim(syllable: str) -> int:
    """The final consonant's index of a Hangul syllable: 0 none, 8 ㄹ."""
    code = ord(syllable) - 0xAC00
    return code % 28 if 0 <= code < 11172 else 0


def rename(value, mapping):
    """Each old name to its new one, and the particle right after it agreeing with the new name's last syllable
    ("김민준이" -> "이영희가", "김민준을" -> "이영희를"; 으로/로 takes 로 after ㄹ)."""
    if not mapping:
        return value
    pattern = re.compile("(" + "|".join(re.escape(k) for k in sorted(mapping, key=len, reverse=True)) + ")"
                         r"(으로|로|이|가|은|는|을|를|과|와|아|야)?(?=[^가-힣]|$)")

    def sub(m):
        new, josa = mapping[m.group(1)], m.group(2)
        if not josa or not ("가" <= new[-1] <= "힣"):
            return new + (josa or "")
        final = batchim(new[-1])
        closed, open_ = JOSA[josa]
        if josa in ("으로", "로"):
            return new + ("로" if final in (0, 8) else "으로")
        return new + (open_ if final == 0 else closed)
    value = pattern.sub(sub, value)
    # a name followed by more Hangul (a longer ending, a compound): the name alone
    plain = re.compile("|".join(re.escape(k) for k in sorted(mapping, key=len, reverse=True)))
    return plain.sub(lambda m: mapping[m.group(0)], value)


def apply(d, min_convos):
    ko, en = count(d)
    recurring = {n for n, c in ko.items() if c >= min_convos} | {n for n, c in en.items() if c >= min_convos}
    rng = random.Random(20260919)
    stats = Counter()

    def mapping_for(ko_names, en_names):
        taken, mapping = set(ko_names) | set(en_names), {}
        people = [p for p in PEOPLE if p not in taken]
        rng.shuffle(people)
        for n in sorted(ko_names):                                 # every Korean name: the operator's people first
            if n in PEOPLE:
                continue
            mapping[n] = people.pop() if people else fresh_ko(rng, taken)
            taken.add(mapping[n])
            stats["korean names"] += 1
        for n in sorted(en_names & recurring):
            mapping[n] = fresh_en(rng, taken)
            taken.add(mapping[n])
        stats["renamed"] += bool(mapping)
        return mapping

    out = []
    for r in load(f"{d}/conversations.jsonl"):
        k, e = names_in(texts_of(r))
        m = mapping_for(k, e)
        for msg in r["messages"]:
            for key in ("content", "reasoning_content"):
                if msg.get(key):
                    msg[key] = rename(msg[key], m)
            for call in msg.get("tool_calls") or []:
                call["function"]["arguments"] = rename(call["function"].get("arguments") or "", m)
        out.append(r)
    with open(f"{d}/conversations.named.jsonl", "w") as fh:
        for r in out:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")
    docs = []
    for doc in load(f"{d}/raw.jsonl"):
        k, e = names_in([doc["text"]])
        doc["text"] = rename(doc["text"], mapping_for(k, e))
        docs.append(doc)
    with open(f"{d}/raw.named.jsonl", "w") as fh:
        for doc in docs:
            fh.write(json.dumps(doc, ensure_ascii=False) + "\n")
    print(json.dumps({"recurring_names": len(recurring), "units": len(out) + len(docs), **stats}, ensure_ascii=False))
    ko2, en2 = Counter(), Counter()
    for r in out:
        k, e = names_in(texts_of(r))
        ko2.update(k)
        en2.update(e)
    print(json.dumps({"after_top_korean": ko2.most_common(8), "after_top_english": en2.most_common(5)}, ensure_ascii=False))


if __name__ == "__main__":
    if sys.argv[1] == "count":
        count(sys.argv[2])
    else:
        apply(sys.argv[2], int(sys.argv[3]) if len(sys.argv) > 3 else 5)
