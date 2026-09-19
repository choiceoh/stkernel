"""A diverse chat prompt set for the MTP head's fine-tuning window: Korean and English, explanations, code, math,
reasoning, summaries of given text, translation, writing, structured output; some with thinking on. The held-out eval
prompts are drawn first and never used for data. Deterministic (seeded)."""
import json
import random
import sys

rng = random.Random(20260919)

TOPICS_KO = ["태양광 발전의 원리", "풍력 발전과 태양광의 차이", "전력망의 주파수 조정", "배터리 에너지 저장 장치", "인플레이션",
             "금리와 환율의 관계", "광합성", "블록체인", "양자 컴퓨터", "기후 변화의 원인", "조선 시대의 과거 제도", "한글 창제",
             "뉴턴의 운동 법칙", "상대성 이론", "백신의 작동 원리", "머신러닝과 딥러닝의 차이", "트랜스포머 모델", "데이터베이스 인덱스",
             "HTTP와 HTTPS", "운영체제의 스케줄러", "커피의 역사", "김치 발효", "마라톤 훈련", "수면의 과학", "화산 폭발", "지진의 규모",
             "주식과 채권", "부동산 계약 시 주의점", "재생에너지 보조금 제도", "탄소 배출권 거래제", "전기차 충전 인프라", "반도체 공정",
             "GPU와 CPU의 차이", "인터넷 라우팅", "암호화의 기초", "게임 이론", "행동경제학", "고대 로마의 멸망", "르네상스", "산업혁명"]
TOPICS_EN = ["how solar panels work", "grid-scale battery storage", "the causes of inflation", "photosynthesis",
             "how vaccines train the immune system", "the transformer architecture", "database indexing", "TCP vs UDP",
             "garbage collection in managed runtimes", "the history of the printing press", "plate tectonics",
             "compound interest", "the prisoner's dilemma", "how GPS works", "the water cycle", "black holes",
             "public-key cryptography", "the French Revolution", "how neural networks learn", "supply and demand",
             "the Krebs cycle", "why the sky is blue", "how wind turbines generate power", "carbon capture", "CRISPR",
             "the difference between weather and climate", "container orchestration", "consensus algorithms",
             "the Big O notation", "how compilers optimize code"]
EXPLAIN_KO = ["{t}에 대해 초보자도 이해할 수 있게 설명해 주세요.", "{t}을(를) 세 문단으로 설명해 주세요.",
              "{t}의 장점과 단점을 표로 정리해 주세요.", "{t}에 대해 자주 묻는 질문 다섯 개와 답을 써 주세요.",
              "{t}을(를) 중학생에게 비유를 들어 설명해 주세요.", "{t}에 관한 핵심 개념 다섯 가지를 bullet로 정리해 주세요."]
EXPLAIN_EN = ["Explain {t} to a beginner.", "Write a three-paragraph explanation of {t}.", "What are common misconceptions about {t}?",
              "Summarize {t} in five bullet points, then give one concrete example.", "Compare {t} with a related idea and explain the differences.",
              "Write a short FAQ (five questions) about {t}."]
CODE = ["Write a Python function that {c}. Include a docstring and two example calls.",
        "Write a JavaScript function that {c}, with comments.", "Implement {c} in Rust and explain the ownership choices.",
        "파이썬으로 {c_ko} 함수를 작성하고 시간 복잡도를 설명해 주세요.", "Write a SQL query that {s}. Explain each clause.",
        "Write a bash script that {b}.", "Review this code and point out bugs:\n```python\n{snippet}\n```"]
CODE_TASKS = [("checks whether a string is a palindrome", "문자열이 회문인지 확인하는"), ("merges two sorted lists", "정렬된 두 리스트를 병합하는"),
              ("computes the n-th Fibonacci number iteratively", "n번째 피보나치 수를 반복문으로 계산하는"),
              ("parses a CSV line with quoted fields", "따옴표가 있는 CSV 한 줄을 파싱하는"),
              ("finds the longest common prefix of a list of strings", "문자열 리스트의 최장 공통 접두사를 찾는"),
              ("groups words that are anagrams", "애너그램끼리 묶는"), ("implements binary search", "이진 탐색을 구현하는"),
              ("validates an email address with a simple rule", "간단한 규칙으로 이메일 주소를 검증하는"),
              ("rate-limits calls with a token bucket", "토큰 버킷으로 호출 빈도를 제한하는"),
              ("computes a moving average over a stream", "스트림의 이동 평균을 계산하는"),
              ("deduplicates records by a key while keeping order", "순서를 유지하면서 키 기준으로 중복을 제거하는"),
              ("converts Roman numerals to integers", "로마 숫자를 정수로 바꾸는")]
SQL = ["returns the top five customers by total order amount in 2025", "counts daily active users for the last 30 days",
       "finds products that were never ordered", "computes month-over-month revenue growth"]
BASH = ["backs up a directory into a dated tar.gz and keeps the last seven", "finds the ten largest files under a directory",
        "watches a log file and prints lines containing ERROR with a timestamp", "renames all .jpeg files to .jpg recursively"]
SNIPPETS = ["def avg(xs):\n    return sum(xs) / len(xs)\n\nprint(avg([]))",
            "def fib(n):\n    if n <= 1: return n\n    return fib(n-1) + fib(n-2)\n\nprint([fib(i) for i in range(40)])",
            "items = [1, 2, 3]\nfor i in range(len(items)):\n    if items[i] % 2 == 0:\n        items.remove(items[i])",
            "import threading\ncount = 0\ndef inc():\n    global count\n    for _ in range(100000): count += 1\nts=[threading.Thread(target=inc) for _ in range(4)]\n[t.start() for t in ts]; [t.join() for t in ts]; print(count)"]
MATH = ["A train leaves at {h}:{m:02d} and travels {d} km at {v} km/h. When does it arrive? Show your steps.",
        "{a}개의 사과를 {b}명이 똑같이 나누면 한 사람당 몇 개이고 몇 개가 남나요? 풀이 과정을 보여 주세요.",
        "What is {a} × {b} − {c}? Explain the calculation.", "Solve for x: {a}x + {b} = {c}. Show each step.",
        "연 이율 {r}%로 {p}만 원을 {n}년 동안 복리로 예치하면 얼마가 되나요? 계산 과정을 보여 주세요.",
        "A rectangle has perimeter {p} cm and one side {a} cm. What is its area?"]
REASON = ["Three friends — Ana, Ben and Chloe — each own a different pet (cat, dog, fish). Ana doesn't own the dog, Ben owns the fish. Who owns what? Explain.",
          "If all bloops are razzies and some razzies are lazzies, must some bloops be lazzies? Explain your reasoning.",
          "철수는 영희보다 키가 크고, 영희는 민수보다 크다. 민수는 지훈보다 크다. 가장 작은 사람은 누구인가? 이유를 설명하라.",
          "A bat and a ball cost $1.10 in total. The bat costs $1.00 more than the ball. How much is the ball? Explain carefully.",
          "You have a 3-liter jug and a 5-liter jug. How do you measure exactly 4 liters? List the steps.",
          "어떤 수에 3을 곱하고 7을 더하면 34가 된다. 그 수는? 풀이를 보여라."]
WRITE_KO = ["{t}을(를) 주제로 짧은 시를 써 주세요.", "{t}에 관한 블로그 글의 도입부를 써 주세요.",
            "고객에게 {t} 관련 서비스 점검을 안내하는 공지문을 써 주세요.", "{t}을(를) 주제로 한 짧은 이야기를 써 주세요."]
WRITE_EN = ["Write a short poem about {t}.", "Write the opening paragraph of a blog post about {t}.",
            "Write a polite email to a colleague asking for feedback on a report about {t}.", "Write a short story that involves {t}."]
TRANSLATE = [("Translate into Korean: ", ["The meeting has been moved to Thursday afternoon because the client requested more time.",
                                          "Renewable energy installations grew faster than expected last year.",
                                          "Please make sure every document is signed before submitting the application."]),
             ("다음 문장을 영어로 번역해 주세요: ", ["이번 분기 발전량은 일사량 증가로 전년 대비 12% 늘었습니다.",
                                           "계약서 초안을 검토한 뒤 수정 사항을 금요일까지 보내 주세요.",
                                           "새 인버터는 설치가 간단하고 효율이 높습니다."])]
JSON_TASKS = ["다음 정보를 JSON으로 정리해 주세요: 이름 김민수, 나이 34, 직업 엔지니어, 취미 등산과 사진.",
              "Return a JSON object describing three fictional products with fields name, price, and tags.",
              "Extract the dates and amounts from this text as a JSON list: 'Paid 120,000 KRW on 2026-03-02 and 45,500 KRW on 2026-03-15.'",
              "태양광 발전소 세 곳의 이름, 용량(MW), 지역을 가진 JSON 배열을 만들어 주세요."]
SUMMARY_SRC = ["태양광 발전소의 인허가 절차는 발전사업허가, 개발행위허가, 환경영향평가 협의, 계통연계 신청, 공사계획 신고, 사용전검사의 순서로 "
               "진행되며 각 단계마다 관할 기관과 제출 서류, 처리 기간이 다릅니다. 풍력은 여기에 해상교통안전진단과 군 전파영향 협의가 더해집니다. ",
               "The committee reviewed the budget proposal in detail, noting that infrastructure costs rose by eleven percent while "
               "operating expenses stayed flat. Several members asked for a phased approach that would defer part of the capital "
               "spending to the next fiscal year, and the chair agreed to circulate a revised draft within two weeks. "]


def explain():
    if rng.random() < 0.5:
        return rng.choice(EXPLAIN_KO).format(t=rng.choice(TOPICS_KO))
    return rng.choice(EXPLAIN_EN).format(t=rng.choice(TOPICS_EN))


def code():
    template = rng.choice(CODE)
    c, c_ko = rng.choice(CODE_TASKS)
    return template.format(c=c, c_ko=c_ko, s=rng.choice(SQL), b=rng.choice(BASH), snippet=rng.choice(SNIPPETS))


def math_():
    return rng.choice(MATH).format(h=rng.randint(5, 20), m=rng.randint(0, 59), d=rng.randint(40, 600), v=rng.choice([60, 80, 90, 120]),
                                   a=rng.randint(3, 97), b=rng.randint(2, 40), c=rng.randint(1, 500), r=rng.choice([2, 3, 3.5, 4, 5]),
                                   p=rng.choice([100, 500, 1000, 3000]), n=rng.randint(2, 10))


def write():
    if rng.random() < 0.5:
        return rng.choice(WRITE_KO).format(t=rng.choice(TOPICS_KO))
    return rng.choice(WRITE_EN).format(t=rng.choice(TOPICS_EN))


def translate():
    prefix, sentences = rng.choice(TRANSLATE)
    return prefix + rng.choice(sentences)


def summary():
    src = rng.choice(SUMMARY_SRC) * rng.randint(4, 30)
    ask = rng.choice(["다음 글을 세 문장으로 요약해 주세요.\n\n", "Summarize the following text in three sentences.\n\n",
                      "다음 글의 핵심을 bullet 다섯 개로 정리해 주세요.\n\n"])
    return ask + src


KINDS = [("explain", explain, 0.28, 384), ("code", code, 0.2, 512), ("math", math_, 0.12, 384), ("reason", lambda: rng.choice(REASON), 0.06, 512),
         ("write", write, 0.12, 384), ("translate", translate, 0.06, 128), ("json", lambda: rng.choice(JSON_TASKS), 0.06, 256),
         ("summary", summary, 0.1, 256)]


def draw(i):
    r, acc = rng.random(), 0.0
    for kind, make, weight, max_tokens in KINDS:
        acc += weight
        if r <= acc:
            break
    thinking = kind in ("math", "reason", "code") and rng.random() < 0.35
    return {"id": f"{kind}-{i:04d}", "kind": kind, "content": make(), "max_tokens": max_tokens * (3 if thinking else 1),
            "thinking": thinking, "temperature": 0.0 if rng.random() < 0.35 else 0.7}


prompts = [draw(i) for i in range(int(sys.argv[1]) if len(sys.argv) > 1 else 700)]
seen, unique = set(), []
for p in prompts:
    if p["content"] not in seen:
        seen.add(p["content"])
        unique.append(p)
evals = []
for p in unique:
    if len(evals) < 40 and not p["thinking"] and p["kind"] != "summary":
        evals.append(dict(p, split="eval", temperature=0.0, max_tokens=256))
eval_ids = {p["id"] for p in evals}
train = [dict(p, split="train") for p in unique if p["id"] not in eval_ids]
with open("prompts.jsonl", "w") as fh:
    for p in evals + train:
        fh.write(json.dumps(p, ensure_ascii=False) + "\n")
print(json.dumps({"eval": len(evals), "train": len(train), "kinds": {k: sum(1 for p in train if p["kind"] == k) for k, *_ in KINDS},
                  "thinking": sum(p["thinking"] for p in train)}))
