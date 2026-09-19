#!/usr/bin/env python3
"""Hand-written live-eval prompts for the MTP head (operator 2026-09-19: "아니면 너가 하나하나 만들어"). Never prefilled
by any data boot. Spread over the twenty categories of datagen2.py, Korean and English, single turns with short and long
answers, multi-turn conversations under a system prompt, pasted text, thinking on for the problems that call for it.
No real names. Greedy; ids "s-<category>-<n>".

    crafted_eval.py > crafted_eval.jsonl
"""
import json

MEETING = """[주간 운영 회의록 — 9월 셋째 주]
참석: 운영팀 4명, 설계팀 2명, 영업 1명
1. 영광 2호기 인버터 3번 트립 반복(주 2회). 제조사 A/S 방문은 다음 주 화요일로 확정. 그때까지 원격 리셋으로 대응하되,
   트립 시각과 당시 일사량을 기록해 두기로 함. 담당: 운영팀.
2. 모듈 세척 일정: 황사 이후 발전량이 약 4% 떨어진 단지부터 우선. 세척 업체 견적 두 곳 비교 후 금요일까지 결정.
3. 신규 부지(해남) 계통 연계 검토: 한전 회신이 늦어 착공이 한 달 밀릴 수 있음. 영업은 고객에게 일정 변동 가능성을 미리
   알리기로 함. 설계팀은 변압기 용량을 두 안(1,000 kVA / 1,500 kVA)으로 준비.
4. 모니터링 시스템 알림이 너무 많다는 의견. 경보 기준을 '10분 이상 지속'으로 바꾸는 안을 다음 회의에서 결정.
5. 기타: 10월 정기 안전교육 일정 공지 예정."""

SPEC_KO = """다음은 우리 팀 내부 문서의 한 단락입니다.
"본 시스템은 발전소별 계측 데이터를 5분 간격으로 수집하여 중앙 서버에 적재하며, 적재 지연이 15분을 초과할 경우 담당자에게
문자 알림을 발송한다. 또한 월말에는 발전량, 가동률, 고장 이력을 집계한 보고서를 자동 생성하여 고객사에 전달한다.\""""

TEXT_EN = """Heat pumps move heat instead of generating it. In winter, a heat pump extracts low-grade heat from outdoor air,
compresses the refrigerant to raise its temperature, and releases that heat indoors. Because it moves more energy than it
consumes, its coefficient of performance is typically between 2 and 4, although efficiency drops as outdoor temperatures fall."""

BUGGY_PY = '''def add_item(item, basket=[]):
    basket.append(item)
    return basket

def last_n(xs, n):
    return xs[len(xs) - n - 1:]

print(add_item("사과"))
print(add_item("배"))
print(last_n([1, 2, 3, 4, 5], 2))'''

BUGGY_JS = '''async function uploadAll(files) {
  files.forEach(async (f) => {
    await upload(f);
    console.log("uploaded", f.name);
  });
  console.log("done");
}'''

FLASK = '''@app.route("/user")
def user():
    name = request.args.get("name")
    cur = db.cursor()
    cur.execute("SELECT * FROM users WHERE name = '" + name + "'")
    row = cur.fetchone()
    return str(row)'''

INVOICE = """INVOICE #A-2291
Bill to: Northwind Solar Ltd.
Date: 2026-08-14
Items: Inverter maintenance (2 units) ........ 1,200.00 USD
       Replacement fuse set ..................    85.50 USD
Total due: 1,285.50 USD   Due date: 2026-09-13"""

P = []


def single(cat, lang, text, max_tokens=512, thinking=False):
    P.append({"cat": cat, "lang": lang, "shape": "single", "messages": [{"role": "user", "content": text}],
              "max_tokens": max_tokens, "thinking": thinking})


def convo(cat, lang, messages, max_tokens=512, thinking=False):
    P.append({"cat": cat, "lang": lang, "shape": "multi", "messages": messages, "max_tokens": max_tokens,
              "thinking": thinking})


# coding
single("coding_write", "en", "Write a Python function that parses an ISO 8601 duration such as 'P3DT4H5M' or 'PT45S' into "
       "total seconds. Raise ValueError with a clear message on bad input, and include pytest tests.", 1024)
single("coding_write", "ko", "Go로 토큰 버킷 방식의 레이트 리미터를 구현해줘. 여러 고루틴에서 동시에 불러도 안전해야 하고, "
       "사용 예시와 테스트 코드도 같이 보여줘.", 1024)
single("coding_debug", "ko", "이 코드가 두 번째 호출부터 이상한 결과를 내고, last_n 도 원하는 개수보다 하나 더 돌려줘. "
       "원인과 고친 코드를 알려줘.\n\n```python\n" + BUGGY_PY + "\n```")
single("coding_debug", "en", "Why does this print \"done\" before the uploads finish, and how should I fix it?\n\n```js\n"
       + BUGGY_JS + "\n```")
single("coding_review", "en", "Review this Flask route for security and style problems, then show a corrected version.\n\n"
       "```python\n" + FLASK + "\n```")
single("data_sql", "ko", "테이블이 orders(order_id, customer_id, ordered_at, amount) 와 customers(customer_id, joined_at) 두 개야. "
       "월별 신규 고객 수와, 그 달에 가입한 고객 중 90일 안에 두 번 이상 주문한 비율을 구하는 SQL(PostgreSQL)을 짜줘.")
# math and reasoning
single("math", "en", "A bag holds 5 red and 7 blue marbles. Three are drawn without replacement. What is the probability "
       "that exactly two are red? Show the steps.", 512, thinking=True)
single("math", "ko", "수열 a1 = 2, a(n+1) = 3·a(n) − 2 의 일반항을 구하고 a10 을 계산해줘.", 512, thinking=True)
single("reasoning", "ko", "다섯 팀(A~E)이 월~금 하루씩 회의실을 쓴다. A는 월요일이 안 되고, B는 C 바로 다음 날, D는 수요일 "
       "아니면 금요일, E는 A보다 먼저다. 가능한 배정을 모두 찾아줘.", 1024, thinking=True)
single("reasoning", "en", "Three switches outside a closed room control three bulbs inside. You may flip switches as you "
       "like but enter the room only once. How do you tell which switch controls which bulb? Explain why it works.", 256)
# writing
single("writing_business", "ko", "태양광 발전소 O&M(운영·유지보수) 계약 갱신을 알리는 공문을 써줘. 수신은 발주처 담당 부서, "
       "계약 기간은 2026년 10월부터 1년, 바뀌는 점은 점검 주기 월 1회→격주, 원격 모니터링 추가, 단가 3% 인상.", 1024)
single("writing_business", "en", "Draft a short, polite email declining a vendor's proposal for this quarter while keeping "
       "the door open for the next one.", 512)
single("writing_creative", "ko", "비 오는 밤, 태양광 발전소 관제실에서 혼자 야간 근무를 하는 사람을 주인공으로 짧은 소설의 "
       "첫 장면을 써줘.", 512)
single("writing_creative", "en", "Write a four-stanza poem about a lighthouse keeper's last night before automation.", 256)
# summarize, translate, extract
single("summarize", "ko", "아래 회의록을 세 줄로 요약하고, 담당자와 기한이 있는 할 일만 따로 목록으로 뽑아줘.\n\n" + MEETING)
single("translate_edit", "ko", SPEC_KO + "\n\n이 단락을 해외 고객에게 보낼 자연스러운 영어로 옮겨줘.")
single("translate_edit", "en", "Translate this into natural Korean for a customer brochure (polite style):\n\n" + TEXT_EN)
single("general_qa", "en", "Extract the invoice number, bill-to company, date, total and due date from this text as JSON "
       "only.\n\n" + INVOICE, 256)
# domains
single("business_finance", "ko", "SMP가 떨어지는 시기에 REC 가격까지 같이 내려가면 100kW 규모 개인 태양광 사업자는 "
       "수익을 어떻게 지킬 수 있을까? 현실적인 선택지를 비교해줘.")
single("legal_tax_admin", "ko", "개인사업자로 소규모 태양광 발전사업을 시작할 때 부가가치세 환급 절차와 놓치기 쉬운 "
       "주의점을 알려줘.")
single("energy_solar", "en", "Explain how inverter clipping affects annual energy yield and how designers choose the "
       "DC/AC ratio.")
single("energy_solar", "ko", "용량 100kW 발전소의 연간 발전량을 추정해줘. 하루 평균 일사량 3.6kWh/m², 성능비 0.8 을 가정하고 "
       "계산 과정을 보여줘.", 1024, thinking=True)
single("tech_ops", "en", "Our Kubernetes pods get OOMKilled every few hours, but memory graphs look flat. How do I "
       "diagnose this step by step?")
single("tech_ops", "ko", "리눅스 서버 디스크 사용률이 99%인데 du 로 합계를 내면 절반도 안 돼. 원인과 확인 방법을 알려줘.")
single("ai_ml", "ko", "LLM의 추측 디코딩(speculative decoding)이 뭔지, 왜 원래 모델과 같은 결과를 낸다고 하는지 쉽게 "
       "설명해줘.")
single("ai_ml", "en", "What is the difference between top-k and top-p sampling, and when would you prefer each?", 256)
single("life_health", "ko", "저녁에 운동하면 잠이 잘 안 와. 운동 시간과 강도를 어떻게 조절하면 좋을까?", 256)
single("howto", "ko", "엑셀에서 시트 두 개에 있는 고객 목록을 비교해서 한쪽에만 있는 사람을 찾는 방법을 함수로 알려줘.")
single("chitchat", "ko", "오늘 좀 지치네. 퇴근길에 들을 만한 잔잔한 노래 몇 곡 추천해줄래?", 256)
# conversations under a system prompt
convo("tech_ops", "ko", [
    {"role": "system", "content": "너는 사내 IT 헬프데스크 도우미다. 짧게, 번호를 매긴 단계로 답한다."},
    {"role": "user", "content": "VPN은 연결됐다고 나오는데 사내 사이트가 하나도 안 열려요."},
    {"role": "assistant", "content": "1. 브라우저에서 사내 사이트 주소를 IP로 직접 열어 보세요.\n2. 열리면 DNS 문제입니다.\n"
                                     "3. 안 열리면 VPN을 끊었다가 다시 연결하고, 그래도 같으면 알려 주세요."},
    {"role": "user", "content": "IP로는 열려요. DNS는 어떻게 바꿔요? 윈도우 11이에요."}], 512)
convo("coding_write", "en", [
    {"role": "system", "content": "You are a senior Python engineer. Answer with code first, then a short explanation."},
    {"role": "user", "content": "Write a function fetch_json(url) that GETs a URL with requests and returns the parsed JSON."},
    {"role": "assistant", "content": "```python\nimport requests\n\ndef fetch_json(url):\n    resp = requests.get(url, timeout=10)\n"
                                     "    resp.raise_for_status()\n    return resp.json()\n```\nIt raises on HTTP errors and times out after 10 s."},
    {"role": "user", "content": "Now make it async with httpx and add retries with exponential backoff on 5xx and timeouts."}], 1024)
convo("life_health", "ko", [
    {"role": "system", "content": "너는 여행 계획을 도와주는 친절한 비서다."},
    {"role": "user", "content": "10월에 부모님 모시고 2박 3일로 남해 쪽 여행 가려고 해. 두 분 다 70대셔."},
    {"role": "assistant", "content": "좋아요! 이동이 적고 걷기 편한 코스가 좋겠어요. 출발 지역과 차량 이용 여부를 알려 주시면 "
                                     "일정을 짜 드릴게요."},
    {"role": "user", "content": "서울에서 차로 가. 무릎이 안 좋으셔서 계단 많은 곳은 피하고 싶어."}], 1024)
convo("explain", "ko", [
    {"role": "system", "content": "너는 중학생을 가르치는 과외 선생님이다. 쉬운 말과 예시로 설명한다."},
    {"role": "user", "content": "주식이랑 채권이 뭐가 달라요?"}], 512)

for i, p in enumerate(P):
    p["id"] = f"s-{p['cat']}-{i:02d}"
print(json.dumps({"_count": len(P)}), file=__import__("sys").stderr)
for p in P:
    print(json.dumps({"id": p["id"], "kind": f"s-{p['shape']}", "suite": "crafted", "messages": p["messages"],
                      "content": p["messages"][-1]["content"], "max_tokens": p["max_tokens"], "thinking": p["thinking"],
                      "temperature": 0.0, "split": "eval", "lang": p["lang"], "category": p["cat"],
                      "system": p["messages"][0]["role"] == "system"}, ensure_ascii=False))
