#!/usr/bin/env python3
"""The second batch of hand-written live-eval prompts (operator: "32개는 좀 적지않아?"): longer pasted text, more
conversations under a system prompt, long code answers, problems that call for thinking. Same fields as crafted_eval.py,
ids "s2-<category>-<n>".

    crafted_eval2.py > crafted_eval2.jsonl
"""
import json
import sys

TRACEBACK = '''Traceback (most recent call last):
  File "report.py", line 41, in <module>
    summary = df.groupby("site")["kwh"].sum()["Haenam-2"]
  File ".../pandas/core/series.py", line 1040, in __getitem__
    return self._get_value(key)
KeyError: 'Haenam-2'

sites in the csv: "Haenam-1", "Haenam-2 ", "Yeonggwang-1"'''

DOCKERFILE = '''FROM python:3.12-slim
WORKDIR /app
COPY . .
RUN pip install -r requirements.txt
CMD ["python", "main.py"]'''

GO_SNIPPET = '''func firstResult(urls []string) string {
    ch := make(chan string)
    for _, u := range urls {
        go func(u string) {
            ch <- fetch(u)
        }(u)
    }
    return <-ch
}'''

DF_SAMPLE = '''   date        store  sales
0  2026-09-01  S01    1320
1  2026-09-01  S02     980
2  2026-09-02  S01    1250
...  (90 days x 12 stores)'''

WEEKLY = """- 전체 발전량 412 MWh (계획 대비 97%)
- 영광 2호기 인버터 트립 2회, 원격 리셋, A/S 화요일 예정
- 해남 부지 계통 연계 회신 지연 (착공 한 달 연기 가능)
- 모듈 세척: 3개 단지 완료, 발전량 약 3% 회복 추정
- 다음 주: A/S 입회, 세척 업체 선정, 경보 기준 변경안 상정"""

NOTES_EN = """- attendees: ops (3), finance (1), vendor rep
- Q3 downtime 1.8% vs 1.2% target; main cause: two transformer faults in July
- vendor offers extended warranty, +4% on annual fee; finance wants a 3-year cost comparison first
- decision deferred to Oct 2; ops to send fault logs to vendor by Friday
- next review: monitoring alert thresholds"""

ARTICLE = """Cities are experimenting with 'cool pavements' — road coatings that reflect more sunlight than conventional asphalt.
Early pilots in several warm cities report surface temperatures 5 to 10 degrees Celsius lower on treated streets during
summer afternoons. The effect on air temperature is smaller and harder to measure, often less than one degree, because
wind and building shade mix the air quickly. Critics point out that reflected light can bounce onto nearby walls and
pedestrians, raising the temperature people actually feel even as the pavement cools. Maintenance is another question:
coatings wear down under heavy traffic and may need reapplication every five to seven years, which adds to their cost.
Supporters argue that cool pavements work best combined with trees and shade structures, and that the lower surface
temperatures still extend the life of the road itself. Researchers say the next step is measuring health outcomes, such as
heat-related emergency visits, rather than surface temperatures alone."""

COMPLAINTS = """[고객 문의 스레드 요약 — 모니터링 앱]
- 9/12: 발전량 그래프가 하루씩 늦게 반영된다는 문의 3건
- 9/13: 알림이 새벽에 너무 자주 온다(구름 낀 날 출력 저하 알림), '끄는 방법을 모르겠다'
- 9/14: 월간 보고서 PDF의 단위가 kW와 kWh로 섞여 있어 헷갈린다
- 9/15: 앱 업데이트 후 로그인 유지가 안 돼 매번 비밀번호 입력
- 9/16: 발전량이 0으로 표시됐다가 몇 시간 뒤 정상으로 돌아옴(2건)"""

MESSY_KO = "안녕하세요 지난번에 말씀드린 건 관련해서 연락드립니다 저희쪽 에서 검토해본결과 일정이 좀 어려울것 같아서요 " \
           "혹시 다음달 둘째주 쯤으로 미룰수 있을지 여쭤봐도 될까요 안되면 어쩔수 없구요"

README_EN = """## Configuration
The agent reads its settings from `agent.yaml` at startup. Each site entry needs an `id`, the inverter `protocol`
(`modbus-tcp` or `sunspec`), and a polling interval in seconds. Intervals below 10 seconds are rejected to protect the
inverters' controllers. When a site is unreachable for three consecutive polls, the agent marks it `stale` and raises a
single alert instead of one per poll. Restart the agent after editing the file; it does not reload settings on its own."""

P = []


def single(cat, lang, text, max_tokens=512, thinking=False):
    P.append({"cat": cat, "lang": lang, "shape": "single", "messages": [{"role": "user", "content": text}],
              "max_tokens": max_tokens, "thinking": thinking})


def convo(cat, lang, messages, max_tokens=512, thinking=False):
    P.append({"cat": cat, "lang": lang, "shape": "multi", "messages": messages, "max_tokens": max_tokens,
              "thinking": thinking})


single("coding_write", "ko", "TypeScript로 제네릭 debounce 함수를 만들어줘. 대기 중인 호출을 취소하는 cancel()과 즉시 실행하는 "
       "flush()를 지원하고, jest 테스트도 붙여줘.", 1024)
single("coding_write", "en", "Write a bash script that gzips log files in /var/log/myapp older than 7 days, deletes archives "
       "older than 30 days, and is safe for filenames with spaces. Explain each step briefly.", 1024)
convo("coding_write", "en", [
    {"role": "system", "content": "You are a meticulous Python reviewer. Prefer standard library solutions and type hints."},
    {"role": "user", "content": "Implement an LRU cache class with O(1) get and put."},
    {"role": "assistant", "content": "```python\nfrom collections import OrderedDict\n\nclass LRU:\n    def __init__(self, cap: int):\n"
                                     "        self.cap, self.d = cap, OrderedDict()\n\n    def get(self, k):\n        if k not in self.d:\n"
                                     "            return None\n        self.d.move_to_end(k)\n        return self.d[k]\n\n    def put(self, k, v):\n"
                                     "        self.d[k] = v\n        self.d.move_to_end(k)\n        if len(self.d) > self.cap:\n"
                                     "            self.d.popitem(last=False)\n```"},
    {"role": "user", "content": "Now do it without OrderedDict: a dict plus your own doubly linked list, fully typed, with a "
                                "few unit tests."}], 1024)
single("coding_debug", "en", "I get this error and I don't understand why the key is missing:\n\n```\n" + TRACEBACK + "\n```")
single("coding_debug", "ko", "도커 이미지를 빌드할 때마다 pip install 이 처음부터 다시 돌아서 10분씩 걸려. 캐시가 안 먹는 이유와 "
       "고친 Dockerfile 을 알려줘.\n\n```dockerfile\n" + DOCKERFILE + "\n```")
single("coding_review", "ko", "이 Go 함수 리뷰해줘. 운영에서 메모리가 조금씩 늘어나는 것 같아.\n\n```go\n" + GO_SNIPPET + "\n```")
single("data_sql", "en", "With a pandas DataFrame like this:\n\n```\n" + DF_SAMPLE + "\n```\n\ncompute each store's 7-day "
       "rolling mean of sales and flag days that fall more than 30% below it. Show the code.")
single("math", "en", "Find every real x with sqrt(x + 3) + sqrt(x - 2) = 5.", 512, thinking=True)
single("math", "ko", "주사위 세 개를 동시에 던질 때 눈의 합이 10 이 될 확률을 구해줘.", 512, thinking=True)
single("reasoning", "en", "Four friends — Ana, Ben, Chloe and Dev — each own one pet: cat, dog, parrot, fish. Ana is allergic "
       "to fur. Ben's pet can talk. Chloe does not own the dog. Who owns which pet?", 512, thinking=True)
single("reasoning", "ko", "견적이 두 개야. A안: 초기비용 1억 2천, 연간 유지비 300만 원, 보증 5년. B안: 초기비용 9천 5백, 연간 "
       "유지비 700만 원, 보증 2년. 10년 쓴다고 할 때 어느 쪽이 나은지 근거와 함께 판단해줘.")
single("writing_business", "ko", "아래 메모로 주간 운영 보고서를 써줘. 실적, 이슈, 다음 주 계획으로 나누고 표를 하나 넣어줘.\n\n"
       + WEEKLY, 1024)
single("writing_business", "en", "Turn these notes into formal meeting minutes with decisions and action items:\n\n" + NOTES_EN)
single("writing_creative", "en", "Write a short bedtime story for a five-year-old about a little solar panel who is afraid of "
       "clouds.", 512)
single("summarize", "en", "Summarize this article in three bullet points, then give one open question it raises.\n\n" + ARTICLE)
single("summarize", "ko", "아래 고객 문의를 보고 핵심 불만 세 가지와 각각의 대응 방향을 정리해줘.\n\n" + COMPLAINTS)
single("translate_edit", "ko", "다음 문장의 맞춤법과 띄어쓰기를 고치고, 거래처에 보내도 될 만큼 공손하게 다듬어줘.\n\n" + MESSY_KO, 256)
single("translate_edit", "en", "Translate this README section into Korean for our field engineers. Keep code terms as they "
       "are.\n\n" + README_EN, 1024)
single("business_finance", "en", "For a small bakery, compare leasing versus buying a delivery van over five years. Build a "
       "simple cost model with assumptions stated, and say which you'd pick.")
single("legal_tax_admin", "ko", "근로계약서에 반드시 들어가야 하는 항목과, 수습 기간 급여를 어떻게 정할 수 있는지 알려줘.")
single("energy_solar", "ko", "ESS를 붙인 태양광 발전소에서 SMP 시간대별 가격에 맞춰 충방전 일정을 짜는 원리를 설명해줘.")
single("energy_solar", "en", "A 500 kW plant loses about 3% of output to soiling between cleanings. Each cleaning costs "
       "$1,200, energy sells at $0.09/kWh and the site yields 1,300 kWh per kWp per year. How many cleanings a year make "
       "sense? Show the reasoning.", 512, thinking=True)
single("tech_ops", "en", "Our nginx returns 502 errors a few times an hour behind a cloud load balancer. Give me a "
       "diagnosis checklist in order.")
single("tech_ops", "ko", "PostgreSQL 쿼리가 갑자기 느려졌어. EXPLAIN ANALYZE 결과를 읽는 법과 흔한 원인을 알려줘.")
single("ai_ml", "en", "Explain KL divergence with an intuitive example, and why KL(P||Q) differs from KL(Q||P).")
single("ai_ml", "ko", "RAG 시스템에서 문서를 자르는 청크 크기는 어떤 기준으로 정해야 해?")
single("life_health", "en", "Give me five high-protein breakfast ideas without eggs.", 256)
single("howto", "en", "How do I set up SSH key login on Ubuntu and then disable password login without locking myself out?")
single("general_qa", "ko", "왜 겨울에 정전기가 더 잘 일어나?", 256)
single("chitchat", "en", "We just shipped a big project at work. Any ideas for a small way to celebrate with the team?", 256)
convo("general_qa", "ko", [
    {"role": "system", "content": "너는 태양광 모니터링 앱의 고객센터 상담원이다. 공손하고 간결하게 답한다."},
    {"role": "user", "content": "앱에서 우리 발전소 발전량이 계속 0으로 나와요."},
    {"role": "assistant", "content": "불편을 드려 죄송합니다. 확인을 위해 여쭤볼게요. 인버터 화면이나 LED 상태는 어떤가요? "
                                     "그리고 언제부터 0으로 보이셨나요?"},
    {"role": "user", "content": "인버터 LED는 초록불이에요. 어제 오후부터 그래요."}], 512)
convo("howto", "en", [
    {"role": "system", "content": "You are a project planning assistant. Be concrete: owners, dates, checkpoints."},
    {"role": "user", "content": "Plan a three-week rollout of a new monitoring dashboard to 40 sites."},
    {"role": "assistant", "content": "Week 1: pilot at 5 sites and fix issues. Week 2: roll out to 20 more sites with daily "
                                     "check-ins. Week 3: the last 15 sites, training sessions, and a retrospective."},
    {"role": "user", "content": "Turn week 1 into a day-by-day checklist with an owner for each item."}], 1024)
single("explain", "en", "Explain, step by step, how HTTPS protects a login form — for a non-technical manager.")
single("explain", "ko", "양자컴퓨터가 암호를 깬다는 말이 무슨 뜻인지 RSA를 예로 들어 설명해줘.", 1024, thinking=True)

for i, p in enumerate(P):
    p["id"] = f"s2-{p['cat']}-{i:02d}"
print(json.dumps({"_count": len(P)}), file=sys.stderr)
for p in P:
    print(json.dumps({"id": p["id"], "kind": f"s-{p['shape']}", "suite": "crafted", "messages": p["messages"],
                      "content": p["messages"][-1]["content"], "max_tokens": p["max_tokens"], "thinking": p["thinking"],
                      "temperature": 0.0, "split": "eval", "lang": p["lang"], "category": p["cat"],
                      "system": p["messages"][0]["role"] == "system"}, ensure_ascii=False))
