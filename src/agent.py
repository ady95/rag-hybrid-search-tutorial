# -*- coding: utf-8 -*-
"""검색을 도구로 노출한 질의응답 에이전트 (09-2, 09-4).

모델이 스스로 검색어를 정해 `search_chunks` 를 부르고, 필요하면 여러 번
부른 뒤 답한다. 답변의 인용은 **실제로 건넨 청크**만 허용된다.

    python -m src.agent --budget-scan        # LLM 없이 예산별 청크 수만 확인
    python -m src.agent "AI 규제 법안이 통과된 나라"
"""
import argparse
import json
import os
import sys
from datetime import date

from src import embedder, tokenizer_ko
from src.build_sqlite import connect
from src.context import BLOCK, build_context, count_tokens
from src.search_sqlite import fetch, search
from src.verify import verify_citations

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

MODEL = os.environ.get("RAG_MODEL", "claude-sonnet-5")
MAX_TURNS = 6
ALLOWED_MODES = {"hybrid", "keyword", "semantic"}

# 09-4: 모델은 오늘이 며칠인지 모른다. "지난달"·"상반기"를 계산하려면
# 오늘 날짜와 자료 범위를 알려 줘야 한다.
CORPUS_RANGE = ("2026-01", "2026-07")

SYSTEM = f"""오늘은 {date.today():%Y-%m-%d} 입니다.
자료는 SPRi AI 브리프 {CORPUS_RANGE[0]} ~ {CORPUS_RANGE[1]} 호입니다.
상대적 날짜 표현(지난달, 최근, 상반기)은 오늘 날짜를 기준으로 계산하되,
자료 범위를 벗어나면 범위 밖이라고 알려 주세요.

당신은 SPRi AI 브리프를 근거로 답하는 조사 도우미입니다.

인용 규칙
- 사실을 서술하는 모든 문장 끝에 근거 자료의 대괄호 식별자를 붙입니다.
  예: 캘리포니아주 상원이 법안을 가결했습니다 [SPRi_AI_Brief_202603#0014].
- 한 문장이 여러 자료에 근거하면 나란히 붙입니다. [A#0001][B#0002]
- <자료> 안에 없는 식별자는 절대 만들어 내지 않습니다.

모를 때의 규칙
- <자료>에서 답을 찾을 수 없으면 "제공된 자료에서 확인되지 않습니다"라고만
  답하고, 어떤 자료가 있으면 답할 수 있는지 한 줄로 알려 줍니다.
- 자료가 질문과 부분적으로만 맞으면, 확인된 부분만 답하고
  나머지는 확인되지 않았다고 명시합니다.
- 추측·일반 상식·사전 학습 지식으로 빈칸을 메우지 않습니다."""

# 도구 설명이 곧 프롬프트다. 모델이 언제 부를지, 무엇을 넣을지를 여기서 정한다.
SEARCH_TOOL = {
    "name": "search_chunks",
    "description": (
        "SPRi AI 브리프 2026년 1~7월호에서 관련 문단을 찾는다. "
        "사실·수치·기관명·날짜가 필요한 질문에는 반드시 먼저 이 도구를 부른다. "
        "질문 문장을 그대로 넣지 말고 핵심 키워드 위주로 다시 쓴다. "
        "특정 월에 한정된 질문이면 year_months 를 채운다."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": "검색어. 예: 'AI 규제 법안 캘리포니아'",
            },
            "year_months": {
                "type": "array",
                "items": {"type": "string"},
                "description": ("특정 호로 좁힐 때만. 형식 2026-03. 여러 달이면 "
                                "나열한다. '상반기'는 2026-01 ~ 2026-06 으로 "
                                "펼쳐서 넣는다. 달을 특정할 수 없으면 생략한다 "
                                "— 비워 두는 편이 낫다."),
            },
            "mode": {
                "type": "string",
                "enum": ["hybrid", "keyword", "semantic"],
                "description": "기본은 hybrid. 고유명사가 정확히 기억날 때만 keyword",
            },
            "top_n": {"type": "integer", "description": "기본 8, 최대 20"},
            "exclude_toc": {
                "type": "boolean",
                "description": "기본 true. 목차·표지 구간을 검색에서 뺀다",
            },
        },
        "required": ["query"],
    },
}

# OpenAI 규격은 키 이름만 다르다.
OPENAI_TOOL = {
    "type": "function",
    "function": {
        "name": SEARCH_TOOL["name"],
        "description": SEARCH_TOOL["description"],
        "parameters": SEARCH_TOOL["input_schema"],
    },
}


def run_tool_with_ids(conn, name, args, budget=2500, count_tokens=count_tokens):
    """도구를 실행하고 (본문, 건넨 chunk_id 목록)을 돌려준다.

    id 목록을 함께 돌려주는 것이 핵심이다. 이것이 인용 검증의 허용 목록이
    되므로, 모델에게 건네지 않은 청크를 인용하면 잡아낼 수 있다.
    """
    if name != "search_chunks":
        return f"알 수 없는 도구: {name}", []

    query = (args.get("query") or "").strip()
    if not query:
        return "query 가 비었습니다. 검색어를 넣어 다시 부르세요.", []

    mode = args.get("mode") or "hybrid"
    if mode not in ALLOWED_MODES:
        mode = "hybrid"
    top_n = min(int(args.get("top_n") or 8), 20)
    ym = args.get("year_months") or args.get("year_month") or None
    no_toc = args.get("exclude_toc", True)

    res = search(conn, query, mode=mode, ym=ym, top_n=top_n, no_toc=no_toc)
    if not res and ym:
        # 필터가 정답을 통째로 날렸을 수 있다. 한 번은 풀고 다시 찾는다 (09-4).
        res = search(conn, query, mode=mode, top_n=top_n, no_toc=no_toc)
        if res:
            shown = ",".join(ym) if isinstance(ym, list) else ym
            print(f"  [필터 해제] {shown} 에는 없어 전체에서 다시 찾았습니다")
    if not res:
        return f"'{query}' 로 찾은 결과가 없습니다. 다른 검색어로 다시 시도하세요.", []

    meta = fetch(conn, [c for c, _, _ in res])
    out, ids, used = [], [], 0
    for cid, _s, _h in res:
        m = meta.get(cid)
        if not m:
            continue
        block = BLOCK.format(**m)
        n = count_tokens(block)
        if used + n > budget:
            break
        out.append(block)
        ids.append(cid)
        used += n
    return "\n---\n".join(out), ids


def compare(conn, question, months, per=5):
    """축마다 따로 담는다. RRF로 섞으면 어느 달 것인지 사라져 비교를 못 한다 (09-4)."""
    out = []
    for ym in months:
        res = search(conn, question, mode="hybrid", ym=ym, top_n=per, no_toc=True)
        meta = fetch(conn, [c for c, _, _ in res])
        body = "\n---\n".join(BLOCK.format(**meta[c])
                              for c, _, _ in res if c in meta)
        out.append(f'<자료 기간="{ym}">\n{body}\n</자료>')
    return "\n\n".join(out)


def ask(conn, question):
    """도구 호출 루프. (답변, 인용 검증 결과)를 돌려준다."""
    from anthropic import Anthropic
    client = Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])
    messages = [{"role": "user", "content": question}]
    served = []                    # 이번 대화에서 실제로 건넨 chunk_id

    for turn in range(MAX_TURNS):
        r = client.messages.create(
            model=MODEL, max_tokens=1500, system=SYSTEM,
            tools=[SEARCH_TOOL], messages=messages)
        messages.append({"role": "assistant", "content": r.content})

        if r.stop_reason != "tool_use":
            answer = "".join(b.text for b in r.content if b.type == "text")
            return answer, verify_citations(answer, served)

        results = []
        for b in (x for x in r.content if x.type == "tool_use"):
            print(f"  [turn {turn+1}] {json.dumps(b.input, ensure_ascii=False)}")
            text, ids = run_tool_with_ids(conn, b.name, b.input)
            served.extend(ids)      # 검증에 쓸 허용 목록을 누적한다
            results.append({"type": "tool_result",
                            "tool_use_id": b.id, "content": text})
        messages.append({"role": "user", "content": results})

    return "검색 횟수를 초과했습니다.", None


def budget_scan(conn, query="AI 규제 법안"):
    """LLM 없이 확인하는 부분 — 예산을 바꾸면 청크가 몇 개 들어가는가."""
    for budget in (1000, 2000, 3000, 6000):
        _ctx, used, kept = build_context(conn, query, mode="keyword",
                                         top_n=12, budget=budget,
                                         count_tokens=count_tokens)
        print(f"예산 {budget:5d} -> 청크 {len(kept):2d}개 / {used:5d}토큰")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("question", nargs="*")
    ap.add_argument("--budget-scan", action="store_true",
                    help="LLM 없이 예산별 청크 수만 확인한다")
    a = ap.parse_args()

    tokenizer_ko.warmup()
    embedder.warmup()
    conn = connect()
    try:
        if a.budget_scan:
            budget_scan(conn)
            return
        q = " ".join(a.question) or "AI 규제 법안이 통과된 나라"
        print(f"질문> {q}")
        answer, v = ask(conn, q)
        print(f"\n{answer}\n")
        if v:
            print(f'  인용 {len(v["cited"])}건 · 커버리지 {v["coverage"]} · '
                  f'{"검증 통과" if v["ok"] else "검증 실패"}')
            if v["invalid"]:
                print(f'  [경고] 존재하지 않는 자료: {v["invalid"]}')
    finally:
        conn.close()


if __name__ == "__main__":
    main()
