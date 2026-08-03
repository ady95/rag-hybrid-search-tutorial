# -*- coding: utf-8 -*-
"""프로젝트 2 — 출처를 밝히는 질의응답 봇 (10-2).

모든 문장에 근거를 붙이게 하고, 그 근거가 실제로 준 자료인지 프로그램이 확인한다.
검색·컨텍스트 조립·인용 검증·로그 적재는 API 키 없이 돌아간다.

    python -m src.bot --retrieve-only "AI 규제 법안이 통과된 나라"   # 키 불필요
    python -m src.bot                                              # 대화 루프
"""
import argparse
import os
import sys
import time
from datetime import datetime

from src import embedder, tokenizer_ko
from src.build_sqlite import connect
from src.context import build_context, count_tokens
from src.search_sqlite import fetch
from src.verify import verify_citations

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

# 봇을 만들 때 가장 먼저 할 일은 로그다. 나중에 붙이려면 결국 안 붙인다.
# 10-3의 대시보드가 이 테이블을 그대로 읽는다.
LOG_SCHEMA = """
CREATE TABLE IF NOT EXISTS query_log (
    id          INTEGER PRIMARY KEY,
    ts          TEXT NOT NULL,
    query       TEXT NOT NULL,
    mode        TEXT,
    n_results   INTEGER,
    n_tokens    INTEGER,
    latency_ms  REAL,
    top_ids     TEXT,      -- 검색 상위 chunk_id (쉼표 구분)
    cited_ids   TEXT,      -- 답변이 인용한 chunk_id
    coverage    REAL,
    invalid     INTEGER,   -- 허위 인용 개수
    refused     INTEGER    -- 근거 없다고 답했으면 1
);
"""

SYSTEM = """당신은 SPRi AI 브리프 2026년 1~7월호를 근거로 답하는 조사 봇입니다.

- 사실을 서술한 모든 문장 끝에 근거 자료의 대괄호 식별자를 붙입니다.
- <자료>에 없는 식별자는 절대 만들지 않습니다.
- 답을 찾을 수 없으면 "제공된 자료에서 확인되지 않습니다."라고 답하고,
  이 봇이 다루는 범위를 한 줄로 안내합니다.
- 추측하지 않습니다. 자료에 없는 숫자·날짜·기관명을 만들지 않습니다.
- 답변은 5문장을 넘기지 않습니다."""


def retrieve(conn, question, ym=None, budget=3500, count_tokens=count_tokens):
    t0 = time.perf_counter()
    ctx, used, kept = build_context(conn, question, mode="hybrid", ym=ym,
                                    top_n=10, budget=budget,
                                    count_tokens=count_tokens)
    ms = (time.perf_counter() - t0) * 1000
    print(f"  [검색] -> {len(kept)}청크 / {used:,}토큰 / {ms:.0f}ms")
    return ctx, kept, used, ms


def answer(conn, question, ym=None, model="claude-sonnet-5"):
    ctx, kept, used, ms = retrieve(conn, question, ym)
    if not kept:
        return {"text": "제공된 자료에서 확인되지 않습니다.", "refused": True,
                "kept": [], "v": None, "ms": ms, "tokens": 0}

    from anthropic import Anthropic
    client = Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])
    r = client.messages.create(
        model=model, max_tokens=900, system=SYSTEM,
        messages=[{"role": "user",
                   "content": f"<자료>\n{ctx}\n</자료>\n\n질문: {question}"}])
    text = "".join(b.text for b in r.content if b.type == "text")

    return {"text": text, "refused": "확인되지 않습니다" in text,
            "kept": kept, "v": verify_citations(text, kept),
            "ms": ms, "tokens": used}


def render(conn, res):
    """답변 아래에 근거를 사람이 읽을 수 있게 펼친다."""
    print(f'\n{res["text"]}\n')
    v = res["v"]
    if not v:
        return
    print(f'  근거 {len(v["cited"])}건 · 인용 커버리지 {v["coverage"]} · '
          f'{"검증 통과" if v["ok"] else "검증 실패"}')
    if v["invalid"]:
        print(f'  [경고] 존재하지 않는 자료를 인용했습니다: {v["invalid"]}')
    meta = fetch(conn, v["cited"])
    for i, cid in enumerate(v["cited"], 1):
        m = meta.get(cid)
        if m:
            print(f'  [{i}] {m["year_month"]} {m["title"][:60]}')


def log(conn, question, res):
    v = res["v"] or {}
    conn.execute(
        "INSERT INTO query_log(ts,query,mode,n_results,n_tokens,latency_ms,"
        "top_ids,cited_ids,coverage,invalid,refused)"
        " VALUES (?,?,?,?,?,?,?,?,?,?,?)",
        (datetime.now().isoformat(timespec="seconds"), question, "hybrid",
         len(res["kept"]), res["tokens"], res["ms"],
         ",".join(res["kept"]), ",".join(v.get("cited", [])),
         v.get("coverage"), len(v.get("invalid", [])), int(res["refused"])))
    conn.commit()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("question", nargs="*")
    ap.add_argument("--ym", default=None)
    ap.add_argument("--retrieve-only", action="store_true",
                    help="모델을 부르지 않고 검색과 컨텍스트 조립까지만")
    a = ap.parse_args()

    tokenizer_ko.warmup()             # 없으면 첫 질문이 2초 느려진다
    embedder.warmup()
    conn = connect()
    conn.executescript(LOG_SCHEMA)
    try:
        if a.retrieve_only:
            q = " ".join(a.question) or "AI 규제 법안이 통과된 나라"
            print(f"질문> {q}")
            _ctx, kept, _used, _ms = retrieve(conn, q, a.ym)
            meta = fetch(conn, kept[:5])
            for i, cid in enumerate(kept[:5], 1):
                m = meta.get(cid, {})
                print(f'    {i}. {m.get("year_month","")}  {m.get("title","")[:56]}')
            print("\n  (여기서 모델을 호출합니다)")
            return

        print('질문을 입력하세요. 종료는 빈 줄이나 Ctrl+C.')
        while True:
            try:
                q = input("\n질문> ").strip()
            except (EOFError, KeyboardInterrupt):
                break
            if not q:
                break
            res = answer(conn, q, ym=a.ym)
            render(conn, res)
            log(conn, q, res)
    finally:
        conn.close()


if __name__ == "__main__":
    main()
