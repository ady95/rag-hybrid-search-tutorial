# -*- coding: utf-8 -*-
"""프로젝트 1 — 월간 AI 동향 리포트 생성기 (10-1).

한 달치 문서에서 주제별로 소재를 모으고 절 단위로 요약을 생성한다.
소재 수집과 토큰 집계까지는 LLM 없이 돌아간다.

    python -m src.report --ym 2026-03 --collect-only   # 수집만 (API 키 불필요)
    python -m src.report --ym 2026-03                  # 리포트 생성
"""
import argparse
import os
import sys
import time
from pathlib import Path

from src import config, embedder, tokenizer_ko
from src.build_sqlite import connect
from src.context import BLOCK, count_tokens
from src.search_sqlite import fetch, search
from src.verify import verify_citations

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

# 앞이 리포트에 찍힐 제목, 뒤가 검색어. 둘을 분리한 것이 요령이다 —
# 사람이 읽을 제목과 검색에 잘 걸리는 질의는 다르다 (04-2).
TOPICS = [
    ("AI 규제와 정책", "AI 규제 법안 정책 정부"),
    ("모델 출시와 성능", "AI 모델 출시 성능 벤치마크"),
    ("투자와 시장", "AI 투자 시장 매출 기업"),
    ("AI 안전과 윤리", "AI 안전 윤리 위험 오정렬"),
]

SECTION_SYSTEM = """당신은 AI 동향 리포트를 작성하는 애널리스트입니다.
- <자료>에 있는 내용만으로 3~5문장의 문단을 씁니다.
- 사실을 서술한 모든 문장 끝에 근거 자료의 대괄호 식별자를 붙입니다.
- 자료가 부족하면 "이달 자료에서는 관련 내용이 확인되지 않습니다"라고만 씁니다.
- 없는 숫자·날짜·기관명을 만들지 않습니다."""


def collect(conn, ym, topics=TOPICS, per_topic=5, count_tokens=len):
    out, seen = [], set()
    for title, query in topics:
        res = search(conn, query, mode="hybrid", ym=ym, top_n=per_topic)
        meta = fetch(conn, [c for c, _, _ in res])
        blocks, ids, used = [], [], 0
        for cid, _s, _h in res:
            if cid in seen or cid not in meta:
                continue        # 다른 주제에 이미 쓴 청크는 건너뛴다
            seen.add(cid)
            b = BLOCK.format(**meta[cid])
            blocks.append(b); ids.append(cid); used += count_tokens(b)
        out.append({"title": title, "context": "\n---\n".join(blocks),
                    "ids": ids, "tokens": used})
    return out


def write_section(section, model="claude-sonnet-5"):
    """절 하나를 생성한다. 주제마다 따로 부르는 이유는 10-1 관찰 포인트 참고."""
    from anthropic import Anthropic
    if not section["ids"]:
        return "이달 자료에서는 관련 내용이 확인되지 않습니다."
    client = Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])
    r = client.messages.create(
        model=model, max_tokens=800, system=SECTION_SYSTEM,
        messages=[{"role": "user", "content":
                   f"<자료>\n{section['context']}\n</자료>\n\n"
                   f"주제: {section['title']}\n이 주제의 이달 동향을 정리하세요."}])
    return "".join(b.text for b in r.content if b.type == "text")


def build_report(conn, ym, out_path):
    sections = collect(conn, ym, count_tokens=count_tokens)
    lines = [f"# {ym[:4]}년 {int(ym[5:]):d}월 AI 동향 리포트", ""]
    total_ids, flagged = [], []

    for s in sections:
        text = write_section(s)
        v = verify_citations(text, s["ids"])
        if v["invalid"]:
            flagged.append((s["title"], v["invalid"]))
            text += f"\n\n> 검증 경고: 확인되지 않는 인용 {v['invalid']}"
        total_ids += v["cited"]
        lines += [f"## {s['title']}", "", text, ""]

    lines += ["---",
              f"근거 자료 {len(set(total_ids))}건 · SPRi AI 브리프 {ym} · 생성 자동"]
    Path(out_path).write_text("\n".join(lines), encoding="utf-8")
    return flagged


def show_collect(conn, ym):
    """LLM 없이 수집 결과만 표로 찍는다."""
    seen, tc, tt = set(), 0, 0
    print(f'{"주제":<18}{"청크":>6}{"토큰":>9}{"소요":>11}')
    print("-" * 46)
    for title, query in TOPICS:
        t0 = time.perf_counter()
        res = search(conn, query, mode="hybrid", ym=ym, top_n=5)
        ms = (time.perf_counter() - t0) * 1000
        meta = fetch(conn, [c for c, _, _ in res])
        used = n = 0
        for cid, _s, _h in res:
            if cid in seen or cid not in meta:
                continue
            seen.add(cid)
            used += count_tokens(BLOCK.format(**meta[cid]))
            n += 1
        tc += n; tt += used
        print(f"{title:<18}{n:>6}{used:>9,}{ms:>9.1f} ms")
    print("-" * 46)
    print(f'{"합계":<18}{tc:>6}{tt:>9,}\n')
    print(f"중복 제거 후 {tc}청크 / {tt:,}토큰 — 주제별 요약 생성 준비 완료")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ym", default="2026-03", help="예: 2026-03")
    ap.add_argument("--collect-only", action="store_true",
                    help="LLM 없이 소재 수집만 확인한다")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()

    tokenizer_ko.warmup()
    embedder.warmup()
    conn = connect()
    try:
        if a.collect_only:
            show_collect(conn, a.ym)
            return
        out = a.out or config.DATA / f"report-{a.ym}.md"
        flagged = build_report(conn, a.ym, out)
        print(f"리포트 저장: {out}")
        for title, bad in flagged:
            print(f"  [검증 경고] {title}: {bad}")
    finally:
        conn.close()


if __name__ == "__main__":
    main()
