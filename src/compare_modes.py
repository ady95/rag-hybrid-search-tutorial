# -*- coding: utf-8 -*-
"""같은 질의를 키워드·의미·폴백·하이브리드로 돌려 결과와 시간을 비교한다.

06-4 실습에서 쓰는 도구다.

사용:
  python -m src.compare_modes
  python -m src.compare_modes --query "AI 규제 법안"
"""
import argparse
import sys
import time

from src import config
from src.build_sqlite import connect
from src.search_sqlite import fetch, search

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

# 유형별로 고른 질의 — 각 검색 방식의 강점과 약점이 드러나도록 구성
QUERIES = [
    ("정확한 용어",   "제네시스 미션 행정명령"),
    ("동의어/의역",   "인공지능에 돈이 얼마나 몰리고 있나"),
    ("영문 고유명사", "K-EXAONE"),
    ("약어",         "AGI"),
    ("서술형 질문",   "AI 때문에 일자리가 줄어들까"),
    ("오타 포함",     "앤트로픽 클로두 오퍼스"),
    ("복합 주제",     "반도체 수출 규제와 중국 AI"),
]

MODES = ["keyword", "semantic", "fallback", "hybrid"]


def run_one(conn, query, backend=None):
    row = {}
    for mode in MODES:
        t0 = time.perf_counter()
        res = search(conn, query, mode=mode, top_n=5, backend=backend)
        row[mode] = {
            "ms": (time.perf_counter() - t0) * 1000,
            "ids": [c for c, _, _ in res],
            "res": res,
        }
    return row


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--query", default=None)
    ap.add_argument("--backend", default=None)
    ap.add_argument("--detail", action="store_true", help="상위 결과 본문까지 출력")
    a = ap.parse_args()

    conn = connect()
    queries = [("사용자 질의", a.query)] if a.query else QUERIES

    # 예열 — Kiwi 사전 적재(첫 tokenize에 약 2초)와 임베딩 모델 적재가
    # 첫 질의 측정에 섞이지 않도록 미리 한 번씩 호출한다.
    search(conn, "예열", mode="keyword", top_n=1)
    search(conn, "예열", mode="semantic", top_n=1, backend=a.backend)

    print(f"{'유형':<12} {'질의':<26} " + "".join(f"{m:>11}" for m in MODES))
    print("-" * 90)
    timing = {m: [] for m in MODES}
    for label, q in queries:
        row = run_one(conn, q, backend=a.backend)
        print(f"{label:<12} {q[:24]:<26} " +
              "".join(f"{row[m]['ms']:9.0f}ms" for m in MODES))
        for m in MODES:
            timing[m].append(row[m]["ms"])

        if a.detail:
            for m in MODES:
                meta = fetch(conn, row[m]["ids"])
                print(f"    [{m}]")
                for i, cid in enumerate(row[m]["ids"][:3], 1):
                    mt = meta.get(cid, {})
                    body = " ".join((mt.get("body") or "").split())
                    print(f"      {i}. {mt.get('year_month','')} {mt.get('title','')[:34]} | {body[:70]}")
            print()

    print("-" * 90)
    for m in MODES:
        ts = sorted(timing[m])
        print(f"  {m:<10} 평균 {sum(ts)/len(ts):7.0f}ms   중앙값 {ts[len(ts)//2]:7.0f}ms   최대 {ts[-1]:7.0f}ms")

    # 모드 간 결과 겹침
    print("\n상위 5건 겹침 (하이브리드 기준)")
    for label, q in queries:
        row = run_one(conn, q, backend=a.backend)
        h = set(row["hybrid"]["ids"])
        ov = {m: len(h & set(row[m]["ids"])) for m in ["keyword", "semantic", "fallback"]}
        print(f"  {q[:26]:<28} keyword {ov['keyword']}/5   semantic {ov['semantic']}/5   fallback {ov['fallback']}/5")
    conn.close()


if __name__ == "__main__":
    main()
