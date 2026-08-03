# -*- coding: utf-8 -*-
"""SQLite와 PostgreSQL 비교 벤치마크 + ROW_NUMBER 함정 실증.

07-5, 08-3 실습에서 쓴다.

사용:
  python -m src.bench
"""
import json
import statistics
import sys
import time

import psycopg

from src import config
from src.build_sqlite import connect as sq_connect
from src.evaluate import hit_at_k, load, recall_at_k, rr
from src.search_pg import RRF_SQL, build_params
from src.search_pg import connect as pg_connect
from src.search_pg import search as pg_search
from src.search_sqlite import search as sq_search

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

QUERIES = [
    "AI 규제 법안이 통과된 나라",
    "한국이 개발한 대규모 언어 모델",
    "AI 반도체와 컴퓨팅 인프라 투자",
    "AI 안전성과 정렬 문제 연구",
    "기업의 AI 도입과 생산성",
    "영상 생성 AI 모델 경쟁",
]


def timeit(fn, n=7):
    fn()                       # 예열
    ts = []
    for _ in range(n):
        t0 = time.perf_counter()
        fn()
        ts.append((time.perf_counter() - t0) * 1000)
    ts.sort()
    return ts[len(ts) // 2], ts[-1]


def bench_latency(sq, pg):
    print("=" * 78)
    print("1. 같은 질의, 같은 모드 — SQLite vs PostgreSQL 지연 (ms, p50/p95)")
    print("=" * 78)
    print(f"{'질의':<26}{'SQLite hybrid':>18}{'PG hybrid':>16}{'PG keyword':>14}")
    print("-" * 78)
    rows = []
    for q in QUERIES:
        s50, s95 = timeit(lambda: sq_search(sq, q, mode="hybrid", top_n=10))
        p50, p95 = timeit(lambda: pg_search(pg, q, mode="hybrid", top_n=10))
        k50, _ = timeit(lambda: pg_search(pg, q, mode="keyword", top_n=10))
        print(f"{q[:24]:<26}{s50:8.0f}/{s95:<9.0f}{p50:7.0f}/{p95:<8.0f}{k50:13.0f}")
        rows.append((s50, p50, k50))
    print("-" * 78)
    print(f"{'평균':<26}{statistics.mean(r[0] for r in rows):8.0f}"
          f"{statistics.mean(r[1] for r in rows):17.0f}"
          f"{statistics.mean(r[2] for r in rows):16.0f}")
    print("\n  주: hybrid 지연의 대부분은 질의 임베딩 생성(HTTP 왕복)이다.")
    print("     PG keyword는 임베딩이 필요 없어 순수 DB 시간을 보여 준다.")


def bench_overlap(sq, pg):
    print("\n" + "=" * 78)
    print("2. 두 DB의 상위 10건 일치도")
    print("=" * 78)
    tot = 0
    for q in QUERIES:
        a = [r[0] for r in sq_search(sq, q, mode="hybrid", top_n=10)]
        b = [r[0] for r in pg_search(pg, q, mode="hybrid", top_n=10)]
        ov = len(set(a) & set(b))
        tot += ov
        print(f"  {q[:30]:<32} {ov:2d}/10")
    print(f"\n  평균 겹침 {tot/len(QUERIES):.1f}/10")
    print("  주: BM25(SQLite)와 ts_rank_cd(PG)는 다른 알고리즘이라 순위가 달라진다.")


def bench_rownumber_trap(pg):
    print("\n" + "=" * 78)
    print("3. ROW_NUMBER 함정 — LIMIT은 윈도우 함수 계산 뒤에 적용된다")
    print("=" * 78)
    q = "AI 규제 법안"
    p = build_params(q, top_n=10, need_vec=False)

    trap = """
        SELECT id, ROW_NUMBER() OVER (
                 ORDER BY ts_rank_cd(body_tsv, to_tsquery('simple', %(tsq)s), 1) DESC) rk
        FROM rag_chunks WHERE body_tsv @@ to_tsquery('simple', %(tsq)s) LIMIT 30
    """
    fixed = """
        SELECT id, ROW_NUMBER() OVER () rk FROM (
          SELECT id FROM rag_chunks WHERE body_tsv @@ to_tsquery('simple', %(tsq)s)
          ORDER BY ts_rank_cd(body_tsv, to_tsquery('simple', %(tsq)s), 1) DESC
          LIMIT 30) k
    """
    with pg.cursor() as cur:
        cur.execute("SELECT count(*) FROM rag_chunks WHERE body_tsv @@ to_tsquery('simple', %(tsq)s)", p)
        matched = cur.fetchone()[0]
        print(f"  이 질의에 매칭되는 행: {matched}개 (전체 489개)\n")
        for label, sql in [("함정(전량 정렬)", trap), ("교정(후보 먼저 자름)", fixed)]:
            cur.execute("EXPLAIN (ANALYZE, COSTS OFF) " + sql, p)
            plan = [r[0] for r in cur.fetchall()]
            sort_rows = [l.strip() for l in plan if "Sort" in l or "WindowAgg" in l]
            exec_ms = [l for l in plan if "Execution Time" in l]
            print(f"  [{label}]")
            for s in sort_rows[:3]:
                print(f"    {s[:96]}")
            print(f"    {exec_ms[0].strip() if exec_ms else ''}\n")


def bench_quality(sq, pg):
    print("=" * 78)
    print("4. 검색 품질 — 같은 평가셋을 두 DB에서")
    print("=" * 78)
    items = [x for x in load() if x.get("kind") == "manual"]
    if not items:
        print("  수동 평가셋 없음 — 건너뜀")
        return
    for name, fn in [("SQLite", lambda q: sq_search(sq, q, mode="hybrid", top_n=10)),
                     ("PostgreSQL", lambda q: pg_search(pg, q, mode="hybrid", top_n=10))]:
        h1 = h5 = r10 = mrr = 0.0
        for it in items:
            ranked = [r[0] for r in fn(it["query"])]
            h1 += hit_at_k(ranked, it["gold"], 1)
            h5 += hit_at_k(ranked, it["gold"], 5)
            r10 += recall_at_k(ranked, it["gold"], 10)
            mrr += rr(ranked, it["gold"])
        n = len(items)
        print(f"  {name:<12} Hit@1 {h1/n:.3f}   Hit@5 {h5/n:.3f}   "
              f"Recall@10 {r10/n:.3f}   MRR {mrr/n:.3f}")


def main():
    sq = sq_connect()
    pg = pg_connect()
    sq_search(sq, "예열", mode="hybrid", top_n=1)
    pg_search(pg, "예열", mode="hybrid", top_n=1)

    bench_latency(sq, pg)
    bench_overlap(sq, pg)
    bench_rownumber_trap(pg)
    bench_quality(sq, pg)

    sq.close()
    pg.close()


if __name__ == "__main__":
    main()
