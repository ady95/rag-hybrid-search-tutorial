# -*- coding: utf-8 -*-
"""규모를 키웠을 때 ROW_NUMBER 함정이 얼마나 커지는지 실증한다.

489개 청크로는 차이가 안 보인다. 그래서 같은 청크를 복제해 행 수만 늘린
인위적 테이블(rag_scale)을 만들어 규모 효과만 분리해서 본다.
본문 내용은 원본과 같으므로 검색 품질 실험에는 쓸 수 없고,
오직 "행이 많아지면 무슨 일이 생기는가"를 보는 용도다.

사용:
  python -m src.bench_scale --factor 400   # 489 x 400 = 약 19.6만 행
  python -m src.bench_scale --measure
"""
import argparse
import sys
import time

import psycopg

from src import config
from src.search_pg import connect

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")


def build(conn, factor):
    cur = conn.cursor()
    print(f"복제 배수 {factor} — rag_scale 생성 중...")
    t0 = time.perf_counter()
    cur.execute("DROP TABLE IF EXISTS rag_scale")
    cur.execute("""
        CREATE TABLE rag_scale AS
        SELECT (g.i * 100000 + c.id)        AS id,
               c.chunk_id || '-' || g.i     AS chunk_id,
               c.doc_id, c.year_month, c.title, c.body, c.body_tsv,
               c.seq, c.source, c.embedding
        FROM rag_chunks c, generate_series(1, %s) AS g(i)
    """, (factor,))
    conn.commit()
    cur.execute("SELECT count(*) FROM rag_scale")
    n = cur.fetchone()[0]
    print(f"  적재 {time.perf_counter()-t0:6.1f}s  ({n:,}행)")

    for label, sql in [
        ("GIN(body_tsv)", "CREATE INDEX rag_scale_tsv_idx ON rag_scale USING GIN (body_tsv)"),
        ("GIN(bigm)", "CREATE INDEX rag_scale_bigm_idx ON rag_scale USING GIN (body gin_bigm_ops)"),
        ("HNSW", "CREATE INDEX rag_scale_hnsw_idx ON rag_scale "
                 "USING hnsw (embedding vector_cosine_ops) WITH (m=16, ef_construction=64)"),
    ]:
        t0 = time.perf_counter()
        cur.execute(sql)
        conn.commit()
        print(f"  인덱스 {label:16s} {time.perf_counter()-t0:7.1f}s")

    conn.autocommit = True
    cur.execute("VACUUM ANALYZE rag_scale")
    conn.autocommit = False
    cur.execute("SELECT pg_size_pretty(pg_total_relation_size('rag_scale'))")
    print(f"  테이블 크기 {cur.fetchone()[0]}")


TRAP = """
SELECT id, ROW_NUMBER() OVER (
         ORDER BY ts_rank_cd(body_tsv, to_tsquery('simple', %(tsq)s), 1) DESC) rk
FROM rag_scale WHERE body_tsv @@ to_tsquery('simple', %(tsq)s) LIMIT 30
"""

FIXED = """
SELECT id, ROW_NUMBER() OVER () rk FROM (
  SELECT id FROM rag_scale WHERE body_tsv @@ to_tsquery('simple', %(tsq)s)
  ORDER BY ts_rank_cd(body_tsv, to_tsquery('simple', %(tsq)s), 1) DESC
  LIMIT 30) k
"""

CAPPED = """
SELECT id, ROW_NUMBER() OVER () rk FROM (
  SELECT id FROM (
    SELECT id, body_tsv FROM rag_scale
    WHERE body_tsv @@ to_tsquery('simple', %(tsq)s) LIMIT %(cand)s) c
  ORDER BY ts_rank_cd(body_tsv, to_tsquery('simple', %(tsq)s), 1) DESC
  LIMIT 30) k
"""


def timeit(cur, sql, params, n=5):
    cur.execute(sql, params)
    cur.fetchall()
    ts = []
    for _ in range(n):
        t0 = time.perf_counter()
        cur.execute(sql, params)
        cur.fetchall()
        ts.append((time.perf_counter() - t0) * 1000)
    ts.sort()
    return ts[len(ts) // 2]


def measure(conn):
    cur = conn.cursor()
    cur.execute("SELECT count(*) FROM rag_scale")
    total = cur.fetchone()[0]
    print(f"rag_scale 총 {total:,}행\n")

    for label, tsq in [("넓은 질의", "'ai' | '규제' | '법안'"),
                       ("좁은 질의", "'제네시스' & '미션'")]:
        p = {"tsq": tsq, "cand": 2000}
        cur.execute("SELECT count(*) FROM rag_scale WHERE body_tsv @@ to_tsquery('simple', %(tsq)s)", p)
        matched = cur.fetchone()[0]
        t_trap = timeit(cur, TRAP, p)
        t_fixed = timeit(cur, FIXED, p)
        t_cap = timeit(cur, CAPPED, p)
        print(f"[{label}]  tsquery = {tsq}")
        print(f"  매칭 행 수                      {matched:,}")
        print(f"  (a) ROW_NUMBER + LIMIT (함정)   {t_trap:8.1f} ms")
        print(f"  (b) ORDER BY..LIMIT 후 번호     {t_fixed:8.1f} ms   ({t_trap/max(t_fixed,0.01):5.1f}배 빠름)")
        print(f"  (c) 후보 2000 제한 + 랭킹       {t_cap:8.1f} ms   ({t_trap/max(t_cap,0.01):5.1f}배 빠름)")
        print()

    # EXPLAIN 으로 근거 확인
    p = {"tsq": "'ai' | '규제' | '법안'", "cand": 2000}
    for label, sql in [("함정", TRAP), ("교정", FIXED), ("후보제한", CAPPED)]:
        cur.execute("EXPLAIN (ANALYZE, COSTS OFF) " + sql, p)
        plan = [r[0] for r in cur.fetchall()]
        srt = [l.strip() for l in plan if "Sort " in l or "Sort Method" in l]
        ex = [l.strip() for l in plan if "Execution Time" in l]
        print(f"  [{label}] {srt[0][:80] if srt else ''}")
        for s in srt[1:2]:
            print(f"          {s[:80]}")
        print(f"          {ex[0] if ex else ''}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--factor", type=int, default=0)
    ap.add_argument("--measure", action="store_true")
    ap.add_argument("--drop", action="store_true")
    a = ap.parse_args()

    conn = connect()
    if a.drop:
        conn.cursor().execute("DROP TABLE IF EXISTS rag_scale")
        conn.commit()
        print("rag_scale 삭제 완료")
    if a.factor:
        build(conn, a.factor)
    if a.measure or not (a.factor or a.drop):
        measure(conn)
    conn.close()


if __name__ == "__main__":
    main()
