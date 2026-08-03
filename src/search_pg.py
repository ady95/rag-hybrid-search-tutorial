# -*- coding: utf-8 -*-
"""PostgreSQL 하이브리드 검색 — RRF까지 SQL 한 번에 처리한다.

SQLite판(search_sqlite.py)은 세 번 질의하고 파이썬에서 합쳤지만,
여기서는 CTE 세 개를 UNION ALL 로 묶어 왕복 한 번에 끝낸다.

핵심 주의: ROW_NUMBER() OVER (ORDER BY ...) 뒤에 LIMIT 을 걸면
LIMIT이 윈도우 함수 계산 뒤에 적용되어 매칭된 행 전부를 정렬한다.
반드시 서브쿼리에서 ORDER BY ... LIMIT 으로 후보를 먼저 자른 뒤
바깥에서 번호를 매긴다 (08-3 참조).

사용:
  python -m src.search_pg "생성형 AI 투자 동향"
  python -m src.search_pg "AI 규제" --mode keyword --ym 2026-03
"""
import argparse
import sys
import time

import psycopg

from src import config, tokenizer_ko
from src.embedder import embed

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")


def connect(dsn=None):
    return psycopg.connect(dsn or config.DATABASE_URL)


def vec_literal(v):
    return "[" + ",".join(f"{x:.6f}" for x in v) + "]"


RRF_SQL = """
WITH keyword AS (
  SELECT id, ROW_NUMBER() OVER () AS rk FROM (
    SELECT id FROM rag_chunks
    WHERE body_tsv @@ to_tsquery('simple', %(tsq)s)
      AND (%(ym)s::text IS NULL OR year_month = %(ym)s::text)
    ORDER BY ts_rank_cd(body_tsv, to_tsquery('simple', %(tsq)s), 1) DESC
    LIMIT %(k)s) k),
-- 폴백은 질의 전체가 아니라 어절 단위로 찾는다.
-- 질의 전체를 LIKE 패턴으로 감싸면 띄어쓰기까지 그대로 일치해야 해서 거의 0건이 된다.
-- (주석에 백분율 기호를 쓰면 psycopg가 플레이스홀더로 오인하므로 피한다)
fallback AS (
  SELECT id, ROW_NUMBER() OVER (ORDER BY sim DESC) AS rk FROM (
    SELECT c.id, MAX(bigm_similarity(c.body, w.word)) AS sim
    FROM unnest(%(words)s::text[]) AS w(word)
    JOIN LATERAL (
      SELECT id, body FROM rag_chunks
      WHERE body LIKE '%%' || w.word || '%%'
        AND (%(ym)s::text IS NULL OR year_month = %(ym)s::text)
      LIMIT %(fcand)s) c ON TRUE
    GROUP BY c.id
    ORDER BY sim DESC
    LIMIT %(k)s) f),
semantic AS (
  SELECT id, ROW_NUMBER() OVER () AS rk FROM (
    SELECT id FROM rag_chunks
    WHERE (%(ym)s::text IS NULL OR year_month = %(ym)s::text)
    ORDER BY embedding <=> %(vec)s::vector
    LIMIT %(k)s) s)
SELECT c.chunk_id, c.year_month, c.title, c.body,
       SUM(u.w / (%(rrf_k)s + u.rk)) AS rrf,
       string_agg(u.src || '#' || u.rk, ', ' ORDER BY u.rk) AS hits
FROM (
  SELECT id, rk, %(w_kw)s::float AS w, 'keyword'  AS src FROM keyword
  UNION ALL
  SELECT id, rk, %(w_fb)s::float,      'fallback'      FROM fallback
  UNION ALL
  SELECT id, rk, %(w_se)s::float,      'semantic'      FROM semantic
) u
JOIN rag_chunks c ON c.id = u.id
GROUP BY c.chunk_id, c.year_month, c.title, c.body
ORDER BY rrf DESC
LIMIT %(top_n)s
"""

SINGLE_SQL = {
    "keyword": """
        SELECT chunk_id, year_month, title, body,
               ts_rank_cd(body_tsv, to_tsquery('simple', %(tsq)s), 1) AS score
        FROM rag_chunks
        WHERE body_tsv @@ to_tsquery('simple', %(tsq)s)
          AND (%(ym)s::text IS NULL OR year_month = %(ym)s::text)
        ORDER BY score DESC LIMIT %(top_n)s
    """,
    "semantic": """
        SELECT chunk_id, year_month, title, body,
               1 - (embedding <=> %(vec)s::vector) AS score
        FROM rag_chunks
        WHERE (%(ym)s::text IS NULL OR year_month = %(ym)s::text)
        ORDER BY embedding <=> %(vec)s::vector LIMIT %(top_n)s
    """,
    "fallback": """
        SELECT c.chunk_id, c.year_month, c.title, c.body,
               MAX(bigm_similarity(c.body, w.word)) AS score
        FROM unnest(%(words)s::text[]) AS w(word)
        JOIN LATERAL (
          SELECT id, chunk_id, year_month, title, body FROM rag_chunks
          WHERE body LIKE '%%' || w.word || '%%'
            AND (%(ym)s::text IS NULL OR year_month = %(ym)s::text)
          LIMIT %(fcand)s) c ON TRUE
        GROUP BY c.chunk_id, c.year_month, c.title, c.body
        ORDER BY score DESC LIMIT %(top_n)s
    """,
}


def build_params(query, ym=None, top_n=None, backend=None, need_vec=True):
    top_n = top_n or config.TOP_N
    words = [w for w in query.split() if len(w) >= 3] or [query.strip()]
    return {
        "tsq": tokenizer_ko.to_tsquery(query, op="|") or "''",
        "raw": query.strip(),
        "words": words,
        "like": "%" + query.strip() + "%",
        "vec": vec_literal(embed([query], backend=backend)[0]) if need_vec else None,
        "ym": ym,
        "k": config.TOPK_PER_MODE,
        "fcand": config.FALLBACK_CANDIDATES,
        "rrf_k": config.RRF_K,
        "w_kw": config.W_KEYWORD,
        "w_fb": config.W_FALLBACK,
        "w_se": config.W_SEMANTIC,
        "top_n": top_n,
    }


def search(conn, query, mode="hybrid", ym=None, top_n=None, backend=None):
    p = build_params(query, ym, top_n, backend,
                     need_vec=(mode in ("hybrid", "semantic")))
    with conn.cursor() as cur:
        if mode == "hybrid":
            cur.execute(RRF_SQL, p)
            return [(r[0], r[1], r[2], r[3], float(r[4]), r[5]) for r in cur.fetchall()]
        cur.execute(SINGLE_SQL[mode], p)
        return [(r[0], r[1], r[2], r[3], float(r[4]), mode) for r in cur.fetchall()]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("query")
    ap.add_argument("--mode", default="hybrid",
                    choices=["hybrid", "keyword", "semantic", "fallback"])
    ap.add_argument("--ym", default=None)
    ap.add_argument("--top", type=int, default=8)
    ap.add_argument("--backend", default=None)
    a = ap.parse_args()

    conn = connect()
    # 예열
    search(conn, "예열", mode="keyword", top_n=1)
    if a.mode in ("hybrid", "semantic"):
        search(conn, "예열", mode="semantic", top_n=1, backend=a.backend)

    t0 = time.perf_counter()
    res = search(conn, a.query, mode=a.mode, ym=a.ym, top_n=a.top, backend=a.backend)
    ms = (time.perf_counter() - t0) * 1000

    print(f'질의: "{a.query}"  모드: {a.mode}'
          + (f"  필터: {a.ym}" if a.ym else "") + f"  ({ms:.0f} ms)\n")
    for i, (cid, ym, title, body, score, hits) in enumerate(res, 1):
        body = " ".join((body or "").split())
        print(f"{i:2d}. [{score:8.5f}] {ym}  {(title or '')[:44]}")
        print(f"     {body[:110]}...")
        print(f"     ({hits})")
    conn.close()


if __name__ == "__main__":
    main()
