# -*- coding: utf-8 -*-
"""SQLite 하이브리드 검색 — 키워드(FTS5) + 의미(sqlite-vec) + 폴백(trigram) + RRF.

각 검색기는 (chunk_id, 점수) 목록을 돌려주고, RRF가 순위만 보고 합친다.

사용:
  python -m src.search_sqlite "생성형 AI 투자 동향"
  python -m src.search_sqlite "생성형 AI 투자 동향" --mode keyword
  python -m src.search_sqlite "AI 규제" --ym 2026-03
"""
import argparse
import sys
import time

from src import config, tokenizer_ko
from src.build_sqlite import connect, pack
from src.embedder import embed

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")


def _ym_filter(ym, alias="c", no_toc=False):
    """메타데이터 조건절을 만든다 (09-4).

    ym 은 문자열 하나도 되고 목록도 된다 — "상반기" 같은 요청을 받으려면
    목록이 필요하다. no_toc 는 파생 메타데이터(src/meta.py)로 표시해 둔
    목차 구간을 뺀다. 열이 없는 옛 색인에서는 조용히 무시한다.
    """
    sql, params = "", []
    if ym:
        if isinstance(ym, str):
            ym = [ym]
        sql += f" AND {alias}.year_month IN ({','.join('?' * len(ym))}) "
        params += list(ym)
    if no_toc:
        sql += f" AND COALESCE({alias}.is_toc, 0) = 0 "
    return sql, params


def search_keyword(conn, query, k=None, ym=None, no_toc=False):
    """FTS5 + BM25. 질의도 색인과 같은 Kiwi 토크나이저를 통과시킨다."""
    k = k or config.TOPK_PER_MODE
    match = tokenizer_ko.to_fts_query(query, op="OR")
    if not match:
        return []
    where, params = _ym_filter(ym, no_toc=no_toc)
    sql = f"""
        SELECT c.id, c.chunk_id, bm25(chunks_fts, 3.0, 1.0) AS score
        FROM chunks_fts f
        JOIN chunks c ON c.id = f.rowid
        WHERE chunks_fts MATCH ? {where}
        ORDER BY score ASC
        LIMIT ?
    """
    rows = conn.execute(sql, [match] + params + [k]).fetchall()
    # bm25()는 값이 작을수록 좋다 -> 부호를 뒤집어 "클수록 좋음"으로 통일
    return [(r["chunk_id"], -r["score"]) for r in rows]


def search_semantic(conn, query, k=None, ym=None, backend=None, no_toc=False):
    """sqlite-vec KNN. 거리(작을수록 좋음)를 유사도로 뒤집어 돌려준다."""
    k = k or config.TOPK_PER_MODE
    qv = embed([query], backend=backend)[0]
    # 필터가 있으면 후보를 넉넉히 뽑아 파이썬에서 거른다
    # (sqlite-vec은 KNN에 조건을 함께 걸 수 없다)
    fetch = k * 8 if (ym or no_toc) else k
    rows = conn.execute("""
        SELECT v.rowid AS id, distance
        FROM chunks_vec v
        WHERE v.embedding MATCH ? AND k = ?
        ORDER BY distance
    """, (pack(qv), fetch)).fetchall()
    if not rows:
        return []
    id_list = [r["id"] for r in rows]
    where, params = _ym_filter(ym, no_toc=no_toc)
    q = f"SELECT id, chunk_id FROM chunks WHERE id IN ({','.join('?' * len(id_list))})"
    meta = {r["id"]: r["chunk_id"] for r in conn.execute(q, id_list)}
    if where:
        ok = {r["id"] for r in conn.execute(
            f"SELECT id FROM chunks c WHERE c.id IN ({','.join('?' * len(id_list))}){where}",
            id_list + params)}
        rows = [r for r in rows if r["id"] in ok]
    return [(meta[r["id"]], -r["distance"]) for r in rows[:k] if r["id"] in meta]


def search_fallback(conn, query, k=None, ym=None, no_toc=False):
    """trigram 폴백. 형태소가 실패하는 질의(신조어·오타·영문 혼합)를 건진다.

    trigram은 부분 문자열 검색이므로 질의 전체를 한 구(phrase)로 넣으면
    띄어쓰기까지 그대로 일치해야 하고, 다단어 한국어 질의는 거의 0건이 된다.
    그래서 어절로 나눠 3글자 이상인 것만 각각 찾고 순위를 합친다.
    """
    k = k or config.TOPK_PER_MODE
    words = [w.strip() for w in query.split() if len(w.strip()) >= 3]
    if not words:
        w = query.strip()
        words = [w] if len(w) >= 3 else []
    if not words:
        return []

    where, params = _ym_filter(ym, no_toc=no_toc)
    sql = f"""
        SELECT c.chunk_id, bm25(chunks_tri) AS score
        FROM chunks_tri t
        JOIN chunks c ON c.id = t.rowid
        WHERE chunks_tri MATCH ? {where}
        ORDER BY score ASC
        LIMIT ?
    """
    merged = {}
    for w in words:
        try:
            rows = conn.execute(
                sql, ['"' + w.replace('"', '""') + '"'] + params + [k]).fetchall()
        except Exception:
            continue
        for rank, r in enumerate(rows, start=1):
            # 어절별 순위의 역수를 더해 합산 (RRF와 같은 발상)
            merged[r["chunk_id"]] = merged.get(r["chunk_id"], 0.0) + 1.0 / (10 + rank)
    return sorted(merged.items(), key=lambda x: -x[1])[:k]


def rrf_fuse(ranked_lists, k=None, weights=None, top_n=None):
    """Reciprocal Rank Fusion.

    ranked_lists: [(이름, [(chunk_id, 점수), ...]), ...]
    각 목록에서 순위(1부터)만 사용한다 — 점수 체계가 달라도 섞을 수 있다.
    """
    k = k or config.RRF_K
    top_n = top_n or config.TOP_N
    weights = weights or {}
    scores, hits = {}, {}
    for name, lst in ranked_lists:
        w = weights.get(name, 1.0)
        for rank, (cid, _s) in enumerate(lst, start=1):
            scores[cid] = scores.get(cid, 0.0) + w / (k + rank)
            hits.setdefault(cid, []).append(f"{name}#{rank}")
    out = sorted(scores.items(), key=lambda x: -x[1])[:top_n]
    return [(cid, sc, hits[cid]) for cid, sc in out]


def search(conn, query, mode="hybrid", ym=None, top_n=None, backend=None,
           no_toc=False):
    """ym 은 "2026-03" 도 되고 ["2026-01", "2026-02"] 도 된다 (09-4)."""
    top_n = top_n or config.TOP_N
    if mode == "keyword":
        res = search_keyword(conn, query, ym=ym, no_toc=no_toc)
        return [(cid, sc, ["keyword"]) for cid, sc in res[:top_n]]
    if mode == "semantic":
        res = search_semantic(conn, query, ym=ym, backend=backend, no_toc=no_toc)
        return [(cid, sc, ["semantic"]) for cid, sc in res[:top_n]]
    if mode == "fallback":
        res = search_fallback(conn, query, ym=ym, no_toc=no_toc)
        return [(cid, sc, ["fallback"]) for cid, sc in res[:top_n]]

    lists = [
        ("keyword", search_keyword(conn, query, ym=ym, no_toc=no_toc)),
        ("semantic", search_semantic(conn, query, ym=ym, backend=backend,
                                     no_toc=no_toc)),
        ("fallback", search_fallback(conn, query, ym=ym, no_toc=no_toc)),
    ]
    return rrf_fuse(lists, weights={
        "keyword": config.W_KEYWORD,
        "semantic": config.W_SEMANTIC,
        "fallback": config.W_FALLBACK,
    }, top_n=top_n)


def fetch(conn, chunk_ids):
    if not chunk_ids:
        return {}
    q = f"SELECT * FROM chunks WHERE chunk_id IN ({','.join('?' * len(chunk_ids))})"
    return {r["chunk_id"]: dict(r) for r in conn.execute(q, list(chunk_ids))}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("query")
    ap.add_argument("--mode", default="hybrid",
                    choices=["hybrid", "keyword", "semantic", "fallback"])
    ap.add_argument("--ym", default=None, help="예: 2026-03")
    ap.add_argument("--top", type=int, default=8)
    ap.add_argument("--backend", default=None)
    a = ap.parse_args()

    conn = connect()
    t0 = time.perf_counter()
    res = search(conn, a.query, mode=a.mode, ym=a.ym, top_n=a.top, backend=a.backend)
    ms = (time.perf_counter() - t0) * 1000
    meta = fetch(conn, [c for c, _, _ in res])

    print(f'질의: "{a.query}"  모드: {a.mode}'
          + (f"  필터: {a.ym}" if a.ym else "")
          + f"  ({ms:.0f} ms)\n")
    if not res:
        print("  결과 없음")
    for i, (cid, score, hits) in enumerate(res, 1):
        m = meta.get(cid, {})
        body = " ".join((m.get("body") or "").split())
        print(f"{i:2d}. [{score:8.5f}] {m.get('year_month','')}  {m.get('title','')[:40]}")
        print(f"     {body[:110]}...")
        print(f"     ({', '.join(hits)})")
    conn.close()


if __name__ == "__main__":
    main()
