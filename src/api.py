# -*- coding: utf-8 -*-
"""검색 비교 데모 API (10-4).

같은 질의를 네 가지 모드로 동시에 돌려 결과를 나란히 돌려준다.
SQLite와 PostgreSQL 두 백엔드를 같은 응답 형태로 감싸므로, 화면에서
DB를 바꿔 가며 같은 질의를 비교할 수 있다.

    uvicorn src.api:app --port 8765
    http://localhost:8765
"""
import sys
import threading
import time
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.responses import FileResponse
from pydantic import BaseModel

from src import config, embedder, tokenizer_ko

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

MODES = ["keyword", "semantic", "fallback", "hybrid"]
STATIC = config.ROOT / "src" / "static"

_local = threading.local()          # SQLite 연결은 스레드마다 따로 (아래 설명)
_pg_lock = threading.Lock()


@asynccontextmanager
async def lifespan(app):
    """Kiwi 사전 적재와 임베딩 첫 호출을 서버가 뜰 때 끝낸다 (04-4).

    빼먹으면 첫 검색만 2초가 넘게 걸려 화면의 지연 숫자가 거짓말을 한다.
    """
    tokenizer_ko.warmup()
    try:
        embedder.warmup()
    except Exception as e:
        print(f"[경고] 임베딩 예열 실패 — 의미·하이브리드 모드가 동작하지 않습니다: {e}")
    yield
    for conn in (getattr(app.state, "pg", None),):
        if conn is not None:
            conn.close()


app = FastAPI(title="하이브리드 검색 비교 데모", lifespan=lifespan)


# ── 백엔드 어댑터 ────────────────────────────────────────────
# 두 검색 모듈의 반환 형태가 다르다. SQLite판은 (chunk_id, score, hits)만
# 주고 본문은 fetch()로 따로 꺼내야 하고, PostgreSQL판은 SQL 한 번에
# 본문까지 실어 온다. 화면이 이 차이를 알 필요는 없으므로 여기서 맞춘다.

def _sqlite_conn():
    if getattr(_local, "conn", None) is None:
        from src.build_sqlite import connect
        _local.conn = connect()
    return _local.conn


def _pg_conn():
    if getattr(app.state, "pg", None) is None:
        from src.search_pg import connect
        app.state.pg = connect()
    return app.state.pg


def _norm_hits(hits):
    """['keyword#3', 'semantic#1'] 형태로 통일한다.

    SQLite판은 리스트로, PostgreSQL판은 쉼표로 이은 문자열로 준다.
    """
    if isinstance(hits, str):
        return [h.strip() for h in hits.split(",") if h.strip()]
    return list(hits or [])


def _pick(m):
    """화면이 쓰는 항목만 추린다."""
    return {"doc_id": m.get("doc_id"), "year_month": m.get("year_month"),
            "title": m.get("title") or "", "body": m.get("body") or ""}


def search_sqlite(query, mode, ym, top_n):
    from src.search_sqlite import fetch, search
    conn = _sqlite_conn()
    res = search(conn, query, mode=mode, ym=ym, top_n=top_n)
    meta = fetch(conn, [c for c, _, _ in res])          # 본문을 따로 꺼낸다
    return [{"rank": i, "chunk_id": cid, "score": float(sc),
             "hits": _norm_hits(hits), **_pick(meta.get(cid, {}))}
            for i, (cid, sc, hits) in enumerate(res, 1)]


def search_postgres(query, mode, ym, top_n):
    from src.search_pg import search
    with _pg_lock:                  # psycopg 연결 하나를 여러 요청이 쓰지 않도록
        res = search(_pg_conn(), query, mode=mode, ym=ym, top_n=top_n)
    return [{"rank": i, "chunk_id": cid, "score": float(sc),
             "hits": _norm_hits(hits),
             **_pick({"doc_id": (cid or "").split("#")[0], "year_month": y,
                      "title": title, "body": body})}
            for i, (cid, y, title, body, sc, hits) in enumerate(res, 1)]


BACKENDS = {"sqlite": search_sqlite, "postgres": search_postgres}


# ── 요청/응답 ────────────────────────────────────────────────
class SearchRequest(BaseModel):
    query: str
    mode: str = "compare"           # keyword | semantic | fallback | hybrid | compare
    backend: str = "sqlite"
    top_n: int = 10
    ym: str | None = None


def _run(fn, query, mode, ym, top_n):
    t0 = time.perf_counter()
    rows = fn(query, mode, ym, top_n)
    return rows, round((time.perf_counter() - t0) * 1000, 1)


def _agreement(results):
    """모드끼리 상위 결과가 얼마나 겹치는지.

    네 모드가 전혀 다른 답을 주는 질의는 검색기가 흔들리고 있다는 신호이고,
    그대로 08장 평가셋 후보가 된다.
    """
    sets = {m: {r["chunk_id"] for r in rows} for m, rows in results.items()}
    single = sets.get("keyword", set()) | sets.get("semantic", set()) \
        | sets.get("fallback", set())
    hybrid = sets.get("hybrid", set())
    out = {m: {"n": len(sets.get(m, ())),
               "overlap_hybrid": len(sets.get(m, set()) & hybrid)}
           for m in ("keyword", "semantic", "fallback")}
    # 하이브리드에만 있는 결과 = 단일 모드 상위에는 없었는데 RRF가 끌어올린 것
    out["hybrid"] = {"n": len(hybrid), "lifted": len(hybrid - single)}
    return out


@app.post("/api/search")
def api_search(req: SearchRequest):
    fn = BACKENDS.get(req.backend)
    if fn is None:
        return {"error": f"알 수 없는 백엔드: {req.backend}"}
    q = (req.query or "").strip()
    if not q:
        return {"error": "질의가 비어 있습니다"}

    modes = MODES if req.mode == "compare" else [req.mode]
    results, timings = {}, {}
    for m in modes:
        try:
            results[m], timings[m] = _run(fn, q, m, req.ym, req.top_n)
        except Exception as e:      # 한 모드가 죽어도 나머지는 보여 준다
            results[m], timings[m] = [], None
            timings[f"{m}_error"] = str(e)[:200]

    return {"query": q, "backend": req.backend, "ym": req.ym,
            "tokens": tokenizer_ko.tokenize(q),
            "fts_query": tokenizer_ko.to_fts_query(q, op="OR"),
            "results": results, "timings_ms": timings,
            "agreement": _agreement(results) if req.mode == "compare" else None}


@app.get("/api/options")
def api_options():
    conn = _sqlite_conn()
    rows = conn.execute("SELECT year_month, count(*) n FROM chunks"
                        " GROUP BY 1 ORDER BY 1").fetchall()
    return {"year_months": [{"value": r["year_month"], "count": r["n"]} for r in rows],
            "modes": MODES,
            "weights": {"keyword": config.W_KEYWORD, "semantic": config.W_SEMANTIC,
                        "fallback": config.W_FALLBACK, "rrf_k": config.RRF_K}}


@app.get("/api/health")
def api_health():
    out = {"embed_backend": config.EMBED_BACKEND, "embed_dim": config.EMBED_DIM}
    try:
        out["sqlite"] = _sqlite_conn().execute("SELECT count(*) FROM chunks").fetchone()[0]
    except Exception as e:
        out["sqlite"], out["sqlite_error"] = None, str(e)[:200]
    try:
        with _pg_lock, _pg_conn().cursor() as cur:
            cur.execute("SELECT count(*) FROM rag_chunks")
            out["postgres"] = cur.fetchone()[0]
    except Exception as e:
        out["postgres"], out["postgres_error"] = None, str(e)[:200]
    try:
        embedder.embed(["ping"])
        out["embed"] = "ok"
    except Exception as e:
        out["embed"], out["embed_error"] = "down", str(e)[:200]
    return out


@app.get("/")
def index():
    return FileResponse(STATIC / "index.html")
