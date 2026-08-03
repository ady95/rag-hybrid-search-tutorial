# -*- coding: utf-8 -*-
"""청크를 SQLite에 색인한다 — 원문 + FTS5(키워드) + trigram(폴백) + sqlite-vec(벡터).

테이블 구성
  chunks      원문과 메타데이터 (일반 테이블)
  chunks_fts  FTS5. Kiwi로 토큰화한 텍스트를 넣는다 (키워드 검색)
  chunks_tri  FTS5 trigram. 원문 그대로 넣는다 (미등록어·오타 폴백)
  chunks_vec  sqlite-vec vec0. 1024차원 임베딩 (의미 검색)

사용:
  python -m src.build_sqlite            # 전체 재구축
  python -m src.build_sqlite --no-embed # 임베딩 건너뛰기(키워드만)
"""
import argparse
import json
import sqlite3
import struct
import sys
import time

import sqlite_vec

from src import config, tokenizer_ko
from src.embedder import embed_batched

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")


def connect(path=None):
    conn = sqlite3.connect(path or config.DB_PATH)
    conn.enable_load_extension(True)
    sqlite_vec.load(conn)
    conn.enable_load_extension(False)
    conn.row_factory = sqlite3.Row
    return conn


def pack(vec):
    """float 목록을 sqlite-vec이 받는 바이너리로."""
    return struct.pack(f"{len(vec)}f", *vec)


SCHEMA = """
DROP TABLE IF EXISTS chunks;
DROP TABLE IF EXISTS chunks_fts;
DROP TABLE IF EXISTS chunks_tri;
DROP TABLE IF EXISTS chunks_vec;

CREATE TABLE chunks (
    id          INTEGER PRIMARY KEY,
    chunk_id    TEXT UNIQUE NOT NULL,
    doc_id      TEXT NOT NULL,
    year_month  TEXT,
    title       TEXT,
    body        TEXT NOT NULL,
    seq         INTEGER,
    source      TEXT
);
CREATE INDEX chunks_ym_idx  ON chunks(year_month);
CREATE INDEX chunks_doc_idx ON chunks(doc_id);

-- 키워드: Kiwi 토큰화 텍스트를 색인한다.
-- content='' 로 두어 원문은 chunks 에만 두고 중복 저장을 피한다.
CREATE VIRTUAL TABLE chunks_fts USING fts5(
    title, body, content='', tokenize='unicode61'
);

-- 폴백: 원문 그대로 trigram 색인 (형태소가 실패하는 질의용)
CREATE VIRTUAL TABLE chunks_tri USING fts5(
    body, content='', tokenize='trigram'
);
"""


def build(no_embed=False, backend=None):
    src = config.DATA / "chunks.jsonl"
    if not src.exists():
        print("chunks.jsonl 이 없습니다. python -m src.chunker 를 먼저 실행하세요.")
        return
    rows = [json.loads(l) for l in src.read_text(encoding="utf-8").splitlines() if l.strip()]
    print(f"청크 {len(rows):,}개 적재 시작")

    if config.DB_PATH.exists():
        config.DB_PATH.unlink()
    conn = connect()
    conn.executescript(SCHEMA)

    # 1) 원문
    t0 = time.perf_counter()
    conn.executemany(
        "INSERT INTO chunks(chunk_id, doc_id, year_month, title, body, seq, source)"
        " VALUES (:chunk_id, :doc_id, :year_month, :title, :body, :seq, :source)", rows)
    conn.commit()
    print(f"  원문 적재      {time.perf_counter()-t0:6.2f}s")

    ids = [r[0] for r in conn.execute("SELECT id FROM chunks ORDER BY id")]

    # 2) 키워드 색인 (Kiwi 토큰화)
    t0 = time.perf_counter()
    fts_rows = []
    for rid, r in zip(ids, rows):
        fts_rows.append((rid,
                         tokenizer_ko.tokenized(r["title"] or ""),
                         tokenizer_ko.tokenized(r["body"])))
    conn.executemany("INSERT INTO chunks_fts(rowid, title, body) VALUES (?,?,?)", fts_rows)
    conn.commit()
    tok_time = time.perf_counter() - t0
    avg_tok = sum(len(x[2].split()) for x in fts_rows) / max(1, len(fts_rows))
    print(f"  FTS5 색인      {tok_time:6.2f}s  (청크당 평균 토큰 {avg_tok:.0f}개)")

    # 3) trigram 폴백 색인 (원문 그대로)
    t0 = time.perf_counter()
    conn.executemany("INSERT INTO chunks_tri(rowid, body) VALUES (?,?)",
                     [(rid, r["body"]) for rid, r in zip(ids, rows)])
    conn.commit()
    print(f"  trigram 색인   {time.perf_counter()-t0:6.2f}s")

    # 4) 벡터
    if not no_embed:
        t0 = time.perf_counter()
        texts = [(r["title"] + "\n" + r["body"]).strip() for r in rows]
        vecs = embed_batched(texts, batch=32, backend=backend)
        dim = len(vecs[0])
        conn.execute(f"CREATE VIRTUAL TABLE chunks_vec USING vec0(embedding float[{dim}])")
        conn.executemany("INSERT INTO chunks_vec(rowid, embedding) VALUES (?,?)",
                         [(rid, pack(v)) for rid, v in zip(ids, vecs)])
        conn.commit()
        print(f"  벡터 색인      {time.perf_counter()-t0:6.2f}s  (차원 {dim})")

    # 통계
    n = conn.execute("SELECT count(*) FROM chunks").fetchone()[0]
    size_mb = config.DB_PATH.stat().st_size / 1024 / 1024
    print(f"\n완료: {n:,}행 / {size_mb:.1f} MB / {config.DB_PATH}")
    conn.close()


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-embed", action="store_true")
    ap.add_argument("--backend", default=None, help="local|server|openai")
    a = ap.parse_args()
    build(no_embed=a.no_embed, backend=a.backend)
