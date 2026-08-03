# -*- coding: utf-8 -*-
"""SQLite에 만든 색인을 PostgreSQL로 옮긴다.

  tsvector + GIN     키워드 (Kiwi 토큰화 텍스트, setweight A/B)
  pgvector + HNSW    의미
  pg_bigm + GIN      폴백 (원문 그대로)

사용:
  python -m src.build_pg                 # 테이블 생성 + 적재 + 인덱스
  python -m src.build_pg --skip-index    # 적재만
"""
import argparse
import json
import sys
import time

import psycopg

from src import config, tokenizer_ko
from src.embedder import embed_batched

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

DDL = """
CREATE EXTENSION IF NOT EXISTS vector;
CREATE EXTENSION IF NOT EXISTS pg_bigm;

DROP TABLE IF EXISTS rag_chunks;
CREATE TABLE rag_chunks (
    id          BIGSERIAL PRIMARY KEY,
    chunk_id    TEXT UNIQUE NOT NULL,
    doc_id      TEXT NOT NULL,
    year_month  TEXT,
    title       TEXT,
    body        TEXT NOT NULL,
    body_tsv    TSVECTOR,
    seq         INTEGER,
    source      TEXT,
    embedding   VECTOR(%(dim)s)
);
"""

INDEXES = [
    ("GIN(body_tsv)",
     "CREATE INDEX rag_chunks_tsv_idx ON rag_chunks USING GIN (body_tsv)"),
    ("GIN(body gin_bigm_ops)",
     "CREATE INDEX rag_chunks_bigm_idx ON rag_chunks USING GIN (body gin_bigm_ops)"),
    ("HNSW(embedding)",
     "CREATE INDEX rag_chunks_hnsw_idx ON rag_chunks "
     "USING hnsw (embedding vector_cosine_ops) WITH (m=16, ef_construction=64)"),
    ("btree(year_month)",
     "CREATE INDEX rag_chunks_ym_idx ON rag_chunks (year_month)"),
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--skip-index", action="store_true")
    ap.add_argument("--backend", default=None)
    a = ap.parse_args()

    src = config.DATA / "chunks.jsonl"
    rows = [json.loads(l) for l in src.read_text(encoding="utf-8").splitlines() if l.strip()]
    print(f"청크 {len(rows):,}개")

    # 임베딩 (SQLite와 같은 모델·같은 텍스트)
    t0 = time.perf_counter()
    texts = [(r["title"] + "\n" + r["body"]).strip() for r in rows]
    vecs = embed_batched(texts, batch=32, backend=a.backend)
    dim = len(vecs[0])
    print(f"  임베딩 {time.perf_counter()-t0:6.2f}s  (차원 {dim})")

    conn = psycopg.connect(config.DATABASE_URL, autocommit=False)
    cur = conn.cursor()
    cur.execute(DDL % {"dim": dim})
    conn.commit()
    print("  스키마 생성 완료")

    # 적재 — 인덱스 없는 상태에서 넣는다 (HNSW 빌드 가속)
    t0 = time.perf_counter()
    batch = []
    for r, v in zip(rows, vecs):
        batch.append((
            r["chunk_id"], r["doc_id"], r["year_month"], r["title"], r["body"],
            tokenizer_ko.tokenized(r["title"] or ""),
            tokenizer_ko.tokenized(r["body"]),
            r["seq"], r["source"], "[" + ",".join(f"{x:.6f}" for x in v) + "]",
        ))
    cur.executemany("""
        INSERT INTO rag_chunks
          (chunk_id, doc_id, year_month, title, body, body_tsv, seq, source, embedding)
        VALUES (%s,%s,%s,%s,%s,
                setweight(to_tsvector('simple', %s), 'A') ||
                setweight(to_tsvector('simple', %s), 'B'),
                %s,%s,%s::vector)
    """, batch)
    conn.commit()
    print(f"  적재 {time.perf_counter()-t0:6.2f}s")

    if not a.skip_index:
        for label, sql in INDEXES:
            t0 = time.perf_counter()
            cur.execute(sql)
            conn.commit()
            print(f"  인덱스 {label:26s} {time.perf_counter()-t0:6.2f}s")
        # VACUUM은 트랜잭션 블록 안에서 실행할 수 없다.
        # psycopg는 기본이 트랜잭션 모드이므로 잠깐 autocommit으로 바꾼다.
        conn.commit()
        conn.autocommit = True
        cur.execute("VACUUM ANALYZE rag_chunks")
        conn.autocommit = False
        print("  VACUUM ANALYZE 완료")

    cur.execute("SELECT count(*), pg_size_pretty(pg_total_relation_size('rag_chunks')) FROM rag_chunks")
    n, size = cur.fetchone()
    print(f"\n완료: {n:,}행 / {size}")
    conn.close()


if __name__ == "__main__":
    main()
