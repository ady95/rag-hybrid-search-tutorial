# -*- coding: utf-8 -*-
"""파생 메타데이터 (09-4).

원문에 없는 메타데이터를 색인 시점에 만들어 넣는다. 지금은 목차 구간
표시 하나뿐이지만, 문서 종류·부서·기밀 등급 같은 축도 같은 자리에 붙인다.

    python -m src.meta            # chunks.is_toc 를 채운다 (재실행 안전)
"""
import re
import sys

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

PAGE = re.compile(r"\s\d{1,3}$")
FRONT_SEQ = 8          # 이 위치보다 앞쪽만 목차 후보로 본다


def is_toc(seq, title, body):
    """목차 항목인가.

    두 조건을 함께 건다. 페이지 번호 조건만 쓰면 제목이 우연히 숫자로 끝난
    본문까지 걸리고(489개 중 79개, 그중 45개가 오탐), 앞부분 조건만 쓰면
    문서마다 목차 길이가 달라 놓친다. 결합하면 42개가 남는다.
    """
    if seq is None or seq >= FRONT_SEQ:
        return False
    tail = (title or "").split(">")[-1].strip()
    lines = [l.strip() for l in (body or "").split("\n") if l.strip()]
    hits = sum(1 for l in lines if PAGE.search(l))
    return bool(PAGE.search(tail)) or bool(lines and hits / len(lines) >= 0.6)


def has_column(conn, table="chunks", column="is_toc"):
    return any(r[1] == column for r in conn.execute(f"PRAGMA table_info({table})"))


def mark_toc(conn):
    """chunks.is_toc 를 만들고 채운다. 여러 번 불러도 안전하다."""
    if not has_column(conn):
        conn.execute("ALTER TABLE chunks ADD COLUMN is_toc INTEGER NOT NULL DEFAULT 0")
        conn.execute("CREATE INDEX IF NOT EXISTS chunks_toc_idx ON chunks(is_toc)")
    rows = conn.execute("SELECT id, seq, title, body FROM chunks").fetchall()
    flags = [(1 if is_toc(r["seq"], r["title"], r["body"]) else 0, r["id"])
             for r in rows]
    conn.executemany("UPDATE chunks SET is_toc = ? WHERE id = ?", flags)
    conn.commit()
    return sum(f for f, _ in flags), len(flags)


def main():
    from src.build_sqlite import connect
    conn = connect()
    n, total = mark_toc(conn)
    print(f"목차로 표시한 청크 {n} / {total}")
    for r in conn.execute("SELECT chunk_id, seq, substr(title,1,46) t"
                          " FROM chunks WHERE is_toc=1 ORDER BY chunk_id LIMIT 6"):
        print(f'  seq={r["seq"]}  {r["chunk_id"]}  {r["t"]}')
    conn.close()


if __name__ == "__main__":
    main()
