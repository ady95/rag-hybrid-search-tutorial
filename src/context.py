# -*- coding: utf-8 -*-
"""검색 결과를 LLM 컨텍스트로 조립한다 (09-1).

토큰 예산을 넘기 직전에 자르므로 top_n 을 넉넉히 줘도 안전하다.
글자 수로 세면 안 된다 — 한국어는 모델마다 토큰화 비율이 크게 다르므로
count_tokens 에 tiktoken 기반 함수를 넘겨 쓴다.
"""
import sys

from src.search_sqlite import fetch, search

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

BLOCK = """[{chunk_id}]
발행: {year_month}  출처: {source}
제목: {title}
{body}
"""


def get_encoder():
    """최신 세대 모델은 o200k 계열 인코딩을 쓴다.

    tiktoken 0.5 이하에는 없으므로 cl100k 로 물러선다.
    """
    import tiktoken
    for name in ("o200k_base", "cl100k_base"):
        try:
            return tiktoken.get_encoding(name)
        except ValueError:
            continue
    raise RuntimeError("tiktoken 인코딩을 찾지 못했습니다")


_enc = None


def count_tokens(s):
    """토큰 수를 센다. tiktoken 은 OpenAI 계열이라 Claude에는 어림값이다."""
    global _enc
    if _enc is None:
        _enc = get_encoder()
    return len(_enc.encode(s))


def build_context(conn, query, mode="hybrid", ym=None,
                  top_n=8, budget=3000, count_tokens=len):
    """검색 결과를 컨텍스트 문자열로 조립한다.

    budget 을 넘기 직전에 멈추므로 top_n 을 넉넉히 줘도 안전하다.
    """
    res = search(conn, query, mode=mode, ym=ym, top_n=top_n)
    meta = fetch(conn, [cid for cid, _, _ in res])

    blocks, used, kept = [], 0, []
    for cid, score, hits in res:
        m = meta.get(cid)
        if not m:
            continue
        block = BLOCK.format(**m)
        n = count_tokens(block)
        if used + n > budget:
            break            # 예산 초과 — 여기서 자른다
        blocks.append(block)
        used += n
        kept.append(cid)
    return "\n---\n".join(blocks), used, kept


if __name__ == "__main__":
    from src import embedder, tokenizer_ko
    from src.build_sqlite import connect

    tokenizer_ko.warmup()
    embedder.warmup()
    conn = connect()
    q = " ".join(sys.argv[1:]) or "AI 규제 법안"
    ctx, used, kept = build_context(conn, q, count_tokens=count_tokens)
    print(f'질의: "{q}"  ->  {len(kept)}청크 / {used:,}토큰\n')
    print(ctx[:1200] + ("..." if len(ctx) > 1200 else ""))
    conn.close()
