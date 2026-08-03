# -*- coding: utf-8 -*-
"""자료 밖 질의 판정 — 외부 검색으로 넘길지 정한다 (09-5).

3층으로 나눈다. 층마다 잡는 것이 다르고, 아래 층으로 못 내려가는 것이 있다.

  1층  날짜 파싱      확정 판정. 명시적 연·월만. 상대 날짜는 모델이 정규화해 준다
  2층  검색 신호      확률 판정. 주제 이탈은 잡고 근접 이탈은 못 잡는다
  3층  프롬프트       나머지. "확인되지 않습니다"로 답하게 한다 (09-3)

RRF 점수는 쓰지 않는다. 순위만 담아 관련도의 크기가 없어서
자료 안과 밖이 구분되지 않는다 (09-5 실측).

    python -m src.router "2027년 EU AI법 개정안"
"""
import argparse
import re
import sys

from src.search_sqlite import search_keyword, search_semantic

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

# 이 색인이 다루는 기간. 자료를 갈아 끼우면 여기부터 고친다.
CORPUS_RANGE = ("2026-01", "2026-07")

# 임계값은 코퍼스마다 다시 재야 한다 (09-5 관찰 포인트 ②).
# 489청크 / bge-m3 기준: 자료 안 질의의 의미 거리 최댓값이 0.97 이었다.
DIST_MAX = 1.00
KW_MIN = 25

YM = re.compile(r"(20\d\d)\s*년?\s*(\d{1,2})\s*월")
YEAR = re.compile(r"(20\d\d)\s*년")


def out_of_range(query, corpus=CORPUS_RANGE):
    """1층 — 질의에 박힌 연·월이 자료 범위를 벗어나는지 확정 판정."""
    for y, m in YM.findall(query):
        ym = f"{y}-{int(m):02d}"
        if not (corpus[0] <= ym <= corpus[1]):
            return ym
    if not YM.search(query):
        m = YEAR.search(query)
        if m and m.group(1) != corpus[0][:4]:
            return m.group(1)
    return None


def topic_off(conn, query, dist_max=DIST_MAX, kw_min=KW_MIN):
    """2층 — 주제가 자료 밖인지 확률 판정. (판정, 근거) 를 돌려준다."""
    se = search_semantic(conn, query, k=5)
    kw = search_keyword(conn, query, k=30)
    dist = -se[0][1] if se else 99.0
    why = []
    if dist > dist_max:
        why.append(f"의미 거리 {dist:.2f} > {dist_max}")
    if len(kw) < kw_min:
        why.append(f"키워드 {len(kw)}건 < {kw_min}")
    return bool(why), {"dist": round(dist, 3), "n_kw": len(kw), "why": why}


def route(conn, query, corpus=CORPUS_RANGE):
    """외부 검색이 필요한지 판정한다.

    오탐(자료 안을 밖으로 보냄)이 미탐보다 비싸다. 임계값은 자료 안을
    놓치지 않는 쪽으로 잡고 남는 것은 3층에 맡긴다.
    """
    bad_ym = out_of_range(query, corpus)
    if bad_ym:
        return {"external": True, "layer": 1,
                "reason": f"{bad_ym} 은 자료 범위({corpus[0]}~{corpus[1]}) 밖"}
    off, info = topic_off(conn, query)
    if off:
        return {"external": True, "layer": 2,
                "reason": " / ".join(info["why"]), **info}
    return {"external": False, "layer": 3,
            "reason": "내부 자료로 시도한다 (자료 부족 판정은 생성 단계에서)", **info}


def external_blocks(hits):
    """외부 결과에 내부와 같은 형식의 ID를 붙인다 (09-5).

    이걸 하지 않으면 09-3의 인용 검증이 깨진다. URL 로 인용하면
    커버리지가 반토막 나고, 임의 ID 로 인용하면 허위로 오판된다.
    발급한 ID 목록을 verify_citations 의 허용 목록에 함께 넘겨야 한다.
    """
    out, ids = [], []
    for i, h in enumerate(hits, 1):
        cid = f"WEB_{i:04d}#0001"          # CITE 정규식과 같은 모양
        ids.append(cid)
        out.append(f"[{cid}]\n출처: {h.get('url', '')}\n"
                   f"발행: {h.get('date') or '미상'}\n"
                   f"제목: {h.get('title', '')}\n{h.get('snippet', '')}\n")
    return "\n---\n".join(out), ids


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("query", nargs="*")
    a = ap.parse_args()
    from src import embedder, tokenizer_ko
    from src.build_sqlite import connect

    tokenizer_ko.warmup()
    embedder.warmup()
    conn = connect()
    qs = [" ".join(a.query)] if a.query else [
        "AI 규제 법안이 통과된 나라",
        "2027년 EU AI법 개정안 내용은",
        "김치찌개 맛있게 끓이는 방법",
        "온디바이스 sLLM 성능 비교",
    ]
    for q in qs:
        r = route(conn, q)
        mark = "외부" if r["external"] else "내부"
        print(f'[{mark}] {q}\n     {r["layer"]}층 · {r["reason"]}')
    conn.close()


if __name__ == "__main__":
    main()
