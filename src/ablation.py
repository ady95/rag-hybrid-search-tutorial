# -*- coding: utf-8 -*-
"""검색기 조합별 기여도 측정 (ablation) — 어떤 검색기를 빼면 얼마나 나빠지는가.

"세 갈래를 합치면 좋다"는 말은 쉽지만, 각 갈래가 실제로 얼마나 보태는지는
빼 보기 전에는 모른다. 이 스크립트는 조합을 바꿔 가며 같은 평가셋을 돌린다.

평가셋 세 종류를 지원한다.
  manual  사람이 쓴 자연어 질문 (data/evalset.json 의 kind=manual)
  term    단독 고유명사 — 형태소 분석이 깨지기 쉬운 유형
  typo    오타가 섞인 질의
term/typo 는 정답을 "원문에 그 문자열이 실제로 든 청크"로 정의해 즉석에서 만든다.

사용:
  python -m src.ablation                # 세 평가셋 모두
  python -m src.ablation --set manual
"""
import argparse
import sys

from src import config, embedder, tokenizer_ko
from src.build_sqlite import connect
from src.evaluate import hit_at_k, load, recall_at_k, rr
from src.search_sqlite import (rrf_fuse, search_fallback, search_keyword,
                               search_semantic)

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

WEIGHTS = {"keyword": config.W_KEYWORD,
           "semantic": config.W_SEMANTIC,
           "fallback": config.W_FALLBACK}

COMBOS = [
    ("keyword",                       ["keyword"]),
    ("semantic",                      ["semantic"]),
    ("fallback",                      ["fallback"]),
    ("keyword + semantic",            ["keyword", "semantic"]),
    ("keyword + fallback",            ["keyword", "fallback"]),
    ("semantic + fallback",           ["semantic", "fallback"]),
    ("keyword + semantic + fallback", ["keyword", "semantic", "fallback"]),
]
FULL = "keyword + semantic + fallback"

# 형태소 분석기가 흔들리는 단독 고유명사
TERMS = ["오퍼스", "제미나이", "엑사원", "코덱스", "딥시크", "젠-4.5", "그록"]

# (오타 질의, 정답 판정에 쓸 올바른 표기)
TYPOS = [
    ("앤트로픽 클로두", "클로드"),
    ("제네시스 미숀", "제네시스"),
    ("반도체 수출규재", "수출 규제"),
    ("허깅페이스", "허깅"),
    ("트럼푸 행정명령", "행정명령"),
]


def run_combo(conn, query, names):
    lists = []
    if "keyword" in names:
        lists.append(("keyword", search_keyword(conn, query)))
    if "semantic" in names:
        lists.append(("semantic", search_semantic(conn, query)))
    if "fallback" in names:
        lists.append(("fallback", search_fallback(conn, query)))
    if len(lists) == 1:
        return [c for c, _ in lists[0][1]][:10]
    return [c for c, _, _ in rrf_fuse(lists, weights=WEIGHTS, top_n=10)]


def gold_by_substring(conn, text, lo=1, hi=60):
    """원문에 문자열이 든 청크를 정답으로 삼는다 (너무 흔하거나 없으면 제외)."""
    rows = [r[0] for r in conn.execute(
        "SELECT chunk_id FROM chunks WHERE body LIKE ?", (f"%{text}%",))]
    return rows if lo <= len(rows) <= hi else None


def build_set(conn, kind):
    if kind == "manual":
        return [x for x in load() if x.get("kind") == "manual"]
    if kind == "term":
        out = []
        for t in TERMS:
            g = gold_by_substring(conn, t, hi=40)
            if g:
                out.append({"query": t, "gold": g})
        return out
    if kind == "typo":
        out = []
        for typo, correct in TYPOS:
            g = gold_by_substring(conn, correct)
            if g:
                out.append({"query": typo, "gold": g})
        return out
    raise ValueError(kind)


def evaluate_set(conn, items):
    res = {}
    for label, names in COMBOS:
        h1 = h5 = r10 = mrr = 0.0
        for it in items:
            ranked = run_combo(conn, it["query"], names)
            h1 += hit_at_k(ranked, it["gold"], 1)
            h5 += hit_at_k(ranked, it["gold"], 5)
            r10 += recall_at_k(ranked, it["gold"], 10)
            mrr += rr(ranked, it["gold"])
        n = max(1, len(items))
        res[label] = (h1 / n, h5 / n, r10 / n, mrr / n)
    return res


def report(name, items, res):
    print(f"\n{'=' * 70}\n{name}  (질의 {len(items)}개)\n{'=' * 70}")
    print(f"{'조합':<32}{'Hit@1':>8}{'Hit@5':>8}{'Recall@10':>11}{'MRR':>8}")
    print("-" * 67)
    for label, _ in COMBOS:
        h1, h5, r10, mrr = res[label]
        print(f"{label:<32}{h1:8.3f}{h5:8.3f}{r10:11.3f}{mrr:8.3f}")

    full = res[FULL]
    print("\n  전체(3갈래) 대비 각 검색기를 뺐을 때")
    for drop, keep in [("fallback", "keyword + semantic"),
                       ("semantic", "keyword + fallback"),
                       ("keyword", "semantic + fallback")]:
        r = res[keep]
        print(f"    {drop:<9} 제거 -> MRR {r[3]:.3f} ({r[3]-full[3]:+.3f})"
              f"   Recall@10 {r[2]:.3f} ({r[2]-full[2]:+.3f})")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--set", dest="kinds", default="manual,term,typo",
                    help="manual,term,typo 중 쉼표로 나열")
    a = ap.parse_args()

    conn = connect()
    tokenizer_ko.warmup()
    embedder.warmup()

    names = {"manual": "수동 평가셋 (자연어 질문)",
             "term": "미등록어 평가셋 (단독 고유명사)",
             "typo": "오타 평가셋"}

    for kind in [k.strip() for k in a.kinds.split(",") if k.strip()]:
        items = build_set(conn, kind)
        if not items:
            print(f"\n[{kind}] 질의가 없어 건너뜁니다")
            continue
        if kind == "term":
            print("\n  단독 고유명사 토큰화 결과")
            for it in items:
                print(f"    {it['query']:<10} -> {tokenizer_ko.tokenize(it['query'])}")
        report(names[kind], items, evaluate_set(conn, items))

    print("\n주의: 평가셋이 작으면 0.03 안팎의 차이는 잡음일 수 있습니다. "
          "경향을 보고, 확신이 필요하면 질의 수를 늘리세요.")
    conn.close()


if __name__ == "__main__":
    main()
