# -*- coding: utf-8 -*-
"""검색 품질 평가 — Recall@k 와 MRR.

평가셋은 두 가지를 섞는다.
  auto  청크의 헤딩과 첫 문장을 질의로 삼고, 그 청크를 정답으로 둔다.
        만들기 쉽지만 "질의에 정답 단어가 그대로 들어 있는" 편향이 있어
        키워드 검색에 유리하다. 이 한계를 알고 봐야 한다.
  manual 사람이 쓴 질의 + 손으로 확인한 정답 청크 목록.

사용:
  python -m src.evaluate --make-auto 30    # 자동 평가셋 생성
  python -m src.evaluate                   # 평가 실행
"""
import argparse
import json
import random
import re
import sys
import time

from src import config
from src.build_sqlite import connect
from src.search_sqlite import search

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

EVAL_PATH = config.DATA / "evalset.json"
MODES = ["keyword", "semantic", "fallback", "hybrid"]


def make_auto(conn, n=30, seed=42):
    """본문이 충분히 길고 제목이 있는 청크를 골라 질의를 만든다."""
    rows = conn.execute("""
        SELECT chunk_id, title, body, year_month FROM chunks
        WHERE length(body) > 300 AND title != '' AND length(title) > 6
    """).fetchall()
    random.Random(seed).shuffle(rows)

    items, seen_titles = [], set()
    for r in rows:
        title = re.sub(r"^.*>\s*", "", r["title"]).strip()
        if len(title) < 8 or title in seen_titles:
            continue
        # 제목이 그대로 본문에 반복되는 경우는 제외 (너무 쉬움)
        seen_titles.add(title)
        items.append({
            "id": f"auto-{len(items)+1:02d}",
            "query": title,
            "gold": [r["chunk_id"]],
            "kind": "auto",
            "year_month": r["year_month"],
        })
        if len(items) >= n:
            break
    return items


def load():
    if not EVAL_PATH.exists():
        return []
    return json.loads(EVAL_PATH.read_text(encoding="utf-8"))


def save(items):
    EVAL_PATH.write_text(json.dumps(items, ensure_ascii=False, indent=2),
                         encoding="utf-8")


def hit_at_k(ranked, gold, k):
    """상위 k 안에 정답이 하나라도 있으면 1 (적중률)."""
    return 1.0 if set(ranked[:k]) & set(gold) else 0.0


def recall_at_k(ranked, gold, k):
    """정답 중 몇 개를 상위 k 안에서 건졌는가 (표준 Recall)."""
    if not gold:
        return 0.0
    return len(set(ranked[:k]) & set(gold)) / len(gold)


def rr(ranked, gold):
    for i, cid in enumerate(ranked, 1):
        if cid in gold:
            return 1.0 / i
    return 0.0


def evaluate(conn, items, modes=None, ks=(1, 3, 5, 10), backend=None):
    modes = modes or MODES
    out = {}
    for mode in modes:
        agg = {f"hit@{k}": 0.0 for k in ks}
        agg.update({f"recall@{k}": 0.0 for k in ks})
        agg["mrr"] = 0.0
        lat = []
        for it in items:
            t0 = time.perf_counter()
            res = search(conn, it["query"], mode=mode, top_n=max(ks), backend=backend)
            lat.append((time.perf_counter() - t0) * 1000)
            ranked = [c for c, _, _ in res]
            for k in ks:
                agg[f"hit@{k}"] += hit_at_k(ranked, it["gold"], k)
                agg[f"recall@{k}"] += recall_at_k(ranked, it["gold"], k)
            agg["mrr"] += rr(ranked, it["gold"])
        n = max(1, len(items))
        agg = {k: v / n for k, v in agg.items()}
        lat.sort()
        agg["p50_ms"] = lat[len(lat) // 2]
        agg["p95_ms"] = lat[min(len(lat) - 1, int(len(lat) * 0.95))]
        out[mode] = agg
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--make-auto", type=int, default=0)
    ap.add_argument("--backend", default=None)
    ap.add_argument("--kind", default=None, help="auto | manual 만 평가")
    a = ap.parse_args()

    conn = connect()

    if a.make_auto:
        items = make_auto(conn, a.make_auto)
        existing = [x for x in load() if x.get("kind") != "auto"]
        save(existing + items)
        print(f"자동 평가셋 {len(items)}개 생성 -> {EVAL_PATH}")
        for it in items[:5]:
            print(f"  {it['id']}  {it['query'][:50]}")
        return

    items = load()
    if a.kind:
        items = [x for x in items if x.get("kind") == a.kind]
    if not items:
        print("평가셋이 없습니다. --make-auto 30 으로 먼저 만드세요.")
        return

    # 예열
    search(conn, "예열", mode="keyword", top_n=1)
    search(conn, "예열", mode="semantic", top_n=1, backend=a.backend)

    print(f"평가셋 {len(items)}개 "
          f"(auto {sum(1 for x in items if x.get('kind')=='auto')} / "
          f"manual {sum(1 for x in items if x.get('kind')=='manual')})\n")
    res = evaluate(conn, items, backend=a.backend)

    cols = ["Hit@1", "Hit@5", "Hit@10", "Recall@5", "Recall@10", "MRR", "p50(ms)"]
    hdr = f"{'모드':<10}" + "".join(f"{c:>11}" for c in cols)
    print(hdr)
    print("-" * len(hdr))
    for mode in MODES:
        r = res[mode]
        print(f"{mode:<10}"
              f"{r['hit@1']:11.3f}{r['hit@5']:11.3f}{r['hit@10']:11.3f}"
              f"{r['recall@5']:11.3f}{r['recall@10']:11.3f}"
              f"{r['mrr']:11.3f}{r['p50_ms']:11.0f}")

    best = max(MODES, key=lambda m: res[m]["mrr"])
    print(f"\nMRR 최고: {best} ({res[best]['mrr']:.3f})")
    conn.close()


if __name__ == "__main__":
    main()
