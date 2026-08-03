# -*- coding: utf-8 -*-
"""RAGAS로 검색·생성 품질 재기 (08-6).

RAGAS는 RAG 평가 프레임워크다. 흔히 judge-LLM으로 채점하는 도구로 알려져
있지만, **LLM이 전혀 필요 없는 검색 지표**도 함께 제공한다. 이 모듈은
기본적으로 그 LLM-free 지표만 돌리고, `--llm` 을 주면 생성 품질까지 잰다.

  LLM 없이 (기본)
    NonLLMContextRecall                  정답 컨텍스트를 얼마나 건졌나
    NonLLMContextPrecisionWithReference   가져온 것 중 쓸모 있는 게 얼마나 위에 있나

  --llm (judge-LLM 필요, 과금)
    Faithfulness       답변이 컨텍스트에 근거하는가 (환각 탐지)
    ResponseRelevancy  답변이 질문에 답하고 있는가

사용:
  python -m src.eval_ragas                 # LLM 없이, 우리 지표와 대조
  python -m src.eval_ragas --kind manual
  python -m src.eval_ragas --llm           # 생성 품질까지 (API 키 필요)
"""
import argparse
import os
import sys

from src import embedder, tokenizer_ko
from src.build_sqlite import connect
from src.evaluate import hit_at_k, load, recall_at_k, rr
from src.search_sqlite import fetch, search

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")


def build_samples(conn, items, top_n=10, backend=None):
    """검색을 돌려 RAGAS 입력과 우리 지표를 동시에 만든다."""
    from ragas import SingleTurnSample

    samples, ours = [], {"recall@10": 0.0, "mrr": 0.0, "hit@1": 0.0}
    for it in items:
        res = search(conn, it["query"], mode="hybrid", top_n=top_n, backend=backend)
        ranked = [c for c, _, _ in res]

        ours["recall@10"] += recall_at_k(ranked, it["gold"], 10)
        ours["mrr"] += rr(ranked, it["gold"])
        ours["hit@1"] += hit_at_k(ranked, it["gold"], 1)

        got = fetch(conn, ranked)
        gold = fetch(conn, it["gold"])
        samples.append(SingleTurnSample(
            user_input=it["query"],
            retrieved_contexts=[got[c]["body"] for c in ranked if c in got],
            reference_contexts=[gold[c]["body"] for c in it["gold"] if c in gold],
        ))

    n = max(1, len(items))
    return samples, {k: v / n for k, v in ours.items()}


def make_judge():
    """judge-LLM과 임베딩을 준비한다 (--llm 일 때만).

    `base_url` 을 주면 02-4에서 만든 자체 API를 judge로 쓸 수도 있다.
    다만 판정 품질은 모델 성능에 그대로 좌우되므로, 채점에는 충분히 큰
    모델을 쓰는 편이 안전하다.
    """
    from ragas.llms import llm_factory

    model = os.environ.get("RAGAS_JUDGE_MODEL", "gpt-5.2")
    base_url = os.environ.get("RAGAS_JUDGE_BASE_URL")     # 없으면 OpenAI
    return llm_factory(model=model, base_url=base_url)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--kind", default="manual", help="manual | auto")
    ap.add_argument("--llm", action="store_true",
                    help="judge-LLM 지표까지 실행 (과금 발생)")
    ap.add_argument("--backend", default=None)
    a = ap.parse_args()

    items = [x for x in load() if x.get("kind") == a.kind]
    if not items:
        print(f"'{a.kind}' 평가셋이 없습니다. python -m src.evaluate --make-auto 30")
        return

    conn = connect()
    tokenizer_ko.warmup()
    embedder.warmup(backend=a.backend)

    print(f"평가셋 {len(items)}개 ({a.kind})\n")
    samples, ours = build_samples(conn, items, backend=a.backend)

    from ragas import EvaluationDataset, evaluate
    from ragas.metrics import (NonLLMContextPrecisionWithReference,
                               NonLLMContextRecall)

    metrics = [NonLLMContextRecall(), NonLLMContextPrecisionWithReference()]
    judge = None
    if a.llm:
        from ragas.metrics import Faithfulness
        judge = make_judge()
        metrics.append(Faithfulness(llm=judge))
        print("judge-LLM 지표 포함 — 질의 수 x 지표 수만큼 API 호출이 발생합니다.\n")

    result = evaluate(dataset=EvaluationDataset(samples=samples), metrics=metrics)

    print("=" * 62)
    print(f"{'지표':<42}{'값':>10}")
    print("-" * 62)
    print(f"{'(우리 구현) Recall@10':<42}{ours['recall@10']:>10.3f}")
    print(f"{'(우리 구현) MRR':<42}{ours['mrr']:>10.3f}")
    print(f"{'(우리 구현) Hit@1':<42}{ours['hit@1']:>10.3f}")
    print("-" * 62)
    for k, v in result._repr_dict.items() if hasattr(result, "_repr_dict") else []:
        print(f"{'(RAGAS) ' + k:<42}{v:>10.3f}")
    if not hasattr(result, "_repr_dict"):
        print(" ", result)
    print("=" * 62)

    print("\n질의별 (RAGAS)")
    df = result.to_pandas()
    SHORT = {
        "non_llm_context_recall": "recall",
        "non_llm_context_precision_with_reference": "precision",
        "faithfulness": "faithfulness",
    }
    cols = [c for c in df.columns if c in SHORT]
    for _, row in df.iterrows():
        vals = "  ".join(f"{SHORT[c]} {row[c]:.3f}" for c in cols)
        print(f"  {str(row['user_input'])[:30]:<32} {vals}")

    print("\n주: reference_contexts 에 정답 청크 본문을 그대로 넣었으므로 "
          "non_llm_context_recall 은 우리 Recall@10 과 같은 값이 나오는 것이 정상입니다. "
          "구현이 어긋나지 않았다는 확인으로 보세요.")
    conn.close()


if __name__ == "__main__":
    main()
