# -*- coding: utf-8 -*-
"""인용 검증 — 답변이 실제로 준 자료만 인용했는지 검사한다 (09-3).

한계를 분명히 해 둔다. 이 검증기는 "없는 식별자"는 잡지만
"있는 식별자인데 내용이 다름"은 잡지 못한다. 그쪽은 08-6의 RAGAS
Faithfulness 처럼 문장 단위로 근거를 대조하는 지표가 필요하다.
"""
import re
import sys

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

CITE = re.compile(r"\[([A-Za-z0-9_]+#\d{4})\]")


def verify_citations(answer, allowed_ids):
    """답변의 인용이 실제 컨텍스트에 있었는지 검사한다.

    allowed_ids: build_context() 가 돌려준 kept 목록
    """
    allowed = set(allowed_ids)
    found = CITE.findall(answer)

    sentences = [s for s in re.split(r"(?<=[.!?])\s+", answer.strip()) if s]
    uncited = [s for s in sentences if not CITE.search(s)]

    return {
        "ok": bool(found) and not (set(found) - allowed),
        "cited": sorted(set(found)),
        "invalid": sorted(set(found) - allowed),      # 없는 자료를 인용
        "unused": sorted(allowed - set(found)),       # 줬는데 안 쓴 자료
        "uncited_sentences": len(uncited),
        "coverage": round(1 - len(uncited) / max(1, len(sentences)), 2),
    }


def gate(answer, allowed_ids, min_coverage=0.8):
    """통과·재시도·거부 셋 중 하나를 돌려준다."""
    r = verify_citations(answer, allowed_ids)
    if r["invalid"]:
        return "reject", f"존재하지 않는 자료를 인용했습니다: {r['invalid']}"
    if r["coverage"] < min_coverage:
        return "retry", f"근거 없는 문장이 {r['uncited_sentences']}개 있습니다"
    return "pass", ""


def _demo():
    """LLM 없이 판정 동작만 확인한다 (10-2의 캡처와 같은 네 가지)."""
    a, b = "SPRi_AI_Brief_202603#0014", "SPRi_AI_Brief_202602#0031"
    cases = [
        ("정상 인용", f"법안이 통과됐습니다 [{a}]. 뉴욕주도 서명했습니다 [{b}]."),
        ("허위 인용", f"법안이 통과됐습니다 [{a}]. 프랑스도 통과시켰습니다 "
                     f"[SPRi_AI_Brief_209912#9999]."),
        ("무인용", "법안이 여러 나라에서 통과됐습니다."),
        ("부분 인용", f"법안이 통과됐습니다 [{a}]. 그리고 다른 나라들도 뒤따랐습니다."),
    ]
    print(f'{"입력":<11}{"ok":>7}{"커버리지":>10}{"허위":>6}   비고')
    print("-" * 58)
    note = {"정상 인용": "", "허위 인용": "없는 ID 검출", "무인용": "근거 없음",
            "부분 인용": "ok=True 인데 절반만 인용"}
    for name, text in cases:
        v = verify_citations(text, [a, b])
        print(f'{name:<11}{str(v["ok"]):>7}{v["coverage"]:>10.1f}'
              f'{len(v["invalid"]):>6}   {note[name]}')
    print("\n검증기는 '없는 ID'는 잡지만 '있는 ID인데 근거가 모자란 답변'은")
    print("잡지 못합니다. 커버리지를 함께 봐야 하는 이유입니다.")


if __name__ == "__main__":
    _demo()
