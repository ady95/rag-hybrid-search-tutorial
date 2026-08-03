# -*- coding: utf-8 -*-
"""사용자 사전 후보 채굴 (04-5).

색인할 문서를 훑어 "Kiwi가 쪼개는데 원문에는 붙어 있는 말"을 찾는다.
자동으로 등록하지는 않는다. 후보 목록을 만들어 줄 뿐이고, 무엇을 넣을지는
사람이 고른다. 잘못 넣으면 재색인해야 하고 검색이 조용히 나빠진다.

    python -m src.dict_miner                 # 후보 목록 보기
    python -m src.dict_miner --out data/userdict.json
"""
import argparse
import json
import re
import sys
from collections import Counter, defaultdict

from src import config
from src.tokenizer_ko import get_kiwi

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

# 색인에 들어가는 품사 (tokenizer_ko.KEEP_TAGS 와 같은 계열)
CONTENT = {"NNG", "NNP", "NNB", "NR", "VV", "VA", "XR", "SL", "SN", "SH"}
# 어절 안에서 단어를 이어 주는 것들 — 여기서 끊지 않는다
GLUE = {"XPN", "XSN", "MM"}
# 후보로 받아 줄 품사. 동사·형용사가 섞이면 활용형이라 사전에 넣을 것이 아니다
NOUNY = {"NNG", "NNP", "SL", "SH", "SN"}

WORDY = CONTENT | GLUE
STRIP = "()[]{}<>‘’“”\"'·,.…:;/|~!?"
STARTS_OK = re.compile(r"^[가-힣A-Za-z]")

DEFAULT_PATH = config.DATA / "userdict.json"


def analyze(kiwi, word):
    """어절 앞쪽에서 '한 단어'인 부분을 잘라 낸다.

    returns (stem, kept, tags)
      stem — 원문 표면형 (예: '데이터센터')
      kept — tokenize() 가 실제로 색인할 토큰 (예: ['데이터', '센터'])
      tags — 그 구간의 품사
    조사·어미가 나오면 거기서 끊는다. Kiwi가 원형으로 바꾸는 경우가 있어
    표면 길이는 form 이 아니라 t.end 로 잰다 ('피지컬'의 '컬' -> '크').
    """
    end, kept, tags = 0, [], []
    for t in kiwi.tokenize(word):
        if t.tag not in WORDY or t.start > end:
            break
        end = t.end
        tags.append(t.tag)
        if t.tag in CONTENT:
            kept.append(t.form.lower())
    return word[:end], kept, tags


def mine(texts, min_len=3, min_count=3):
    """후보 목록을 돌려준다. 충돌이 큰 것부터 정렬한다.

    충돌(collide)이 이 도구의 핵심입니다. 쪼개진 조각이 **이 단어 바깥에서도**
    자주 쓰이면, 색인이 그 조각으로 들어가는 순간 엉뚱한 문서가 함께 걸립니다.
    '데이터센터'가 '데이터'로 쪼개지면 데이터를 다루는 모든 문서와 섞입니다.
    충돌이 0에 가까우면 쪼개져도 실질 피해가 없으므로 등록할 이유가 적습니다.
    """
    kiwi = get_kiwi()
    count = Counter()                 # 후보 -> 등장 횟수
    pieces, kind = {}, {}
    tok_docs = defaultdict(set)       # 토큰 -> 등장 문서 번호
    cand_docs = defaultdict(set)      # 후보 -> 등장 문서 번호

    for di, text in enumerate(texts or []):
        for raw in (text or "").split():
            w = raw.strip(STRIP)
            if len(w) < min_len or not STARTS_OK.match(w):
                continue
            stem, kept, tags = analyze(kiwi, w)
            if len(stem) < min_len or not kept:
                continue
            for k in kept:
                tok_docs[k].add(di)

            nouny = [g for g in tags if g not in GLUE]
            if not nouny or any(g not in NOUNY for g in nouny):
                continue
            # ① 명사가 둘 이상으로 쪼개졌다 (숫자+단위는 뺀다)
            split = len(kept) >= 2 and not kept[0].isdigit()
            # ② 관형사·접두사가 명사에 붙어 있는데 색인에서 사라졌다
            glued = tags[0] in ("MM", "XPN") and len(tags) > 1
            if not (split or glued):
                continue
            count[stem] += 1
            cand_docs[stem].add(di)
            pieces[stem], kind[stem] = kept, ("split" if split else "glued")

    out = []
    for stem, n in count.items():
        if n < min_count:
            continue
        ps, own = pieces[stem], cand_docs[stem]
        collide = {p: len(tok_docs[p] - own) for p in ps}
        out.append({"word": stem, "kind": kind[stem], "count": n,
                    "docs": len(own), "pieces": ps,
                    "collide": max(collide.values(), default=0),
                    "worst": max(collide, key=collide.get) if collide else ""})
    out.sort(key=lambda d: (-d["collide"], -d["count"]))
    return out


def load_user_words(path=None):
    """승인된 사전을 읽는다. 파일이 없으면 빈 목록."""
    p = config.ROOT / (path or DEFAULT_PATH)
    if not p.exists():
        return []
    data = json.loads(p.read_text(encoding="utf-8"))
    return [w for w, ok in data.get("words", {}).items() if ok]


def save_template(cands, path=None, top=40):
    """후보를 승인 파일로 내려쓴다. 값이 true 인 것만 실제로 등록된다."""
    p = config.ROOT / (path or DEFAULT_PATH)
    old = json.loads(p.read_text(encoding="utf-8")).get("words", {}) \
        if p.exists() else {}
    words = dict(old)
    for c in cands[:top]:
        words.setdefault(c["word"], False)      # 기본은 '보류'
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps({"words": words}, ensure_ascii=False, indent=2),
                 encoding="utf-8")
    return p, sum(1 for v in words.values() if v), len(words)


def _corpus():
    from src.build_sqlite import connect
    conn = connect()
    rows = conn.execute("SELECT title, body FROM chunks").fetchall()
    conn.close()
    return [f'{r["title"] or ""}\n{r["body"] or ""}' for r in rows]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--min-count", type=int, default=3)
    ap.add_argument("--top", type=int, default=25)
    ap.add_argument("--out", default=None, help="승인 파일 경로")
    ap.add_argument("--no-write", action="store_true", help="목록만 보고 파일은 두기")
    a = ap.parse_args()

    texts = _corpus()
    cands = mine(texts, min_count=a.min_count)
    labels = {"split": "A. 복합명사가 쪼개진 것 — 등록하면 정밀도가 오릅니다",
              "glued": "B. 접두사·관형사가 사라진 것 — 맞교환이라 판단이 필요합니다"}
    for k, label in labels.items():
        sub = [c for c in cands if c["kind"] == k]
        print(f"\n=== {label} ({len(sub)}종) ===")
        print(f'{"후보":<16}{"빈도":>5}{"문서":>5}{"충돌":>5}  색인되는 모습')
        print("-" * 70)
        for c in sub[:a.top]:
            print(f'{c["word"]:<16}{c["count"]:>5}{c["docs"]:>5}{c["collide"]:>5}'
                  f'  {" + ".join(c["pieces"])}')

    print(f"\n청크 {len(texts)}개 -> 후보 {len(cands)}종")
    if a.no_write:
        return
    p, on, total = save_template(cands, a.out, top=a.top * 2)
    print(f"승인 파일: {p}  (등록 {on}종 / 후보 {total}종)")
    print("true 로 바꾼 단어만 등록됩니다. 바꾼 뒤에는 반드시 재색인하세요.")


if __name__ == "__main__":
    main()
