# -*- coding: utf-8 -*-
"""Kiwi 기반 한국어 색인 토큰화.

FTS5(SQLite)와 tsvector(PostgreSQL)는 둘 다 공백으로 토큰을 나눈다.
한국어는 조사가 붙어 있어 그대로 넣으면 "신고를"과 "신고가"가 다른 단어가
된다. 그래서 색인 전에 형태소로 나누고, 검색어도 같은 함수를 통과시킨다.

핵심 규칙: 색인과 질의는 반드시 같은 토크나이저를 쓴다.

사용:
  python -m src.tokenizer_ko "부가가치세를 신고했습니다"
"""
import re
import sys
import unicodedata

from kiwipiepy import Kiwi

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

# 색인할 품사
#   NNG 일반명사 / NNP 고유명사 / NNB 의존명사 / NR 수사
#   VV  동사     / VA  형용사   / XR  어근
#   SL  외국어   / SN  숫자     / SH 한자
KEEP_TAGS = {"NNG", "NNP", "NR", "VV", "VA", "XR", "SL", "SN", "SH"}

# 조사(J*), 어미(E*), 접사 일부, 기호(S*)는 버린다.

_kiwi = None
_loaded_words = ()


def get_kiwi(user_words=None):
    """Kiwi 인스턴스를 재사용한다 (생성 비용이 크다 — 약 1.1초).

    주의: `add_user_word()` 는 **첫 토큰화 이전에만** 반영된다.
    이미 토큰화를 한 인스턴스에 뒤늦게 부르면 예외 없이 False를 반환하고
    조용히 무시된다. 그래서 사전이 바뀌면 인스턴스를 새로 만든다.
    """
    global _kiwi, _loaded_words
    # user_words 를 생략하면 "현재 인스턴스를 그대로 쓴다"는 뜻이다.
    # 빈 목록([])을 넘기는 것과 구분해야 한다 — 그건 "사전을 비운다"는 뜻이다.
    if user_words is None:
        if _kiwi is None:
            _kiwi = Kiwi()
            _loaded_words = ()
        return _kiwi

    want = tuple(sorted(user_words))
    if _kiwi is None or want != _loaded_words:
        _kiwi = Kiwi()
        for w in want:
            _kiwi.add_user_word(w, "NNP")
        _loaded_words = want
    return _kiwi


def set_user_words(words):
    """도메인 용어를 등록한다. 색인·검색 양쪽에서 같은 목록을 써야 한다.

    사전이 바뀌면 색인된 토큰도 달라지므로 `body_tsv`/FTS5 재색인이 필요하다.
    """
    get_kiwi(words)
    return _loaded_words


def warmup():
    """형태소 사전을 미리 메모리에 올린다.

    Kiwi는 **첫 tokenize() 호출** 때 사전을 적재한다. 인스턴스를 만드는 것만으로는
    부족해서, 실제로 한 번 돌려 줘야 한다. 이 비용이 1초 남짓이라 예열하지 않으면
    첫 질의만 유독 느리게 찍히고 벤치마크 평균이 통째로 망가진다.

    넘기는 문자열의 내용은 결과에 영향을 주지 않는다 (한 글자든 긴 문장이든 같다).
    그래서 최소 길이인 "가"를 쓴다.

    기동 시 한 번, 그리고 성능을 측정하기 전에 반드시 호출할 것.
    """
    tokenize("가")


def tokenize(text, keep_tags=None):
    """색인용 토큰 목록. 원형(lemma)으로 돌려준다."""
    if not text:
        return []
    keep = keep_tags or KEEP_TAGS
    text = unicodedata.normalize("NFC", text)
    out = []
    for t in get_kiwi().tokenize(text):
        if t.tag not in keep:
            continue
        form = t.form.strip()
        if not form or len(form) == 1 and t.tag in {"NNB", "NR"}:
            continue
        out.append(form.lower())
    return out


def tokenized(text):
    """공백으로 이어 붙인 색인 문자열."""
    return " ".join(tokenize(text))


def to_fts_query(text, op="OR"):
    """FTS5 MATCH 식으로 변환. 토큰을 큰따옴표로 감싸 특수문자를 무력화한다."""
    toks = tokenize(text)
    if not toks:
        return None
    return f" {op} ".join('"' + t.replace('"', '""') + '"' for t in toks)


def to_tsquery(text, op="|"):
    """PostgreSQL to_tsquery 식으로 변환.

    토큰을 작은따옴표로 감싼다. 점을 지우면 안 된다 —
    to_tsvector('simple', ...)는 "4.5"를 한 토큰으로 유지하므로
    "45"로 바꿔 질의하면 매칭이 실패한다(07-4의 함정).
    """
    toks = []
    for t in tokenize(text):
        t = t.strip()
        if not t:
            continue
        toks.append("'" + t.replace("'", "''") + "'")
    if not toks:
        return None
    return f" {op} ".join(toks)


if __name__ == "__main__":
    samples = sys.argv[1:] or [
        "부가가치세를 신고했습니다",
        "생성형 AI 시장이 급성장하고 있다",
        "오픈AI가 GPT-5.2를 출시했다",
        "앤스로픽의 클로드 오퍼스 4.5 출시 소식",
    ]
    for s in samples:
        print(f"입력   : {s}")
        print(f"토큰   : {tokenize(s)}")
        print(f"FTS5   : {to_fts_query(s)}")
        print(f"tsquery: {to_tsquery(s)}")
        print()
