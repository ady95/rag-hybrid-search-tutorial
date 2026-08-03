# -*- coding: utf-8 -*-
"""SPRi AI 브리프 PDF를 마크다운으로 변환한다.

pdfplumber로 줄 단위 텍스트와 폰트 크기를 함께 뽑아
  - 세로쓰기 장식 문자 제거
  - 머리말/꼬리말(페이지 절반 이상 반복되는 줄) 제거
  - 폰트 크기로 헤딩 추정
  - 줄바꿈으로 끊긴 문장 결합
을 거쳐 마크다운으로 저장한다.

PyMuPDF가 아니라 pdfplumber를 쓰는 이유는 03-2 참조 —
이 PDF는 PyMuPDF에서 한글과 라틴 문자의 읽기 순서가 뒤섞인다.

사용:
  python -m src.pdf_to_md
"""
import io
import re
import sys
from collections import Counter

import pdfplumber

from src import config

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

BOILERPLATE = [
    re.compile(r"^\s*SPRi\s*AI\s*Brief\s*$", re.I),
    re.compile(r"^\s*소프트웨어정책연구소\s*$"),
    re.compile(r"^\s*\d{1,3}\s*$"),
    re.compile(r"^\s*Ⅰ?\s*20\d\d년\s*\d{1,2}월호\s*Ⅰ?\s*$"),
    re.compile(r"^\s*CONTENTS\s*$", re.I),
    re.compile(r"^\s*[･·∙\-–—|]\s*$"),
]

# 사설 영역(PUA) 불릿과 특수 공백 정리. 빈 문자열 키를 넣으면
# str.replace가 모든 글자 사이에 삽입되므로 반드시 이스케이프로 적는다.
REPLACE = {
    "\uf06e": "- ", "\uf0a1": "- ", "\uf0b7": "- ",   # PUA 불릿
    "\u2022": "- ", "\u2219": "- ",                    # 불릿 기호
    "\u00a0": " ", "\u200b": "",                       # NBSP, ZWSP
}

# 본문 첫 글자로 쓰이는 사설 불릿 (SPRi 고유 마커)
LEADING_MARKER = re.compile(r"^\s*(n|£|◇|□|■)\s+")


def clean(s):
    for a, b in REPLACE.items():
        s = s.replace(a, b)
    return re.sub(r"[ \t]+", " ", s).strip()


def is_noise(s):
    if not s:
        return True
    if any(p.match(s) for p in BOILERPLATE):
        return True
    # 세로쓰기로 한 글자씩 떨어진 장식 텍스트
    if len(s) <= 2 and not re.search(r"[가-힣A-Za-z]{2}", s):
        return True
    return False


def page_lines(page):
    out = []
    for ln in page.extract_text_lines():
        txt = clean(ln["text"])
        if not txt:
            continue
        size = round(max(c["size"] for c in ln["chars"]), 1)
        out.append((txt, size))
    return out


def body_font_size(all_lines):
    c = Counter()
    for txt, size in all_lines:
        c[size] += len(txt)
    return c.most_common(1)[0][0] if c else 10.0


def repeated_lines(pages, min_ratio=0.4):
    c = Counter()
    for lines in pages:
        for txt, _ in set(lines):
            if 0 < len(txt) <= 40:
                c[txt] += 1
    threshold = max(3, int(len(pages) * min_ratio))
    return {t for t, n in c.items() if n >= threshold}


SENT_END = re.compile(r"(?:[.!?…]|다\.|음\.|함\.|임\.|[」』”\)])\s*$")


def to_markdown(pages):
    all_lines = [ln for p in pages for ln in p]
    body = body_font_size(all_lines)
    rep = repeated_lines(pages)

    blocks, buf = [], []

    def flush():
        if buf:
            blocks.append(" ".join(buf))
            buf.clear()

    for lines in pages:
        for txt, size in lines:
            if is_noise(txt) or txt in rep:
                continue
            # 헤딩 판정: 본문보다 뚜렷하게 큰 글자 + 짧은 줄
            if size >= body + 2.5 and len(txt) <= 70:
                flush()
                blocks.append(f"## {txt}")
                continue
            if size >= body + 1.0 and len(txt) <= 70:
                flush()
                blocks.append(f"### {txt}")
                continue
            # 사설 불릿으로 시작하면 새 문단
            m = LEADING_MARKER.match(txt)
            if m:
                flush()
                txt = txt[m.end():]
            buf.append(txt)
            if SENT_END.search(txt):
                flush()
    flush()

    md = "\n\n".join(b for b in blocks if b.strip())
    md = re.sub(r"\n{3,}", "\n\n", md)
    return md


def convert(pdf_path):
    with pdfplumber.open(pdf_path) as pdf:
        pages = [page_lines(p) for p in pdf.pages]
    return to_markdown(pages), len(pages)


def main():
    config.MD_DIR.mkdir(parents=True, exist_ok=True)
    pdfs = sorted(config.PDF_DIR.glob("*.pdf"))
    if not pdfs:
        print(f"PDF가 없습니다: {config.PDF_DIR}")
        return
    total = 0
    for p in pdfs:
        md, npages = convert(p)
        out = config.MD_DIR / (p.stem + ".md")
        io.open(out, "w", encoding="utf-8", newline="\n").write(md)
        h2 = len(re.findall(r"^## ", md, re.M))
        h3 = len(re.findall(r"^### ", md, re.M))
        print(f"{p.name:30s} {npages:3d}p -> {len(md):7,d}자  H2 {h2:3d} / H3 {h3:3d}")
        total += len(md)
    print(f"\n총 {len(pdfs)}개 문서 / {total:,}자")


if __name__ == "__main__":
    main()
