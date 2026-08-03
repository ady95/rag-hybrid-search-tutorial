# -*- coding: utf-8 -*-
"""마크다운을 검색용 청크로 자른다.

헤딩을 경계로 삼되, 한 절이 너무 길면 문단 단위로 더 자르고
너무 짧은 절은 앞뒤와 합친다. 각 청크는 자기가 속한 헤딩 경로를
title 로 함께 들고 다닌다 — 검색 결과에 맥락을 보여 주고,
키워드 색인에서 가중치를 더 주기 위해서다.

사용:
  python -m src.chunker
"""
import io
import json
import re
import sys

from src import config

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

HEADING = re.compile(r"^(#{2,3})\s+(.*)$")
MIN_CHUNK = 120          # 이보다 짧으면 앞 청크에 붙인다
DROP_SHORTER_THAN = 40   # 이보다 짧으면 버린다 (표지·구분면 잔재)


def split_sections(md):
    """마크다운을 (헤딩경로, 본문) 목록으로 나눈다."""
    h2 = h3 = ""
    buf, out = [], []

    def flush():
        body = "\n".join(buf).strip()
        if body:
            path = " > ".join(x for x in (h2, h3) if x)
            out.append((path, body))
        buf.clear()

    for line in md.split("\n"):
        m = HEADING.match(line)
        if m:
            flush()
            if len(m.group(1)) == 2:
                h2, h3 = m.group(2).strip(), ""
            else:
                h3 = m.group(2).strip()
            continue
        buf.append(line)
    flush()
    return out


def split_long(text, target, overlap, hard_max):
    """긴 본문을 문단 경계에서 target 자 내외로 자른다."""
    paras = [p.strip() for p in re.split(r"\n{2,}", text) if p.strip()]
    chunks, cur = [], ""
    for p in paras:
        # 문단 하나가 hard_max를 넘으면 문장 단위로 쪼갠다
        pieces = [p]
        if len(p) > hard_max:
            # 파이썬 re는 가변 길이 lookbehind를 지원하지 않으므로
            # 종결부호 한 글자만 본다 ("~다." "~음." 도 마침표로 끝난다)
            pieces = re.split(r"(?<=[.!?…])\s+", p)
        for piece in pieces:
            if not piece:
                continue
            if cur and len(cur) + len(piece) + 1 > target:
                chunks.append(cur)
                tail = cur[-overlap:] if overlap else ""
                cur = (tail + " " + piece).strip() if tail else piece
            else:
                cur = (cur + " " + piece).strip() if cur else piece
    if cur:
        chunks.append(cur)
    return chunks


def chunk_document(doc_id, md, meta):
    out = []
    for path, body in split_sections(md):
        for i, piece in enumerate(split_long(body, config.CHUNK_TARGET,
                                             config.CHUNK_OVERLAP, config.CHUNK_MAX)):
            if len(piece) < DROP_SHORTER_THAN:
                continue
            out.append({
                "doc_id": doc_id,
                "title": path,
                "body": piece,
                "seq": len(out),
                **meta,
            })
    # 너무 짧은 청크는 직전 청크에 병합
    merged = []
    for c in out:
        if merged and len(c["body"]) < MIN_CHUNK and merged[-1]["title"] == c["title"]:
            merged[-1]["body"] += " " + c["body"]
        else:
            merged.append(c)
    for i, c in enumerate(merged):
        c["seq"] = i
        c["chunk_id"] = f"{doc_id}#{i:04d}"
    return merged


def main():
    mds = sorted(config.MD_DIR.glob("*.md"))
    if not mds:
        print("마크다운이 없습니다. 먼저 python -m src.pdf_to_md 를 실행하세요.")
        return
    all_chunks = []
    for p in mds:
        doc_id = p.stem                       # SPRi_AI_Brief_202601
        ym = re.search(r"(\d{4})(\d{2})$", doc_id)
        meta = {
            "year_month": f"{ym.group(1)}-{ym.group(2)}" if ym else "",
            "source": p.name,
        }
        chunks = chunk_document(doc_id, p.read_text(encoding="utf-8"), meta)
        all_chunks.extend(chunks)
        lens = [len(c["body"]) for c in chunks]
        print(f"{p.name:30s} 청크 {len(chunks):4d}개  "
              f"평균 {sum(lens)//max(1,len(lens)):4d}자  "
              f"최소 {min(lens):4d}  최대 {max(lens):4d}")

    out = config.DATA / "chunks.jsonl"
    with io.open(out, "w", encoding="utf-8", newline="\n") as f:
        for c in all_chunks:
            f.write(json.dumps(c, ensure_ascii=False) + "\n")

    lens = [len(c["body"]) for c in all_chunks]
    lens.sort()
    print(f"\n총 청크 {len(all_chunks):,}개 -> {out}")
    print(f"길이 분포  p10={lens[len(lens)//10]}  p50={lens[len(lens)//2]}  "
          f"p90={lens[len(lens)*9//10]}  max={lens[-1]}")


if __name__ == "__main__":
    main()
