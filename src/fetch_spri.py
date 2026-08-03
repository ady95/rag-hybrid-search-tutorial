# -*- coding: utf-8 -*-
"""SPRi AI 브리프 PDF를 내려받는다 (03-1).

소프트웨어정책연구소(SPRi)가 공개하는 'AI 브리프' 월간 보고서를
data/pdf/ 에 저장한다.

상세 페이지 HTML에는 PDF 주소가 문자열로 없다. 다운로드 버튼이
`file_down('<파일ID>')` 자바스크립트를 호출하고 그 함수가
`/download/<파일ID>` 로 이동시키는 구조라, 파일 ID를 먼저 뽑아야 한다.

사용:
  python -m src.fetch_spri              # 2026년 1~7월호 (책 기준 데이터)
  python -m src.fetch_spri --list       # 최근 게시글 목록만 확인
  python -m src.fetch_spri --post 23943 --name 202601   # 특정 호만
"""
import argparse
import re
import sys
import time
import urllib.request

from src import config

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

BASE = "https://spri.kr"
LIST_URL = BASE + "/posts?code=AI-Brief"
UA = {"User-Agent": "Mozilla/5.0 (compatible; rag-hybrid-search-tutorial/1.0)"}

# 이 책이 사용한 2026년 1~7월호. (게시글 ID, 저장할 이름)
# 게시글 ID가 바뀌면 --list 로 다시 확인하면 된다.
ISSUES = [
    (23943, "202601"),
    (23950, "202602"),
    (23956, "202603"),
    (23964, "202604"),
    (23982, "202605"),
    (23992, "202606"),
    (24003, "202607"),
]

FILE_ID_RE = re.compile(r"file_down\('(\d+)'\)")
POST_RE = re.compile(r"/posts/view/(\d+)\?code=AI-Brief")
TITLE_RE = re.compile(r"<title>(.*?)</title>", re.S | re.I)


def get(url, timeout=30):
    req = urllib.request.Request(url, headers=UA)
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return r.read()


def find_file_id(post_id):
    """상세 페이지에서 다운로드 파일 ID를 뽑는다."""
    html = get(f"{BASE}/posts/view/{post_id}?code=AI-Brief").decode("utf-8", "replace")
    ids = FILE_ID_RE.findall(html)
    if not ids:
        return None, None
    title = TITLE_RE.search(html)
    return ids[0], (title.group(1).strip() if title else "")


def download(post_id, name, out_dir):
    file_id, title = find_file_id(post_id)
    if not file_id:
        print(f"  [{name}] 파일 ID를 찾지 못했습니다 (게시글 {post_id})")
        return False

    out = out_dir / f"SPRi_AI_Brief_{name}.pdf"
    if out.exists():
        print(f"  [{name}] 이미 있음 — 건너뜀 ({out.stat().st_size/1024/1024:.2f} MB)")
        return True

    data = get(f"{BASE}/download/{file_id}", timeout=120)
    if not data.startswith(b"%PDF"):
        print(f"  [{name}] PDF가 아닙니다 (앞 8바이트: {data[:8]!r})")
        return False

    out.write_bytes(data)
    print(f"  [{name}] 저장 {out.stat().st_size/1024/1024:5.2f} MB  <- 파일ID {file_id}")
    return True


def list_recent(n=15):
    html = get(LIST_URL).decode("utf-8", "replace")
    seen, rows = set(), []
    for pid in POST_RE.findall(html):
        if pid in seen:
            continue
        seen.add(pid)
        rows.append(pid)
        if len(rows) >= n:
            break
    print(f"최근 게시글 {len(rows)}개 (최신순)")
    for pid in rows:
        print(f"  게시글 ID {pid}   {BASE}/posts/view/{pid}?code=AI-Brief")
    print("\n원하는 호의 게시글 ID를 --post 로 넘기면 내려받습니다.")


def main():
    ap = argparse.ArgumentParser(description="SPRi AI 브리프 PDF 내려받기")
    ap.add_argument("--list", action="store_true", help="최근 게시글 목록만 출력")
    ap.add_argument("--post", type=int, help="특정 게시글 ID")
    ap.add_argument("--name", help="--post 와 함께 쓸 저장 이름 (예: 202601)")
    a = ap.parse_args()

    if a.list:
        list_recent()
        return

    config.PDF_DIR.mkdir(parents=True, exist_ok=True)
    targets = [(a.post, a.name or str(a.post))] if a.post else ISSUES

    print(f"저장 위치: {config.PDF_DIR}")
    ok = 0
    for post_id, name in targets:
        try:
            if download(post_id, name, config.PDF_DIR):
                ok += 1
        except Exception as e:
            print(f"  [{name}] 실패: {type(e).__name__} {e}")
        time.sleep(0.5)          # 서버 예의

    print(f"\n{ok}/{len(targets)}개 완료")
    if ok:
        print("다음: python -m src.pdf_to_md")


if __name__ == "__main__":
    main()
