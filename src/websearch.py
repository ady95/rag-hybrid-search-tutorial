# -*- coding: utf-8 -*-
"""외부 웹 검색 — Tavily (09-5).

내부 자료에 없는 질문을 밖에서 찾는다. router.py 가 "외부로 나가야 한다"고
판정했을 때만 부른다.

실측에서 확인한 것 (2026-08, 09-5 참조):
  - topic="general" 은 published_date 를 주지 않는다. news 만 준다.
    최신 정보를 얻으려고 나갔는데 날짜가 없으면 나간 이유가 없어지므로
    이 모듈은 news 를 기본으로 쓴다.
  - search_depth="advanced" 는 3배 느린데(5.7초) 날짜도 없다. 쓰지 않는다.
  - score 는 관련도지 신뢰도가 아니다. 주식 홍보 블로그가 0.83~0.87 로
    진짜 유용한 기사(0.807)보다 높게 나왔다. 도메인 제한이 더 효과적이다.

    TAVILY_API_KEY=... python -m src.websearch "2027년 EU AI Act 개정안"
"""
import argparse
import json
import sys
import time
import urllib.error
import urllib.request

from src import config

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

API = "https://api.tavily.com/search"

# 근거로 쓸 만한 곳만 남기는 편이 score 임계값보다 낫다.
# 각자의 도메인 목록으로 바꿔 쓸 것.
EXCLUDE_DEFAULT = ["brunch.co.kr", "tistory.com", "blog.naver.com",
                   "cafe.naver.com", "choicestock.co.kr"]


class WebSearchError(RuntimeError):
    pass


def search_web(query, max_results=5, days=180, topic="news",
               include_domains=None, exclude_domains=None, timeout=40):
    """Tavily 검색. (결과 목록, 지연 ms) 를 돌려준다."""
    key = config.get("TAVILY_API_KEY")
    if not key:
        raise WebSearchError("TAVILY_API_KEY 가 없습니다 (.env 확인)")

    body = {"query": query, "max_results": max_results, "topic": topic}
    if topic == "news" and days:
        body["days"] = days
    if include_domains:
        body["include_domains"] = list(include_domains)
    body["exclude_domains"] = list(
        exclude_domains if exclude_domains is not None else EXCLUDE_DEFAULT)

    req = urllib.request.Request(
        API, data=json.dumps(body).encode("utf-8"),
        headers={"Content-Type": "application/json",
                 "Authorization": f"Bearer {key}"},   # 본문 api_key 방식도 동작한다
        method="POST")
    t0 = time.perf_counter()
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            data = json.load(r)
    except urllib.error.HTTPError as e:
        raise WebSearchError(f"HTTP {e.code}: {e.read()[:200].decode(errors='replace')}")
    ms = (time.perf_counter() - t0) * 1000

    hits = []
    for r in data.get("results") or []:
        hits.append({"title": (r.get("title") or "").strip(),
                     "url": r.get("url") or "",
                     "snippet": (r.get("content") or "").strip(),
                     "date": r.get("published_date") or None,
                     "score": r.get("score")})
    return hits, ms


def drop_undated(hits):
    """발행일 없는 결과를 버린다.

    최신 정보를 얻으려고 나갔는데 날짜를 모르면 근거로 쓸 수 없다.
    topic="general" 로 부르면 전부 여기서 걸린다.
    """
    return [h for h in hits if h.get("date")]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("query", nargs="*")
    ap.add_argument("--days", type=int, default=180)
    ap.add_argument("--topic", default="news", choices=["news", "general"])
    ap.add_argument("--max", type=int, default=5)
    a = ap.parse_args()
    q = " ".join(a.query) or "2027년 EU AI Act 개정안 내용"

    hits, ms = search_web(q, max_results=a.max, days=a.days, topic=a.topic)
    dated = drop_undated(hits)
    print(f'질의: "{q}"  ({a.topic}, {ms:.0f}ms)')
    print(f'결과 {len(hits)}건 · 발행일 있는 것 {len(dated)}건\n')
    for h in hits:
        dom = h["url"].split("/")[2] if "//" in h["url"] else h["url"]
        print(f'  [{h["date"] or "날짜없음":<18}] score {h["score"]}  {dom}')
        print(f'      {h["title"][:60]}')
        print(f'      {h["snippet"][:100]}...')


if __name__ == "__main__":
    main()
