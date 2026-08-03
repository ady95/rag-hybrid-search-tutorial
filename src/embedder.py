# -*- coding: utf-8 -*-
"""임베딩 생성기. 세 가지 백엔드를 같은 인터페이스로 제공한다.

  local   sentence-transformers 로 bge-m3 를 직접 돌린다 (기본, 무료)
  server  OpenAI 호환 임베딩 서버에 HTTP 요청 (GPU 서버가 있을 때)
  openai  OpenAI 임베딩 API

EMBED_BACKEND 환경변수로 고른다. 어느 쪽을 쓰든 embed(texts) -> list[list[float]]
형태로 같은 결과를 돌려주므로 이후 코드는 바뀌지 않는다.

사용:
  python -m src.embedder
"""
import json
import os
import sys
import urllib.request

from src import config

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

BACKEND = os.environ.get("EMBED_BACKEND", "local")
LOCAL_MODEL = os.environ.get("EMBED_LOCAL_MODEL", "BAAI/bge-m3")
OPENAI_MODEL = os.environ.get("EMBED_OPENAI_MODEL", "text-embedding-3-small")

_st_model = None


def _embed_local(texts, batch_size=8):
    global _st_model
    from sentence_transformers import SentenceTransformer
    if _st_model is None:
        _st_model = SentenceTransformer(LOCAL_MODEL)
    vecs = _st_model.encode(texts, batch_size=batch_size,
                            normalize_embeddings=True, show_progress_bar=False)
    return [v.tolist() for v in vecs]


def _embed_server(texts):
    url = config.EMBED_BASE_URL.rstrip("/") + config.EMBED_ENDPOINT
    req = urllib.request.Request(
        url,
        data=json.dumps({"texts": texts}).encode("utf-8"),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=config.EMBED_TIMEOUT) as r:
        payload = json.loads(r.read())
    for key in ("dense", "embeddings", "dense_vecs"):
        if key in payload:
            return payload[key]
    raise RuntimeError(f"임베딩 응답에서 벡터를 찾지 못했습니다: {list(payload)}")


def _embed_openai(texts):
    from openai import OpenAI
    client = OpenAI()
    r = client.embeddings.create(model=OPENAI_MODEL, input=texts)
    return [d.embedding for d in r.data]


def embed(texts, backend=None):
    """문자열 목록 -> 벡터 목록."""
    if isinstance(texts, str):
        texts = [texts]
    b = backend or BACKEND
    if b == "local":
        return _embed_local(texts)
    if b == "server":
        return _embed_server(texts)
    if b == "openai":
        return _embed_openai(texts)
    raise ValueError(f"알 수 없는 백엔드: {b}")


def embed_batched(texts, batch=32, backend=None, progress=True):
    out = []
    for i in range(0, len(texts), batch):
        out.extend(embed(texts[i:i + batch], backend=backend))
        if progress:
            print(f"\r  임베딩 {min(i + batch, len(texts)):,}/{len(texts):,}",
                  end="", flush=True)
    if progress:
        print()
    return out


if __name__ == "__main__":
    v = embed(["생성형 AI 시장이 급성장하고 있다", "인공지능 투자가 늘고 있다"])
    print(f"백엔드: {BACKEND}  개수: {len(v)}  차원: {len(v[0])}")
    import math
    a, b = v[0], v[1]
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(x * x for x in b))
    print(f"두 문장 코사인 유사도: {dot / (na * nb):.4f}")
