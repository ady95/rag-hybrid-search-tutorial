# -*- coding: utf-8 -*-
"""임베딩 생성기 — OpenAI API와 자체 API를 같은 코드로 다룬다.

자체 API(`deploy/embed_openai_server.py`)가 OpenAI와 **같은 규격**
(`POST /v1/embeddings`)으로 응답하므로, 바뀌는 것은 `base_url` 하나뿐이다.
분기문도 응답 파싱 코드도 따로 둘 필요가 없다.

  EMBED_BACKEND=openai   OpenAI API        (text-embedding-3-small, 1536차원)
  EMBED_BACKEND=server   자체 bge-m3 API    (1024차원, GPU 없이 CPU로도 동작)

사용:
  python -m src.embedder
"""
import os
import sys

from src import config

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

BACKEND = os.environ.get("EMBED_BACKEND", "server")
OPENAI_MODEL = os.environ.get("EMBED_OPENAI_MODEL", "text-embedding-3-small")
SERVER_MODEL = os.environ.get("EMBED_SERVER_MODEL", "bge-m3")

_client = None
_model = None


def get_client(backend=None):
    """백엔드에 맞는 OpenAI 클라이언트를 만든다 (한 번만).

    자체 API는 인증을 요구하지 않지만 SDK가 키를 요구하므로
    아무 문자열이나 넣어 준다.
    """
    global _client, _model
    b = backend or BACKEND
    if _client is None:
        from openai import OpenAI
        if b == "server":
            _client = OpenAI(
                base_url=config.EMBED_BASE_URL.rstrip("/") + "/v1",
                api_key=os.environ.get("EMBED_SERVER_API_KEY", "not-needed"),
                timeout=config.EMBED_TIMEOUT,
            )
            _model = SERVER_MODEL
        elif b == "openai":
            _client = OpenAI(timeout=config.EMBED_TIMEOUT)   # OPENAI_API_KEY 사용
            _model = OPENAI_MODEL
        else:
            raise ValueError(f"알 수 없는 백엔드: {b} (openai | server)")
    return _client, _model


def embed(texts, backend=None):
    """문자열 목록 -> 벡터 목록. 백엔드가 달라도 결과 형태는 같다."""
    if isinstance(texts, str):
        texts = [texts]
    client, model = get_client(backend)
    r = client.embeddings.create(model=model, input=texts)
    # data 순서가 보장되지만 index로 한 번 더 정렬해 안전하게 만든다
    return [d.embedding for d in sorted(r.data, key=lambda d: d.index)]


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
    import math
    import time

    pairs = ["생성형 AI 시장이 급성장하고 있다", "인공지능 투자가 늘고 있다"]

    t0 = time.perf_counter()
    v = embed(pairs)
    first = (time.perf_counter() - t0) * 1000
    t0 = time.perf_counter()
    embed(pairs)
    warm = (time.perf_counter() - t0) * 1000

    a, b = v
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(x * x for x in b))

    print(f"백엔드   : {BACKEND}")
    print(f"차원     : {len(a)}  (설정값 EMBED_DIM={config.EMBED_DIM})")
    print(f"벡터 노름: {na:.6f}  (1.0이면 정규화된 벡터)")
    print(f"소요     : 첫 호출 {first:.0f} ms / 두 번째 {warm:.0f} ms")
    print(f"두 문장 코사인 유사도: {dot / (na * nb):.4f}")

    if len(a) != config.EMBED_DIM:
        print(f"\n[주의] 실제 차원({len(a)})과 EMBED_DIM({config.EMBED_DIM})이 다릅니다. "
              f".env를 맞춰 주세요.")
