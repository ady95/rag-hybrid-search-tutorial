# -*- coding: utf-8 -*-
"""bge-m3 임베딩을 OpenAI 호환 API로 제공한다 (02-4).

OpenAI와 같은 규격(`POST /v1/embeddings`)으로 응답하므로, 클라이언트는
`base_url` 만 바꾸면 OpenAI API와 이 API를 그대로 오갈 수 있다.

GPU가 없어도 된다. `EMBED_DEVICE=cpu` 로 노트북에서도 돌아간다.
코드 변경 없이 환경변수만으로 전환된다.

실행 (CPU):
  pip install torch --index-url https://download.pytorch.org/whl/cpu
  pip install fastapi uvicorn FlagEmbedding
  EMBED_DEVICE=cpu OMP_NUM_THREADS=8 uvicorn deploy.embed_openai_server:app --port 8000

실행 (GPU):
  EMBED_DEVICE=cuda:0 uvicorn deploy.embed_openai_server:app --host 0.0.0.0 --port 8000

확인:
  curl http://localhost:8000/health
  curl -X POST http://localhost:8000/v1/embeddings \
       -H "Content-Type: application/json" \
       -d '{"model":"bge-m3","input":["생성형 AI 시장이 급성장하고 있다"]}'
"""
import os
import threading

import torch
from fastapi import FastAPI
from pydantic import BaseModel

from FlagEmbedding import BGEM3FlagModel

MODEL_NAME = "bge-m3"
MAX_LENGTH = 1024
DEVICE = os.getenv("EMBED_DEVICE", "cpu")      # "cpu" 또는 "cuda:0"

app = FastAPI(title="bge-m3 embedding API (OpenAI compatible)")

if DEVICE == "cpu":
    # torch는 컨테이너의 --cpus 제한이 아니라 호스트 코어 수를 보고 스레드를 띄운다.
    # 오버서브스크립션되면 최대 3배까지 느려지므로 실제 할당 코어 수로 고정한다.
    torch.set_num_threads(int(os.getenv("OMP_NUM_THREADS", os.cpu_count())))

# CPU에서는 fp16을 쓸 수 없다. FlagEmbedding이 내부적으로 fp32로 전환하지만
# 의도를 드러내기 위해 여기서도 끈다.
# GPU일 때 devices를 단일 장치로 고정해야 멀티프로세싱 풀(spawn 자식 프로세스)이
# 뜨지 않는다. 풀이 장치를 점유하면 메인 스레드의 model.to(device)가
# "CUDA-capable device(s) is/are busy or unavailable"로 실패한다.
model = BGEM3FlagModel("BAAI/bge-m3", use_fp16=(DEVICE != "cpu"), devices=DEVICE)

# model.encode()는 스레드 세이프하지 않다. FastAPI는 def 엔드포인트를 스레드풀에서
# 병렬 실행하므로, 동시 요청이 같은 model 객체로 encode를 호출하면 상태가 깨진다.
# CPU에서도 반드시 유지해야 한다 (GPU 전용 안전장치가 아니다).
_encode_lock = threading.Lock()


class EmbeddingRequest(BaseModel):
    input: str | list[str]
    model: str = MODEL_NAME
    encoding_format: str = "float"


@app.post("/v1/embeddings")
def embeddings(req: EmbeddingRequest):
    texts = [req.input] if isinstance(req.input, str) else req.input
    with _encode_lock:
        out = model.encode(texts, return_dense=True, max_length=MAX_LENGTH)
    return {
        "object": "list",
        "data": [
            {"object": "embedding", "index": i, "embedding": vec.tolist()}
            for i, vec in enumerate(out["dense_vecs"])
        ],
        "model": req.model,
        "usage": {"prompt_tokens": 0, "total_tokens": 0},
    }


# 일부 OpenAI 호환 클라이언트가 기동 시 모델 목록을 조회한다.
@app.get("/v1/models")
def models():
    return {
        "object": "list",
        "data": [{"id": MODEL_NAME, "object": "model", "owned_by": "BAAI"}],
    }


@app.get("/health")
def health():
    return {"status": "ok", "device": DEVICE}
