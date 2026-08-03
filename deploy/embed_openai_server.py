# -*- coding: utf-8 -*-
"""bge-m3 임베딩을 OpenAI 호환 API로 제공하는 서버 (02-4).

OpenAI와 같은 규격(`POST /v1/embeddings`)으로 응답하므로, 클라이언트는
`base_url` 만 바꾸면 OpenAI API와 이 서버를 그대로 오갈 수 있다.

실행:
  pip install fastapi uvicorn FlagEmbedding
  uvicorn deploy.embed_openai_server:app --host 0.0.0.0 --port 8000

확인:
  curl http://localhost:8000/health
  curl -X POST http://localhost:8000/v1/embeddings \
       -H "Content-Type: application/json" \
       -d '{"model":"bge-m3","input":["생성형 AI 시장이 급성장하고 있다"]}'
"""
import threading

from fastapi import FastAPI
from pydantic import BaseModel

from FlagEmbedding import BGEM3FlagModel

MODEL_NAME = "bge-m3"
MAX_LENGTH = 1024

app = FastAPI(title="bge-m3 embedding server (OpenAI compatible)")

# devices를 단일 GPU로 고정해야 FlagEmbedding이 멀티프로세싱 풀(spawn 자식 프로세스)을
# 띄우지 않는다. 풀이 디바이스를 점유하면 메인 스레드의 model.to(device)가
# "CUDA-capable device(s) is/are busy or unavailable"로 실패한다.
model = BGEM3FlagModel("BAAI/bge-m3", use_fp16=True, devices="cuda:0")

# model.encode()는 스레드 세이프하지 않다. FastAPI는 def 엔드포인트를 스레드풀에서
# 병렬 실행하므로, 동시 요청이 같은 model 객체로 encode를 호출하면 CUDA 상태가 깨진다.
# 락으로 GPU 접근을 직렬화한다.
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
    return {"status": "ok"}
