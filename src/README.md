# 예제 코드 — RAG 하이브리드 검색 시스템 구축 따라하기

위키독스 책 「RAG 하이브리드 검색 시스템 구축 따라하기 (SQLite부터 PostgreSQL까지)」의 예제 코드입니다.
각 모듈은 책의 특정 장에 대응하며, 순서대로 실행하면 검색 시스템이 완성됩니다.

## 모듈 구성

| 파일 | 역할 | 관련 장 |
|------|------|---------|
| `config.py` | `.env` 로딩과 공통 상수 | 02, 07-2 |
| `pdf_to_md.py` | PDF → 마크다운 변환 (pdfplumber, 폰트 크기로 헤딩 추정) | 03-2 |
| `chunker.py` | 헤딩 경계 기반 청킹 | 03-3 |
| `tokenizer_ko.py` | Kiwi 형태소 토큰화, FTS5/tsquery 변환 | 02-3, 04-2 |
| `dict_miner.py` | 사용자 사전 후보 채굴 (승인은 `data/userdict.json`) | 04-5 |
| `embedder.py` | 임베딩 생성 (자체 서버 API / OpenAI API 2종 백엔드) | 02-4, 05-1 |
| `build_sqlite.py` | SQLite 색인 (FTS5 + trigram + sqlite-vec) | 04-3, 05-3 |
| `search_sqlite.py` | SQLite 하이브리드 검색 + RRF 융합 | 04, 05, 06 |
| `compare_modes.py` | 네 가지 검색 모드 비교 도구 | 06-4 |
| `build_pg.py` | PostgreSQL 색인 (tsvector + pgvector + pg_bigm) | 07-3 |
| `search_pg.py` | PostgreSQL 검색, RRF를 SQL 한 번에 | 07-4 |
| `evaluate.py` | Recall@k / Hit@k / MRR 측정 | 08-1, 08-2 |
| `bench.py` | SQLite vs PostgreSQL 비교 벤치마크 | 07-5, 08-3 |
| `bench_scale.py` | 규모를 키웠을 때의 성능 관찰 | 08-3 |
| `ablation.py` | 구성 요소를 하나씩 빼 보는 절제 실험 | 06-3 |
| `eval_ragas.py` | RAGAS 기반 검색·생성 품질 평가 | 08-6 |
| `dashboard.py` | Streamlit 검색 품질 대시보드 | 10-3 |
| `api.py` + `static/` | 네 모드를 나란히 보는 검색 비교 데모 (FastAPI) | 10-4 |
| `fetch_spri.py` | 샘플 PDF(SPRi AI 브리프) 내려받기 | 03-1 |

## 실행 순서

```bash
# 0) 준비
pip install -r requirements.txt
cp .env.example .env          # 값 채우기

# 1) 샘플 PDF를 data/pdf/ 에 넣고 마크다운으로 변환
python -m src.pdf_to_md

# 2) 청킹
python -m src.chunker

# 3) SQLite 색인 (임베딩 생성 포함)
python -m src.build_sqlite

# 3-1) 사용자 사전 후보 뽑아 보기 (04-5)
#      data/userdict.json 에서 true 로 바꾼 뒤 재색인해야 반영된다
python -m src.dict_miner

# 4) 검색해 보기
python -m src.search_sqlite "AI 규제 법안이 통과된 나라"
python -m src.search_sqlite "AI 규제" --mode keyword --ym 2026-03

# 5) 네 가지 모드 비교
python -m src.compare_modes

# 6) 품질 평가
python -m src.evaluate --make-auto 30
python -m src.evaluate

# 7) PostgreSQL로 확장 (docker/postgres 에서 컨테이너 먼저 기동)
python -m src.build_pg
python -m src.search_pg "AI 규제 법안이 통과된 나라"

# 8) 두 DB 비교
python -m src.bench

# 9) 품질 대시보드 (10-3)
streamlit run src/dashboard.py

# 10) 검색 비교 데모 (10-4) — http://localhost:8765
uvicorn src.api:app --port 8765
```

## 임베딩 백엔드 고르기

`EMBED_BACKEND` 환경변수로 선택합니다. **둘 다 OpenAI 규격(`/v1/embeddings`)** 이라
바뀌는 것은 `base_url` 하나뿐이고 이후 코드는 동일합니다.

| 값 | 방식 | 모델 | 차원 | 비고 |
|----|------|------|------|------|
| `server` (기본) | 자체 API | BAAI/bge-m3 | 1024 | `deploy/embed_openai_server.py` 로 기동. GPU 없이 CPU로도 동작 |
| `openai` | OpenAI 임베딩 API | text-embedding-3-small | 1536 | `OPENAI_API_KEY` 필요 |

백엔드를 바꾸면 차원이 달라지므로 `EMBED_DIM` 도 함께 바꾸고 **재색인**해야 합니다.

자체 API 기동 — 별도 장비가 필요하지 않습니다. 노트북에서 띄워도 됩니다.

```bash
pip install fastapi uvicorn FlagEmbedding
pip install torch --index-url https://download.pytorch.org/whl/cpu   # CPU로 돌릴 때

EMBED_DEVICE=cpu OMP_NUM_THREADS=8 uvicorn deploy.embed_openai_server:app --port 8000
curl http://localhost:8000/health      # {"status":"ok","device":"cpu"}
```

GPU가 있으면 `EMBED_DEVICE=cuda:0` 로 바꾸면 됩니다. 도커로 띄우려면
`deploy/Dockerfile.cpu` 와 `deploy/docker-compose.embed-cpu.yml` 을 쓰세요
(CPU 전용 이미지 약 1.69 GB).

## 주의

- **색인과 질의는 반드시 같은 토크나이저를 통과해야 합니다.** `tokenizer_ko.tokenize()` 를 양쪽에서 쓰세요. 어긋나면 검색이 조용히 0건이 됩니다
- 벤치마크 전에는 예열이 필요합니다. Kiwi 사전 적재에 약 2초가 걸립니다
- `.env` 는 커밋하지 마세요
