# RAG 하이브리드 검색 예제 코드

한국어 문서로 **하이브리드 검색(Hybrid Search)** 을 처음부터 만들어 보는 예제입니다.
키워드 검색(FTS5 · tsvector), 의미 검색(sqlite-vec · pgvector), 폴백 검색(trigram · pg_bigm)을
**RRF(Reciprocal Rank Fusion)** 로 합칩니다. SQLite로 끝까지 만든 뒤 PostgreSQL로 옮깁니다.

> 위키독스 책 **[RAG 하이브리드 검색 시스템 구축 따라하기 (SQLite부터 PostgreSQL까지)](https://wikidocs.net/book/20783)** 의 예제 코드입니다.
> 각 모듈이 책의 어느 장에 대응하는지는 아래 표에 있습니다.

---

## 왜 하이브리드인가

한국어 문서 489개 청크로 실측한 결과입니다. 사람이 만든 질의 12개와 손으로 라벨링한 정답으로 쟀습니다.

| 검색 방식 | Hit@1 | Recall@10 | MRR |
|---|---|---|---|
| 키워드 (FTS5 + BM25) | 0.500 | 0.542 | 0.708 |
| 의미 (sqlite-vec KNN) | 0.750 | 0.624 | 0.875 |
| 폴백 (FTS5 trigram) | 0.417 | 0.220 | 0.492 |
| **하이브리드 (RRF 융합)** | **0.833** | **0.837** | **0.917** |

가장 좋은 하나(의미 검색 0.875)보다 셋을 합친 쪽(0.917)이 낫습니다.

한국어에서 특히 중요한 지점이 있습니다. `오퍼스` 를 단독으로 검색하면 형태소 분석기가
`퍼스` 로 잘못 잘라 **결과가 0건**이 됩니다. 색인에는 `오퍼스` 가 5건 멀쩡히 있는데도요.
문맥이 있으면(`클로드 오퍼스`) 제대로 자릅니다. 색인은 문장을 받아 유리하고 질의는 단어
하나를 받아 불리한 이 비대칭이 한국어 검색에서 조용한 0건을 만듭니다.
trigram 폴백이 필수인 이유입니다.

---

## 빠른 시작

```bash
git clone https://github.com/ady95/rag-hybrid-search-tutorial.git
cd rag-hybrid-search-tutorial

python -m venv .venv && .venv\Scripts\activate     # Windows
# python3 -m venv .venv && source .venv/bin/activate  # macOS / Linux

pip install -r requirements.txt
cp .env.example .env          # 값 채우기 (임베딩 백엔드 등)

python -m src.fetch_spri      # 샘플 PDF 7개 내려받기
python -m src.pdf_to_md       # PDF -> 마크다운
python -m src.chunker         # 마크다운 -> 청크 489개
python -m src.build_sqlite    # 색인 (FTS5 + trigram + 벡터)

python -m src.search_sqlite "AI 규제 법안이 통과된 나라"
```

검색 모드를 바꿔 가며 비교해 봅니다.

```bash
python -m src.search_sqlite "AI 규제" --mode keyword --ym 2026-03
python -m src.compare_modes            # 네 모드 나란히 비교
python -m src.evaluate --make-auto 30  # 평가셋 생성
python -m src.evaluate                 # Recall / MRR 측정
```

PostgreSQL로 확장합니다.

```bash
cd docker/postgres && docker compose up -d && cd ../..
python -m src.build_pg
python -m src.search_pg "AI 규제 법안이 통과된 나라"
python -m src.bench                    # SQLite vs PostgreSQL 비교
```

---

## 모듈 구성

| 파일 | 역할 | 책의 장 |
|---|---|---|
| `config.py` | `.env` 로딩과 공통 상수 | 02, 07-2 |
| `fetch_spri.py` | 샘플 PDF 내려받기 | 03-1 |
| `pdf_to_md.py` | PDF → 마크다운 (pdfplumber, 폰트 크기로 헤딩 추정) | 03-2 |
| `chunker.py` | 헤딩 경계 기반 청킹 | 03-3 |
| `tokenizer_ko.py` | Kiwi 형태소 토큰화, FTS5/tsquery 변환 | 02-3, 04-2 |
| `embedder.py` | 임베딩 생성 (local / server / openai) | 02-4, 05-1 |
| `build_sqlite.py` | SQLite 색인 (FTS5 + trigram + sqlite-vec) | 04-3, 05-3 |
| `search_sqlite.py` | 하이브리드 검색 + RRF 융합 | 04~06 |
| `compare_modes.py` | 네 가지 검색 모드 비교 | 06-4 |
| `build_pg.py` | PostgreSQL 색인 (tsvector + pgvector + pg_bigm) | 07-3 |
| `search_pg.py` | PostgreSQL 검색, RRF를 SQL 한 번에 | 07-4 |
| `evaluate.py` | Recall@k / Hit@k / MRR 측정 | 08-1, 08-2 |
| `bench.py` | SQLite vs PostgreSQL 벤치마크 | 07-5, 08-3 |
| `bench_scale.py` | 규모를 키웠을 때의 성능 관찰 | 08-3 |

---

## 실행 환경

| 구성 요소 | 검증 버전 |
|---|---|
| Python | 3.10.18 |
| SQLite | 3.50.2 (FTS5 · trigram 내장) |
| sqlite-vec | 0.1.6 |
| kiwipiepy (Kiwi) | 0.20.1 |
| pdfplumber | 0.11.9 |
| PostgreSQL | 16.14 |
| pgvector / pg_bigm | 0.7.4 / 1.2 |
| psycopg | 3.2.3 |
| 임베딩 모델 | BAAI/bge-m3 (1024차원) |

Windows · macOS · Linux에서 동작합니다. GPU는 필요하지 않습니다.

### 임베딩 백엔드 고르기

`EMBED_BACKEND` 환경변수로 선택합니다. 어느 쪽을 쓰든 이후 코드는 바뀌지 않습니다.

| 값 | 방식 | 비고 |
|---|---|---|
| `local` | sentence-transformers로 `BAAI/bge-m3` 직접 실행 | 무료, 최초 2.2GB 다운로드 |
| `server` | OpenAI 호환 임베딩 서버에 HTTP 요청 | GPU 서버가 있을 때 가장 빠름 |
| `openai` | OpenAI 임베딩 API | `OPENAI_API_KEY` 필요 |

---

## 샘플 데이터

소프트웨어정책연구소(SPRi)가 공개하는 **AI 브리프 2026년 1~7월호**(PDF 7개, 202페이지)를 사용합니다.

PDF 본문은 SPRi의 저작물이므로 **이 저장소에 포함하지 않습니다.** `python -m src.fetch_spri` 가
공개된 배포 주소에서 직접 내려받습니다. 원문과 이용 조건은 아래에서 확인하세요.

- SPRi AI 브리프: <https://spri.kr/posts?code=AI-Brief>

`data/evalset.json`(질의 42개와 정답 라벨)은 이 저장소의 산출물이라 함께 배포합니다.
청크 ID는 위 파이프라인을 그대로 돌리면 동일하게 재현됩니다.

---

## 알아두면 좋은 함정

만들면서 실제로 밟은 것들입니다. 자세한 내용은 책 부록 A에 있습니다.

- **색인과 질의는 반드시 같은 토크나이저를 통과해야 합니다.** 어긋나면 예외도 경고도 없이 결과만 0건이 됩니다
- `add_user_word()` 는 **첫 토큰화 이전에만** 반영됩니다. 이후에 부르면 `False` 를 반환하고 조용히 무시됩니다
- 벤치마크 전에 예열하세요. Kiwi 사전 적재에 약 2초가 걸립니다 (첫 질의 1,996ms → 이후 1ms)
- trigram/pg_bigm 폴백에 질의 전체를 구(phrase)로 넣으면 다단어 한국어 질의가 거의 0건입니다. 어절로 나눠야 합니다
- sqlite-vec은 KNN 질의에 `WHERE` 조건을 함께 걸 수 없습니다
- PostgreSQL에서 `VACUUM` 은 트랜잭션 블록 안에서 실행할 수 없습니다 (`conn.autocommit = True`)
- 규모가 커지면 `ROW_NUMBER() OVER (ORDER BY ...)` 뒤의 `LIMIT` 은 소용없습니다. **랭킹 대상 자체를 줄여야** 합니다 (19.5만 행에서 845ms → 24.9ms)

---

## 라이선스

MIT. 자유롭게 가져다 쓰세요.

샘플 데이터(SPRi AI 브리프)의 저작권은 소프트웨어정책연구소에 있으며 이 라이선스와 무관합니다.
