-- 컨테이너 첫 기동 시 1회 자동 실행된다.
CREATE EXTENSION IF NOT EXISTS vector;    -- pgvector: 벡터 타입 + HNSW
CREATE EXTENSION IF NOT EXISTS pg_bigm;   -- 한국어 bigram GIN (폴백 검색)
CREATE EXTENSION IF NOT EXISTS pg_trgm;   -- 보조 trigram (영문·숫자 매칭)

-- 설치 확인용
SELECT extname, extversion FROM pg_extension
WHERE extname IN ('vector', 'pg_bigm', 'pg_trgm')
ORDER BY extname;
