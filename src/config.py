# -*- coding: utf-8 -*-
"""프로젝트 공통 설정. .env 를 읽어 상수로 노출한다."""
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
PDF_DIR = DATA / "pdf"
MD_DIR = DATA / "md"
DB_PATH = DATA / "aibrief.db"


def load_env(path=None):
    """의존성 없이 .env 를 읽는다 (python-dotenv 미설치 환경 대비)."""
    path = Path(path or ROOT / ".env")
    env = {}
    if not path.exists():
        return env
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, v = line.split("=", 1)
        env[k.strip()] = v.strip()
    os.environ.update({k: v for k, v in env.items() if k not in os.environ})
    return env


ENV = load_env()


def get(key, default=None):
    return os.environ.get(key, default)


# 임베딩
EMBED_BASE_URL = get("EMBED_BASE_URL", "http://localhost:8000")
EMBED_ENDPOINT = get("EMBED_ENDPOINT", "/embed")
EMBED_DIM = int(get("EMBED_DIM", "1024"))
EMBED_TIMEOUT = int(get("EMBED_TIMEOUT", "60"))

# PostgreSQL — 기본값은 docker/postgres/docker-compose.yml 이 만드는 것과 같다
DATABASE_URL = get("DATABASE_URL", "postgresql://raguser:ragpass@localhost:5432/ragbook")

# 검색 튜닝
TOPK_PER_MODE = int(get("SEARCH_TOPK_PER_MODE", "30"))
TOP_N = int(get("SEARCH_TOP_N", "10"))
RRF_K = int(get("RRF_K", "60"))
W_KEYWORD = float(get("RRF_WEIGHT_KEYWORD", "1.0"))
W_FALLBACK = float(get("RRF_WEIGHT_FALLBACK", "0.5"))
W_SEMANTIC = float(get("RRF_WEIGHT_SEMANTIC", "1.0"))
KEYWORD_CANDIDATES = int(get("SEARCH_KEYWORD_CANDIDATES", "2000"))
FALLBACK_CANDIDATES = int(get("SEARCH_FALLBACK_CANDIDATES", "300"))

# 청킹
CHUNK_TARGET = int(get("CHUNK_TARGET_CHARS", "900"))
CHUNK_OVERLAP = int(get("CHUNK_OVERLAP_CHARS", "150"))
CHUNK_MAX = int(get("CHUNK_MAX_CHARS", "1600"))
