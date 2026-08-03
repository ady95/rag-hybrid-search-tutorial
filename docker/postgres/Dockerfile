# PostgreSQL 16 + pgvector + pg_bigm
# pgvector는 공식 이미지에 포함되어 있고, pg_bigm만 소스에서 빌드한다.
FROM pgvector/pgvector:pg16

ARG PGBIGM_VERSION=1.2-20240606

USER root
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential wget ca-certificates postgresql-server-dev-16 \
    && wget -q "https://github.com/pgbigm/pg_bigm/archive/refs/tags/v${PGBIGM_VERSION}.tar.gz" \
    && tar xzf "v${PGBIGM_VERSION}.tar.gz" \
    && cd "pg_bigm-${PGBIGM_VERSION}" \
    && make USE_PGXS=1 \
    && make USE_PGXS=1 install \
    && cd .. && rm -rf "pg_bigm-${PGBIGM_VERSION}" "v${PGBIGM_VERSION}.tar.gz" \
    && apt-get purge -y build-essential wget postgresql-server-dev-16 \
    && apt-get autoremove -y \
    && rm -rf /var/lib/apt/lists/*

USER postgres
