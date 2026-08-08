# PostgreSQL 16 + pgvector + PGroonga (Japanese full-text search)
FROM pgvector/pgvector:pg16

RUN apt-get update && \
    apt-get install -y --no-install-recommends curl ca-certificates lsb-release wget && \
    wget -q "https://packages.groonga.org/debian/groonga-apt-source-latest-$(lsb_release -cs).deb" && \
    apt-get install -y --no-install-recommends "./groonga-apt-source-latest-$(lsb_release -cs).deb" && \
    rm "groonga-apt-source-latest-$(lsb_release -cs).deb" && \
    apt-get update && \
    apt-get install -y --no-install-recommends postgresql-16-pgdg-pgroonga && \
    rm -rf /var/lib/apt/lists/*
