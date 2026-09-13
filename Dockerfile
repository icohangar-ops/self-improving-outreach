FROM python:3.12-slim

WORKDIR /app
COPY pyproject.toml README.md ./
COPY src ./src
COPY migrations ./migrations
COPY data ./data

RUN pip install --no-cache-dir . \
    && groupadd --gid 10001 app \
    && useradd --uid 10001 --gid app --home-dir /app --shell /usr/sbin/nologin app \
    && chown app:app /app /app/data

ENV MOCK_MODE=true
USER 10001
ENTRYPOINT ["python", "-m", "self_improving_outreach"]
CMD ["--help"]
