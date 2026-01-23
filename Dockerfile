FROM python:3.11-slim

COPY --from=ghcr.io/astral-sh/uv:0.9.22 /uv /uvx /bin/

COPY . /app

ENV UV_NO_DEV=1

WORKDIR /app
RUN uv sync --frozen

EXPOSE 8000

CMD ["uv", "run", "--", "fastapi", "run", "--host", "0.0.0.0", "--port", "8000", "main.py"]