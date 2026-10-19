FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PORT=8000

WORKDIR /app

COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

COPY sop_agent ./sop_agent
COPY apps ./apps
COPY ui ./ui
COPY scripts ./scripts

# Emails the agent "sends" are written here (mounted as a volume in docker-compose)
RUN mkdir -p /app/data/outbox

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=10s \
  CMD python -c "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://127.0.0.1:8000/api/health').status == 200 else 1)"

CMD ["uvicorn", "sop_agent.api.app:app", "--host", "0.0.0.0", "--port", "8000"]
