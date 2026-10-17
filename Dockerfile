FROM python:3.11-slim
WORKDIR /app
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PYTHONPATH=/app/src
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY src ./src
COPY app ./app
COPY data ./data
COPY knowledge ./knowledge
COPY eval ./eval
# train inside the image so the container is self-contained and reproducible (seed 42)
RUN python -m churnlab.train --fast
RUN useradd -m appuser && chown -R appuser /app
USER appuser
EXPOSE 8000 8501
HEALTHCHECK CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')" || exit 1
CMD ["sh", "-c", "uvicorn churnlab.api:app --host 0.0.0.0 --port ${PORT:-8000}"]
