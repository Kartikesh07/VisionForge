FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1 \
    VIRTUAL_ENV=/opt/venv PATH=/opt/venv/bin:$PATH \
    VISIONFORGE_DEVICE=cpu MPLBACKEND=Agg OMP_NUM_THREADS=2

RUN apt-get update && apt-get install -y --no-install-recommends libgl1 libglib2.0-0 libgomp1 \
    && rm -rf /var/lib/apt/lists/* \
    && python -m venv /opt/venv
COPY deployment/requirements.txt /tmp/requirements.txt
RUN /opt/venv/bin/python -m pip install --no-cache-dir torch==2.14.1 torchvision==0.29.1 --index-url https://download.pytorch.org/whl/cpu \
    && /opt/venv/bin/python -m pip install --no-cache-dir -r /tmp/requirements.txt

RUN useradd --create-home --uid 1000 appuser
WORKDIR /app
COPY --chown=appuser:appuser app.py ./
COPY --chown=appuser:appuser src ./src
COPY --chown=appuser:appuser .streamlit ./.streamlit
COPY --chown=appuser:appuser deployment ./deployment
RUN mkdir -p /app/artifacts/models && chown -R appuser:appuser /app/artifacts
USER appuser
EXPOSE 7860
HEALTHCHECK --interval=30s --timeout=5s --start-period=120s \
    CMD ["/opt/venv/bin/python", "-c", "import urllib.request; urllib.request.urlopen('http://127.0.0.1:7860/_stcore/health', timeout=4)"]
CMD ["/opt/venv/bin/python", "-m", "deployment.start"]
