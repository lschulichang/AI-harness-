FROM python:3.11-slim-bookworm

ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PYTHONIOENCODING=utf-8 \
    BASELINE_CPU_THREADS=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1

WORKDIR /app
COPY requirements.txt /opt/harness/requirements.txt
COPY sdk /opt/contest-harness-sdk
RUN python3 -m pip install --no-cache-dir -r /opt/harness/requirements.txt /opt/contest-harness-sdk

COPY harness.py contest_submission.py contest_data.py /app/
COPY harness_core /app/harness_core
COPY wrappers /app/wrappers
COPY baseline /app/baseline

USER 65534:65534
ENTRYPOINT ["python3", "/app/harness.py"]
