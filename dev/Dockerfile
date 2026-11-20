FROM python:3.11-slim
ENV PYTHONUNBUFFERED=1 PYTHONIOENCODING=utf-8 LANGSMITH_TRACING=false OMP_NUM_THREADS=1
WORKDIR /app
COPY requirements-harness.txt ./
RUN pip install --no-cache-dir -r requirements-harness.txt
COPY harness_core ./harness_core
COPY wrappers ./wrappers
COPY run.py ./run.py
VOLUME ["/app/data", "/app/runs"]
ENTRYPOINT ["python", "run.py"]
CMD ["run", "--strategy", "rules"]
