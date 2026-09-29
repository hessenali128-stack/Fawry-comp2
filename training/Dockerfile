FROM python:3.11-slim

RUN apt-get update \
    && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY training/requirements.txt /app/training/requirements.txt
RUN pip install --no-cache-dir -r /app/training/requirements.txt

COPY notebook /app/notebook
COPY training /app/training

RUN mkdir -p /app/artifacts /app/dataset

CMD ["python", "training/train.py"]