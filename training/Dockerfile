FROM python:3.11-slim
WORKDIR /app
COPY training/requirements.txt /app/training/requirements.txt
RUN pip install --no-cache-dir -r /app/training/requirements.txt
COPY notebook /app/notebook
COPY training /app/training
RUN mkdir -p /app/artifacts /app/dataset
CMD ["python", "training/train.py"]
