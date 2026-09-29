FROM python:3.11-slim
WORKDIR /app
COPY production/requirements.txt /app/production/requirements.txt
RUN pip install --no-cache-dir -r /app/production/requirements.txt
COPY production /app/production
COPY ui /app/ui
COPY artifacts /app/artifacts
COPY data /app/data
ENV PYTHONPATH=/app/production:/app:/app/ui
EXPOSE 7860
CMD ["uvicorn", "production.app:app", "--host", "0.0.0.0", "--port", "7860"]
