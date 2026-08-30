FROM python:3.11-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt psycopg2-binary
COPY . .
EXPOSE 8080
CMD ["uvicorn", "chip_eng:app", "--host", "0.0.0.0", "--port", "8080"]
