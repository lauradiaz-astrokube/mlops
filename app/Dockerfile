FROM python:3.9-slim

WORKDIR /app

COPY app.py .
COPY inference_test.py .

RUN pip install fastapi uvicorn scikit-learn joblib numpy mlflow

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
