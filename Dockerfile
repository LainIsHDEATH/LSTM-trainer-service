FROM pytorch/pytorch:2.2.2-cuda12.1-cudnn8-runtime

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY app/ ./app
COPY models/ ./models

EXPOSE 7001

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "7001"]
