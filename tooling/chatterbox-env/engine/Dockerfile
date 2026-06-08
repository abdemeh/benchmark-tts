FROM python:3.10-slim
ENV PYTHONUNBUFFERED=1

RUN apt-get update && apt-get install -y \
    git \
    libsndfile1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .

RUN pip3 install --no-cache-dir torch torchaudio --index-url https://download.pytorch.org/whl/cpu
RUN pip3 install --no-cache-dir fastapi uvicorn pydantic
RUN pip3 install --no-cache-dir git+https://github.com/resemble-ai/chatterbox.git

COPY main.py .

EXPOSE 8000

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]

