FROM python:3.11-slim

# set working dir
WORKDIR /app

# install system deps for OpenCV and pdf2image
RUN apt-get update && apt-get install -y \
    libgl1 libglib2.0-0 poppler-utils tesseract-ocr && \
    rm -rf /var/lib/apt/lists/*

# copy and install
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# copy code
COPY . .

# expose default HF port
EXPOSE 7860

# start FastAPI
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "7860"]