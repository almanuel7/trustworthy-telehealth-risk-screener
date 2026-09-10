FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Download model & tokenizer explicitly into a fixed folder
RUN python3 -c "\
from transformers import AutoTokenizer, AutoModelForSequenceClassification; \
model_name = 'cardiffnlp/twitter-roberta-base-sentiment-latest'; \
tokenizer = AutoTokenizer.from_pretrained(model_name); \
model = AutoModelForSequenceClassification.from_pretrained(model_name); \
tokenizer.save_pretrained('/app/sentiment_model'); \
model.save_pretrained('/app/sentiment_model')"

ENV TRANSFORMERS_OFFLINE=1
ENV HF_HUB_OFFLINE=1

COPY app.py index.html ./

ENV PORT=8080
EXPOSE 8080

CMD ["python3", "app.py"]