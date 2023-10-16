# syntax=docker/dockerfile:1

FROM python:3.8-slim-buster

WORKDIR /python-docker3

COPY requirements_service.txt requirements.txt

RUN pip install -r requirements.txt --default-timeout=3600

COPY . .

RUN python -c 'from transformers import AutoModelForSequenceClassification,AutoTokenizer; \
    model = AutoModelForSequenceClassification.from_pretrained("microsoft/codebert-base", num_labels=10); \
    tokenizer = AutoTokenizer.from_pretrained("microsoft/codebert-base"); \
    model.save_pretrained("data/models/codebert");tokenizer.save_pretrained("data/models/codebert");'


CMD [ "python", "-m" , "flask", "run", "--host=0.0.0.0"]
