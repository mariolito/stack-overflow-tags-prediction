export PYTHONPATH=.

python -c 'from transformers import AutoModelForSequenceClassification,AutoTokenizer;
model = AutoModelForSequenceClassification.from_pretrained("microsoft/codebert-base", num_labels=10);
tokenizer = AutoTokenizer.from_pretrained("microsoft/codebert-base");
model.save_pretrained("data/models/codebert");tokenizer.save_pretrained("data/models/codebert");'
