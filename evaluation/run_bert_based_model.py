import torch
from torch.utils.data import DataLoader
from transformers import AutoModelForSequenceClassification, AdamW, AutoTokenizer
import os
import time
from tqdm import tqdm
from data_processor_pt import SentenceDataset
import numpy as np
from sklearn.metrics import f1_score, roc_auc_score, accuracy_score
import logging
import json


logging.basicConfig(format='%(asctime)s - %(levelname)s - %(filename)s  - %(message)s', level=logging.INFO)

device = 'cpu'

with open(os.path.join(os.path.dirname(__file__), "..", "data", 'tags_counter_list.json'), "r") as openfile:
    tags_counter_list = json.load(openfile)

label2id = {tags_counter_list[i]: i for i in range(len(tags_counter_list))}
id2label = {j: i for i, j in label2id.items()}


def multi_label_metrics(predictions, labels, threshold=0.5):
    # first, apply sigmoid on predictions which are of shape (batch_size, num_labels)
    sigmoid = torch.nn.Sigmoid()
    probs = sigmoid(torch.Tensor(predictions))

    # next, use threshold to turn them into integer predictions
    y_pred = np.zeros(probs.shape)
    y_pred[np.where(probs >= threshold)] = 1
    # finally, compute metrics
    y_true = labels
    f1_micro_average = f1_score(y_true=y_true, y_pred=y_pred, average='micro', zero_division=0)
    f1_macro_average = f1_score(y_true=y_true, y_pred=y_pred, average='macro', zero_division=0)

    roc_auc = roc_auc_score(y_true, y_pred, average='micro')
    accuracy = accuracy_score(y_true, y_pred)
    metrics = {'f1-micro': f1_micro_average,
               'f1-macro': f1_macro_average,
               'roc_auc': roc_auc,
               'accuracy': accuracy}
    return metrics


def train(model, train_loader, val_loader, optimizer, num_epochs):

    for epoch in range(num_epochs):
        start = time.time()
        model.train()
        total_train_loss = 0
        total_train_f1_micro = 0
        total_train_f1_macro = 0
        total_train_roc_auc = 0
        total_train_accuracy = 0

        for batch_idx, sample in tqdm(
                enumerate(train_loader), desc="Epoch {}/{} Processing batches".format(epoch, num_epochs),
                total=len(train_loader)):

            input_ids = sample.input_ids.to(device)
            attention_mask = sample.attention_mask.to(device)
            labels = sample.labels.to(device)
            optimizer.zero_grad()

            loss, logits, = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels
            ).values()
            metrics = multi_label_metrics(logits, labels)

            loss.backward()
            optimizer.step()

            total_train_loss += loss.item()
            total_train_f1_micro += metrics['f1-micro']
            total_train_f1_macro += metrics['f1-macro']
            total_train_roc_auc += metrics['roc_auc']
            total_train_accuracy += metrics['accuracy']

        train_f1_micro = total_train_f1_micro / len(train_loader)
        train_f1_macro = total_train_f1_macro / len(train_loader)
        train_roc_auc = total_train_roc_auc / len(train_loader)
        train_accuracy = total_train_accuracy / len(train_loader)
        train_loss = total_train_loss / len(train_loader)


        model.eval()
        total_val_f1_micro = 0
        total_val_f1_macro = 0
        total_val_roc_auc = 0
        total_val_accuracy = 0
        total_val_loss = 0
        with torch.no_grad():
            for batch_idx, sample in enumerate(val_loader):
                input_ids = sample.input_ids.to(device)
                attention_mask = sample.attention_mask.to(device)
                labels = sample.labels.to(device)
                optimizer.zero_grad()

                loss, logits, = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels).values()

                metrics = multi_label_metrics(logits, labels)

                total_val_loss += loss.item()
                total_val_f1_micro += metrics['f1-micro']
                total_val_f1_macro += metrics['f1-macro']
                total_val_roc_auc += metrics['roc_auc']
                total_val_accuracy += metrics['accuracy']

        val_f1_micro = total_val_f1_micro / len(val_loader)
        val_f1_macro = total_val_f1_macro / len(val_loader)
        val_roc_auc = total_val_roc_auc / len(val_loader)
        val_accuracy = total_val_accuracy / len(val_loader)
        val_loss = total_val_loss / len(val_loader)
        end = time.time()
        hours, rem = divmod(end - start, 3600)
        minutes, seconds = divmod(rem, 60)


        message = f'Epoch {epoch + 1}: train_loss: {train_loss:.4f} ' \
                  f'train_acc: {train_accuracy:.4f} ' \
                  f'train_f1_micro: {train_f1_micro:.4f} '\
                  f'train_f1_macro: {train_f1_macro:.4f} ' \
                  f'train_roc_auc: {train_roc_auc:.4f} ' \
                  f'| val_loss: {val_loss:.4f} ' \
                  f'val_acc: {val_accuracy:.4f} ' \
                  f'val_f1_micro: {val_f1_micro:.4f} '\
                  f'val_f1_macro: {val_f1_macro:.4f} ' \
                  f'val_roc_auc: {val_roc_auc:.4f} ' \
                  f'total epoch time: {"{:0>2}:{:0>2}:{:05.2f}".format(int(hours), int(minutes), seconds)}'
        logging.info(message)

    return model


def get_datasets(
        train_split, val_split, batch_size_train, batch_size_val, max_train_samples, max_val_samples, tokenizer):

    logging.info("Loading Train Dataset")

    train_dataset = SentenceDataset(
        split=train_split,
        tokenizer=tokenizer,
        labels_given=True,
        shuffle_data=True,
        max_samples=max_train_samples
    )

    logging.info("Loading Validation Dataset")

    train_loader = DataLoader(
        train_dataset, batch_size=batch_size_train, collate_fn=train_dataset.collate_fn
    )

    val_dataset = SentenceDataset(
        split=val_split,
        tokenizer=tokenizer,
        labels_given=True,
        shuffle_data=True,
        max_samples=max_val_samples
    )

    val_loader = DataLoader(
        val_dataset, batch_size=batch_size_val, collate_fn=val_dataset.collate_fn
    )

    return train_loader, val_loader


def main(
        train_split, val_split, num_epochs, model_max_length, batch_size_train, batch_size_val,
        max_train_samples, max_val_samples, output_name
    ):

    config = {
        'num_epochs': num_epochs,
        'model_max_length': model_max_length,
        'batch_size_train': batch_size_train,
        "model_filename": "model.pth",
        "label2id": label2id
    }

    model = AutoModelForSequenceClassification.from_pretrained(
        os.path.join(os.path.dirname(__file__), "..", "data", "models", "codebert"),
        local_files_only=True, num_labels=len(id2label),
        problem_type="multi_label_classification"
    )

    tokenizer = AutoTokenizer.from_pretrained(
        os.path.join(os.path.dirname(__file__), "..", "data", "models", "codebert"),
        local_files_only=True
    )
    tokenizer.model_max_length = model_max_length

    param_optimizer = list(model.named_parameters())
    no_decay = ['bias', 'gamma', 'beta']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)],
         'weight_decay_rate': 0.01},
        {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)],
         'weight_decay_rate': 0.0}
    ]
    optimizer = AdamW(optimizer_grouped_parameters, lr=2e-5, correct_bias=True)

    train_loader, val_loader = get_datasets(
        train_split, val_split, batch_size_train, batch_size_val, max_train_samples, max_val_samples, tokenizer
    )

    logging.info("Training")

    model = train(model, train_loader, val_loader, optimizer, num_epochs)

    store_utils(
        model, output_name, config
    )


def store_utils(model, output_name, config):

    model_results_path = os.path.join(
        os.path.dirname(__file__), "..", "data", "results", output_name
    )

    os.mkdir(model_results_path)

    torch.save(
        model.state_dict(),
        os.path.join(model_results_path, config['model_filename'])
    )

    json.dump(
        config, open(
            os.path.join(model_results_path, 'config.json'), "w"),
        indent=2
    )

    logging.info("Model utils stored")


if __name__ == '__main__':

    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_split', required=True)
    parser.add_argument('--val_split', required=True, help='validation data split')
    parser.add_argument('--num_epochs', required=True)
    parser.add_argument('--model_max_length', required=True, help='max number of tokens')
    parser.add_argument('--batch_size_train', required=True)
    parser.add_argument('--batch_size_val', required=True)
    parser.add_argument('--max_train_samples', default=None)
    parser.add_argument('--max_val_samples', default=None)
    parser.add_argument('--output_name', help='output folder name to store')

    args = parser.parse_args()

    max_train_samples = args.max_train_samples
    max_val_samples = args.max_val_samples

    if max_train_samples:
        max_train_samples = int(max_train_samples)
    if max_val_samples:
        max_val_samples = int(max_val_samples)

    main(
        train_split=args.train_split,
        val_split=args.val_split,
        num_epochs=int(args.num_epochs),
        model_max_length=int(args.model_max_length),
        batch_size_train=int(args.batch_size_train),
        batch_size_val=int(args.batch_size_val),
        max_train_samples=max_train_samples,
        max_val_samples=max_val_samples,
        output_name=args.output_name
    )
