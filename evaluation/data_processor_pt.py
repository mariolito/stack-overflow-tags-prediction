from collections import namedtuple
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence
import os
import pandas as pd
import torch
import ast
import json
from collections import Counter
from sklearn.utils import shuffle
import logging
logging.basicConfig(format='%(asctime)s - %(levelname)s - %(filename)s  - %(message)s', level=logging.INFO)


class SentenceDataset(Dataset):
    def __init__(self, split, tokenizer, shuffle_data=False,
                 labels_given=False, max_samples=None):

        self.split = split
        if labels_given:
            self.dataset_sample_fields = "Id", "sentences", "input_ids", "attention_mask", "labels"
        else:
            self.dataset_sample_fields = "Id", "sentences", "input_ids", "attention_mask"

        self.SentenceDatasetSample = namedtuple("SentenceDatasetSample", ("id", *self.dataset_sample_fields))
        self.SentenceDatasetBatch = namedtuple(
            "SentenceDatasetBatch", ("id", "sample_lengths", *self.dataset_sample_fields)
        )
        with open(os.path.join(os.path.dirname(__file__), "..", "data", 'tags_counter_list.json'), "r") as openfile:
            self.tags_counter_list = json.load(openfile)

        self.label2id = {self.tags_counter_list[i]: i for i in range(len(self.tags_counter_list))}
        self.id2label = {j: i for i, j in self.label2id.items()}
        self.tokenizer = tokenizer
        self.data_df = self._read_data(split, shuffle_data, max_samples)
        self.samples = self.parse_samples()

    @staticmethod
    def _read_data(split, shuffle_data, max_samples):
        df = pd.read_csv(
            os.path.join(os.path.dirname(__file__), "..", "data", '{}.csv'.format(split)
                         ),
            encoding='latin-1').rename(columns={'text_with_code': 'sentences', 'Tag': 'labels'})[
            ['Id', 'sentences', 'labels']]
        df["labels"] = df["labels"].apply(ast.literal_eval)
        df['labels_len'] = df["labels"].apply(len)
        all_labels = []
        for i in range(len(df)):
            all_labels += df.iloc[i]['labels']
        logging.info("Labels distribution: {}".format(str(dict(Counter(all_labels)))))
        logging.info("Total labels per example distribution: {}".format(str(dict(Counter(df['labels_len'].values)))))

        if shuffle_data:
            df = shuffle(df)
        if max_samples:
            df = df.sample(max_samples)
        return df

    def parse_samples(self):
        data_list = list(self.data_df.to_dict(orient='index').values())
        data_dict = {}
        c = 0
        for sample in data_list:
            data_dict[c] = {}
            for k, v in sample.items():
                if k == 'sentences':
                    data_dict[c]['input_ids'] = self.tokenizer.encode(v, return_tensors="pt", truncation=True)[0]
                    data_dict[c]['attention_mask'] = torch.tensor([1] * len(data_dict[c]['input_ids']))
                    data_dict[c]['sentences'] = v
                elif k == 'labels':
                    data_dict[c]['labels'] = [1.0 if label in v else 0.0 for label in self.tags_counter_list] #self.label2id[v[0]]#
                else:
                    data_dict[c][k] = v
            c += 1
        return data_dict

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        """Get a sample from the dataset.
        Args:
            idx (int): Index of the sample in the full dataset.
        Returns:
            SentenceDatasetSample: Sample from the dataset.
        """
        data_dict = self.samples[idx]

        sample = self.SentenceDatasetSample(idx, *[data_dict.get(key) for key in self.dataset_sample_fields])

        return sample

    def collate_fn(self, data):
        """Create a batch of variable length tensors.
        Args:
            data (list): List of SentenceDatasetSample objects.
        Returns:
            SentenceDatasetBatch: Batch of samples.
        """
        batch = dict(
            id=[sample.id for sample in data],
            sample_lengths=torch.IntTensor([len(sample.input_ids) for sample in data])
        )

        # pad data such that all samples in the batch have the same length
        for key in self.dataset_sample_fields:
            if key == "labels":
                batch[key] = torch.tensor([getattr(sample, key) for sample in data])
            elif key in ("input_ids", "attention_mask"):
                batch[key] = pad_sequence(
                    [getattr(sample, key) for sample in data],
                    padding_value=self.tokenizer.pad_token_id,
                    batch_first=True
                )
            else:
                batch[key] = [getattr(sample, key) for sample in data]
        return self.SentenceDatasetBatch(**batch)

    @staticmethod
    def batch_to_device(self, batch, device):
        """Move batch to device.
        Args:
            batch (SentenceDatasetBatch): Batch of samples.
            device (torch.device): Device to move the batch to.
        Returns:
            SentenceDatasetBatch: Batch of samples on the device.
        """
        device_tensors = [getattr(batch, key).to(device) for key in self.dataset_batch_fields]
        return self.SentenceDatasetBatch(batch.id, *device_tensors)
