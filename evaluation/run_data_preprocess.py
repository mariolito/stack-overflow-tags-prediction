import os
import pandas as pd
from bs4 import BeautifulSoup
import json
from tqdm import tqdm
import re
from string import punctuation
from nltk.corpus import stopwords
import logging
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
import nltk
nltk.download('wordnet')
nltk.download('stopwords')
nltk.download('omw-1.4')
logging.basicConfig(format='%(asctime)s - %(levelname)s - %(filename)s  - %(message)s', level=logging.INFO)
stop = stopwords.words('english')
lemmatizer = WordNetLemmatizer()
tqdm.pandas()


class Preprocessor(object):

    def __init__(
            self, top_common_tags=300, max_num_questions=100000, min_question_score=2
    ):
        self.top_common_tags = top_common_tags
        self.max_num_questions = max_num_questions
        self.min_question_score = min_question_score

    def _read_data(self):

        questions = pd.read_csv(os.path.join(os.path.dirname(__file__), "..", "data", 'Questions.csv'),
                                encoding='latin-1')
        tags = pd.read_csv(os.path.join(os.path.dirname(__file__), "..", "data", 'Tags.csv'))
        return questions, tags

    def _process_tags(self, questions, tags):
        logging.info("Input Data contain {} questions".format(len(questions)))
        tags_counter = tags.groupby("Tag", as_index=False).agg(tag_count=("Tag", "count")).\
            sort_values('tag_count', ascending=False).reset_index(drop=True).iloc[:self.top_common_tags]
        tags_counter_list = list(tags_counter['Tag'].values)
        tags_counter_list.sort()
        tags = pd.merge(tags, tags_counter['Tag'], how='inner', on="Tag")
        tags = tags.groupby("Id")['Tag'].apply(list)
        data = pd.merge(tags, questions, how='inner', on="Id")
        data = data[data['Score'] >= self.min_question_score].reset_index(drop=True)
        logging.info("After taking top common tags, Data contain {} questions".format(len(data)))
        logging.info("We sample {} questions".format(self.max_num_questions))
        data = data.sample(self.max_num_questions).reset_index(drop=True)
        return data, tags_counter_list

    @staticmethod
    def _text_cleaning(text):

        # case normalization
        text = text.lower()

        # remove unicodes
        text = re.sub(r"(@\[A-Za-z0-9]+)|([^0-9A-Za-z \t])|(\w+:\/\/\S+)|^rt|http.+?", "", text)

        # remove punctuation
        text = re.sub(f"[{re.escape(punctuation)}]", "", text)  # Remove
        text = " ".join(text.split(" "))

        # Removing stopwords
        text = " ".join([word for word in text.split() if word not in (stop)])

        tokens = word_tokenize(text)
        text = " ".join([lemmatizer.lemmatize(token) for token in tokens])
        return text

    @staticmethod
    def _parse_html(html):
        soup = BeautifulSoup(html, features="html.parser")
        # remove all code parts
        code_parts = soup.find_all(['code', 'script'])
        text_with_code = soup.get_text()
        # break into lines and remove leading and trailing space on each
        lines = (line.strip() for line in text_with_code.splitlines())
        # break multi-headlines into a line each
        chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
        # drop blank lines
        text_with_code = ' '.join(chunk for chunk in chunks if chunk)
        for code in code_parts:
            html = html.replace(str(code), "")
        if code_parts:
            soup = BeautifulSoup(html, features="html.parser")
            # get text
            text_no_code = soup.get_text()
            # break into lines and remove leading and trailing space on each
            lines = (line.strip() for line in text_no_code.splitlines())
            # break multi-headlines into a line each
            chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
            # drop blank lines
            text_no_code = ' '.join(chunk for chunk in chunks if chunk)
        else:
            text_no_code = text_with_code
        return text_with_code, text_no_code

    def _store(self, data, tags_counter_list):
        data.to_csv(os.path.join(os.path.dirname(__file__), "..", "data", 'data.csv'), index=False)

        json_object = json.dumps(tags_counter_list, indent=2)

        # Writing to sample.json
        with open(os.path.join(os.path.dirname(__file__), "..", "data", 'tags_counter_list.json'), "w") as outfile:
            outfile.write(json_object)

        train_dataset = data.sample(frac=0.80, random_state=200)
        val_dataset = data.drop(train_dataset.index).sample(frac=1.00, random_state=200, ignore_index=True).copy()
        train_dataset = train_dataset.sample(frac=1.00, random_state=200, ignore_index=True).copy()
        train_dataset.  to_csv(os.path.join(os.path.dirname(__file__), "..", "data", 'train.csv'), index=False)
        val_dataset.to_csv(os.path.join(os.path.dirname(__file__), "..", "data", 'val.csv'), index=False)

    def _process_text(self, data):
        data[['Body_with_code', 'Body_no_code']] = data.apply(
            lambda x: self._parse_html(x['Body']), result_type='expand', axis=1
        )
        data['Body_clean_no_code'] = data['Body_no_code'].progress_apply(self._text_cleaning)
        data['Title_clean'] = data['Title'].progress_apply(self._text_cleaning)
        data['text_with_code'] = data.apply(lambda x: x['Title'] + ' ' + x['Body_with_code'], axis=1)
        data['text_clean'] = data.apply(lambda x: x['Title_clean'] + ' ' + x['Body_clean_no_code'], axis=1)
        return data

    def parse(self):
        logging.info("Preprocessing data")
        questions, tags = self._read_data()
        data, tags_counter_list = self._process_tags(questions, tags)
        data = self._process_text(data)

        self._store(data, tags_counter_list)
        logging.info("Data stored.")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--top_common_tags', help='Include only top k taggs for the rest of the analysis.')
    parser.add_argument('--max_num_questions', help='Sample max_num_questions if we end up for more.')
    parser.add_argument('--min_question_score', help='Minimum question score to take into account.')

    args = parser.parse_args()
    Preprocessor(
        top_common_tags=int(args.top_common_tags),
        max_num_questions=int(args.max_num_questions),
        min_question_score=int(args.min_question_score)
    ).parse()
