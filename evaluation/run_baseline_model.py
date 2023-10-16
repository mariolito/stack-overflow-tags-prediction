import pandas as pd
import ast
from sklearn.model_selection import train_test_split
from sklearn.linear_model import RidgeClassifier
from sklearn.multioutput import MultiOutputClassifier
import logging
from sklearn.preprocessing import MultiLabelBinarizer
import os
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, roc_auc_score, accuracy_score
logging.basicConfig(format='%(asctime)s - %(levelname)s - %(filename)s  - %(message)s', level=logging.INFO)

device = 'cpu'


def _read_data():
    df = pd.read_csv(os.path.join(os.path.dirname(__file__), "..", "data", 'data.csv'),  encoding='latin-1')
    return df


def _preprocess(df, tfidf):
    df["Tag"] = df["Tag"].apply(ast.literal_eval)
    multilabel_binarizer = MultiLabelBinarizer()
    y_bin = multilabel_binarizer.fit_transform(df['Tag'])
    if tfidf:
        vectorizer = TfidfVectorizer(analyzer='word',
                                     strip_accents=None,
                                     encoding='utf-8',
                                     preprocessor=None,
                                     token_pattern=r"(?u)\S\S+")
    else:
        vectorizer = CountVectorizer(analyzer='word',
                                     strip_accents=None,
                                     encoding='utf-8',
                                     preprocessor=None,
                                     token_pattern=r"(?u)\S\S+")
    X = vectorizer.fit_transform(df['text_clean'])
    X_train, X_test, y_train, y_test = train_test_split(X, y_bin, test_size=0.2, random_state=0)
    return X_train, X_test, y_train, y_test


def multi_label_metrics(y_pred, y_true):
    f1_micro_average = f1_score(y_true=y_true, y_pred=y_pred, average='micro', zero_division=0)
    f1_macro_average = f1_score(y_true=y_true, y_pred=y_pred, average='macro', zero_division=0)

    roc_auc = roc_auc_score(y_true, y_pred, average='micro')
    accuracy = accuracy_score(y_true, y_pred)
    metrics = {'f1-micro': f1_micro_average,
               'f1-macro': f1_macro_average,
               'roc_auc': roc_auc,
               'accuracy': accuracy}
    return metrics


def _evaluate(X_train, X_test, y_train, y_test, classifier):

    if classifier == 'logistic':
        clf = MultiOutputClassifier(LogisticRegression())
    elif classifier == 'ridge':
        clf = MultiOutputClassifier(RidgeClassifier())
    clf.fit(X_train, y_train)
    y_test_pred = clf.predict(X_test)
    metrics = multi_label_metrics(y_test_pred, y_test)
    from pprint import pprint
    pprint(metrics)
    return clf


def main(classifier, tfidf):
    df = _read_data()
    logging.info("Data Preprocessing")
    X_train, X_test, y_train, y_test = _preprocess(df, tfidf=tfidf)
    logging.info("Train & Evaluate")
    clf = _evaluate(X_train, X_test, y_train, y_test, classifier)
    _store_utils(clf)


def _store_utils(clf):
    import pickle
    logging.info("Data stored")
    pickle.dump(
        clf,
        open(
            os.path.join(os.path.dirname(__file__), "..", "data", "results", "baseline", 'clf.p'), "wb"))


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--tfidf', action='store_true', default=False)
    parser.add_argument('--classifier', help='type of classifier. You may use logistic or ridge')
    args = parser.parse_args()
    main(
        classifier=args.classifier,
        tfidf=args.tfidf
    )