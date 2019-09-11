import re
import os
import glob
import pickle
import logging

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import accuracy_score
from sklearn.svm import LinearSVC
from sklearn.model_selection import train_test_split

from nltk.corpus import stopwords


DATA_PATH = './data/rated_bash'
MODEL_FILE = 'phrase_classifier.pickle'
VECTORIZER_FILE = 'vectorizer.pk'
REPLACE_NO_SPACE = re.compile("(\.)|(\;)|(\:)|(\!)|(\')|(\?)|(\,)|(\")|(\()|(\))|(\[)|(\])")
REPLACE_WITH_SPACE = re.compile("(<br\s*/><br\s*/>)|(\-)|(\/)|(<)|(>)")
MIN_FUN_RATING = 2700


logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger('Classifier')
stop_words = stopwords.words('russian')
stop_words = stop_words + [';)', 'xxx', 'yyy', 'в', 'и', 'с', 'а', ':-)', 'на', 'по', 'меня', 'мне', 'то', 'когда',
                           'то', 'сегодня', 'ты', 'вот', 'только', 'это', 'если', 'тут', 'тебя', 'ну', 'так', 'не',
                           'что', 'да', 'есть', 'теперь', 'вчера', 'тебе', 'он', 'там', 'она', 'всё', 'ещё', 'те'
                           'дтп', 'ДТП', 'погиб']


def preprocess_txt(txt) -> str:
    txt = REPLACE_NO_SPACE.sub("", txt.lower())
    txt = REPLACE_WITH_SPACE.sub(" ", txt)
    return txt.strip()


def parse_csv_line(line: str) -> (str, int):
    line_splitted = line.split(';')
    if len(line_splitted) != 2:
        raise ParseError()
    txt, label = line_splitted
    label = label.strip()
    if not label.isdigit():
        raise ParseError()
    label = int(label)
    txt = preprocess_txt(txt)

    return txt, label


class ParseError(Exception):
    pass


class Classifier:

    labels = []
    quotes_train_clean = []
    classifier = None
    vectorizer = None
    best_accuracy = None
    rated_csv_files: {str, } = set()

    def __init__(
            self,
            data_path: str,
            stop_words: [str],
            show_stats: bool = False,
    ) -> None:
        self.data_path = data_path
        self.model_file = f'{data_path}/{MODEL_FILE}'
        self.vectorizer_file = f'{data_path}/{VECTORIZER_FILE}'
        self.show_stats = show_stats
        self.stop_words = stop_words

    def add_rated_csv_file(self, file_name: str):
        self.rated_csv_files.add(file_name)

    def _print(self, text: str) -> None:
        if self.show_stats:
            logger.info(text)

    def _load_csv_data(self, csv_files: [], yes_no_file=False):
        for csv_file in csv_files:
            with open(csv_file, encoding='utf-8') as cf:
                for line in cf:
                    try:
                        txt, label = parse_csv_line(line)
                    except ParseError:
                        continue

                    if not yes_no_file:
                        label = 1 if label > MIN_FUN_RATING else 0

                    self.quotes_train_clean.append(txt)
                    self.labels.append(label)

    def _load_train_data(self) -> None:
        csv_file_path_pattern = f'{self.data_path}/*.csv'
        csv_files = glob.glob(csv_file_path_pattern)
        self._load_csv_data(csv_files)
        if self.rated_csv_files:
            self._load_csv_data(self.rated_csv_files, yes_no_file=True)

    def train(self):
        if not self.quotes_train_clean:
            self._load_train_data()
        self.vectorizer = CountVectorizer(binary=True, ngram_range=(3, 10), stop_words=stop_words)
        self.vectorizer.fit(self.quotes_train_clean)

        x = self.vectorizer.transform(self.quotes_train_clean)
        x_train, x_val, y_train, y_val = train_test_split(x, self.labels, train_size=0.75)

        max_accuracy = 0
        best_step = 0
        for c in [0.01, 0.05, 0.25, 0.5, 1]:
            svm = LinearSVC(C=c)
            svm.fit(x_train, y_train)
            accuracy = accuracy_score(y_val, svm.predict(x_val))
            if accuracy > max_accuracy:
                max_accuracy = accuracy
                best_step = c
            self.best_accuracy = accuracy
            self._print("Accuracy for C=%s: %s" % (c, accuracy))

        self._print(f'Best accuracy: {max_accuracy}')

        self.classifier = LinearSVC(C=best_step)
        self.classifier.fit(x, self.labels)

    def cleanup(self):
        if os.path.isfile(self.model_file):
            os.remove(self.model_file)
        if os.path.isfile(self.vectorizer_file):
            os.remove(self.vectorizer_file)
        self.quotes_train_clean = []
        self.labels = []

    def get_train_stats(self) -> str:
        feature_to_coef = {
            word: coef for word, coef in zip(self.vectorizer.get_feature_names(), self.classifier.coef_[0])
        }
        result = " Top positive: \n"
        for best_positive in sorted(feature_to_coef.items(), key=lambda x: x[1], reverse=True)[:5]:
            result += f"{best_positive[0]}: {best_positive[1]} \n"

        result += "\n Top negative \n"
        for best_negative in sorted(feature_to_coef.items(), key=lambda x: x[1])[:5]:
            result += f"{best_negative[0]}: {best_negative[1]} \n"

        return result

    def show_train_stats(self):
        stats = self.get_train_stats()
        logger.info(stats)

    def save_model(self):
        if not self.classifier:
            raise Exception('Run train first')

        with open(self.model_file, 'wb') as f:
            pickle.dump(self.classifier, f)

        with open(self.vectorizer_file, 'wb') as f:
            pickle.dump(self.vectorizer, f)

    def load_model(self):
        if not os.path.isfile(self.model_file) or not os.path.isfile(self.vectorizer_file):
            return False
        with open(self.model_file, 'rb') as f:
            self.classifier = pickle.load(f)

        with open(self.vectorizer_file, 'rb') as f:
            self.vectorizer = pickle.load(f)

        return True

    def check_phrases(self, phrases: [str]) -> [int]:
        phrases_clean = [preprocess_txt(t) for t in phrases]
        x_test = self.vectorizer.transform(phrases_clean)
        predictions = self.classifier.predict(x_test)
        return predictions


classifier = Classifier(DATA_PATH, stop_words, show_stats=True)
