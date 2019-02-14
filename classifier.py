import re
import os
import pickle

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.model_selection import train_test_split

from nltk.corpus import stopwords

stop_words = stopwords.words('russian')
DATA_FILE = './data/bash_logs.csv'
NEWS_FILE = './data/news.txt'
MODEL_FILE = './data/phrase_classifier.pickle'
VECTORIZER_FILE = './data/vectorizer.pk'
REPLACE_NO_SPACE = re.compile("(\.)|(\;)|(\:)|(\!)|(\')|(\?)|(\,)|(\")|(\()|(\))|(\[)|(\])")
REPLACE_WITH_SPACE = re.compile("(<br\s*/><br\s*/>)|(\-)|(\/)|(<)|(>)")
stop_words = stop_words + [';)', 'xxx', 'yyy', 'в', 'и', 'с', 'а', ':-)', 'на', 'по', 'меня', 'мне', 'то', 'когда',
                           'то', 'сегодня', 'ты', 'вот', 'только', 'это', 'если', 'тут', 'тебя', 'ну', 'так', 'не',
                           'что', 'да', 'есть', 'теперь', 'вчера', 'тебе', 'он', 'там', 'она', 'всё', 'ещё', 'те']


def preprocess_txt(txt):
    txt = REPLACE_NO_SPACE.sub("", txt.lower())
    txt = REPLACE_WITH_SPACE.sub(" ", txt)
    return txt.strip()


class Classifier:

    labels = []
    quotes_train_clean = []
    classifier = None
    vectorizer = None

    def __init__(
            self,
            data_file: str,
            news_file: str,
            model_file: str,
            vectorizer_file: str,
            stop_words: [str],
            show_stats: bool = False
    ) -> None:
        self.data_file = data_file
        self.news_file = news_file
        self.model_file = model_file
        self.vectorizer_file = vectorizer_file
        self.show_stats = show_stats
        self.stop_words = stop_words

    def _print(self, text: str) -> None:
        if self.show_stats:
            print(text)

    def _load_train_data(self) -> None:
        for line in open(self.data_file, 'r', encoding='utf-8'):
            txt, label = line.split(';')
            self.quotes_train_clean.append(preprocess_txt(txt))
            self.labels.append(int(label))

        for line in open(self.news_file, 'r', encoding='utf-8'):
            self.quotes_train_clean.append(preprocess_txt(line))
            self.labels.append(0)

    def train(self):
        if not self.quotes_train_clean:
            self._load_train_data()
        self.vectorizer = CountVectorizer(binary=True, ngram_range=(1, 10), stop_words=stop_words)
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
                self._print("Accuracy for C=%s: %s" % (c, accuracy))

        self._print(f'\n Best accuracy: {max_accuracy}')

        self.classifier = LogisticRegression(C=best_step)
        self.classifier.fit(x, self.labels)

    def show_train_stats(self):
        feature_to_coef = {
            word: coef for word, coef in zip(self.vectorizer.get_feature_names(), self.classifier.coef_[0])
        }

        print('\n Best positive:')
        for best_positive in sorted(
                feature_to_coef.items(),
                key=lambda x: x[1],
                reverse=True
        )[:5]: print(best_positive)

        print('\n Best negative:')
        for best_negative in sorted(
                feature_to_coef.items(),
                key=lambda x: x[1]
        )[:5]: print(best_negative)

    def save_model(self):
        if not self.classifier:
            raise Exception('Run train first')

        with open(self.model_file, 'wb') as f:
            pickle.dump(self.classifier, f)

        with open(self.vectorizer_file, 'wb') as f:
            pickle.dump(self.vectorizer, f)

    def load_model(self):
        if not os.path.isfile(self.model_file):
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


classifier = Classifier(DATA_FILE, NEWS_FILE, MODEL_FILE, VECTORIZER_FILE, stop_words, show_stats=True)
