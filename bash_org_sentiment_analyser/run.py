import os

from bash_org_sentiment_analyser.classifier import Classifier

if __name__ == '__main__':
    data_path = os.environ.get('DATA_PATH')
    if not data_path:
        raise Exception("Please setup environment var DATA_PATH")

    classifier = Classifier(data_path, show_stats=True)
    if not classifier.load_model():
        classifier.train()
        classifier.show_train_stats()
        classifier.save_model()

    test_labels = [1, 1, 0, 0]
    test_phrases = [
        "Американская демократия распространяется воздушно-ракетным путем",
        "пытался сделать девушке массаж, а получилось фаталити",
        "Ты кто по гироскопу?",
        "Судя по отступам, Маяковский писал на питоне",
    ]

    predictions = classifier.check_phrases(test_phrases)
    print('\n Predictions:')
    prediction_errors = 0
    for i, p in enumerate(predictions):
        if p != test_labels[i]:
            print(f'Error in {test_phrases[i]} real: {test_labels[i]} predicted: {p}')
            prediction_errors += 1
    print(f'Prediction errors: {prediction_errors}')