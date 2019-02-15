from bash_org_sentiment_analyser.classifier import classifier

if __name__ == '__main__':
    if not classifier.load_model():
        classifier.train()
        classifier.show_train_stats()
        classifier.save_model()

    test_labels = []
    test_phrases = [
    ]

    predictions = classifier.check_phrases(test_phrases)
    print('\n Predictions:')
    prediction_errors = 0
    for i, p in enumerate(predictions):
        if p != test_labels[i]:
            print(f'Error in {test_phrases[i]} real: {test_labels[i]} predicted: {p}')
            prediction_errors += 1
    print(f'Prediction errors: {prediction_errors}')