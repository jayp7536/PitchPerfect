# WHO = Lyndsey
# Unigram + Bigram Bag of Words model for artist classification from lyrics

import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score
from dataloader import dataloaderLANG
from metrics import evaluate_model, print_conf_matrix

#note: can you add an option for n? ie: 2-gram 3-gram etc  -jonah
#   also include args for epochs, trainingtestingsplit
def train_ngram_model(n=2, epoch=1000, datasplit=0.7):
    # Load data
    data = dataloaderLANG()
    X_train, y_train = data.TrainX, data.TrainY
    X_val, y_val = data.ValX, data.ValY

    # -----------------------------
    # MODEL: Unigram + Bigram BOW
    # -----------------------------
    vectorizer = CountVectorizer(max_features=10000, ngram_range=(1, n))
    X_train_ngram = vectorizer.fit_transform(X_train)
    X_val_ngram = vectorizer.transform(X_val)

    model = LogisticRegression(max_iter=epoch)
    model.fit(X_train_ngram, y_train)

    pred = model.predict(X_val_ngram)

    # print("MODEL: UNIGRAM + BIGRAM BOW")
    # print("Accuracy:", accuracy_score(y_val, pred))
    # print(classification_report(y_val, pred, target_names=data.encoder.classes_))

    # -----------------------------
    # PRINT CONFUSION MATRIX IN TERMINAL
    # -----------------------------
    print_conf_matrix(
    y_val,
    pred,
    class_names=data.encoder.classes_
    )

    # -----------------------------
    # SHOWS CONFUSION MATRIX IN NEW WINDOW (VISUAL)
    # -----------------------------
    # evaluate_model(
    # y_val,
    # pred,
    # class_names=data.encoder.classes_,
    # model_name="UNIGRAM + BIGRAM BOW"
    # )
    
    return model, vectorizer

if __name__ == "__main__":
    train_ngram_model()
