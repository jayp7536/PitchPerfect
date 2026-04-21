# WHO = Lyndsey
# TF-IDF with character n-grams model for artist classification from lyrics

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score
from dataloader import dataloaderLANG
from metrics import evaluate_model, print_conf_matrix

#note: can you add an option for n? ie: 2-gram 3-gram etc  -jonah
#   also include args for epochs, trainingtestingsplit
def train_tfidf_model():
    # Load data
    data = dataloaderLANG()
    X_train, y_train = data.TrainX, data.TrainY
    X_val, y_val = data.ValX, data.ValY

    # -----------------------------
    # MODEL: TF-IDF Character N-Grams
    # -----------------------------
    vectorizer = TfidfVectorizer(max_features=10000, analyzer='char', ngram_range=(2, 4))
    X_train_tfidf = vectorizer.fit_transform(X_train)
    X_val_tfidf = vectorizer.transform(X_val)

    model = LogisticRegression(max_iter=1000)
    model.fit(X_train_tfidf, y_train)

    pred = model.predict(X_val_tfidf)

    print("MODEL: TF-IDF CHARACTER N-GRAMS")
    print("Accuracy:", accuracy_score(y_val, pred))
    print(classification_report(y_val, pred, target_names=data.encoder.classes_))

    # -----------------------------
    # PRINT CONFUSION MATRIX IN TERMINAL
    # -----------------------------
    # print_conf_matrix(
    # y_val,
    # pred,
    # class_names=data.encoder.classes_
    # )

    # -----------------------------
    # SHOWS CONFUSION MATRIX IN NEW WINDOW (VISUAL)
    # -----------------------------
    evaluate_model(
    y_val,
    pred,
    class_names=data.encoder.classes_,
    model_name="TF-IDF CHARACTER N-GRAMS"
    )
    
    return model, vectorizer

if __name__ == "__main__":
    train_tfidf_model()
