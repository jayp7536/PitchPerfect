# WHO = Lyndsey
# Combined word n-grams + character n-grams TF-IDF model for artist classification from lyrics

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score
from scipy.sparse import hstack
from dataloader import dataloaderLANG
from metrics import evaluate_model, print_conf_matrix

#note: can you add an option for n? ie: 2-gram 3-gram etc  -jonah
#   also include args for epochs, trainingtestingsplit

def train_combined_model(n=2, epoch=1000, datasplit=0.7):
    data = dataloaderLANG(datasplit)
    X_train, y_train = data.TrainX, data.TrainY
    X_val, y_val = data.ValX, data.ValY

    # -----------------------------
    # VECTORIZER 1: Word N-Grams
    # -----------------------------
    # word n-grams depend on n
    word_vectorizer = TfidfVectorizer(
        max_features=10000,
        analyzer="word",
        ngram_range=(1, n)
    )
    X_train_word = word_vectorizer.fit_transform(X_train)
    X_val_word = word_vectorizer.transform(X_val)

    # -----------------------------
    # VECTORIZER 2: Character N-Grams
    # -----------------------------
    # keep char n-grams fixed
    char_vectorizer = TfidfVectorizer(
        max_features=10000,
        analyzer="char",
        ngram_range=(2, 4)
    )
    X_train_char = char_vectorizer.fit_transform(X_train)
    X_val_char = char_vectorizer.transform(X_val)

    # -----------------------------
    # COMBINE BOTH
    # -----------------------------
    X_train_combined = hstack([X_train_word, X_train_char])
    X_val_combined = hstack([X_val_word, X_val_char])

    model = LogisticRegression(max_iter=epoch)
    model.fit(X_train_combined, y_train)

    pred = model.predict(X_val_combined)

    # print("MODEL: COMBINED WORD + CHARACTER N-GRAMS")
    # print("Accuracy:", accuracy_score(y_val, pred))
    # print(classification_report(y_val, pred, target_names=data.encoder.classes_))
    
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
    model_name="MODEL: COMBINED WORD + CHARACTER N-GRAMS"
    )
    
    return model, word_vectorizer, char_vectorizer

if __name__ == "__main__":
    train_combined_model()
