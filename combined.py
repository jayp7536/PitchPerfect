# WHO = Lyndsey
# Combined word n-grams + character n-grams TF-IDF model for artist classification from lyrics

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score
from scipy.sparse import hstack
from dataloader import dataloaderLANG

def train_combined_model():
    # Load data
    data = dataloaderLANG()
    X_train, y_train = data.TrainX, data.TrainY
    X_val, y_val = data.ValX, data.ValY

    # -----------------------------
    # VECTORIZER 1: Word N-Grams
    # -----------------------------
    word_vectorizer = TfidfVectorizer(max_features=10000, analyzer='word', ngram_range=(1, 2))
    X_train_word = word_vectorizer.fit_transform(X_train)
    X_val_word = word_vectorizer.transform(X_val)

    # -----------------------------
    # VECTORIZER 2: Character N-Grams
    # -----------------------------
    char_vectorizer = TfidfVectorizer(max_features=10000, analyzer='char', ngram_range=(2, 4))
    X_train_char = char_vectorizer.fit_transform(X_train)
    X_val_char = char_vectorizer.transform(X_val)

    # -----------------------------
    # COMBINE BOTH
    # -----------------------------
    X_train_combined = hstack([X_train_word, X_train_char])
    X_val_combined = hstack([X_val_word, X_val_char])

    model = LogisticRegression(max_iter=1000)
    model.fit(X_train_combined, y_train)

    pred = model.predict(X_val_combined)

    print("MODEL: COMBINED WORD + CHARACTER N-GRAMS")
    print("Accuracy:", accuracy_score(y_val, pred))
    print(classification_report(y_val, pred, target_names=data.encoder.classes_))

    return model, word_vectorizer, char_vectorizer

if __name__ == "__main__":
    train_combined_model()
