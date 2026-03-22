import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score

# Step 1: load your filtered lyrics dataset
df = pd.read_csv("filtered_lyrics_top20_pop.csv")

# Make sure needed columns exist
df = df.dropna(subset=["lyrics", "artist"])

print("Dataset shape:", df.shape)
print("\nArtists:")
print(df["artist"].value_counts())

# Step 2: define input/output
X = df["lyrics"]
y = df["artist"]

# Step 3: split into train/dev
X_train, X_dev, y_train, y_dev = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print("\nTrain size:", len(X_train))
print("Dev size:", len(X_dev))

# -----------------------------
# MODEL 1: Unigram Bag of Words
# -----------------------------
vectorizer_uni = CountVectorizer()
X_train_uni = vectorizer_uni.fit_transform(X_train)
X_dev_uni = vectorizer_uni.transform(X_dev)

model_uni = LogisticRegression(max_iter=1000)
model_uni.fit(X_train_uni, y_train)

pred_uni = model_uni.predict(X_dev_uni)

print("\nMODEL 1: UNIGRAM BOW")
print("Accuracy:", accuracy_score(y_dev, pred_uni))
print(classification_report(y_dev, pred_uni))

# -----------------------------
# MODEL 2: TF-IDF
# -----------------------------
vectorizer_tfidf = TfidfVectorizer()
X_train_tfidf = vectorizer_tfidf.fit_transform(X_train)
X_dev_tfidf = vectorizer_tfidf.transform(X_dev)

model_tfidf = LogisticRegression(max_iter=1000)
model_tfidf.fit(X_train_tfidf, y_train)

pred_tfidf = model_tfidf.predict(X_dev_tfidf)

print("\nMODEL 2: TF-IDF")
print("Accuracy:", accuracy_score(y_dev, pred_tfidf))
print(classification_report(y_dev, pred_tfidf))

# -----------------------------
# Example input/output
# -----------------------------
sample_idx = 0
print("\nSAMPLE INPUT:")
print(X_dev.iloc[sample_idx][:500])

print("\nTRUE ARTIST:")
print(y_dev.iloc[sample_idx])

print("\nPREDICTED ARTIST (TF-IDF):")
print(pred_tfidf[sample_idx])
