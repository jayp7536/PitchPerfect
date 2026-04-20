# WHO= Saung and Jay
#   container for dataloader class
#       load data
#       do split of testing/training (Saung)
#       clean data (Jay)

import pandas as pd
import re
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder    





class dataloaderLANG():
    def __init__(self, trainsplit=0.7):
        # load dataset
        df = pd.read_csv("filtered_lyrics_top20_pop.csv")

        # filter to english
        lang_col = None
        for c in df.columns:
            if c.lower().startswith("language"):
                lang_col = c
                break

        if lang_col is not None:
            df = df[df[lang_col].str.lower() == "en"]

        df = df[["lyrics", "artist"]].dropna()

        # remove duplicates
        df = df.drop_duplicates(subset=["lyrics", "artist"])

        # clean lyrics
        df["lyrics"] = df["lyrics"].apply(self.clean_text)

        # (100 songs per artist)
        df = df.groupby("artist").filter(lambda x: len(x) >= 100)
        df = df.groupby("artist").sample(n=100, random_state=42)

        # split data
        X = df["lyrics"]
        y = df["artist"]

        # encode labels
        self.encoder = LabelEncoder()
        y_encoded = self.encoder.fit_transform(y)

        # train(70%)/val(15%)/test(15%) split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y_encoded, test_size=1-trainsplit, random_state=42, stratify=y_encoded
        )
        X_val, X_test, y_val, y_test = train_test_split(
            X_test, y_test, test_size=0.5, random_state=42, stratify=y_test
        )

        # save splits
        self.TrainX=X_train
        self.TrainY=y_train
        self.ValX=X_val     
        self.ValY=y_val
        self.TestX=X_test
        self.TestY=y_test





    def clean_text(self, text):
        text = text.lower()

        #remove bracketed text
        text = re.sub(r"\[.*?\]", "", text)
        #remove newlines
        text = text.replace("\n", " ")
        #remove punctuation 
        text = re.sub(r"[^\w\s]", "", text)
        # remove extra spaces
        text = re.sub(r"\s+", " ", text).strip()

        return text





if __name__ == "__main__":
    data = dataloaderLANG()

    print("#------------------------------------------------- DATASET INFO #-------------------------------------------------")
    print("Train size:", len(data.TrainX))
    print("Validation size:", len(data.ValX))
    print("Test size:", len(data.TestX))

    print("\n#------------------------------------------------- SAMPLE INPUT #-------------------------------------------------")
    print(list(data.TrainX)[0])

    print("\n#------------------------------------------------- LABEL INFO #-------------------------------------------------")
    print("Encoded label:", data.TrainY[0])
    print("Actual artist:", data.encoder.inverse_transform([data.TrainY[0]])[0])

    print("\n#-------------------------------------------------#")