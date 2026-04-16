# WHO= Saung and Jay
#   container for dataloader class
#       load data
#       do split of testing/training (Saung)
#       clean data (Jay)

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder    

class dataloaderLANG():
    def __init__(self, trainsplit=0.7):
        # load dataset
        df = pd.read_csv("filtered_lyrics_top20_pop.csv")
        # filter
        df = df[["lyrics", "artist"]].dropna()

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

if __name__ == "__main__":
    data = dataloaderLANG()

    print("Train size:", len(data.TrainX))
    print("Validation size:", len(data.ValX))
    print("Test size:", len(data.TestX))