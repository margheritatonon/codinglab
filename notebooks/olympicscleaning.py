import pandas as pd
import re


def load_and_preprocess(path):
    df = pd.read_csv(path)

    #filling null values:
    df["medal"].fillna("No", inplace=True)

    col = df["edition"].str.split(" ").str[1]
    df["season"] = col

    col2 = df["edition"].str.split(" ").str[0]
    df["year"] = col2

    df.drop(columns = "pos", inplace = True)

    df["gender"] = df["event"].str.split(",").str[-1]
    
    #merging the boys and men
    #genderdict = {"Women":"Female", "Men":"Male", "Boys":"Male"}
    #df["gender"] = df["gender"].map(genderdict)

    return df