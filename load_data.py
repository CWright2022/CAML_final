# loads dataset from kagglehub
import kagglehub
import pandas as pd
import shutil
import os

DATASET_PATH = "./dataset/malicious_phish.csv"
MAXIMUM_URL_LENGTH = 500

def clean_dataset(df: pd.DataFrame) -> pd.DataFrame:
    """Ensure URLs are formed correctly and remove duplicates"""
    df_clean = df.copy()
    df_clean = df_clean[df_clean['url'].str.match(r'^[\x00-\x7F]+$', na=False)]  # only ascii characters allowed
    df_clean = df_clean[df_clean['url'].str.len() < MAXIMUM_URL_LENGTH]  # only less than a max length
    df_clean = df_clean.drop_duplicates(subset=['url'])  # remove duplicates
    return df_clean

def load_dataset() -> pd.DataFrame:
    if os.path.exists(DATASET_PATH):
        print("Datset exists in folder. Skipping download.")
    else:
        # Download latest version
        path = kagglehub.dataset_download("sid321axn/malicious-urls-dataset")
        print("Downloaded dataset to:", path)
        shutil.move(path+"/malicious_phish.csv", DATASET_PATH)
    data_frame = pd.read_csv(DATASET_PATH)
    return data_frame

def describe_dataset(df: pd.DataFrame) -> None:
    print("Dataset Description:")
    print(df.describe())