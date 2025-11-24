# loads dataset from kagglehub
import kagglehub
import pandas as pd
import shutil
import os

DATASET_PATH = "./dataset/malicious_phish.csv"

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