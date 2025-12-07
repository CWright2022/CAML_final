# loads dataset from kagglehub
import kagglehub
import pandas as pd
import shutil
import os
from typing import Callable

DATASET_PATH = "./dataset/malicious_phish.csv"
MAXIMUM_URL_LENGTH = 500

def clean_dataset(df: pd.DataFrame) -> pd.DataFrame:
    """Ensure URLs are formed correctly and remove duplicates"""
    df_clean = df.copy()
    df_clean = df_clean[df_clean['url'].str.match(r'^[\x00-\x7F]+$', na=False)]  # only ascii characters allowed
    df_clean = df_clean[df_clean['url'].str.len() < MAXIMUM_URL_LENGTH]  # only less than a max length
    df_clean = df_clean.drop_duplicates(subset=['url'])  # remove duplicates
    return df_clean

def balance_dataset(df: pd.DataFrame, class_distinguisher: Callable[[str], int], class_size) -> pd.DataFrame | None:
    """
    Balance the dataframe so it has equal numbers of each class
    Retains random values from each class
    """
    df = df.copy()
    df['class'] = df['type'].apply(class_distinguisher)  # helper column

    class_labels = sorted(df['class'].unique())

    for cls in class_labels:
        cls_rows = df[df['class'] == cls]
        if len(cls_rows) < class_size:
            return None  # one of the classes does not have enough samples
    
    balanced_cls_rows = []
    for cls in class_labels:
        cls_rows = df[df['class'] == cls]
        sampled = cls_rows.sample(class_size, replace=False, random_state=42)
        balanced_cls_rows.append(sampled)
    
    # balanced_cls_rows is a list of DataFrames. Each df contains class_size rows of each class
    result = pd.concat(balanced_cls_rows).sample(frac=1, random_state=42).reset_index(drop=True)

    # remove the helper column before returning it
    result = result.drop(columns=['class'])
    return result

def remove_class(df: pd.DataFrame, class_to_remove: str) -> pd.DataFrame:
    """Removes all rows of the givne type from thte data frame"""
    df = df.copy()
    return df[df['type'] != class_to_remove]

def load_dataset() -> pd.DataFrame:
    """Loads samples from the dataset. Skips download if the user has already downloaded it"""
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
    """Print a description of the dataset"""
    print("Dataset Description:")
    print(df.describe())