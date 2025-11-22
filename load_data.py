# loads dataset from kagglehub
import kagglehub
import pandas as pd
import shutil
import os

def load_dataset() -> pd.DataFrame:
    if os.path.exists("./dataset/NandakumarMenonAdvait_MT_S2.csv"):
        print("Datset exists in folder. Skipping download.")
    else:
        # Download latest version
        path = kagglehub.dataset_download("advaitnmenon/network-traffic-data-malicious-activity-detection")
        print("Downloaded dataset to:", path)
        shutil.move(path+"/NandakumarMenonAdvait_MT_S2.csv", "./dataset/NandakumarMenonAdvait_MT_S2.csv")
    data_frame = pd.read_csv("./dataset/NandakumarMenonAdvait_MT_S2.csv")
    return data_frame

def describe_dataset(df: pd.DataFrame) -> None:
    print("Dataset Description:")
    print(df.describe())
    print("rows x columns:")
    print(df.shape)
    print("preview:")
    print(df.head())
    print("class overview:")
    print(df["bad_packet"].value_counts())