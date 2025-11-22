# loads dataset from kagglehub
import kagglehub
import pandas as pd

def load_dataset() -> pd.DataFrame:
    # Download latest version
    path = kagglehub.dataset_download("advaitnmenon/network-traffic-data-malicious-activity-detection")
    print("Downloaded dataset to:", path)
    data_frame = pd.read_csv(path+"/NandakumarMenonAdvait_MT_S2.csv")
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