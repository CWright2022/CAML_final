# CAML FINAL
# Zach Riback, Cayden Wright, Ariana Ciaschini, Brett Huber

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report, confusion_matrix
import load_data

def main() -> None:
    data_frame = load_data.load_dataset()
    load_data.describe_dataset(data_frame)
    

if __name__ == "__main__":
    main()