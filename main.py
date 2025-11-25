# CAML FINAL
# Zach Riback, Cayden Wright, Ariana Ciaschini, Brett Huber
import sys
import re
import pandas as pd
import numpy as np
from urllib.parse import urlparse
import matplotlib.pyplot as plt
import argparse
from collections import Counter
import load_data
import feature_extraction
import data_statistics
import nn


def main() -> None:
    parser = argparse.ArgumentParser('Argument parser to help with only running certain parts of the code')
    parser.add_argument('--statistics', '-s', action='store_true', help='Loads data and does statistics')
    parser.add_argument('--decision_tree', '-d', action='store_true', help='Loads data, extracts features, and trains a decision tree')
    parser.add_argument('--nueral_network', '-n', action='store_true', help='Loads data, extracts features, and trains a neural network')

    args = parser.parse_args()

    if args.statistics:
        df = load_data.load_dataset()
        df = load_data.clean_dataset(df)
        data_statistics.do_statistics(df)
    elif args.decision_tree:
        df = load_data.load_dataset()
        df = load_data.clean_dataset(df)

        # get rid of all www subdomains
        df['url'] = df['url'].replace('www.', '', regex=True)

        feature_extraction.do_feature_extraction_decision_tree(df)
    elif args.nueral_network:
        df = load_data.load_dataset()
        df = load_data.clean_dataset(df)

        # label=0 if this is a benign URL, label=1 if this is a malicious URL
        class_distinguisher = lambda x: 0 if x == 'benign' else 1

        df = load_data.balance_dataset(df, class_distinguisher, 200_000)
        if df is None:
            print('Could not balance the dataset! Exiting...')
            sys.exit()

        X, y = feature_extraction.do_feature_extraction_nn(df, class_distinguisher)

        print('Got X and y with the following shapes:')
        print(X.shape)
        print(y.shape)

        # Now train and test the model
        nn.evaluate_nn(X, y)

    else:
        print('Did not do anything :(')


if __name__ == "__main__":
    main()