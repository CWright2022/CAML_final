# CAML FINAL
# Zach Riback, Cayden Wright, Ariana Ciaschini, Brett Huber

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


def main() -> None:
    parser = argparse.ArgumentParser('Argument parser to help with only running certain parts of the code')
    parser.add_argument('--statistics', '-s', action='store_true', help='Loads data and does statistics')
    parser.add_argument('--feature-extraction', '-f', action='store_true', help='Loads data and extracts features')

    args = parser.parse_args()

    if args.statistics:
        df = load_data.load_dataset()
        df = load_data.clean_dataset(df)
        data_statistics.do_statistics(df)
    elif args.feature_extraction:
        df = load_data.load_dataset()
        df = load_data.clean_dataset(df)

        # get rid of all www subdomains
        df['url'] = df['url'].replace('www.', '', regex=True)

        feature_extraction.do_feature_extraction(df)
    else:
        print('Did not do anything :(')


if __name__ == "__main__":
    main()