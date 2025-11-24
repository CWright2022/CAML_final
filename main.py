# CAML FINAL
# Zach Riback, Cayden Wright, Ariana Ciaschini, Brett Huber

import re
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter
from collections import Counter
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report, confusion_matrix
import load_data

tld_pattern = re.compile(r'\.([a-zA-Z0-9-]+)(?=[/:\?]|$)')

def get_tld(url: str) -> str:
    """Extract TLD from a URL. Returns an empty string if it cannot find one"""
    m = tld_pattern.search(url)
    if m is None or m.group(1).isdigit():
        return ''
    return m.group(1)


def do_statistics(df: pd.DataFrame) -> None:
    urls = df['url'].to_numpy(dtype=np.str_)
    types = df['type'].to_numpy(dtype=np.str_)
    
    labels, counts = np.unique_counts(types)
    counts_dict = dict(zip(labels, counts))
    malicious_count = sum([counts_dict[k] for k in counts_dict.keys() if k != 'benign'])
    
    # Text analysis
    print('URL count by label:')
    for label, count in zip(labels, counts):
        print(f'{label:.<20}{count:.>7}')
    print(f'{'Total malicious':.<20}{malicious_count:.>7}')
    print()

    fig, ax = plt.subplots(nrows=1, ncols=2)
    fig.set_figheight(6)
    fig.set_figwidth(12)
    fig.subplots_adjust(bottom=0.25, hspace=0.8, wspace=0.6)
    fig.suptitle('URL Label Counts')

    # First plot
    # Benign vs. Malicious
    this_ax = ax[0]
    this_labels = ['benign', 'malicious']
    this_counts = [counts_dict['benign'], malicious_count]
    this_ax.bar(this_labels, this_counts)
    this_ax.set_title('Benign vs. Malicious')
    this_ax.set_xlabel('URL Label')
    this_ax.set_ylabel('Count')

    # Second plot
    # Malicious URL types vs. each other
    this_ax = ax[1]
    this_labels = [k for k in counts_dict.keys() if k != 'benign']
    this_counts = [counts_dict[k] for k in counts_dict.keys() if k != 'benign']
    this_ax.bar(this_labels, this_counts)
    this_ax.set_title('Malicious URLs')
    this_ax.set_xlabel('URL Label')
    this_ax.set_ylabel('Count')

    # Work with TLDs
    benign_tlds = df[df['type'] == 'benign']['url'].apply(get_tld).to_list()
    malicious_tlds = df[df['type'] != 'benign']['url'].apply(get_tld).to_list()
    all_tlds = benign_tlds + malicious_tlds
    all_tlds_set = set(benign_tlds) | set(malicious_tlds)
    
    benign_tld_counts = Counter(benign_tlds)
    malicious_tld_counts = Counter(malicious_tlds)
    all_tld_counts = Counter(all_tlds)

    tld_count_requirement = 200

    tld_benign_proportion_dict = dict() 
    for tld in all_tlds_set:
        if all_tld_counts[tld] < tld_count_requirement:  # only consider TLDs with more appearences
            continue
        benign_tld_count = benign_tld_counts.get(tld, 0)
        malicious_tld_count = malicious_tld_counts.get(tld, 0)
        tld_benign_proportion_dict[tld] = benign_tld_count / (benign_tld_count + malicious_tld_count)
    
    sorted_benign_proportion_items = sorted(tld_benign_proportion_dict.items(), key = lambda x: x[1])
    bottom20 = sorted_benign_proportion_items[:20]
    top20 = sorted_benign_proportion_items[-20:]

    b20_labels, b20_vals = zip(*bottom20)
    t20_labels, t20_vals = zip(*top20)

    fig, ax = plt.subplots(nrows=1, ncols=2)
    fig.set_figheight(8)
    fig.set_figwidth(16)
    fig.subplots_adjust(bottom=0.25, hspace=0.8, wspace=0.6)
    fig.suptitle(f'TLD Benign Proportions (>{tld_count_requirement} appearences)')

    # First plot (top 20)
    this_ax = ax[0]
    this_ax.bar(t20_labels, t20_vals)
    this_ax.set_title('Top 20')
    this_ax.set_xlabel('TLD')
    this_ax.set_ylabel('Proportion')
    this_ax.set_xticks(this_ax.get_xticks(), this_ax.get_xticklabels(), rotation=45, ha='right', rotation_mode='anchor', size=8)

    # Second plot (bottom 20)
    this_ax = ax[1]
    this_ax.bar(b20_labels, b20_vals)
    this_ax.set_title('Bottom 20')
    this_ax.set_xlabel('TLD')
    this_ax.set_ylabel('Proportion')
    this_ax.set_xticks(this_ax.get_xticks(), this_ax.get_xticklabels(), rotation=45, ha='right', rotation_mode='anchor', size=8)

    print('Charts should be displayed now...')
    print('Close the plots to continue.')
    plt.show()


def main() -> None:
    df = load_data.load_dataset()
    df = load_data.clean_dataset(df)
    # load_data.describe_dataset(df)

    print('\nStatistics!')
    do_statistics(df)


if __name__ == "__main__":
    main()