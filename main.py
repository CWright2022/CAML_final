# CAML FINAL
# Zach Riback, Cayden Wright, Ariana Ciaschini, Brett Huber

import re
import pandas as pd
import numpy as np
from urllib.parse import urlparse
import matplotlib.pyplot as plt
from collections import Counter
import load_data
import feature_extraction
from matplotlib.ticker import FormatStrFormatter

tld_pattern = re.compile(r'\.([a-zA-Z0-9-]+)(?=[/:\?]|$)')

def get_tld(url: str) -> str:
    """Extract TLD from a URL. Returns an empty string if it cannot find one"""
    m = tld_pattern.search(url)
    if m is None or m.group(1).isdigit():
        return ''
    return m.group(1)

def count_special_chars(url) -> int:
    normal_chars = 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ1234567890'
    return sum(1 for c in url if c not in normal_chars)

def count_subdomains(url):
    if not url.startswith(('http://', 'https://')):
        url = 'http://' + url
    
    parsed = urlparse(url)
    host = parsed.hostname

    if host is None:
        return 0
    
    parts = host.split('.')

    if len(parts) <= 2:
        return 0
    return len(parts) - 2


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
    print(f'{"Total malicious":.<20}{malicious_count:.>7}')
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

    
    # URL Length Stuff
    fig, ax = plt.subplots(nrows=1, ncols=2)
    fig.set_figheight(8)
    fig.set_figwidth(16)
    fig.subplots_adjust(bottom=0.25, hspace=0.8, wspace=0.6)
    fig.suptitle(f'URL Lengths')

    # First plot (benign URLs)
    this_ax = ax[0]
    url_lengths = df[df['type'] == 'benign']['url'].apply(len).to_list()
    counts, bins = this_ax.hist(url_lengths, bins=np.linspace(0, 500, 26), density=False)[:2]
    rel_freq = counts / counts.sum()
    this_ax.cla()
    this_ax.bar(bins[:-1], rel_freq, width=np.diff(bins), align='edge')
    this_ax.set_ylim(0, 0.35)
    this_ax.set_title('Benign URL Lengths')
    this_ax.set_xlabel('Length')
    this_ax.set_ylabel('Relative Frequency')
    this_ax.set_xticks(bins, bins, rotation=45, ha='right', rotation_mode='anchor', size=8)
    this_ax.xaxis.set_major_formatter(FormatStrFormatter('%0d'))

    # Second plot (malicious URLs)
    this_ax = ax[1]
    url_lengths = df[df['type'] != 'benign']['url'].apply(len).to_list()
    counts, bins = this_ax.hist(url_lengths, bins=np.linspace(0, 500, 26), density=False)[:2]
    rel_freq = counts / counts.sum()
    this_ax.cla()
    this_ax.bar(bins[:-1], rel_freq, width=np.diff(bins), align='center')
    this_ax.set_ylim(0, 0.35)
    this_ax.set_title('Malicious URL Lengths')
    this_ax.set_xlabel('Length')
    this_ax.set_ylabel('Relative Frequency')
    this_ax.set_xticks(bins, bins, rotation=45, ha='right', rotation_mode='anchor', size=8)
    this_ax.xaxis.set_major_formatter(FormatStrFormatter('%0d'))

    # Gonna look at special characters
    fig, ax = plt.subplots(nrows=1, ncols=2)
    fig.set_figheight(8)
    fig.set_figwidth(16)
    fig.subplots_adjust(bottom=0.25, hspace=0.8, wspace=0.6)
    fig.suptitle(f'Special Character Counts')
    
    # First plot (benign special counts)
    this_ax = ax[0]
    special_counts = df[df['type'] == 'benign']['url'].apply(count_special_chars).to_list()
    counts, bins = this_ax.hist(special_counts, bins=np.linspace(0, 100, 11), density=False)[:2]
    rel_freq = counts / counts.sum()
    this_ax.cla()
    this_ax.bar(bins[:-1], rel_freq, width=np.diff(bins), align='center')
    this_ax.set_title('Benign URLs')
    this_ax.set_xlabel('Length')
    this_ax.set_ylabel('Relative Frequency')
    this_ax.set_xticks(bins, bins, rotation=45, ha='right', rotation_mode='anchor', size=8)
    this_ax.xaxis.set_major_formatter(FormatStrFormatter('%0d'))

    # Second plot (malicious special counts)
    this_ax = ax[1]
    special_counts = df[df['type'] != 'benign']['url'].apply(count_special_chars).to_list()
    counts, bins = this_ax.hist(special_counts, bins=np.linspace(0, 100, 11), density=False)[:2]
    rel_freq = counts / counts.sum()
    this_ax.cla()
    this_ax.bar(bins[:-1], rel_freq, width=np.diff(bins), align='edge')
    this_ax.set_title('Malicious URLs')
    this_ax.set_xlabel('Length')
    this_ax.set_ylabel('Relative Frequency')
    this_ax.set_xticks(bins, bins, rotation=45, ha='right', rotation_mode='anchor', size=8)
    this_ax.xaxis.set_major_formatter(FormatStrFormatter('%0d'))

    # Sub domain stuff
    fig, ax = plt.subplots(nrows=1, ncols=2)
    fig.set_figheight(8)
    fig.set_figwidth(16)
    fig.subplots_adjust(bottom=0.25, hspace=0.8, wspace=0.6)
    fig.suptitle(f'Subdomain Counts')

    this_ax = ax[0]
    subdomain_counts = df[df['type'] == 'benign']['url'].apply(count_subdomains).to_list()
    counts, bins = this_ax.hist(subdomain_counts, bins=np.linspace(0,10,11), density=False)[:2]
    rel_freq = counts / counts.sum()
    this_ax.cla()
    this_ax.bar(bins[:-1], rel_freq, width=np.diff(bins), align='edge')
    this_ax.set_ylim(0, 0.75)
    this_ax.set_title('Benign Subdomains')
    this_ax.set_xlabel('Number of subdomains')
    this_ax.set_ylabel('Relative Frequency')
    this_ax.set_xticks(bins, bins, rotation=45, ha='right', rotation_mode='anchor', size=8)
    this_ax.xaxis.set_major_formatter(FormatStrFormatter('%0d'))

    this_ax = ax[1]
    subdomain_counts = df[df['type'] != 'benign']['url'].apply(count_subdomains).to_list()
    counts, bins = this_ax.hist(subdomain_counts, bins=np.linspace(0,10,11), density=False)[:2]
    rel_freq = counts / counts.sum()
    this_ax.cla()
    this_ax.bar(bins[:-1], rel_freq, width=np.diff(bins), align='edge')
    this_ax.set_ylim(0, 0.75)
    this_ax.set_title('Malicious Subdomains')
    this_ax.set_xlabel('Number of subdomains')
    this_ax.set_ylabel('Relative Frequency')
    this_ax.set_xticks(bins, bins, rotation=45, ha='right', rotation_mode='anchor', size=8)
    this_ax.xaxis.set_major_formatter(FormatStrFormatter('%0d'))





    print('Charts should be displayed now...')
    print('Close the plots to continue.')
    plt.show()


def main() -> None:
    df = load_data.load_dataset()
    df = load_data.clean_dataset(df)
    # load_data.describe_dataset(df)

    print('\nStatistics!')
    # do_statistics(df)
    # get rid of all www subdomains
    df['url'] = df['url'].replace('www.', '', regex=True)
    
    # do feature extraction
    feature_extraction.do_feature_extraction(df)
    print(df.head())


if __name__ == "__main__":
    main()