import re
import pandas as pd
import numpy as np
from urllib.parse import urlparse
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter
from collections import Counter

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

def count_directory_levels(url) -> int:
    if not url.startswith(('http://', 'https://')):
        url = 'http://' + url 
    parsed = urlparse(url)
    path = parsed.path.strip('/')
    if not path:
        return 0
    return len(path.split('/'))


def benign_malicious_histograms(df: pd.DataFrame) -> None:
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


def tld_counts_histograms(df: pd.DataFrame) -> None:
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


def url_length_histograms(df: pd.DataFrame) -> None:
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


def special_character_histograms(df: pd.DataFrame) -> None:
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


def subdomain_count_histograms(df: pd.DataFrame) -> None:
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
    this_ax.set_title('Benign URLs')
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
    this_ax.set_title('Malicious URLs')
    this_ax.set_xlabel('Number of subdomains')
    this_ax.set_ylabel('Relative Frequency')
    this_ax.set_xticks(bins, bins, rotation=45, ha='right', rotation_mode='anchor', size=8)
    this_ax.xaxis.set_major_formatter(FormatStrFormatter('%0d'))


def directory_level_count_histograms(df: pd.DataFrame) -> None:
    fig, ax = plt.subplots(nrows=1, ncols=2)
    fig.set_figheight(8)
    fig.set_figwidth(16)
    fig.subplots_adjust(bottom=0.25, hspace=0.8, wspace=0.6)
    fig.suptitle(f'Directory Levels Counts')

    this_ax = ax[0]
    subdomain_counts = df[df['type'] == 'benign']['url'].apply(count_directory_levels).to_list()
    counts, bins = this_ax.hist(subdomain_counts, bins=np.linspace(0,12,13), density=False)[:2]
    rel_freq = counts / counts.sum()
    this_ax.cla()
    this_ax.bar(bins[:-1], rel_freq, width=np.diff(bins), align='edge')
    this_ax.set_ylim(0, 0.85)
    this_ax.set_title('Benign URLs')
    this_ax.set_xlabel('Number of subdomains')
    this_ax.set_ylabel('Relative Frequency')
    this_ax.set_xticks(bins, bins, rotation=45, ha='right', rotation_mode='anchor', size=8)
    this_ax.xaxis.set_major_formatter(FormatStrFormatter('%0d'))

    this_ax = ax[1]
    subdomain_counts = df[df['type'] != 'benign']['url'].apply(count_directory_levels).to_list()
    counts, bins = this_ax.hist(subdomain_counts, bins=np.linspace(0,12,13), density=False)[:2]
    rel_freq = counts / counts.sum()
    this_ax.cla()
    this_ax.bar(bins[:-1], rel_freq, width=np.diff(bins), align='edge')
    this_ax.set_ylim(0, 0.85)
    this_ax.set_title('Malicious URLs')
    this_ax.set_xlabel('Number of subdomains')
    this_ax.set_ylabel('Relative Frequency')
    this_ax.set_xticks(bins, bins, rotation=45, ha='right', rotation_mode='anchor', size=8)
    this_ax.xaxis.set_major_formatter(FormatStrFormatter('%0d'))


def do_statistics(df: pd.DataFrame) -> None:
    benign_malicious_histograms(df)
    tld_counts_histograms(df)
    url_length_histograms(df)
    special_character_histograms(df)
    subdomain_count_histograms(df)
    directory_level_count_histograms(df)

    print('Charts should be displayed now...')
    print('Close the plots to continue.')
    plt.show()
