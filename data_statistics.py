import re
import pandas as pd
import numpy as np
from urllib.parse import urlparse, parse_qs
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter
from collections import Counter
from sklearn.metrics import confusion_matrix
import seaborn as sns

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
    """Return the number of subdomains of this URL"""
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
    """Return the directory depth of this URL"""
    if not url.startswith(('http://', 'https://')):
        url = 'http://' + url 
    parsed = urlparse(url)
    path = parsed.path.strip('/')
    if not path:
        return 0
    return len(path.split('/'))

def count_url_params(url) -> int:
    """Return the count of URL parameters of this URL"""
    if not url.startswith(('http://', 'https://')):
        url = 'http://' + url 
    parsed = urlparse(url)
    query = parsed.query
    if not query:
        return 0
    return len(parse_qs(query))

def get_url_entropy(url) -> float:
    """Return the Shannon entropy for this URL"""
    arr = np.frombuffer(url.encode('utf-8'), dtype=np.uint8)

    # Count occurrences of each unique byte
    values, counts = np.unique(arr, return_counts=True)

    p = counts / counts.sum()

    # Shannon entropy
    return float(-np.sum(p * np.log2(p)))

def domain_or_ip(url) -> int:
    """Returns 0 if it is an IP address, 1 if it is a domain name"""
    host = urlparse(url).hostname or ''
    parts = host.split('.')
    if len(parts) == 4 and all(p.isdigit() for p in parts):
        if all(0 <= int(p) <= 255 for p in parts):
            return 0
    return 1

def get_domain_entropy(url) -> float:
    """Return the Shannon entropy of the domain name for this URL"""
    host = urlparse(url).hostname
    if not host:
        return 0
    arr = np.frombuffer(host.encode("utf-8"), dtype=np.uint8)

    values, counts = np.unique(arr, return_counts=True)
    p = counts / counts.sum()

    return float(-np.sum(p * np.log2(p)))

def plot_confusion_matrix(y_true, y_pred, class_names):
    """Plot and display a confusion matrix given these predictions and targets"""
    cm = confusion_matrix(y_true, y_pred, normalize='true')

    plt.figure(figsize=(10,8))
    sns.heatmap(cm, annot=True, fmt='.2f', cmap='Blues', 
                xticklabels=class_names, yticklabels=class_names,
                annot_kws={'size':12},
                cbar=False)
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()

    print('Confusion matrix should be displayed now.')
    print('Close the charts to continue...')

    plt.show()


def benign_malicious_histograms(df: pd.DataFrame) -> None:
    """Create a histogram of benign, malicious, and phishing, malware, and defacement URL counts"""
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
    """Create a histogram showing the top 20 and bottom 20 by benign proportion of top level domain names"""
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
    """Create a histogram of URL lengths"""
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
    """Create a histogram of number of special characters in the URLs"""
    fig, ax = plt.subplots(nrows=1, ncols=2)
    fig.set_figheight(8)
    fig.set_figwidth(16)
    fig.subplots_adjust(bottom=0.25, hspace=0.8, wspace=0.6)
    fig.suptitle(f'Special Character Counts')

    bins = np.linspace(0, 100, 11)

    # --- Compute data first so we can align the Y axis ---
    benign_counts = df[df['type'] == 'benign']['url'] \
        .apply(count_special_chars).to_list()
    malicious_counts = df[df['type'] != 'benign']['url'] \
        .apply(count_special_chars).to_list()

    # Histogram + relative frequency (no plotting yet)
    benign_hist, _ = np.histogram(benign_counts, bins=bins)
    malicious_hist, _ = np.histogram(malicious_counts, bins=bins)

    benign_rel = benign_hist / benign_hist.sum()
    malicious_rel = malicious_hist / malicious_hist.sum()

    # Shared Y-axis limit
    max_y = max(benign_rel.max(), malicious_rel.max()) * 1.1

    # --- First plot (Benign) ---
    this_ax = ax[0]
    this_ax.bar(bins[:-1], benign_rel, width=np.diff(bins), align='center')
    this_ax.set_title('Benign URLs')
    this_ax.set_xlabel('Length')
    this_ax.set_ylabel('Relative Frequency')
    this_ax.set_xticks(bins, bins, rotation=45, ha='right',
                       rotation_mode='anchor', size=8)
    this_ax.xaxis.set_major_formatter(FormatStrFormatter('%0d'))
    this_ax.set_ylim(0, max_y)

    # --- Second plot (Malicious) ---
    this_ax = ax[1]
    this_ax.bar(bins[:-1], malicious_rel, width=np.diff(bins), align='edge')
    this_ax.set_title('Malicious URLs')
    this_ax.set_xlabel('Length')
    this_ax.set_ylabel('Relative Frequency')
    this_ax.set_xticks(bins, bins, rotation=45, ha='right',
                       rotation_mode='anchor', size=8)
    this_ax.xaxis.set_major_formatter(FormatStrFormatter('%0d'))
    this_ax.set_ylim(0, max_y)



def subdomain_count_histograms(df: pd.DataFrame) -> None:
    """Create a histogram of the number of subdomains of the URLs"""
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
    """Create a histogram of the directory depth of the URLs"""
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


def url_param_count_histograms(df: pd.DataFrame) -> None:
    """Create a histogram of the number of URL parameters of the URLs"""
    fig, ax = plt.subplots(nrows=1, ncols=2)
    fig.set_figheight(8)
    fig.set_figwidth(16)
    fig.subplots_adjust(bottom=0.25, hspace=0.8, wspace=0.6)
    fig.suptitle(f'URL Parameter Counts Counts')

    this_ax = ax[0]
    subdomain_counts = df[df['type'] == 'benign']['url'].apply(count_url_params).to_list()
    counts, bins = this_ax.hist(subdomain_counts, bins=np.linspace(0,10,11), density=False)[:2]
    rel_freq = counts / counts.sum()
    this_ax.cla()
    this_ax.bar(bins[:-1], rel_freq, width=np.diff(bins), align='edge')
    this_ax.set_ylim(0, 0.9)
    this_ax.set_title('Benign URLs')
    this_ax.set_xlabel('Number of URL Parameters')
    this_ax.set_ylabel('Relative Frequency')
    this_ax.set_xticks(bins, bins, rotation=45, ha='right', rotation_mode='anchor', size=8)
    this_ax.xaxis.set_major_formatter(FormatStrFormatter('%0d'))

    this_ax = ax[1]
    subdomain_counts = df[df['type'] != 'benign']['url'].apply(count_url_params).to_list()
    counts, bins = this_ax.hist(subdomain_counts, bins=np.linspace(0,10,11), density=False)[:2]
    rel_freq = counts / counts.sum()
    this_ax.cla()
    this_ax.bar(bins[:-1], rel_freq, width=np.diff(bins), align='edge')
    this_ax.set_ylim(0, 0.9)
    this_ax.set_title('Malicious URLs')
    this_ax.set_xlabel('Number of URL Parameters')
    this_ax.set_ylabel('Relative Frequency')
    this_ax.set_xticks(bins, bins, rotation=45, ha='right', rotation_mode='anchor', size=8)
    this_ax.xaxis.set_major_formatter(FormatStrFormatter('%0d'))


def url_entropy_histogram(df: pd.DataFrame) -> None:
    """Create a bar chart of the entropy of benign/malicious URLs"""
    fig, ax = plt.subplots(nrows=1, ncols=1)
    fig.set_figheight(8)
    fig.set_figwidth(16)
    fig.subplots_adjust(bottom=0.25, hspace=0.8, wspace=0.6)
    fig.suptitle(f'URL Entropy by URL Type')

    this_ax = ax  # just one so no subscript now

    averages = []
    averages.append(np.average(df[df['type'] == 'benign']['url'].apply(get_url_entropy).to_list()))
    averages.append(np.average(df[df['type'] == 'phishing']['url'].apply(get_url_entropy).to_list()))
    averages.append(np.average(df[df['type'] == 'malware']['url'].apply(get_url_entropy).to_list()))
    averages.append(np.average(df[df['type'] == 'defacement']['url'].apply(get_url_entropy).to_list()))

    this_ax.bar(['benign', 'phishing', 'malware', 'defacement'], averages)
    this_ax.set_xlabel('URL Type')
    this_ax.set_ylabel('Entropy')


def domain_entropy_histogram(df: pd.DataFrame) -> None:
    """Create a bar chart of the entropy of benign/malicious domain names for the URLs"""
    fig, ax = plt.subplots(nrows=1, ncols=1)
    fig.set_figheight(8)
    fig.set_figwidth(16)
    fig.subplots_adjust(bottom=0.25, hspace=0.8, wspace=0.6)
    fig.suptitle(f'Domain Entropy by URL Type')

    this_ax = ax  # just one so no subscript now

    averages = []
    averages.append(np.average(df[df['type'] == 'benign']['url'].apply(get_domain_entropy).to_list()))
    averages.append(np.average(df[df['type'] == 'phishing']['url'].apply(get_domain_entropy).to_list()))
    averages.append(np.average(df[df['type'] == 'malware']['url'].apply(get_domain_entropy).to_list()))
    averages.append(np.average(df[df['type'] == 'defacement']['url'].apply(get_domain_entropy).to_list()))

    this_ax.bar(['benign', 'phishing', 'malware', 'defacement'], averages)
    this_ax.set_xlabel('URL Type')
    this_ax.set_ylabel('Entropy')


def domain_or_ip_histogram(df: pd.DataFrame) -> None:
    """Create a bar chart displaying the proportion of each class that is an IP address or has a domain name"""
    fig, ax = plt.subplots(nrows=1, ncols=1)
    fig.set_figheight(8)
    fig.set_figwidth(16)
    fig.subplots_adjust(bottom=0.25, hspace=0.8, wspace=0.6)
    fig.suptitle(f'Domain/IP Proportion by URL Type')

    this_ax = ax  # just one so no subscript now

    # domain_or_ip returns 1 if it is a domain, 0 if it is an IP address
    domain_proportions = []
    r = df[df['type'] == 'benign']['url'].apply(domain_or_ip).to_list()
    domain_proportions.append(sum(r)/len(r))
    r = df[df['type'] == 'phishing']['url'].apply(domain_or_ip).to_list()
    domain_proportions.append(sum(r)/len(r))
    r = df[df['type'] == 'malware']['url'].apply(domain_or_ip).to_list()
    domain_proportions.append(sum(r)/len(r))
    r = df[df['type'] == 'defacement']['url'].apply(domain_or_ip).to_list()
    domain_proportions.append(sum(r)/len(r))

    this_ax.bar(['benign', 'phishing', 'malware', 'defacement'], domain_proportions)
    this_ax.set_xlabel('URL Type')
    this_ax.set_ylabel('Domain Proportion')



def do_statistics(df: pd.DataFrame) -> None:
    """Run and display all the charts"""
    benign_malicious_histograms(df)
    tld_counts_histograms(df)
    url_length_histograms(df)
    special_character_histograms(df)
    subdomain_count_histograms(df)
    directory_level_count_histograms(df)
    url_param_count_histograms(df)
    url_entropy_histogram(df)
    domain_entropy_histogram(df)
    domain_or_ip_histogram(df)

    print('Charts should be displayed now...')
    print('Close the plots to continue.')
    plt.show()
