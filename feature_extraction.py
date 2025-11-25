import pandas as pd
import numpy as np
import tldextract
from typing import Callable
import data_statistics


def normalize_arr(arr: np.ndarray) -> np.ndarray:
    mean = arr.mean()
    std = arr.std()
    return (arr - mean) / std


def do_feature_extraction_decision_tree(df: pd.DataFrame) -> None:
    '''
    extracts features and places them in the dataframe (destructive)
    - url_len: length of the url
    - tld: top level domain
    - subdomain: subdomain of the url
    - prefix: "prefix", the part without ".com" or such
    - symbols - whether a given symbol is present
    '''
    df['url_len'] = df['url'].apply(lambda x: len(str(x)))
    df['tld'] = df['url'].apply(lambda x: tldextract.extract(x).suffix)
    df['subdomain'] = df['url'].apply(lambda x: tldextract.extract(x).subdomain)
    df['prefix'] = df['url'].apply(lambda x: tldextract.extract(x).domain)
    symbols = ['@','?','-','=','#','%','+','$','!','*',',']
    for symbol in symbols:
        df[symbol] = df['url'].apply(lambda i: i.count(symbol))


def do_feature_extraction_nn(df: pd.DataFrame, class_distinguisher: Callable[[str], int]) -> tuple[np.ndarray, np.ndarray]:
    """
    Extracts features and returns X and y structures for machine learning in a neural network
    Features extracted:
        URL length
        URL special character count
        URL subdomain count
        URL directory level count
        URL Parameter count
        URL Entropy
        Domain entropy
        IP or Domain Name (0 or 1)
    """
    url_lens = df['url'].apply(len).to_numpy()
    spec_char_counts = df['url'].apply(data_statistics.count_special_chars).to_numpy(dtype=np.float32)
    subdomain_counts = df['url'].apply(data_statistics.count_subdomains).to_numpy(dtype=np.float32)
    directory_level_counts = df['url'].apply(data_statistics.count_directory_levels).to_numpy(dtype=np.float32)
    url_parameter_counts = df['url'].apply(data_statistics.count_url_params).to_numpy(dtype=np.float32)
    url_entropies = df['url'].apply(data_statistics.get_url_entropy).to_numpy(dtype=np.float32)
    domain_entropies = df['url'].apply(data_statistics.get_domain_entropy).to_numpy(dtype=np.float32)
    domain_or_ip = df['url'].apply(data_statistics.domain_or_ip).to_numpy(dtype=np.float32)
    y = df['type'].apply(class_distinguisher).to_numpy()

    # Normalize these vectors
    url_lens = normalize_arr(url_lens)
    spec_char_counts = normalize_arr(spec_char_counts)
    subdomain_counts = normalize_arr(subdomain_counts)
    directory_level_counts = normalize_arr(directory_level_counts)
    url_parameter_counts = normalize_arr(url_parameter_counts)
    url_entropies = normalize_arr(url_entropies)
    domain_entropies = normalize_arr(domain_entropies)
    # domain or ip does NOT need to be normalized

    X = np.stack(
        (url_lens,
         spec_char_counts,
         subdomain_counts,
         directory_level_counts,
         url_parameter_counts,
         url_entropies,
         domain_entropies,
         domain_or_ip
         ), 
         axis=1
    )

    return X, y

    

