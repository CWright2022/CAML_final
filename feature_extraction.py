import pandas as pd
import tldextract
def do_feature_extraction (df: pd.DataFrame) -> None:
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