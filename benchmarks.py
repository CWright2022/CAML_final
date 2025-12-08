import time
import os
import csv
import matplotlib.pyplot as plt
import numpy as np

import load_data
import feature_extraction
import decision_tree
import random_forest
import nn
from sklearn.model_selection import train_test_split


def safe_balance(df, class_distinguisher, cap=200_000):
    mapped = df['type'].apply(class_distinguisher)
    counts = mapped.value_counts()
    if counts.size == 0:
        raise RuntimeError('No classes found')
    class_size = min(cap, int(counts.min()))
    if class_size <= 0:
        raise RuntimeError('Insufficient examples to balance')
    df_bal = load_data.balance_dataset(df, class_distinguisher, class_size)
    if df_bal is None:
        raise RuntimeError(f'Could not balance dataset with class_size={class_size}')
    return df_bal


def prepare_dt_df(df):
    df = df.copy()
    df['url'] = df['url'].replace('www.', '', regex=True)
    feature_extraction.do_feature_extraction_decision_tree(df)
    X = df.drop(['url', 'type', 'Category', 'domain'], axis=1)
    y = df['Category']
    return X, y


def prepare_nn_arrays(df, class_distinguisher):
    X, y = feature_extraction.do_feature_extraction_nn(df, class_distinguisher)
    return X, y


def time_decision_tree(X_train, X_test, y_train, y_test, class_names=None):
    from sklearn.tree import DecisionTreeClassifier
    clf = DecisionTreeClassifier()
    t0 = time.time()
    decision_tree.train_decision_tree(clf, X_train, X_test, y_train, y_test, class_names=class_names)
    t1 = time.time()
    return t1 - t0


def time_random_forest(X_train, X_test, y_train, y_test):
    t0 = time.time()
    random_forest.train_random_forest(X_train, X_test, y_train, y_test, n_estimators=200, max_depth=None, min_samples_leaf=1, n_jobs=-1, random_state=42)
    t1 = time.time()
    return t1 - t0


def time_nn(X, y, class_names=None):
    t0 = time.time()
    nn.evaluate_nn(X, y, test_size=0.1, class_names=class_names)
    t1 = time.time()
    return t1 - t0


def run_all(out_dir='benchmarks_output'):
    os.makedirs(out_dir, exist_ok=True)

    print('Loading dataset...')
    df = load_data.load_dataset()
    df = load_data.clean_dataset(df)

    timings = []

    # Binary classification experiments only
    class_distinguisher_bin = lambda x: 0 if x == 'benign' else 1

    # 1) Neural Net - binary
    print('\nPreparing NN (binary)...')
    df_bin = safe_balance(df, class_distinguisher_bin, cap=50_000)
    X_bin, y_bin = prepare_nn_arrays(df_bin, class_distinguisher_bin)
    print('Running NN binary...')
    t_nn_bin = time_nn(X_bin, y_bin, class_names=['benign', 'malicious'])
    timings.append(('NN (binary)', t_nn_bin))

    # 2) Decision Tree - binary
    print('\nPreparing Decision Tree (binary)...')
    df_dt_bin = safe_balance(df, class_distinguisher_bin, cap=50_000)
    X_dt_bin, y_dt_bin = prepare_dt_df(df_dt_bin)
    X_train, X_test, y_train, y_test = train_test_split(X_dt_bin, y_dt_bin, test_size=0.2, random_state=2)
    print('Running Decision Tree binary...')
    t_dt_bin = time_decision_tree(X_train, X_test, y_train, y_test, class_names=['benign', 'malicious'])
    timings.append(('DecisionTree (binary)', t_dt_bin))

    # 3) Random Forest - binary
    print('\nPreparing Random Forest (binary)...')
    # reuse df_dt_bin (already balanced and feature-extracted)
    X_rf, y_rf = X_dt_bin, y_dt_bin
    X_train, X_test, y_train, y_test = train_test_split(X_rf, y_rf, test_size=0.2, random_state=2)
    print('Running Random Forest binary...')
    t_rf_bin = time_random_forest(X_train, X_test, y_train, y_test)
    timings.append(('RandomForest (binary)', t_rf_bin))

    # Save timings to CSV
    csv_path = os.path.join(out_dir, 'timings.csv')
    with open(csv_path, 'w', newline='') as fh:
        writer = csv.writer(fh)
        writer.writerow(['model', 'seconds'])
        for name, t in timings:
            writer.writerow([name, f'{t:.4f}'])

    # Plot
    names = [n for n, _ in timings]
    values = [v for _, v in timings]

    plt.figure(figsize=(10, 6))
    y_pos = np.arange(len(names))
    bars = plt.barh(y_pos, values, align='center', color='C0')
    plt.yticks(y_pos, names)
    plt.xlabel('Time (seconds)')
    plt.title('Model Training Time Comparison')
    for bar, val in zip(bars, values):
        plt.text(bar.get_width() + 0.02, bar.get_y() + bar.get_height()/2, f'{val:.2f}s', va='center')

    out_png = os.path.join(out_dir, 'timings.png')
    plt.tight_layout()
    plt.savefig(out_png)
    print(f'Benchmark results saved to: {csv_path} and {out_png}')
    plt.show()


if __name__ == '__main__':
    run_all()
