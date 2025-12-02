# CAML FINAL
# Zach Riback, Cayden Wright, Ariana Ciaschini, Brett Huber
import sys
import argparse
import load_data
import feature_extraction
import data_statistics
import nn
import decision_tree
import random_forest
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier

def main() -> None:
    parser = argparse.ArgumentParser('Argument parser to help with only running certain parts of the code')
    parser.add_argument('--statistics', '-s', action='store_true', help='Loads data and does statistics')
    parser.add_argument('--decision_tree', '-d', action='store_true', help='Loads data, extracts features, and trains a decision tree')
    parser.add_argument('--decision_tree_binary', '-db', action='store_true', help='Loads data, extracts features, and trains a decision tree for binary classification (benign vs malicious)')
    parser.add_argument('--random_forest', '-r', action='store_true', help='Loads data, extracts features, and trains a Random Forest') 
    parser.add_argument('--random_forest_binary', '-rb', action='store_true', help='Loads data, extracts features, and trains a Random Forest for binary classification (benign vs malicious)')
    parser.add_argument('--nn_binary', '-n', action='store_true', help='Loads data, extracts features, and trains a neural network to distinguish benign and malicious URLs')
    parser.add_argument('--nn_malicious', '-m', action='store_true', help='Loads data, extracts features, and trains a neural network to distinguish between malicious URLs')
    parser.add_argument('--nn_all', '-a', action='store_true', help='Loads data, extracts features, and trains a neural network to distinguish between all URL types')
    parser.add_argument('--balance', action='store_true', help='Balance the dataset before training')


    args = parser.parse_args()

    if args.statistics:
        df = load_data.load_dataset()
        df = load_data.clean_dataset(df)
        data_statistics.do_statistics(df)

    elif args.decision_tree:
        df = load_data.load_dataset()
        df = load_data.clean_dataset(df)
        # multi-class mapping consistent with feature_extraction.do_feature_extraction_decision_tree
        class_distinguisher = lambda x: 0 if x == 'benign' \
            else 1 if x == 'defacement' \
            else 2 if x == 'phishing' \
            else 3
        # Determine the maximum safe class size: use the smallest available class count, capped at 200k
        mapped = df['type'].apply(class_distinguisher)
        min_count = int(mapped.value_counts().min())
        class_size = min(200_000, min_count)
        if class_size <= 0:
            print('Dataset does not contain examples for all classes. Exiting...')
            sys.exit()

        df = load_data.balance_dataset(df, class_distinguisher, class_size)
        if df is None:
            print('Could not balance the dataset with class_size=', class_size, "Exiting...")
            sys.exit()

        # get rid of all www subdomains
        df['url'] = df['url'].replace('www.', '', regex=True)
        
        # do feature extraction (this will populate `Category` numeric codes)
        feature_extraction.do_feature_extraction_decision_tree(df)
        
        # split into train/test using multi-class `Category` as the target
        X = df.drop(['url','type','Category','domain'], axis=1)
        y = df['Category']
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=2)
        model = DecisionTreeClassifier()
        decision_tree.train_decision_tree(model, X_train, X_test, y_train, y_test, class_names=['benign', 'defacement', 'phishing', 'malware'])
        decision_tree.print_tree(model, X_train.columns)

    elif args.decision_tree_binary:
        df = load_data.load_dataset()
        df = load_data.clean_dataset(df)
        # binary mapping: 0 = benign, 1 = malicious
        class_distinguisher = lambda x: 0 if x == 'benign' else 1

        mapped = df['type'].apply(class_distinguisher)
        min_count = int(mapped.value_counts().min())
        class_size = min(200_000, min_count)
        if class_size <= 0:
            print('Dataset does not contain examples for both classes. Exiting...')
            sys.exit()

        df = load_data.balance_dataset(df, class_distinguisher, class_size)
        if df is None:
            print('Could not balance the dataset with class_size=', class_size, "Exiting...")
            sys.exit()

        # get rid of all www subdomains
        df['url'] = df['url'].replace('www.', '', regex=True)

        # feature extraction (still ok to call; Category will be created but we will use binary labels)
        feature_extraction.do_feature_extraction_decision_tree(df)

        # split into train/test using binary labels
        X = df.drop(['url','type','Category','domain'], axis=1)
        y = df['type'].apply(class_distinguisher)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=2)
        model = DecisionTreeClassifier()
        decision_tree.train_decision_tree(model, X_train, X_test, y_train, y_test, class_names=['benign', 'malicious'])
        decision_tree.print_tree(model, X_train.columns)
        
    elif args.random_forest:  
        df = load_data.load_dataset()
        df = load_data.clean_dataset(df)

        # Balances the benign vs malicious (all non-benign grouped together for balancing)
        class_distinguisher = lambda x: 0 if x == 'benign' else 1
        df = load_data.balance_dataset(df, class_distinguisher, 200_000)
        if df is None:
            print('Could not balance the dataset! Exiting...')
            sys.exit()

        # Gets rid of all www subdomains
        df['url'] = df['url'].replace('www.', '', regex=True)

        # Feature extraction (populates Category: 0=benign,1=defacement,2=phishing,3=malware)
        feature_extraction.do_feature_extraction_decision_tree(df)

        # Train/test split on 4-class labels
        X = df.drop(['url','type','Category','domain'], axis=1)
        y = df['Category']
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=2
        )

        random_forest.train_random_forest(
            X_train,
            X_test,
            y_train,
            y_test,
            n_estimators=200,
            max_depth=None,
            min_samples_leaf=1,
            n_jobs=-1,
            random_state=42,
            class_names=['benign', 'defacement', 'phishing', 'malware'],
        )

    elif args.random_forest_binary:
        df = load_data.load_dataset()
        df = load_data.clean_dataset(df)

        # Binary mapping: 0 = benign, 1 = malicious (any non-benign)
        class_distinguisher = lambda x: 0 if x == 'benign' else 1

        mapped = df['type'].apply(class_distinguisher)
        min_count = int(mapped.value_counts().min())
        class_size = min(200_000, min_count)
        if class_size <= 0:
            print('Dataset does not contain examples for both classes. Exiting...')
            sys.exit()

        df = load_data.balance_dataset(df, class_distinguisher, class_size)
        if df is None:
            print('Could not balance the dataset! Exiting...')
            sys.exit()

        # Gets rid of all www subdomains
        df['url'] = df['url'].replace('www.', '', regex=True)

        # Feature extraction (still uses decision_tree-style features)
        feature_extraction.do_feature_extraction_decision_tree(df)

        # Split using binary labels (0=benign, 1=malicious)
        X = df.drop(['url','type','Category','domain'], axis=1)
        y = df['type'].apply(class_distinguisher)
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=2
        )

        random_forest.train_random_forest(
            X_train,
            X_test,
            y_train,
            y_test,
            n_estimators = 200,
            max_depth = None,
            min_samples_leaf = 1,
            n_jobs = -1,
            random_state = 42,
            class_names=['benign', 'malicious'],
        )

    elif args.nn_binary or args.nn_malicious or args.nn_all:
        df = load_data.load_dataset()
        df = load_data.clean_dataset(df)

        if args.nn_binary:
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
        elif args.nn_malicious:
            # Remove benign data
            df = load_data.remove_class(df, 'benign')


            # label = 0,1,2 for phishing,malware,defacement
            class_distinguisher = lambda x: 0 if x == 'phishing' else 1 if x == 'malware' else 2

            df = load_data.balance_dataset(df, class_distinguisher, 20_000)
            if df is None:
                print('Could not balance the dataset! Exiting...')
                sys.exit()
            
            X, y = feature_extraction.do_feature_extraction_nn(df, class_distinguisher)

            print('Got X and y with the following shapes:')
            print(X.shape)
            print(y.shape)

            # Now train and test the model
            nn.evaluate_nn(X, y, class_names=['phishing', 'malware', 'defacement'])
        
        elif args.nn_all:
            # label = 0,1,2,3 for benign,phishing,malware,defacement
            class_distinguisher = lambda x: 0 if x == 'benign' \
                else 1 if x == 'phishing' \
                else 2 if x == 'malware' \
                else 3  # defacement

            df = load_data.balance_dataset(df, class_distinguisher, 20_000)
            if df is None:
                print('Could not balance the dataset! Exiting...')
                sys.exit()
            
            X, y = feature_extraction.do_feature_extraction_nn(df, class_distinguisher)

            print('Got X and y with the following shapes:')
            print(X.shape)
            print(y.shape)

            nn.evaluate_nn(X, y, class_names=['benign', 'phishing', 'malware', 'defacement'])


    else:
        parser.print_help()


if __name__ == "__main__":
    main()