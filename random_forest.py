from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
import data_statistics

def train_random_forest(
    X_train,
    X_test,
    y_train,
    y_test,
    n_estimators: int = 200,
    max_depth=None,
    min_samples_leaf: int = 1,
    n_jobs: int = -1,
    random_state: int = 42,
) -> None:

    model = RandomForestClassifier(
        n_estimators = n_estimators,
        max_depth = max_depth,
        min_samples_leaf = min_samples_leaf,
        n_jobs = n_jobs,
        random_state = random_state,
    )
    model.fit(X_train, y_train)

    pred = model.predict(X_test)

    accuracy = accuracy_score(pred, y_test)
    print('Random Forest Accuracy :\033[32m \033[01m {:.2f}% \033[30m \033[0m'.format(accuracy * 100))

    print('Classification report:')
    print(classification_report(y_test, pred))

    print('Confusion_matrix:')
    data_statistics.plot_confusion_matrix(y_test, pred,['benign', 'phishing', 'malware', 'defacement'])
