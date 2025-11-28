from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

def train_random_forest(X_train, X_test, y_train, y_test) -> None:
    model = RandomForestClassifier(
        n_estimators=300,
        max_depth=None,
        min_samples_split=2,
        min_samples_leaf=1,
        bootstrap=True,
        n_jobs=-1,
        random_state=42
    )

    print("\nTraining Random Forest...")
    model.fit(X_train, y_train)
    pred = model.predict(X_test)

    accuracy = accuracy_score(y_test, pred)
    print('\nRandom Forest Accuracy: \033[32m\033[01m {:.2f}% \033[30m\033[0m'.format(accuracy * 100))

    print("\nClassification Report:")
    print(classification_report(y_test, pred))

    print("\nConfusion Matrix:")
    cf_matrix = confusion_matrix(y_test, pred)

    sns.heatmap(cf_matrix / np.sum(cf_matrix), annot=True, fmt='0.2%', cmap='Blues')
    plt.title('Random Forest Confusion Matrix (Normalized)')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.show()
