from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np


def train_random_forest(
    X_train,
    X_test,
    y_train,
    y_test,
    n_estimators: int = 200,
    max_depth: int | None = None,
    min_samples_leaf: int = 1,
    n_jobs: int = -1,
    random_state: int = 42,
) -> None:

    rf = RandomForestClassifier(
        n_estimators = n_estimators,
        max_depth = max_depth,
        min_samples_leaf = min_samples_leaf,
        n_jobs = n_jobs,
        random_state = random_state,
        class_weight = None,  
    )

    rf.fit(X_train, y_train)

    y_pred = rf.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    print("Random Forest Accuracy :\033[32m \033[01m {:.2f}% \033[30m \033[0m".format(acc * 100))

    print("\nClassification report:")
    print(classification_report(y_test, y_pred))

    cf_matrix = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cf_matrix / np.sum(cf_matrix), annot=True, fmt="0.2%")
    plt.title("Random Forest - Normalized Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.tight_layout()
    plt.show()

    if hasattr(rf, "feature_importances_"):
        importances = rf.feature_importances_
        print("\nTop feature importances (by index):")
        for idx, imp in sorted(enumerate(importances), key=lambda x: x[1], reverse=True):
            print(f"Feature {idx}: {imp:.4f}")
