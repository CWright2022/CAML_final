from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score, precision_score, recall_score, f1_score
import numpy as np
import data_statistics

def train_random_forest(
    X_train, # Feature matrix for training data
    X_test, #  Feature matrix for testing data
    y_train, # Truth labels for training data
    y_test, # Truth labels for testing data
    n_estimators: int = 200, # Number of trees within the forest
    max_depth = None, # Tree depth limiter (defaults none)
    min_samples_leaf: int = 1, # Controls leaf size to prevent overfitting
    n_jobs: int = -1, # CPU parallelization (if -1, it uses all cores)
    random_state: int = 42, # Helps tree building
    class_names: list | None = None, # Use for plotting and reports
) -> None:

    # Trains the Random Forest model given the specified hyperparameters
    model = RandomForestClassifier(
        n_estimators = n_estimators,
        max_depth = max_depth,
        min_samples_leaf = min_samples_leaf,
        n_jobs = n_jobs,
        random_state = random_state,
    )

    # Fits the model and then generates predictions based on test data
    model.fit(X_train, y_train)
    pred = model.predict(X_test)

    # Computes accuracy (overall correctness)
    accuracy = accuracy_score(y_test, pred)

    # Determines if the classifier is binary or multiclass
    unique_labels = np.unique(np.concatenate((np.asarray(y_test), np.asarray(pred))))
    is_binary = unique_labels.size == 2

    # Computes precision, recall, and F1 score based on binary or multiclass classification and weighting
    if is_binary:
        precision = precision_score(y_test, pred, average = "binary", zero_division = 0)
        recall = recall_score(y_test, pred, average = "binary", zero_division = 0)
        f1 = f1_score(y_test, pred, average = "binary", zero_division = 0)
    else:
        # For multiclass, present macro averages
        precision = precision_score(y_test, pred, average = "macro", zero_division = 0)
        recall = recall_score(y_test, pred, average = "macro", zero_division = 0)
        f1 = f1_score(y_test, pred, average = "macro", zero_division = 0)

    # Prints the evaluation metrics in a structured manner
    print("\n=== Random Forest Evaluation Metrics ===")
    print(f"Accuracy : {accuracy*100:0.2f}%")
    print(
        f'Precision: {precision:0.4f} ("macro" average)'
        if not is_binary
        else f"Precision: {precision:0.4f} (binary)"
    )
    print(
        f'Recall   : {recall:0.4f} ("macro" average)'
        if not is_binary
        else f"Recall   : {recall:0.4f} (binary)"
    )
    print(
        f'F1 Score : {f1:0.4f} ("macro" average)'
        if not is_binary
        else f"F1 Score : {f1:0.4f} (binary)"
    )

    # Complete classification report
    print("\nClassification report:")
    try:
        # If class names are equivalent to number of unique labels, then use them
        if class_names is not None and len(class_names) == len(unique_labels):
            print(
                classification_report(
                    y_test, pred, target_names=class_names, zero_division = 0
                )
            )
        else:
            print(classification_report(y_test, pred, zero_division = 0))
    except Exception:

        print(classification_report(y_test, pred, zero_division = 0))

    # Binary-specific breakdown
    if is_binary:
        cm = confusion_matrix(y_test, pred)  # [[TN, FP], [FN, TP]]
        tn, fp, fn, tp = cm.ravel()
        print("Binary confusion counts:")
        if class_names is not None and len(class_names) >= 2:
            print(f"True Negative ({class_names[0]}): {tn}")
            print(f"False Positive ({class_names[1]}): {fp}")
            print(f"False Negative ({class_names[0]} predicted as {class_names[1]}): {fn}")
            print(f"True Positive ({class_names[1]}): {tp}")
        else:
            print(f"TN: {tn}, FP: {fp}, FN: {fn}, TP: {tp}")

        # Per-class precision/recall/f1
        per_prec = precision_score(y_test, pred, average = None, zero_division = 0)
        per_rec = recall_score(y_test, pred, average = None, zero_division = 0)
        per_f1 = f1_score(y_test, pred, average = None, zero_division = 0)
        print("\nPer-class metrics:")
        for idx, (p, r, f) in enumerate(zip(per_prec, per_rec, per_f1)):
            name = (
                class_names[idx]
                if class_names is not None and idx < len(class_names)
                else str(idx)
            )
            print(f"  {name}: Precision={p:0.4f}, Recall={r:0.4f}, F1={f:0.4f}")

    # Confusion matrix plot
    if class_names is None:
        # If class names are not provided, provides conversion of labels to strings
        class_names = [str(l) for l in sorted(unique_labels.tolist())]

    print("Confusion matrix:")
    # Call the project's plotting function
    data_statistics.plot_confusion_matrix(y_test, pred, class_names)
