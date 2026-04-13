```python
# human_behavior_classification.py

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression


# ==============================
# Load Dataset
# ==============================
def load_data(path):
    df = pd.read_csv(path)
    X = df.iloc[:, :-1]
    y = df.iloc[:, -1]
    return X, y


# ==============================
# Preprocessing
# ==============================
def preprocess_data(X_train, X_test):
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    return X_train_scaled, X_test_scaled


# ==============================
# Evaluation
# ==============================
def evaluate(y_true, y_pred):
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, average='weighted'),
        "recall": recall_score(y_true, y_pred, average='weighted'),
        "f1_score": f1_score(y_true, y_pred, average='weighted')
    }


def print_metrics(name, metrics):
    print(f"\n{name}")
    print("-" * len(name))
    for k, v in metrics.items():
        print(f"{k.capitalize():10}: {v:.4f}")


def plot_cm(y_true, y_pred, title, filename):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, cmap="Blues", annot=False)
    plt.title(title)
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.savefig(filename, dpi=300, bbox_inches="tight")
    plt.close()


# ==============================
# Models
# ==============================
def run_knn(X_train_scaled, X_test_scaled, y_train, y_test):
    print("\n===== KNN =====")

    best_k = 1
    best_acc = 0
    best_pred = None

    for k in range(1, 11):
        model = KNeighborsClassifier(n_neighbors=k)
        model.fit(X_train_scaled, y_train)

        y_pred = model.predict(X_test_scaled)
        metrics = evaluate(y_test, y_pred)

        print(f"\nk = {k}")
        print_metrics("Metrics", metrics)

        if metrics["accuracy"] > best_acc:
            best_acc = metrics["accuracy"]
            best_k = k
            best_pred = y_pred

    print(f"\nBest K: {best_k} | Accuracy: {best_acc:.4f}")
    plot_cm(y_test, best_pred, "KNN Confusion Matrix", "knn_confusion_matrix.png")


def run_decision_tree(X_train, X_test, y_train, y_test):
    print("\n===== Decision Tree =====")

    model = DecisionTreeClassifier(max_depth=14, random_state=42)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    metrics = evaluate(y_test, y_pred)

    print_metrics("Decision Tree", metrics)


def run_logistic_regression(X_train_scaled, X_test_scaled, y_train, y_test):
    print("\n===== Logistic Regression =====")

    model = LogisticRegression(max_iter=1000, solver='lbfgs')
    model.fit(X_train_scaled, y_train)

    y_pred = model.predict(X_test_scaled)
    metrics = evaluate(y_test, y_pred)

    print_metrics("Logistic Regression", metrics)

    plot_cm(y_test, y_pred, "Logistic Regression Confusion Matrix", "lr_confusion_matrix.png")


# ==============================
# Main
# ==============================
def main():
    # Change this path to your dataset file
    data_path = "mhealth_raw_data.csv"

    X, y = load_data(data_path)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    X_train_scaled, X_test_scaled = preprocess_data(X_train, X_test)

    run_knn(X_train_scaled, X_test_scaled, y_train, y_test)
    run_decision_tree(X_train, X_test, y_train, y_test)
    run_logistic_regression(X_train_scaled, X_test_scaled, y_train, y_test)

    print("\nDone.")


if __name__ == "__main__":
    main()
```
