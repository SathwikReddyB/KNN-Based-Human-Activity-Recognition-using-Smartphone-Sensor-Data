import os
import numpy as np
import pandas as pd

from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import GridSearchCV

# 📁 Update this path to your dataset folder
DATASET_PATH = "/Users/sathwik/Development/DMA KNN Project/UCI HAR Dataset"

# -------------------------------

# 1. Load Data
# -------------------------------
def load_data():
    X_train = pd.read_csv(os.path.join(DATASET_PATH, "train/X_train.txt"), sep=r'\s+', header=None)
    y_train = pd.read_csv(os.path.join(DATASET_PATH, "train/y_train.txt"), sep=r'\s+', header=None)

    X_test = pd.read_csv(os.path.join(DATASET_PATH, "test/X_test.txt"), sep=r'\s+', header=None)
    y_test = pd.read_csv(os.path.join(DATASET_PATH, "test/y_test.txt"), sep=r'\s+', header=None)

    return X_train, y_train.values.ravel(), X_test, y_test.values.ravel()

# -------------------------------
# 2. Preprocessing (Scaling)
# -------------------------------
def preprocess(X_train, X_test):
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    return X_train_scaled, X_test_scaled, scaler

# -------------------------------
# 3. Train Model (with tuning)
# -------------------------------
def train_knn(X_train, y_train):
    print("🔍 Tuning KNN hyperparameters...")

    param_grid = {
        'n_neighbors': [3, 5, 7, 9],
        'weights': ['uniform', 'distance'],
        'metric': ['euclidean', 'manhattan']
    }

    knn = KNeighborsClassifier()

    grid = GridSearchCV(knn, param_grid, cv=3, scoring='accuracy', n_jobs=-1, verbose=1)
    grid.fit(X_train, y_train)

    print(f"✅ Best Params: {grid.best_params_}")
    print(f"✅ Best CV Accuracy: {grid.best_score_:.4f}")

    return grid.best_estimator_

# -------------------------------
# 4. Evaluate Model
# -------------------------------
def evaluate(model, X_test, y_test):
    y_pred = model.predict(X_test)

    print("\n📊 Test Results:")
    print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))

    print("\nConfusion Matrix:")
    print(confusion_matrix(y_test, y_pred))

# -------------------------------
# 5. Main Pipeline
# -------------------------------
def main():
    print("🚀 Loading dataset...")
    X_train, y_train, X_test, y_test = load_data()

    print("⚙️ Preprocessing...")
    X_train, X_test, scaler = preprocess(X_train, X_test)

    print("🤖 Training model...")
    model = train_knn(X_train, y_train)

    print("📈 Evaluating model...")
    evaluate(model, X_test, y_test)

    # Save model
    import joblib
    joblib.dump(model, "knn_har_model.pkl")
    joblib.dump(scaler, "scaler.pkl")

    print("\n💾 Model saved as knn_har_model.pkl")

# -------------------------------
if __name__ == "__main__":
    main()