# src/train_rf_explain.py
import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import os

def train(parsed_pkl_path='data/processed/parsed.pkl'):
    """
    Loads processed data, trains a RandomForest classifier, evaluates it,
    and saves the final model and vectorizer.
    """
    print("--- Starting Model Training ---")

    # --- 1. Load Processed Data ---
    if not os.path.exists(parsed_pkl_path):
        print(f"Error: Processed data file not found at '{parsed_pkl_path}'.")
        print("Please run 'python src/dataset_builder.py' first.")
        return

    print(f"Loading data from {parsed_pkl_path}...")
    df = pd.read_pickle(parsed_pkl_path)
    df['norm_msg'] = df['norm_msg'].fillna('') # Handle any potential empty messages

    # --- 2. Create Heuristic Labels ---
    print("Creating heuristic labels for training...")
    # Mark lines with common error-related keywords as anomalies (label=1)
    keywords = r'error|exception|fail|timeout|unable|traceback|critical|denied'
    df['label'] = df['msg'].str.contains(keywords, case=False, na=False).astype(int)

    print("Label distribution:")
    print(df['label'].value_counts())

    # --- 3. Feature Engineering (TF-IDF) ---
    print("Vectorizing text data with TF-IDF...")
    # Use n-grams to capture multi-word patterns
    vec = TfidfVectorizer(max_features=5000, ngram_range=(1, 2))
    X = vec.fit_transform(df['norm_msg'])
    y = df['label'].values

    # --- 4. Train & Evaluate Model ---
    print("Splitting data and training RandomForest Classifier...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # n_jobs=-1 uses all available CPU cores for faster training
    clf = RandomForestClassifier(n_estimators=100, n_jobs=-1, random_state=42)
    clf.fit(X_train, y_train)

    print("\n--- Model Evaluation ---")
    y_pred = clf.predict(X_test)
    print("Classification Report on Test Set:")
    print(classification_report(y_test, y_pred))

    # --- 5. Save the final models ---
    models_dir = 'models'
    os.makedirs(models_dir, exist_ok=True)
    joblib.dump(clf, os.path.join(models_dir, 'rf_anomaly_detector.joblib'))
    joblib.dump(vec, os.path.join(models_dir, 'tfidf_vectorizer.joblib'))

    print(f"\nModels saved to '{models_dir}' directory.")
    print("--- Training Finished ---")

if __name__ == '__main__':
    train()