

import pandas as pd
import joblib
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from imblearn.over_sampling import SMOTE

from src.preprocessing import preprocess_data


def train_initial_model(data_path: str, model_name: str = "rf_attrition_model"):

    print(f"Loading data from {data_path}...")
    df = pd.read_csv(data_path)
    
    print("Preprocessing data...")
    X, y = preprocess_data(df, has_target=True)
    
    print(f"Dataset shape: {X.shape}")
    print(f"Target distribution:\n{y.value_counts()}")
    
    # Split data into train and test sets
    print("\nSplitting data...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, train_size=0.80, random_state=0, stratify=y
    )
    
    # Apply SMOTE for class balancing
    print("Applying SMOTE for class balancing...")
    oversampler = SMOTE(random_state=0)
    X_train_balanced, y_train_balanced = oversampler.fit_resample(X_train, y_train)
    
    print(f"After SMOTE - Training set size: {len(X_train_balanced)}")
    
    # Train Random Forest model with parameters
    print("\nTraining Random Forest model...")
    rf_params = {
        'n_jobs': -1,
        'n_estimators': 1000,
        'max_features': 0.3,
        'max_depth': 4,
        'min_samples_leaf': 2,
        'max_features': 'sqrt',
        'random_state': 0,
        'verbose': 0
    }
    
    model = RandomForestClassifier(**rf_params)
    model.fit(X_train_balanced, y_train_balanced)
    
    # Evaluate on test set
    print("\nEvaluating model on test set...")
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    
    print(f"Accuracy score: {accuracy:.4f}")
    print("=" * 80)
    print(classification_report(y_test, y_pred))
    
    # Save model with feature columns
    models_dir = Path("models")
    models_dir.mkdir(exist_ok=True)
    
    model_path = models_dir / f"{model_name}.pkl"
    print(f"\nSaving model to {model_path}...")
    
    joblib.dump({
        'model': model,
        'feature_columns': list(X.columns)
    }, model_path)
    
    print(f"Model saved successfully as '{model_name}'")
    print(f"Feature columns saved: {len(X.columns)} features")
    
    return model, X.columns


if __name__ == "__main__":
    train_initial_model("data/Employee-Attrition.csv", "rf_attrition_model")
