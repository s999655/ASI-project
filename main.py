import os
import joblib
import pandas as pd
from pathlib import Path
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.responses import JSONResponse
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from imblearn.over_sampling import SMOTE
from typing import List

from src.preprocessing import preprocess_data, align_features

app = FastAPI(title="Employee Attrition Model")

MODELS_DIR = Path("models")
MODELS_DIR.mkdir(exist_ok=True)


@app.get("/")
def root():
    return {"message": "Employee Attrition Model", "status": "running"}


@app.post("/continue-train")
async def continue_train(
    model_name: str = Form(...),
    train_input: UploadFile = File(...),
    new_model_name: str = Form(...)
):
    try:
        # Load existing model
        model_path = MODELS_DIR / f"{model_name}.pkl"
        if not model_path.exists():
            raise HTTPException(status_code=404, detail=f"Model '{model_name}' not found")
        
        model_data = joblib.load(model_path)
        model = model_data['model']
        feature_columns = model_data['feature_columns']
        
        # Read and preprocess training data
        df = pd.read_csv(train_input.file)
        X_train, y_train = preprocess_data(df, has_target=True)
        
        # Align features with the original model's features
        X_train = align_features(X_train, feature_columns)
        
        # Apply SMOTE for class balancing
        oversampler = SMOTE(random_state=0)
        X_train_balanced, y_train_balanced = oversampler.fit_resample(X_train, y_train)
        
        # Continue training 
        model.set_params(warm_start=True)
        model.fit(X_train_balanced, y_train_balanced)
        
        # Make predictions on training data for metrics
        y_pred = model.predict(X_train)
        accuracy = accuracy_score(y_train, y_pred)
        report = classification_report(y_train, y_pred, output_dict=True)
        
        # Save the new model
        new_model_path = MODELS_DIR / f"{new_model_name}.pkl"
        joblib.dump({
            'model': model,
            'feature_columns': feature_columns
        }, new_model_path)
        
        return {
            "status": "success",
            "new_model_name": new_model_name,
            "metrics": {
                "accuracy": accuracy,
                "classification_report": report
            }
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/predict")
async def predict(
    model_name: str = Form(...),
    input: UploadFile = File(...)
):
    try:
        # Load model
        model_path = MODELS_DIR / f"{model_name}.pkl"
        if not model_path.exists():
            raise HTTPException(status_code=404, detail=f"Model '{model_name}' not found")
        
        model_data = joblib.load(model_path)
        model = model_data['model']
        feature_columns = model_data['feature_columns']
        
        # Read and preprocess input data
        df = pd.read_csv(input.file)
        
        if 'Attrition' in df.columns:
            raise HTTPException(
                status_code=400, 
                detail="Input file should NOT contain 'Attrition' column"
            )
        
        X_predict, _ = preprocess_data(df, has_target=False)
        
        # Align features with the model's features
        X_predict = align_features(X_predict, feature_columns)
        
        # Make predictions
        predictions = model.predict(X_predict)
        probabilities = model.predict_proba(X_predict)
        
        # Convert to human-readable format
        prediction_labels = ['No' if p == 0 else 'Yes' for p in predictions]
        
        return {
            "status": "success",
            "predictions": prediction_labels,
            "probabilities": probabilities[:, 1].tolist(),  # Probability of attrition
            "count": len(predictions)
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/models")
def list_models() -> List[str]:
    
    try:
        model_files = list(MODELS_DIR.glob("*.pkl"))
        model_names = [f.stem for f in model_files]
        return model_names
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


def main():
    print("Employee Attrition Model API")
    


if __name__ == "__main__":
    main()

