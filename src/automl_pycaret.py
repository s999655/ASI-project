import typer
import pandas as pd
from pycaret.classification import (
    setup,
    compare_models,
    finalize_model,
    predict_model,
    save_model,
    load_model,
    pull
)

app = typer.Typer()

@app.command()
def train(
    data_path: str,
    target: str = "Attrition",
    model_name: str = "best_attrition_model"
):
    df = pd.read_csv(data_path)

    setup(
        data=df,
        target=target,
        session_id=42,
        verbose=False
    )

    best_model = compare_models()
    final_model = finalize_model(best_model)
    save_model(final_model, model_name)

    print("\n=== MODEL METRICS ===")
    print(pull())


@app.command()
def predict(
    model_name: str,
    input_path: str
):
    model = load_model(model_name)
    df = pd.read_csv(input_path)

    preds = predict_model(model, data=df)
    print(preds[["prediction_label", "prediction_score"]])


if __name__ == "__main__":
    app()
