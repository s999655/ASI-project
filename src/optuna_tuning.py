import optuna
import pandas as pd
import typer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder

app = typer.Typer()

def preprocess(df):
    for col in df.select_dtypes(include="object"):
        df[col] = LabelEncoder().fit_transform(df[col])
    return df

@app.command()
def tune(data_path: str, trials: int = 30):
    df = preprocess(pd.read_csv(data_path))

    X = df.drop(columns=["Attrition"])
    y = df["Attrition"]

    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    def objective(trial):
        model = RandomForestClassifier(
            n_estimators=trial.suggest_int("n_estimators", 50, 300),
            max_depth=trial.suggest_int("max_depth", 3, 20),
            random_state=42
        )
        model.fit(X_train, y_train)
        preds = model.predict(X_val)
        return accuracy_score(y_val, preds)

    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=trials)

    print("Best params:", study.best_params)
    print("Best accuracy:", study.best_value)

if __name__ == "__main__":
    app()
