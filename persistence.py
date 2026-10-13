import json
import os
from datetime import datetime

import joblib
import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder, FunctionTransformer
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import HistGradientBoostingRegressor

RANDOM_STATE = 42
URL = "https://raw.githubusercontent.com/ageron/handson-ml2/master/datasets/housing/housing.csv"

EXPECTED_COLUMNS = ["longitude","latitude","housing_median_age","total_rooms","total_bedrooms","population","households","median_income","ocean_proximity"]
EXPECTED_DTYPES = {
    "longitude":"float64","latitude":"float64","housing_median_age":"float64",
    "total_rooms":"float64","total_bedrooms":"float64","population":"float64",
    "households":"float64","median_income":"float64","ocean_proximity":"object"
}
POST_NUMERIC = ["longitude","latitude","housing_median_age","total_rooms","total_bedrooms",
                "population","households","median_income","rooms_per_household",
                "bedrooms_per_room","population_per_household"]

def load_data():
    df = pd.read_csv(URL)
    df["total_bedrooms"] = df["total_bedrooms"].fillna(df["total_bedrooms"].median())
    X = df.drop(columns=["median_house_value"])
    y = df["median_house_value"]
    return train_test_split(X, y, test_size=0.2, random_state=RANDOM_STATE)

def add_ratios_and_log(X_df):
    out = X_df.copy()
    out["rooms_per_household"] = out["total_rooms"] / out["households"]
    out["bedrooms_per_room"] = out["total_bedrooms"] / out["total_rooms"]
    out["population_per_household"] = out["population"] / out["households"]
    for col in ["total_rooms","total_bedrooms","population","households"]:
        out[col] = np.log1p(out[col])
    return out

def build_production_pipeline():
    return Pipeline([
        ("engineer", FunctionTransformer(add_ratios_and_log, validate=False)),
        ("pre", ColumnTransformer([
            ("num", StandardScaler(), POST_NUMERIC),
            ("cat", OneHotEncoder(handle_unknown="ignore"), ["ocean_proximity"])
        ])),
        ("model", HistGradientBoostingRegressor(
            learning_rate=0.05, max_iter=300, max_leaf_nodes=31,
            random_state=RANDOM_STATE
        ))
    ])

def validate_input(df):
    missing = set(EXPECTED_COLUMNS) - set(df.columns)
    if missing:
        raise ValueError(f"missing columns: {sorted(missing)}")
    extra = set(df.columns) - set(EXPECTED_COLUMNS)
    if extra:
        raise ValueError(f"unexpected columns: {sorted(extra)}")
    for col, expected in EXPECTED_DTYPES.items():
        actual = str(df[col].dtype)
        if not actual.startswith(expected.split("64")[0]):
            raise ValueError(f"column {col!r}: expected dtype {expected}, got {actual}")

def predict_house_value(df, model_path="model.joblib", metadata_path="model.metadata.json"):
    with open(metadata_path, encoding="utf-8") as f:
        meta = json.load(f)
    expected = meta["expected_columns"]
    missing = set(expected) - set(df.columns)
    if missing:
        raise ValueError(f"missing columns: {sorted(missing)}")
    model = joblib.load(model_path)
    return model.predict(df[expected])

def main():
    X_train, X_test, y_train, y_test = load_data()

    print("=== Exercise 0 ===")
    print(f"joblib: {joblib.__version__}")
    print(f"sklearn: {__import__('sklearn').__version__}")
    print(f"mlflow: {mlflow.__version__}")

    production_pipe = build_production_pipeline()
    production_pipe.fit(X_train, y_train)
    test_r2 = production_pipe.score(X_test, y_test)
    print(f"production pipeline test R²: {test_r2:.4f}")

    print("\n=== Exercise 1 ===")
    joblib.dump(production_pipe, "model.joblib")
    print(f"file exists: {os.path.exists('model.joblib')}")
    print(f"file size: {os.path.getsize('model.joblib'):,} bytes")
    loaded = joblib.load("model.joblib")
    original_preds = production_pipe.predict(X_test.head(10))
    loaded_preds = loaded.predict(X_test.head(10))
    print(f"original predictions: {original_preds.round(2)}")
    print(f"loaded predictions:   {loaded_preds.round(2)}")
    print(f"identical: {np.allclose(original_preds, loaded_preds)}")
    print(f"original test R²: {test_r2:.4f}")
    print(f"loaded test R²:   {loaded.score(X_test, y_test):.4f}")

    print("\n=== Exercise 2 ===")
    baseline = Pipeline([
        ("engineer", FunctionTransformer(add_ratios_and_log, validate=False)),
        ("pre", ColumnTransformer([
            ("num", StandardScaler(), POST_NUMERIC),
            ("cat", OneHotEncoder(handle_unknown="ignore"), ["ocean_proximity"])
        ])),
        ("model", LinearRegression())
    ])
    baseline.fit(X_train, y_train)
    joblib.dump(baseline, "baseline.joblib")
    bs, ms = os.path.getsize("baseline.joblib"), os.path.getsize("model.joblib")
    print(f"baseline.joblib: {bs:,} bytes")
    print(f"model.joblib:    {ms:,} bytes")
    print(f"size ratio:      {ms/bs:.1f}x")
    print(f"baseline test R²: {joblib.load('baseline.joblib').score(X_test, y_test):.4f}")

    print("\n=== Exercise 3 ===")
    valid = X_test.head(3).copy()
    try:
        validate_input(valid); print("valid input: OK")
    except ValueError as e: print(f"valid input rejected: {e}")
    for label, bad in [
        ("missing-column", valid.drop(columns=["median_income"])),
        ("extra-column", valid.assign(extra_col=0.0)),
        ("wrong-dtype", valid.assign(median_income=valid["median_income"].astype(str)))
    ]:
        try:
            validate_input(bad); print(f"{label}: validator failed to catch")
        except ValueError as e: print(f"{label} rejected: {str(e)[:100]}")

    print("\n=== Exercise 4 ===")
    metadata = {
        "model_version":"v1.0.0",
        "training_date":datetime.now().isoformat(timespec="seconds"),
        "model_type":"HistGradientBoostingRegressor",
        "preprocessing":"log+ratios+ohe+standardize",
        "expected_columns":EXPECTED_COLUMNS,
        "expected_dtypes":EXPECTED_DTYPES,
        "metrics":{"test_r2":float(test_r2)},
        "training_rows":int(X_train.shape[0]),
        "test_rows":int(X_test.shape[0])
    }
    with open("model.metadata.json","w",encoding="utf-8") as f:
        json.dump(metadata,f,indent=2)
    print("metadata file:", os.path.exists("model.metadata.json"))
    with open("model.metadata.json",encoding="utf-8") as f:
        meta = json.load(f)
    print("loaded model_version:", meta["model_version"])
    print("loaded test_r2:", f'{meta["metrics"]["test_r2"]:.4f}')
    print("expected columns:", len(meta["expected_columns"]))

    print("\n=== Exercise 5 ===")
    try:
        mlflow.set_experiment("california-housing")
        with mlflow.start_run(run_name="hgb-production-v1") as run:
            mlflow.log_param("model_type","HistGradientBoostingRegressor")
            mlflow.log_param("learning_rate",0.05)
            mlflow.log_param("max_iter",300)
            mlflow.log_param("max_leaf_nodes",31)
            mlflow.log_metric("test_r2",float(test_r2))
            mlflow.sklearn.log_model(production_pipe,name="model")
            run_id = run.info.run_id
            print("AUDIT MARKER: experiment_name=california-housing")
            print(f"AUDIT MARKER: run_id={run_id}")
            print("AUDIT MARKER: run_name=hgb-production-v1")
        uri = f"runs:/{run_id}/model"
        mlflow_loaded = mlflow.sklearn.load_model(uri)
        print("identical to in-memory:", np.allclose(
            mlflow_loaded.predict(X_test.head(10)),
            production_pipe.predict(X_test.head(10))
        ))
        print(f"mlflow loaded test R²: {mlflow_loaded.score(X_test,y_test):.4f}")
    except Exception as e:
        print(f"MLflow skipped/failed: {type(e).__name__}: {e}")

    print("\n=== Exercise 6 ===")
    preds = predict_house_value(X_test.head(5))
    print(f"predictions: {preds.round(2)}")
    print(f"actual:      {y_test.head(5).values.round(2)}")
    try:
        predict_house_value(X_test.head(5).drop(columns=["median_income"]))
    except ValueError as e:
        print(f"caught: {e}")
    print("artifacts produced:")
    for name in ["model.joblib","baseline.joblib","model.metadata.json"]:
        if os.path.exists(name):
            print(f"  {name}: {os.path.getsize(name):,} bytes")
        else:
            print(f"  {name}: MISSING")

if __name__ == "__main__":
    main()
