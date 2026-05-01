# import warnings
# warnings.filterwarnings("ignore")

# from pathlib import Path
# import joblib
# import mlflow
# import mlflow.sklearn

# import pandas as pd
# import numpy as np

# from sklearn.model_selection import TimeSeriesSplit, cross_validate
# from sklearn.pipeline import Pipeline
# from sklearn.compose import ColumnTransformer
# from sklearn.preprocessing import StandardScaler, OneHotEncoder
# from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
# from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


# # -----------------------------
# # Settings
# # -----------------------------
# FAST_MODE = True  # True = quick development run, False = full final training

# #INPUT_PATH = Path("data/processed/rul_advanced_features.csv")
# INPUT_PATH = Path("data/processed/rul_fixed_lifecycle_dataset.csv")
# MODEL_DIR = Path("models")
# MODEL_DIR.mkdir(parents=True, exist_ok=True)

# BEST_MODEL_PATH = MODEL_DIR / "best_rul_model.pkl"
# RESULTS_PATH = MODEL_DIR / "model_results.csv"

# MLFLOW_DIR = Path("mlruns")
# mlflow.set_tracking_uri(f"file:{MLFLOW_DIR.resolve()}")
# mlflow.set_experiment("Kataoka_RUL_Prediction")


# # -----------------------------
# # Load dataset
# # -----------------------------
# df = pd.read_csv(INPUT_PATH)
# print("Loaded dataset:", df.shape)

# # -----------------------------
# # Optional fast sampling
# # -----------------------------
# if FAST_MODE:
#     print("FAST_MODE enabled: using reduced training configuration.")
#     if len(df) > 20000:
#         df = df.sample(n=20000, random_state=42).sort_index().reset_index(drop=True)
#         print("Sampled dataset:", df.shape)


# # -----------------------------
# # Sort by time
# # -----------------------------
# if "timestamp" in df.columns:
#     df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
#     df = df.sort_values(["timestamp", "robot_id"]).reset_index(drop=True)
# else:
#     df = df.reset_index(drop=True)


# # -----------------------------
# # Target
# # -----------------------------
# TARGET = "RUL_hours"

# if TARGET not in df.columns:
#     raise ValueError("RUL_hours target column not found.")


# # -----------------------------
# # Remove leakage and identifier columns
# # -----------------------------
# leakage_cols = [
#     "RUL_hours",
#     "next_failure_time",
#     "next_failure_type",
#     "next_failure_root_cause",
#     "failure_time",
#     "failure_type",
#     "root_cause",
#     "rul_health_band",
#     "early_failure_zone",
#     "critical_failure_zone",
#     "is_simulated"
# ]

# identifier_cols = [
#     "reading_id",
#     "robot_id",
#     "timestamp",
#     "last_maintenance_time",
#     "installation_date"
# ]

# drop_cols = [col for col in leakage_cols + identifier_cols if col in df.columns]

# X = df.drop(columns=drop_cols, errors="ignore")
# y = df[TARGET]

# X = X.replace([np.inf, -np.inf], np.nan)

# numeric_features = X.select_dtypes(include=[np.number]).columns.tolist()
# categorical_features = X.select_dtypes(include=["object", "category", "bool"]).columns.tolist()

# X[numeric_features] = X[numeric_features].fillna(0)

# for col in categorical_features:
#     X[col] = X[col].astype(str).fillna("unknown")

# print("Features used:", X.shape[1])
# print("Numeric features:", len(numeric_features))
# print("Categorical features:", len(categorical_features))


# # -----------------------------
# # Time-based holdout split
# # -----------------------------
# split_index = int(len(X) * 0.8)

# X_train = X.iloc[:split_index]
# X_test = X.iloc[split_index:]

# y_train = y.iloc[:split_index]
# y_test = y.iloc[split_index:]

# print("Train shape:", X_train.shape)
# print("Test shape:", X_test.shape)


# # -----------------------------
# # Preprocessor
# # -----------------------------
# preprocessor = ColumnTransformer(
#     transformers=[
#         ("num", StandardScaler(), numeric_features),
#         ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_features)
#     ],
#     remainder="drop"
# )


# # -----------------------------
# # Models
# # -----------------------------
# if FAST_MODE:
#     models = {
#         "RandomForest_FAST": RandomForestRegressor(
#             n_estimators=50,
#             max_depth=10,
#             min_samples_leaf=10,
#             random_state=42,
#             n_jobs=-1
#         ),
#         "GradientBoosting_FAST": GradientBoostingRegressor(
#             n_estimators=80,
#             learning_rate=0.1,
#             max_depth=3,
#             min_samples_leaf=10,
#             random_state=42
#         )
#     }

#     n_splits = 3

# else:
#     models = {
#         "RandomForest_FULL": RandomForestRegressor(
#             n_estimators=200,
#             max_depth=12,
#             min_samples_leaf=8,
#             random_state=42,
#             n_jobs=-1
#         ),
#         "GradientBoosting_FULL": GradientBoostingRegressor(
#             n_estimators=300,
#             learning_rate=0.05,
#             max_depth=4,
#             min_samples_leaf=8,
#             random_state=42
#         )
#     }

#     n_splits = 5


# # -----------------------------
# # Evaluation function
# # -----------------------------
# def evaluate_model(y_true, y_pred):
#     mae = mean_absolute_error(y_true, y_pred)
#     rmse = np.sqrt(mean_squared_error(y_true, y_pred))
#     r2 = r2_score(y_true, y_pred)

#     return {
#         "MAE": mae,
#         "RMSE": rmse,
#         "R2": r2
#     }


# # -----------------------------
# # Train and track experiments
# # -----------------------------
# best_model = None
# best_model_name = None
# best_mae = float("inf")
# results = []

# tscv = TimeSeriesSplit(n_splits=n_splits)

# for model_name, model in models.items():
#     print(f"\nTraining model: {model_name}")

#     pipeline = Pipeline(
#         steps=[
#             ("preprocessor", preprocessor),
#             ("model", model)
#         ]
#     )

#     with mlflow.start_run(run_name=model_name):
#         cv_scores = cross_validate(
#             pipeline,
#             X_train,
#             y_train,
#             cv=tscv,
#             scoring={
#                 "mae": "neg_mean_absolute_error",
#                 "rmse": "neg_root_mean_squared_error",
#                 "r2": "r2"
#             },
#             n_jobs=-1,
#             return_train_score=True
#         )

#         cv_mae = -cv_scores["test_mae"].mean()
#         cv_rmse = -cv_scores["test_rmse"].mean()
#         cv_r2 = cv_scores["test_r2"].mean()

#         pipeline.fit(X_train, y_train)

#         train_preds = pipeline.predict(X_train)
#         test_preds = pipeline.predict(X_test)

#         train_metrics = evaluate_model(y_train, train_preds)
#         test_metrics = evaluate_model(y_test, test_preds)

#         mlflow.log_param("mode", "FAST" if FAST_MODE else "FULL")
#         mlflow.log_param("model_name", model_name)
#         mlflow.log_param("time_series_cv_splits", n_splits)
#         mlflow.log_param("train_rows", len(X_train))
#         mlflow.log_param("test_rows", len(X_test))
#         mlflow.log_param("num_features", len(numeric_features))
#         mlflow.log_param("cat_features", len(categorical_features))

#         mlflow.log_metric("cv_mae", cv_mae)
#         mlflow.log_metric("cv_rmse", cv_rmse)
#         mlflow.log_metric("cv_r2", cv_r2)

#         mlflow.log_metric("train_mae", train_metrics["MAE"])
#         mlflow.log_metric("train_rmse", train_metrics["RMSE"])
#         mlflow.log_metric("train_r2", train_metrics["R2"])

#         mlflow.log_metric("test_mae", test_metrics["MAE"])
#         mlflow.log_metric("test_rmse", test_metrics["RMSE"])
#         mlflow.log_metric("test_r2", test_metrics["R2"])

#         mlflow.sklearn.log_model(pipeline, name="model")

#         results.append({
#             "mode": "FAST" if FAST_MODE else "FULL",
#             "model": model_name,
#             "cv_mae": cv_mae,
#             "cv_rmse": cv_rmse,
#             "cv_r2": cv_r2,
#             "train_mae": train_metrics["MAE"],
#             "train_rmse": train_metrics["RMSE"],
#             "train_r2": train_metrics["R2"],
#             "test_mae": test_metrics["MAE"],
#             "test_rmse": test_metrics["RMSE"],
#             "test_r2": test_metrics["R2"]
#         })

#         print("CV MAE:", round(cv_mae, 3))
#         print("CV RMSE:", round(cv_rmse, 3))
#         print("CV R2:", round(cv_r2, 3))
#         print("Train MAE:", round(train_metrics["MAE"], 3))
#         print("Train R2:", round(train_metrics["R2"], 3))
#         print("Test MAE:", round(test_metrics["MAE"], 3))
#         print("Test RMSE:", round(test_metrics["RMSE"], 3))
#         print("Test R2:", round(test_metrics["R2"], 3))

#         if test_metrics["MAE"] < best_mae:
#             best_mae = test_metrics["MAE"]
#             best_model = pipeline
#             best_model_name = model_name


# # -----------------------------
# # Save outputs
# # -----------------------------
# joblib.dump(best_model, BEST_MODEL_PATH)

# results_df = pd.DataFrame(results)
# results_df.to_csv(RESULTS_PATH, index=False)

# print("\nTraining completed successfully.")
# print("Best model:", best_model_name)
# print("Best test MAE:", round(best_mae, 3))
# print(f"Best model saved to: {BEST_MODEL_PATH}")
# print(f"Results saved to: {RESULTS_PATH}")
# print("\nModel results:")
# print(results_df)


import warnings

warnings.filterwarnings("ignore")

from pathlib import Path
import joblib
import mlflow
import mlflow.sklearn

import pandas as pd
import numpy as np

from sklearn.model_selection import TimeSeriesSplit, cross_validate
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer, TransformedTargetRegressor
from sklearn.preprocessing import StandardScaler, OneHotEncoder, FunctionTransformer
from sklearn.ensemble import (
    RandomForestRegressor,
    GradientBoostingRegressor,
    ExtraTreesRegressor,
)
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# ============================================================
# SETTINGS
# ============================================================
FAST_MODE = True  # True = quick run, False = final full training
USE_LOG_TARGET = True  # Helps stabilise RUL prediction

INPUT_PATH = Path("data/processed/rul_fixed_lifecycle_dataset.csv")

MODEL_DIR = Path("models")
MODEL_DIR.mkdir(parents=True, exist_ok=True)

BEST_MODEL_PATH = MODEL_DIR / "best_rul_model.pkl"
RESULTS_PATH = MODEL_DIR / "model_results.csv"
FEATURE_IMPORTANCE_PATH = MODEL_DIR / "feature_importance.csv"

MLFLOW_DIR = Path("mlruns")
mlflow.set_tracking_uri(f"file:{MLFLOW_DIR.resolve()}")
mlflow.set_experiment("Kataoka_RUL_Prediction")


# ============================================================
# LOAD DATA
# ============================================================
df = pd.read_csv(INPUT_PATH)
print("Loaded dataset:", df.shape)

if "timestamp" in df.columns:
    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    df = df.sort_values(["timestamp", "robot_id"]).reset_index(drop=True)
else:
    df = df.reset_index(drop=True)

if FAST_MODE:
    print("FAST_MODE enabled.")
    if len(df) > 20000:
        # Keep chronological order; do NOT random sample time-series data
        df = df.iloc[:20000].reset_index(drop=True)
        print("Using first 20,000 chronological rows:", df.shape)


# ============================================================
# TARGET
# ============================================================
TARGET = "RUL_hours"

if TARGET not in df.columns:
    raise ValueError("RUL_hours target column not found.")

# Cap only extreme long-horizon values to reduce saturation
df[TARGET] = pd.to_numeric(df[TARGET], errors="coerce")
df = df.dropna(subset=[TARGET])
df = df[df[TARGET] > 0]

# Optional: cap at 95th percentile instead of hard 2000 if needed
# cap_value = df[TARGET].quantile(0.95)
# df[TARGET] = df[TARGET].clip(upper=cap_value)


# ============================================================
# DROP LEAKAGE / IDENTIFIER COLUMNS
# ============================================================
leakage_cols = [
    "RUL_hours",
    "next_failure_time",
    "next_failure_type",
    "next_failure_root_cause",
    "failure_time",
    "failure_type",
    "root_cause",
    "rul_health_band",
    "early_failure_zone",
    "critical_failure_zone",
    "is_simulated",
    "remaining_lifecycle_ratio",  # too directly related to target
]

identifier_cols = [
    "reading_id",
    "robot_id",
    "timestamp",
    "last_maintenance_time",
    "installation_date",
]

drop_cols = [c for c in leakage_cols + identifier_cols if c in df.columns]

X = df.drop(columns=drop_cols, errors="ignore")
y = df[TARGET]

X = X.replace([np.inf, -np.inf], np.nan)

numeric_features = X.select_dtypes(include=[np.number]).columns.tolist()
categorical_features = X.select_dtypes(
    include=["object", "category", "bool"]
).columns.tolist()

X[numeric_features] = X[numeric_features].fillna(0)

for col in categorical_features:
    X[col] = X[col].astype(str).fillna("unknown")

print("Features used:", X.shape[1])
print("Numeric features:", len(numeric_features))
print("Categorical features:", len(categorical_features))


# ============================================================
# TIME-BASED HOLDOUT SPLIT
# ============================================================
split_index = int(len(X) * 0.8)

X_train = X.iloc[:split_index]
X_test = X.iloc[split_index:]
y_train = y.iloc[:split_index]
y_test = y.iloc[split_index:]

print("Train shape:", X_train.shape)
print("Test shape:", X_test.shape)


# ============================================================
# PREPROCESSOR
# ============================================================
preprocessor = ColumnTransformer(
    transformers=[
        ("num", StandardScaler(), numeric_features),
        ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_features),
    ],
    remainder="drop",
)


# ============================================================
# MODELS
# ============================================================
if FAST_MODE:
    n_splits = 3

    base_models = {
        "RandomForest_OPT_FAST": RandomForestRegressor(
            n_estimators=80,
            max_depth=10,
            min_samples_leaf=12,
            min_samples_split=20,
            max_features="sqrt",
            random_state=42,
            n_jobs=-1,
        ),
        "GradientBoosting_OPT_FAST": GradientBoostingRegressor(
            n_estimators=120,
            learning_rate=0.06,
            max_depth=3,
            min_samples_leaf=12,
            min_samples_split=20,
            subsample=0.85,
            random_state=42,
        ),
        "ExtraTrees_OPT_FAST": ExtraTreesRegressor(
            n_estimators=120,
            max_depth=12,
            min_samples_leaf=10,
            min_samples_split=20,
            max_features="sqrt",
            random_state=42,
            n_jobs=-1,
        ),
    }

else:
    n_splits = 5

    base_models = {
        "RandomForest_OPT_FULL": RandomForestRegressor(
            n_estimators=300,
            max_depth=14,
            min_samples_leaf=8,
            min_samples_split=16,
            max_features="sqrt",
            random_state=42,
            n_jobs=-1,
        ),
        "GradientBoosting_OPT_FULL": GradientBoostingRegressor(
            n_estimators=450,
            learning_rate=0.035,
            max_depth=3,
            min_samples_leaf=10,
            min_samples_split=20,
            subsample=0.85,
            random_state=42,
        ),
        "ExtraTrees_OPT_FULL": ExtraTreesRegressor(
            n_estimators=350,
            max_depth=16,
            min_samples_leaf=6,
            min_samples_split=12,
            max_features="sqrt",
            random_state=42,
            n_jobs=-1,
        ),
    }


def maybe_wrap_target(model):
    if not USE_LOG_TARGET:
        return model

    return TransformedTargetRegressor(
        regressor=model, func=np.log1p, inverse_func=np.expm1
    )


models = {name: maybe_wrap_target(model) for name, model in base_models.items()}


# ============================================================
# METRICS
# ============================================================
def evaluate_model(y_true, y_pred):
    y_pred = np.maximum(y_pred, 0)

    return {
        "MAE": mean_absolute_error(y_true, y_pred),
        "RMSE": np.sqrt(mean_squared_error(y_true, y_pred)),
        "R2": r2_score(y_true, y_pred),
    }


# ============================================================
# TRAINING
# ============================================================
best_model = None
best_model_name = None
best_mae = float("inf")
best_r2 = -float("inf")
results = []

tscv = TimeSeriesSplit(n_splits=n_splits)

for model_name, model in models.items():
    print(f"\nTraining model: {model_name}")

    pipeline = Pipeline(steps=[("preprocessor", preprocessor), ("model", model)])

    with mlflow.start_run(run_name=model_name):
        cv_scores = cross_validate(
            pipeline,
            X_train,
            y_train,
            cv=tscv,
            scoring={
                "mae": "neg_mean_absolute_error",
                "rmse": "neg_root_mean_squared_error",
                "r2": "r2",
            },
            n_jobs=-1,
            return_train_score=True,
        )

        cv_mae = -cv_scores["test_mae"].mean()
        cv_rmse = -cv_scores["test_rmse"].mean()
        cv_r2 = cv_scores["test_r2"].mean()

        pipeline.fit(X_train, y_train)

        train_preds = pipeline.predict(X_train)
        test_preds = pipeline.predict(X_test)

        train_metrics = evaluate_model(y_train, train_preds)
        test_metrics = evaluate_model(y_test, test_preds)

        mlflow.log_param("mode", "FAST" if FAST_MODE else "FULL")
        mlflow.log_param("use_log_target", USE_LOG_TARGET)
        mlflow.log_param("model_name", model_name)
        mlflow.log_param("time_series_cv_splits", n_splits)
        mlflow.log_param("train_rows", len(X_train))
        mlflow.log_param("test_rows", len(X_test))
        mlflow.log_param("num_features", len(numeric_features))
        mlflow.log_param("cat_features", len(categorical_features))

        mlflow.log_metric("cv_mae", cv_mae)
        mlflow.log_metric("cv_rmse", cv_rmse)
        mlflow.log_metric("cv_r2", cv_r2)

        mlflow.log_metric("train_mae", train_metrics["MAE"])
        mlflow.log_metric("train_rmse", train_metrics["RMSE"])
        mlflow.log_metric("train_r2", train_metrics["R2"])

        mlflow.log_metric("test_mae", test_metrics["MAE"])
        mlflow.log_metric("test_rmse", test_metrics["RMSE"])
        mlflow.log_metric("test_r2", test_metrics["R2"])

        mlflow.sklearn.log_model(pipeline, name="model")

        results.append(
            {
                "mode": "FAST" if FAST_MODE else "FULL",
                "use_log_target": USE_LOG_TARGET,
                "model": model_name,
                "cv_mae": cv_mae,
                "cv_rmse": cv_rmse,
                "cv_r2": cv_r2,
                "train_mae": train_metrics["MAE"],
                "train_rmse": train_metrics["RMSE"],
                "train_r2": train_metrics["R2"],
                "test_mae": test_metrics["MAE"],
                "test_rmse": test_metrics["RMSE"],
                "test_r2": test_metrics["R2"],
            }
        )

        print("CV MAE:", round(cv_mae, 3))
        print("CV RMSE:", round(cv_rmse, 3))
        print("CV R2:", round(cv_r2, 3))
        print("Train MAE:", round(train_metrics["MAE"], 3))
        print("Train R2:", round(train_metrics["R2"], 3))
        print("Test MAE:", round(test_metrics["MAE"], 3))
        print("Test RMSE:", round(test_metrics["RMSE"], 3))
        print("Test R2:", round(test_metrics["R2"], 3))

        # Prefer better R2, then lower MAE
        if (test_metrics["R2"] > best_r2) or (
            test_metrics["R2"] == best_r2 and test_metrics["MAE"] < best_mae
        ):
            best_r2 = test_metrics["R2"]
            best_mae = test_metrics["MAE"]
            best_model = pipeline
            best_model_name = model_name


# ============================================================
# SAVE BEST MODEL AND RESULTS
# ============================================================
joblib.dump(best_model, BEST_MODEL_PATH)

results_df = pd.DataFrame(results)
results_df.to_csv(RESULTS_PATH, index=False)

print("\nTraining completed successfully.")
print("Best model:", best_model_name)
print("Best test R2:", round(best_r2, 3))
print("Best test MAE:", round(best_mae, 3))
print(f"Best model saved to: {BEST_MODEL_PATH}")
print(f"Results saved to: {RESULTS_PATH}")


# ============================================================
# FEATURE IMPORTANCE
# ============================================================
try:
    fitted_preprocessor = best_model.named_steps["preprocessor"]
    fitted_model = best_model.named_steps["model"]

    feature_names = fitted_preprocessor.get_feature_names_out()

    # If target transformer is used, access underlying regressor
    if isinstance(fitted_model, TransformedTargetRegressor):
        estimator = fitted_model.regressor_
    else:
        estimator = fitted_model

    if hasattr(estimator, "feature_importances_"):
        importances = estimator.feature_importances_

        importance_df = pd.DataFrame(
            {"feature": feature_names, "importance": importances}
        ).sort_values("importance", ascending=False)

        importance_df.to_csv(FEATURE_IMPORTANCE_PATH, index=False)

        print(f"Feature importance saved to: {FEATURE_IMPORTANCE_PATH}")
        print("\nTop 20 important features:")
        print(importance_df.head(20))
    else:
        print("Best model does not expose feature_importances_.")

except Exception as e:
    print("Feature importance extraction failed:", e)


print("\nModel results:")
print(results_df)
