import pandas as pd
import numpy as np
from pathlib import Path
import joblib

from sklearn.preprocessing import StandardScaler

# -----------------------------
# Paths
# -----------------------------
INPUT_PATH = Path("data/processed/rul_feature_table.csv")
OUTPUT_PATH = Path("data/processed/rul_cleaned.csv")
SCALER_PATH = Path("models/scaler.pkl")

SCALER_PATH.parent.mkdir(parents=True, exist_ok=True)

# -----------------------------
# Load data
# -----------------------------
df = pd.read_csv(INPUT_PATH)

print("Initial shape:", df.shape)

# -----------------------------
# Convert timestamps
# -----------------------------
date_cols = ["timestamp", "next_failure_time", "installation_date"]

for col in date_cols:
    if col in df.columns:
        df[col] = pd.to_datetime(df[col], errors="coerce")

# -----------------------------
# Sort data for time-series cleaning
# -----------------------------
df = df.sort_values(["robot_id", "timestamp"]).reset_index(drop=True)

# -----------------------------
# Missing value handling
# Forward-fill and backward-fill per robot
# -----------------------------
print("\nMissing values before cleaning:")
print(df.isna().sum().sort_values(ascending=False).head(20))

df = (
    df.groupby("robot_id", group_keys=False)
    .apply(lambda x: x.ffill().bfill())
    .reset_index(drop=True)
)

# Drop rows without target
df = df.dropna(subset=["RUL_hours"])

# -----------------------------
# Remove invalid target values
# -----------------------------
df = df[df["RUL_hours"] > 0]

# -----------------------------
# Remove impossible sensor values
# -----------------------------
sensor_cols = [
    "vibration_level",
    "motor_temperature",
    "torque_load",
    "power_consumption"
]

for col in sensor_cols:
    if col in df.columns:
        df = df[df[col].notna()]
        df = df[df[col] >= 0]

# -----------------------------
# Outlier handling using IQR capping
# Better than deleting rows because failure signals can be extreme
# -----------------------------
def cap_outliers_iqr(dataframe, column):
    Q1 = dataframe[column].quantile(0.25)
    Q3 = dataframe[column].quantile(0.75)
    IQR = Q3 - Q1

    lower = Q1 - 1.5 * IQR
    upper = Q3 + 1.5 * IQR

    dataframe[column] = dataframe[column].clip(lower=lower, upper=upper)
    return dataframe


outlier_cols = [
    "vibration_level",
    "motor_temperature",
    "torque_load",
    "power_consumption",
    "stress_index",
    "thermal_stress_index",
    "vibration_level_rolling_mean_24h",
    "motor_temperature_rolling_mean_24h",
    "torque_load_rolling_mean_24h",
    "power_consumption_rolling_mean_24h"
]

for col in outlier_cols:
    if col in df.columns:
        df = cap_outliers_iqr(df, col)

print("\nShape after invalid-value filtering and outlier capping:", df.shape)

# -----------------------------
# Feature selection
# -----------------------------
drop_cols = [
    "reading_id",
    "timestamp",
    "next_failure_time",
    "installation_date",
    "failure_id",
    "failure_time",
    "root_cause"
]

df = df.drop(columns=drop_cols, errors="ignore")

# -----------------------------
# Encode categorical variables
# -----------------------------
categorical_cols = df.select_dtypes(include=["object"]).columns.tolist()

# Keep robot_id for traceability? For modelling, encode/drop it.
# Here we drop robot_id to avoid memorising robot-specific behaviour.
if "robot_id" in categorical_cols:
    categorical_cols.remove("robot_id")

df = df.drop(columns=["robot_id"], errors="ignore")

categorical_cols = [
    col for col in categorical_cols
    if col in df.columns
]

df = pd.get_dummies(
    df,
    columns=categorical_cols,
    drop_first=True
)

# -----------------------------
# Separate target before scaling
# -----------------------------
target_col = "RUL_hours"

if target_col not in df.columns:
    raise ValueError("Target column RUL_hours not found.")

X = df.drop(columns=[target_col])
y = df[target_col]

# Convert booleans from get_dummies to integers
X = X.astype(float)

# -----------------------------
# Scale numerical features
# -----------------------------
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_scaled_df = pd.DataFrame(
    X_scaled,
    columns=X.columns
)

cleaned_df = pd.concat(
    [X_scaled_df, y.reset_index(drop=True)],
    axis=1
)

# -----------------------------
# Final validation
# -----------------------------
cleaned_df = cleaned_df.replace([np.inf, -np.inf], np.nan)
cleaned_df = cleaned_df.dropna()

print("\nFinal cleaned shape:", cleaned_df.shape)
print("Total missing values:", cleaned_df.isna().sum().sum())
print("\nTarget summary:")
print(cleaned_df["RUL_hours"].describe())

# -----------------------------
# Save outputs
# -----------------------------
cleaned_df.to_csv(OUTPUT_PATH, index=False)
joblib.dump(scaler, SCALER_PATH)

print("\nPreprocessing completed successfully.")
print(f"Cleaned dataset saved to: {OUTPUT_PATH}")
print(f"Scaler saved to: {SCALER_PATH}")