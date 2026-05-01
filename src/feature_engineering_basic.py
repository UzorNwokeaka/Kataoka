# Basic Feature Engineering for RUL Prediction
# This script takes the aligned timeseries dataset and creates basic features
# like rolling statistics, cumulative runtime, time since installation/maintenance,
# and simple stress indices. It also handles missing values and validates the output.

import pandas as pd
import numpy as np
from pathlib import Path

# -----------------------------
# Paths
# -----------------------------
INPUT_PATH = Path("data/processed/rul_aligned_timeseries.csv")
OUTPUT_PATH = Path("data/processed/rul_basic_features.csv")

# -----------------------------
# Load aligned dataset
# -----------------------------
df = pd.read_csv(INPUT_PATH)

print("Loaded aligned dataset:", df.shape)

# -----------------------------
# Convert datetime columns
# -----------------------------
date_cols = [
    "timestamp",
    "last_maintenance_time",
    "next_failure_time",
    "installation_date",
]

for col in date_cols:
    if col in df.columns:
        df[col] = pd.to_datetime(df[col], errors="coerce")

# -----------------------------
# Sort by robot timeline
# -----------------------------
df = df.sort_values(["robot_id", "timestamp"]).reset_index(drop=True)

# -----------------------------
# Sensor columns
# -----------------------------
sensor_cols = [
    "vibration_level",
    "motor_temperature",
    "torque_load",
    "power_consumption",
]

# Ensure sensor columns are numeric
for col in sensor_cols:
    if col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")

# -----------------------------
# Basic rolling degradation features
# Assumption: readings are every 6 hours
# 4 readings = 24 hours
# 12 readings = 72 hours
# 28 readings = 7 days
# -----------------------------
windows = {"24h": 4, "72h": 12, "7d": 28}

for col in sensor_cols:
    if col in df.columns:
        for label, window in windows.items():
            df[f"{col}_rolling_mean_{label}"] = df.groupby("robot_id")[col].transform(
                lambda x: x.rolling(window=window, min_periods=1).mean()
            )

            df[f"{col}_rolling_std_{label}"] = df.groupby("robot_id")[col].transform(
                lambda x: x.rolling(window=window, min_periods=1).std()
            )

            df[f"{col}_rolling_min_{label}"] = df.groupby("robot_id")[col].transform(
                lambda x: x.rolling(window=window, min_periods=1).min()
            )

            df[f"{col}_rolling_max_{label}"] = df.groupby("robot_id")[col].transform(
                lambda x: x.rolling(window=window, min_periods=1).max()
            )

# -----------------------------
# Cumulative runtime per robot
# Since readings are every 6 hours, each row adds 6 runtime hours
# -----------------------------
df["reading_sequence"] = df.groupby("robot_id").cumcount() + 1
df["cumulative_runtime_hours"] = df["reading_sequence"] * 6

# -----------------------------
# Time since installation
# -----------------------------
if "installation_date" in df.columns and "timestamp" in df.columns:
    df["time_since_installation_hours"] = (
        df["timestamp"] - df["installation_date"]
    ).dt.total_seconds() / 3600
else:
    df["time_since_installation_hours"] = 0

# -----------------------------
# Time since last maintenance
# Preserve existing column if already created during alignment
# Otherwise calculate it
# -----------------------------
if "time_since_last_maintenance_hours" not in df.columns:
    if "last_maintenance_time" in df.columns and "timestamp" in df.columns:
        df["time_since_last_maintenance_hours"] = (
            df["timestamp"] - df["last_maintenance_time"]
        ).dt.total_seconds() / 3600
    else:
        df["time_since_last_maintenance_hours"] = 0

df["time_since_last_maintenance_hours"] = pd.to_numeric(
    df["time_since_last_maintenance_hours"], errors="coerce"
)

# -----------------------------
# Basic stress indicators
# -----------------------------
df["torque_power_stress_index"] = df["torque_load"] * df["power_consumption"]

df["thermal_power_stress_index"] = df["motor_temperature"] * df["power_consumption"]

df["vibration_torque_stress_index"] = df["vibration_level"] * df["torque_load"]

df["combined_basic_stress_index"] = (
    df["vibration_level"] * 0.30
    + df["motor_temperature"] * 0.25
    + df["torque_load"] * 0.25
    + df["power_consumption"] * 0.20
)

# -----------------------------
# Maintenance flags
# -----------------------------
if "maintenance_count_to_date" in df.columns:
    df["maintenance_count_to_date"] = pd.to_numeric(
        df["maintenance_count_to_date"], errors="coerce"
    )
else:
    df["maintenance_count_to_date"] = 0

df["has_previous_maintenance"] = np.where(df["maintenance_count_to_date"] > 0, 1, 0)

df["recent_maintenance_24h"] = np.where(
    (df["time_since_last_maintenance_hours"] > 0)
    & (df["time_since_last_maintenance_hours"] <= 24),
    1,
    0,
)

df["recent_maintenance_72h"] = np.where(
    (df["time_since_last_maintenance_hours"] > 0)
    & (df["time_since_last_maintenance_hours"] <= 72),
    1,
    0,
)

# -----------------------------
# RUL health bands for analysis only
# Do NOT use this as a model input later because it is derived from target
# -----------------------------
if "RUL_hours" in df.columns:
    df["RUL_hours"] = pd.to_numeric(df["RUL_hours"], errors="coerce")

    df["rul_health_band"] = pd.cut(
        df["RUL_hours"],
        bins=[-np.inf, 24, 72, 168, np.inf],
        labels=["Critical", "Warning", "Watch", "Healthy"],
    )

    df["rul_health_band"] = df["rul_health_band"].astype(str)
else:
    raise ValueError("RUL_hours column is missing from the aligned dataset.")

# -----------------------------
# Final cleanup - type-aware missing value handling
# -----------------------------
df = df.replace([np.inf, -np.inf], np.nan)

numeric_cols = df.select_dtypes(include=[np.number]).columns
categorical_cols = df.select_dtypes(include=["object", "category"]).columns

df[numeric_cols] = df[numeric_cols].fillna(0)
df[categorical_cols] = df[categorical_cols].fillna("unknown")

# -----------------------------
# Validate output
# -----------------------------
print("\nBasic feature engineering completed successfully.")
print("Output shape:", df.shape)
print("Total missing values:", df.isna().sum().sum())

new_cols = [
    "cumulative_runtime_hours",
    "time_since_installation_hours",
    "time_since_last_maintenance_hours",
    "torque_power_stress_index",
    "thermal_power_stress_index",
    "vibration_torque_stress_index",
    "combined_basic_stress_index",
    "has_previous_maintenance",
    "recent_maintenance_24h",
    "recent_maintenance_72h",
    "rul_health_band",
]

available_new_cols = [col for col in new_cols if col in df.columns]

print("\nNew feature examples:")
print(df[available_new_cols].head())

# -----------------------------
# Save output
# -----------------------------
df.to_csv(OUTPUT_PATH, index=False)

print(f"\nSaved to: {OUTPUT_PATH}")
