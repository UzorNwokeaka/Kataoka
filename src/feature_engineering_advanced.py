import pandas as pd
import numpy as np
from pathlib import Path

INPUT_PATH = Path("data/processed/rul_basic_features.csv")
OUTPUT_PATH = Path("data/processed/rul_advanced_features.csv")

df = pd.read_csv(INPUT_PATH)

print("Loaded basic feature dataset:", df.shape)

date_cols = [
    "timestamp",
    "last_maintenance_time",
    "next_failure_time",
    "installation_date"
]

for col in date_cols:
    if col in df.columns:
        df[col] = pd.to_datetime(df[col], errors="coerce")

df = df.sort_values(["robot_id", "timestamp"]).reset_index(drop=True)

sensor_cols = [
    "vibration_level",
    "motor_temperature",
    "torque_load",
    "power_consumption"
]

for col in sensor_cols:
    if col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")

# -----------------------------
# Helper: rolling slope
# -----------------------------
def rolling_slope(values):
    values = np.array(values)

    if len(values) < 2:
        return 0

    x = np.arange(len(values))

    if np.isnan(values).any():
        values = pd.Series(values).ffill().bfill().fillna(0).values

    slope = np.polyfit(x, values, 1)[0]
    return slope

windows = {
    "24h": 4,
    "72h": 12,
    "7d": 28
}

# -----------------------------
# 1. Wear trend slope features
# -----------------------------
for col in sensor_cols:
    if col in df.columns:
        for label, window in windows.items():
            df[f"{col}_slope_{label}"] = (
                df.groupby("robot_id")[col]
                .transform(
                    lambda x: x.rolling(window=window, min_periods=2)
                    .apply(rolling_slope, raw=False)
                )
            )

# -----------------------------
# 2. Load stress index: torque × runtime
# -----------------------------
df["load_stress_index"] = (
    df["torque_load"] * df["cumulative_runtime_hours"]
)

df["vibration_runtime_wear_index"] = (
    df["vibration_level"] * df["cumulative_runtime_hours"]
)

df["thermal_runtime_stress_index"] = (
    df["motor_temperature"] * df["cumulative_runtime_hours"]
)

# -----------------------------
# 3. Energy consumption trends
# -----------------------------
for label, window in windows.items():
    df[f"power_consumption_change_{label}"] = (
        df.groupby("robot_id")["power_consumption"]
        .transform(lambda x: x.diff(periods=window))
    )

    df[f"power_consumption_pct_change_{label}"] = (
        df.groupby("robot_id")["power_consumption"]
        .transform(lambda x: x.pct_change(periods=window))
    )

    df[f"energy_trend_slope_{label}"] = (
        df.groupby("robot_id")["power_consumption"]
        .transform(
            lambda x: x.rolling(window=window, min_periods=2)
            .apply(rolling_slope, raw=False)
        )
    )

df["cumulative_energy_consumption"] = (
    df.groupby("robot_id")["power_consumption"].cumsum()
)

df["energy_per_runtime_hour"] = (
    df["cumulative_energy_consumption"]
    / df["cumulative_runtime_hours"].replace(0, np.nan)
)

# -----------------------------
# 4. Early degradation indicators
# -----------------------------
df["vibration_high_flag"] = np.where(
    df["vibration_level"] > df["vibration_level"].quantile(0.75),
    1,
    0
)

df["temperature_high_flag"] = np.where(
    df["motor_temperature"] > df["motor_temperature"].quantile(0.75),
    1,
    0
)

df["torque_high_flag"] = np.where(
    df["torque_load"] > df["torque_load"].quantile(0.75),
    1,
    0
)

df["power_high_flag"] = np.where(
    df["power_consumption"] > df["power_consumption"].quantile(0.75),
    1,
    0
)

df["multi_sensor_stress_flag"] = np.where(
    (
        df["vibration_high_flag"]
        + df["temperature_high_flag"]
        + df["torque_high_flag"]
        + df["power_high_flag"]
    ) >= 2,
    1,
    0
)

df["severe_multi_sensor_stress_flag"] = np.where(
    (
        df["vibration_high_flag"]
        + df["temperature_high_flag"]
        + df["torque_high_flag"]
        + df["power_high_flag"]
    ) >= 3,
    1,
    0
)

# -----------------------------
# 5. Acceleration / instability features
# -----------------------------
for col in sensor_cols:
    df[f"{col}_first_diff"] = (
        df.groupby("robot_id")[col].diff()
    )

    df[f"{col}_second_diff"] = (
        df.groupby("robot_id")[f"{col}_first_diff"].diff()
    )

    df[f"{col}_volatility_72h"] = (
        df.groupby("robot_id")[col]
        .transform(lambda x: x.rolling(window=12, min_periods=2).std())
    )

# -----------------------------
# 6. Maintenance-adjusted degradation features
# -----------------------------
df["stress_since_last_maintenance"] = (
    df["torque_power_stress_index"]
    * df["time_since_last_maintenance_hours"]
)

df["vibration_since_last_maintenance"] = (
    df["vibration_level"]
    * df["time_since_last_maintenance_hours"]
)

df["temperature_since_last_maintenance"] = (
    df["motor_temperature"]
    * df["time_since_last_maintenance_hours"]
)

# -----------------------------
# 7. Normalised degradation pressure score
# -----------------------------
pressure_components = [
    "vibration_level",
    "motor_temperature",
    "torque_load",
    "power_consumption",
    "load_stress_index",
    "energy_per_runtime_hour"
]

for col in pressure_components:
    if col in df.columns:
        min_val = df[col].min()
        max_val = df[col].max()

        if max_val != min_val:
            df[f"{col}_norm"] = (df[col] - min_val) / (max_val - min_val)
        else:
            df[f"{col}_norm"] = 0

df["degradation_pressure_score"] = (
    df["vibration_level_norm"] * 0.25
    + df["motor_temperature_norm"] * 0.20
    + df["torque_load_norm"] * 0.20
    + df["power_consumption_norm"] * 0.15
    + df["load_stress_index_norm"] * 0.10
    + df["energy_per_runtime_hour_norm"] * 0.10
)

# -----------------------------
# 8. Target-safe warning labels for analysis only
# Do NOT use this as model input later
# -----------------------------
df["early_failure_zone"] = np.where(df["RUL_hours"] <= 72, 1, 0)
df["critical_failure_zone"] = np.where(df["RUL_hours"] <= 24, 1, 0)

# -----------------------------
# Cleanup
# -----------------------------
df = df.replace([np.inf, -np.inf], np.nan)

numeric_cols = df.select_dtypes(include=[np.number]).columns
categorical_cols = df.select_dtypes(include=["object", "category"]).columns

df[numeric_cols] = df[numeric_cols].fillna(0)
df[categorical_cols] = df[categorical_cols].fillna("unknown")

# -----------------------------
# Save
# -----------------------------
df.to_csv(OUTPUT_PATH, index=False)

print("Advanced feature engineering completed successfully.")
print("Output shape:", df.shape)
print("Total missing values:", df.isna().sum().sum())
print(f"Saved to: {OUTPUT_PATH}")

advanced_cols = [
    "load_stress_index",
    "vibration_runtime_wear_index",
    "thermal_runtime_stress_index",
    "cumulative_energy_consumption",
    "energy_per_runtime_hour",
    "multi_sensor_stress_flag",
    "severe_multi_sensor_stress_flag",
    "degradation_pressure_score",
    "early_failure_zone",
    "critical_failure_zone"
]

print("\nAdvanced feature examples:")
print(df[advanced_cols].head())