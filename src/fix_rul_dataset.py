import pandas as pd
import numpy as np
from pathlib import Path

np.random.seed(42)

# -----------------------------
# Paths
# -----------------------------
RAW_DIR = Path("data/raw")
PROCESSED_DIR = Path("data/processed")
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

SENSOR_PATH = RAW_DIR / "sensor_readings.csv"
ROBOT_PATH = RAW_DIR / "robot_units.csv"
MAINT_PATH = PROCESSED_DIR / "maintenance_logs_with_timestamps.csv"

OUTPUT_PATH = PROCESSED_DIR / "rul_fixed_lifecycle_dataset.csv"

# -----------------------------
# Load data
# -----------------------------
sensor = pd.read_csv(SENSOR_PATH)
robots = pd.read_csv(ROBOT_PATH)
maintenance = pd.read_csv(MAINT_PATH)

sensor["timestamp"] = pd.to_datetime(sensor["timestamp"], errors="coerce")
robots["installation_date"] = pd.to_datetime(
    robots["installation_date"], errors="coerce"
)
maintenance["maintenance_time"] = pd.to_datetime(
    maintenance["maintenance_time"], errors="coerce"
)

sensor = sensor.dropna(subset=["robot_id", "timestamp"])
sensor = sensor.sort_values(["robot_id", "timestamp"]).reset_index(drop=True)

print("Loaded sensor data:", sensor.shape)

# -----------------------------
# Create lifecycle-aware RUL
# -----------------------------
fixed_rows = []

for robot_id, robot_df in sensor.groupby("robot_id"):
    robot_df = robot_df.sort_values("timestamp").copy()
    n = len(robot_df)

    if n < 10:
        continue

    start_time = robot_df["timestamp"].min()
    end_time = robot_df["timestamp"].max()

    # Sensor frequency assumption: 6-hour readings
    observed_hours = (end_time - start_time).total_seconds() / 3600

    # Simulate failure after observed period but not too far away
    extra_life_hours = np.random.randint(72, 720)
    failure_time = end_time + pd.to_timedelta(extra_life_hours, unit="h")

    robot_df["next_failure_time"] = failure_time
    robot_df["RUL_hours"] = (
        robot_df["next_failure_time"] - robot_df["timestamp"]
    ).dt.total_seconds() / 3600

    # Cap RUL to a business-actionable horizon
    # Anything beyond 2000 hours is treated as long-term healthy
    robot_df["RUL_hours"] = robot_df["RUL_hours"].clip(upper=2000)

    # Lifecycle progression features
    robot_df["time_index"] = np.arange(n)
    robot_df["lifecycle_progress"] = robot_df["time_index"] / max(n - 1, 1)
    robot_df["remaining_lifecycle_ratio"] = 1 - robot_df["lifecycle_progress"]

    fixed_rows.append(robot_df)

df = pd.concat(fixed_rows, ignore_index=True)

# -----------------------------
# Merge robot metadata
# -----------------------------
df = df.merge(robots, on="robot_id", how="left")

# -----------------------------
# Add maintenance features safely
# -----------------------------
maintenance_features = []

for _, row in df.iterrows():
    robot_id = row["robot_id"]
    ts = row["timestamp"]

    past_maintenance = maintenance[
        (maintenance["robot_id"] == robot_id) & (maintenance["maintenance_time"] <= ts)
    ]

    if past_maintenance.empty:
        maintenance_features.append(
            {
                "last_maintenance_type": "none",
                "last_issue_detected": "none",
                "last_downtime_hours": 0,
                "maintenance_count_to_date": 0,
                "time_since_last_maintenance_hours": 0,
            }
        )
    else:
        last_event = past_maintenance.sort_values("maintenance_time").iloc[-1]

        maintenance_features.append(
            {
                "last_maintenance_type": last_event.get("maintenance_type", "unknown"),
                "last_issue_detected": last_event.get("issue_detected", "unknown"),
                "last_downtime_hours": last_event.get("downtime_hours", 0),
                "maintenance_count_to_date": len(past_maintenance),
                "time_since_last_maintenance_hours": (
                    ts - last_event["maintenance_time"]
                ).total_seconds()
                / 3600,
            }
        )

maintenance_features = pd.DataFrame(maintenance_features)
df = pd.concat([df.reset_index(drop=True), maintenance_features], axis=1)

# -----------------------------
# Basic + advanced degradation features
# -----------------------------
df = df.sort_values(["robot_id", "timestamp"]).reset_index(drop=True)

sensor_cols = [
    "vibration_level",
    "motor_temperature",
    "torque_load",
    "power_consumption",
]

for col in sensor_cols:
    df[col] = pd.to_numeric(df[col], errors="coerce")

windows = {"24h": 4, "72h": 12, "7d": 28}


def rolling_slope(values):
    values = np.array(values)

    if len(values) < 2:
        return 0

    values = pd.Series(values).ffill().bfill().fillna(0).values
    x = np.arange(len(values))

    return np.polyfit(x, values, 1)[0]


for col in sensor_cols:
    for label, window in windows.items():
        df[f"{col}_rolling_mean_{label}"] = df.groupby("robot_id")[col].transform(
            lambda x: x.rolling(window, min_periods=1).mean()
        )

        df[f"{col}_rolling_std_{label}"] = df.groupby("robot_id")[col].transform(
            lambda x: x.rolling(window, min_periods=1).std()
        )

        df[f"{col}_slope_{label}"] = df.groupby("robot_id")[col].transform(
            lambda x: x.rolling(window, min_periods=2).apply(rolling_slope, raw=False)
        )

# Runtime features
df["cumulative_runtime_hours"] = df.groupby("robot_id").cumcount() * 6

# Stress features
df["torque_power_stress_index"] = df["torque_load"] * df["power_consumption"]
df["thermal_power_stress_index"] = df["motor_temperature"] * df["power_consumption"]
df["vibration_torque_stress_index"] = df["vibration_level"] * df["torque_load"]

df["load_stress_index"] = df["torque_load"] * df["cumulative_runtime_hours"]
df["vibration_runtime_wear_index"] = (
    df["vibration_level"] * df["cumulative_runtime_hours"]
)
df["thermal_runtime_stress_index"] = (
    df["motor_temperature"] * df["cumulative_runtime_hours"]
)

# Energy features
df["cumulative_energy_consumption"] = df.groupby("robot_id")[
    "power_consumption"
].cumsum()

df["energy_per_runtime_hour"] = df["cumulative_energy_consumption"] / df[
    "cumulative_runtime_hours"
].replace(0, np.nan)

# Sensor change features
for col in sensor_cols:
    df[f"{col}_first_diff"] = df.groupby("robot_id")[col].diff()
    df[f"{col}_second_diff"] = df.groupby("robot_id")[f"{col}_first_diff"].diff()

# Early stress flags
df["vibration_high_flag"] = (
    df["vibration_level"] > df["vibration_level"].quantile(0.75)
).astype(int)

df["temperature_high_flag"] = (
    df["motor_temperature"] > df["motor_temperature"].quantile(0.75)
).astype(int)

df["torque_high_flag"] = (df["torque_load"] > df["torque_load"].quantile(0.75)).astype(
    int
)

df["power_high_flag"] = (
    df["power_consumption"] > df["power_consumption"].quantile(0.75)
).astype(int)

df["multi_sensor_stress_flag"] = (
    df[
        [
            "vibration_high_flag",
            "temperature_high_flag",
            "torque_high_flag",
            "power_high_flag",
        ]
    ].sum(axis=1)
    >= 2
).astype(int)

# Analysis-only labels, not for model input
df["rul_health_band"] = pd.cut(
    df["RUL_hours"],
    bins=[-np.inf, 24, 72, 168, np.inf],
    labels=["Critical", "Warning", "Watch", "Healthy"],
).astype(str)

df["early_failure_zone"] = (df["RUL_hours"] <= 72).astype(int)
df["critical_failure_zone"] = (df["RUL_hours"] <= 24).astype(int)

# -----------------------------
# Cleanup
# -----------------------------
df = df.replace([np.inf, -np.inf], np.nan)

num_cols = df.select_dtypes(include=[np.number]).columns
cat_cols = df.select_dtypes(include=["object", "category"]).columns

df[num_cols] = df[num_cols].fillna(0)
df[cat_cols] = df[cat_cols].fillna("unknown")

# -----------------------------
# Validate RUL monotonicity
# -----------------------------
violations = 0

for robot_id, robot_df in df.groupby("robot_id"):
    rul_diff = robot_df.sort_values("timestamp")["RUL_hours"].diff().dropna()

    if (rul_diff > 0).any():
        violations += 1

print("RUL monotonicity violations:", violations)
print("Final fixed dataset shape:", df.shape)
print("RUL summary:")
print(df["RUL_hours"].describe())

df.to_csv(OUTPUT_PATH, index=False)

print(f"\nFixed lifecycle RUL dataset saved to: {OUTPUT_PATH}")
