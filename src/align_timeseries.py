import pandas as pd
import numpy as np
from pathlib import Path

# -----------------------------
# Paths
# -----------------------------
RAW_DIR = Path("data/raw")
PROCESSED_DIR = Path("data/processed")
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

SENSOR_PATH = RAW_DIR / "sensor_readings.csv"
ROBOT_PATH = RAW_DIR / "robot_units.csv"
MAINT_PATH = PROCESSED_DIR / "maintenance_logs_with_timestamps.csv"
FAILURE_PATH = PROCESSED_DIR / "failure_events_augmented.csv"

OUTPUT_PATH = PROCESSED_DIR / "rul_aligned_timeseries.csv"

# -----------------------------
# Load datasets
# -----------------------------
sensor = pd.read_csv(SENSOR_PATH)
robots = pd.read_csv(ROBOT_PATH)
maintenance = pd.read_csv(MAINT_PATH)
failures = pd.read_csv(FAILURE_PATH)

print("Loaded datasets:")
print("Sensor:", sensor.shape)
print("Robots:", robots.shape)
print("Maintenance:", maintenance.shape)
print("Failures:", failures.shape)

# -----------------------------
# Convert datetime columns
# -----------------------------
sensor["timestamp"] = pd.to_datetime(sensor["timestamp"], errors="coerce")
robots["installation_date"] = pd.to_datetime(robots["installation_date"], errors="coerce")
maintenance["maintenance_time"] = pd.to_datetime(maintenance["maintenance_time"], errors="coerce")
failures["failure_time"] = pd.to_datetime(failures["failure_time"], errors="coerce")

# -----------------------------
# Basic cleaning
# -----------------------------
sensor = sensor.dropna(subset=["robot_id", "timestamp"])
maintenance = maintenance.dropna(subset=["robot_id", "maintenance_time"])
failures = failures.dropna(subset=["robot_id", "failure_time"])

sensor = sensor.sort_values(["robot_id", "timestamp"]).reset_index(drop=True)
maintenance = maintenance.sort_values(["robot_id", "maintenance_time"]).reset_index(drop=True)
failures = failures.sort_values(["robot_id", "failure_time"]).reset_index(drop=True)

# -----------------------------
# Add static robot metadata
# -----------------------------
aligned = sensor.merge(
    robots,
    on="robot_id",
    how="left"
)

print("\nAfter robot metadata merge:", aligned.shape)

# -----------------------------
# Align last maintenance before each sensor reading
# Leakage-safe backward time join
# -----------------------------
maintenance_features = []

for robot_id, robot_sensor in aligned.groupby("robot_id"):
    robot_maint = maintenance[maintenance["robot_id"] == robot_id].copy()

    robot_sensor = robot_sensor.sort_values("timestamp").copy()

    if robot_maint.empty:
        robot_sensor["last_maintenance_time"] = pd.NaT
        robot_sensor["last_maintenance_type"] = "none"
        robot_sensor["last_issue_detected"] = "none"
        robot_sensor["last_downtime_hours"] = 0
        robot_sensor["maintenance_count_to_date"] = 0
        maintenance_features.append(robot_sensor)
        continue

    robot_maint = robot_maint.sort_values("maintenance_time").copy()

    merged = pd.merge_asof(
        robot_sensor,
        robot_maint[
            [
                "maintenance_time",
                "maintenance_type",
                "issue_detected",
                "downtime_hours"
            ]
        ],
        left_on="timestamp",
        right_on="maintenance_time",
        direction="backward"
    )

    merged = merged.rename(columns={
        "maintenance_time": "last_maintenance_time",
        "maintenance_type": "last_maintenance_type",
        "issue_detected": "last_issue_detected",
        "downtime_hours": "last_downtime_hours"
    })

    # Count maintenance events up to current timestamp
    maint_times = robot_maint["maintenance_time"].values
    merged["maintenance_count_to_date"] = np.searchsorted(
        maint_times,
        merged["timestamp"].values,
        side="right"
    )

    maintenance_features.append(merged)

aligned = pd.concat(maintenance_features, ignore_index=True)

# -----------------------------
# Time since last maintenance
# -----------------------------
aligned["time_since_last_maintenance_hours"] = (
    aligned["timestamp"] - aligned["last_maintenance_time"]
).dt.total_seconds() / 3600

aligned["last_maintenance_type"] = aligned["last_maintenance_type"].fillna("none")
aligned["last_issue_detected"] = aligned["last_issue_detected"].fillna("none")
aligned["last_downtime_hours"] = aligned["last_downtime_hours"].fillna(0)
aligned["time_since_last_maintenance_hours"] = (
    aligned["time_since_last_maintenance_hours"].fillna(0)
)

print("After maintenance alignment:", aligned.shape)

# -----------------------------
# Align next failure after each sensor reading
# This creates RUL without future leakage in features
# -----------------------------
failure_aligned = []

for robot_id, robot_sensor in aligned.groupby("robot_id"):
    robot_failures = failures[failures["robot_id"] == robot_id].copy()

    robot_sensor = robot_sensor.sort_values("timestamp").copy()

    if robot_failures.empty:
        continue

    robot_failures = robot_failures.sort_values("failure_time").copy()

    merged = pd.merge_asof(
        robot_sensor,
        robot_failures[
            [
                "failure_time",
                "failure_type",
                "root_cause",
                "is_simulated"
            ]
        ],
        left_on="timestamp",
        right_on="failure_time",
        direction="forward"
    )

    merged = merged.dropna(subset=["failure_time"])

    merged = merged.rename(columns={
        "failure_time": "next_failure_time",
        "failure_type": "next_failure_type",
        "root_cause": "next_failure_root_cause"
    })

    merged["RUL_hours"] = (
        merged["next_failure_time"] - merged["timestamp"]
    ).dt.total_seconds() / 3600

    failure_aligned.append(merged)

aligned = pd.concat(failure_aligned, ignore_index=True)

# -----------------------------
# Integrity checks
# -----------------------------
aligned = aligned[aligned["RUL_hours"] > 0]

aligned = aligned.sort_values(["robot_id", "timestamp"]).reset_index(drop=True)

duplicate_rows = aligned.duplicated(subset=["robot_id", "timestamp"]).sum()
negative_rul = (aligned["RUL_hours"] <= 0).sum()
missing_robot_metadata = aligned["model_type"].isna().sum()

print("\nIntegrity checks:")
print("Duplicate robot/timestamp rows:", duplicate_rows)
print("Non-positive RUL rows:", negative_rul)
print("Missing robot metadata rows:", missing_robot_metadata)
print("Unique robots:", aligned["robot_id"].nunique())
print("Final aligned shape:", aligned.shape)

# -----------------------------
# Save aligned dataset
# -----------------------------
aligned.to_csv(OUTPUT_PATH, index=False)

print("\nTime-series alignment completed successfully.")
print(f"Saved to: {OUTPUT_PATH}")
print(aligned.head())