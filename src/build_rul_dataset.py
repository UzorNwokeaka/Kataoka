import pandas as pd
import numpy as np
from pathlib import Path

np.random.seed(42)

RAW_DIR = Path("data/raw")
PROCESSED_DIR = Path("data/processed")
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

robot_units = pd.read_csv(RAW_DIR / "robot_units.csv")
sensor_readings = pd.read_csv(RAW_DIR / "sensor_readings.csv")
maintenance_logs = pd.read_csv(RAW_DIR / "maintenance_logs.csv")
failure_events = pd.read_csv(RAW_DIR / "failure_events.csv")

sensor_readings["timestamp"] = pd.to_datetime(
    sensor_readings["timestamp"], errors="coerce"
)
robot_units["installation_date"] = pd.to_datetime(
    robot_units["installation_date"], errors="coerce"
)
failure_events["failure_time"] = pd.to_datetime(
    failure_events["failure_time"], errors="coerce"
)

sensor_readings = sensor_readings.sort_values(["robot_id", "timestamp"])


def generate_maintenance_times(maintenance_logs, sensor_readings):
    maintenance_logs = maintenance_logs.copy()
    times = []

    for _, row in maintenance_logs.iterrows():
        robot_id = row["robot_id"]
        robot_sensor = sensor_readings[sensor_readings["robot_id"] == robot_id]

        if robot_sensor.empty:
            times.append(pd.NaT)
            continue

        start_time = robot_sensor["timestamp"].min()
        end_time = robot_sensor["timestamp"].max()

        random_seconds = np.random.uniform(
            0, max((end_time - start_time).total_seconds(), 1)
        )

        times.append(start_time + pd.to_timedelta(random_seconds, unit="s"))

    maintenance_logs["maintenance_time"] = times
    return maintenance_logs


def simulate_failure_events(sensor_readings, failure_events, failure_probability=1.0):
    failure_events = failure_events.copy()
    failure_events["is_simulated"] = 0

    existing_failure_robots = set(failure_events["robot_id"])
    all_sensor_robots = sensor_readings["robot_id"].unique()

    failure_types = [
        "motor_failure",
        "joint_wear",
        "bearing_failure",
        "overheating",
        "encoder_drift",
    ]

    root_causes = [
        "lubrication breakdown",
        "thermal cycling fatigue",
        "overload operation",
        "mechanical misalignment",
        "electrical surge",
        "contamination",
    ]

    new_failures = []
    next_id = len(failure_events) + 1

    for robot_id in all_sensor_robots:
        if robot_id in existing_failure_robots:
            continue

        robot_sensor = sensor_readings[sensor_readings["robot_id"] == robot_id]

        if len(robot_sensor) < 5:
            continue

        if np.random.rand() <= failure_probability:
            max_time = robot_sensor["timestamp"].max()

            simulated_failure_time = max_time + pd.to_timedelta(
                np.random.randint(1, 15), unit="D"
            )

            new_failures.append(
                {
                    "failure_id": f"SIM-FAIL-{next_id:04d}",
                    "robot_id": robot_id,
                    "failure_type": np.random.choice(failure_types),
                    "failure_time": simulated_failure_time,
                    "root_cause": np.random.choice(root_causes),
                    "is_simulated": 1,
                }
            )

            next_id += 1

    simulated_failures = pd.DataFrame(new_failures)

    return pd.concat([failure_events, simulated_failures], ignore_index=True)


def add_rul_target(sensor_readings, failure_events):
    rows = []

    for robot_id, robot_sensor in sensor_readings.groupby("robot_id"):
        robot_failures = failure_events[failure_events["robot_id"] == robot_id]

        if robot_failures.empty:
            continue

        robot_failures = robot_failures.sort_values("failure_time")

        for _, sensor_row in robot_sensor.iterrows():
            future_failures = robot_failures[
                robot_failures["failure_time"] > sensor_row["timestamp"]
            ]

            if future_failures.empty:
                continue

            next_failure_time = future_failures["failure_time"].min()

            row = sensor_row.to_dict()
            row["next_failure_time"] = next_failure_time
            row["RUL_hours"] = (
                next_failure_time - sensor_row["timestamp"]
            ).total_seconds() / 3600

            rows.append(row)

    return pd.DataFrame(rows)


def add_maintenance_features(rul_dataset, maintenance_logs):
    features = []

    for _, row in rul_dataset.iterrows():
        robot_id = row["robot_id"]
        timestamp = row["timestamp"]

        past_maintenance = maintenance_logs[
            (maintenance_logs["robot_id"] == robot_id)
            & (maintenance_logs["maintenance_time"] <= timestamp)
        ]

        if past_maintenance.empty:
            features.append(
                {
                    "time_since_last_maintenance_hours": 0,
                    "maintenance_count_to_date": 0,
                    "last_downtime_hours": 0,
                    "last_maintenance_type": "none",
                }
            )
        else:
            last_event = past_maintenance.sort_values("maintenance_time").iloc[-1]

            features.append(
                {
                    "time_since_last_maintenance_hours": (
                        timestamp - last_event["maintenance_time"]
                    ).total_seconds()
                    / 3600,
                    "maintenance_count_to_date": len(past_maintenance),
                    "last_downtime_hours": last_event["downtime_hours"],
                    "last_maintenance_type": last_event["maintenance_type"],
                }
            )

    return pd.concat(
        [rul_dataset.reset_index(drop=True), pd.DataFrame(features)], axis=1
    )


maintenance_logs = generate_maintenance_times(maintenance_logs, sensor_readings)
failure_events_augmented = simulate_failure_events(sensor_readings, failure_events)

rul_dataset = add_rul_target(sensor_readings, failure_events_augmented)

rul_dataset = rul_dataset.merge(robot_units, on="robot_id", how="left")

rul_dataset = add_maintenance_features(rul_dataset, maintenance_logs)

rul_dataset = rul_dataset.sort_values(["robot_id", "timestamp"])

sensor_cols = [
    "vibration_level",
    "motor_temperature",
    "torque_load",
    "power_consumption",
]

for col in sensor_cols:
    rul_dataset[f"{col}_rolling_mean_24h"] = rul_dataset.groupby("robot_id")[
        col
    ].transform(lambda x: x.rolling(window=4, min_periods=1).mean())

    rul_dataset[f"{col}_rolling_std_24h"] = rul_dataset.groupby("robot_id")[
        col
    ].transform(lambda x: x.rolling(window=4, min_periods=1).std())

    rul_dataset[f"{col}_change_rate"] = rul_dataset.groupby("robot_id")[col].diff()

rul_dataset["stress_index"] = (
    rul_dataset["torque_load"] * rul_dataset["power_consumption"]
)

rul_dataset["thermal_stress_index"] = (
    rul_dataset["motor_temperature"] * rul_dataset["power_consumption"]
)

rul_dataset = rul_dataset.fillna(0)

rul_dataset.to_csv(PROCESSED_DIR / "rul_feature_table.csv", index=False)
maintenance_logs.to_csv(
    PROCESSED_DIR / "maintenance_logs_with_timestamps.csv", index=False
)
failure_events_augmented.to_csv(
    PROCESSED_DIR / "failure_events_augmented.csv", index=False
)

print("Final RUL dataset created successfully.")
print("Shape:", rul_dataset.shape)
print("Saved to: data/processed/rul_feature_table.csv")
print(rul_dataset.head())
