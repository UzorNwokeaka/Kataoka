from pathlib import Path

import gradio as gr
import joblib
import numpy as np
import pandas as pd

# -----------------------------
# Paths
# -----------------------------
MODEL_PATH = Path("models/best_rul_model.pkl")
DATA_PATH = Path("data/processed/rul_fixed_lifecycle_dataset.csv")

# -----------------------------
# Load model and reference dataset
# -----------------------------
model = joblib.load(MODEL_PATH)
reference_df = pd.read_csv(DATA_PATH)

# -----------------------------
# Match training feature structure
# -----------------------------
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
    "remaining_lifecycle_ratio",
]

identifier_cols = [
    "reading_id",
    "robot_id",
    "timestamp",
    "last_maintenance_time",
    "installation_date",
]

drop_cols = [c for c in leakage_cols + identifier_cols if c in reference_df.columns]
X_ref = reference_df.drop(columns=drop_cols, errors="ignore")

numeric_cols = X_ref.select_dtypes(include=[np.number]).columns.tolist()
categorical_cols = X_ref.select_dtypes(
    include=["object", "category", "bool"]
).columns.tolist()

default_values = {}

for col in numeric_cols:
    default_values[col] = float(X_ref[col].median())

for col in categorical_cols:
    default_values[col] = str(X_ref[col].mode()[0])


# -----------------------------
# Helper functions
# -----------------------------
def classify_health(rul_hours):
    if rul_hours > 1500:
        return "🟢 Healthy"
    if rul_hours > 500:
        return "🟡 Moderate Risk"
    if rul_hours > 168:
        return "🟠 Warning"
    return "🔴 Critical"


def recommend_action(rul_hours):
    if rul_hours > 1500:
        return "No immediate action required. Continue normal monitoring."
    if rul_hours > 500:
        return "Plan preventive maintenance within the next maintenance cycle."
    if rul_hours > 168:
        return "Schedule maintenance soon to reduce risk of unplanned downtime."
    return "Immediate inspection recommended. Prioritise this robot for maintenance."


def build_input_row(
    vibration_level,
    motor_temperature,
    torque_load,
    power_consumption,
    model_type,
    factory_location,
    operating_environment,
    cumulative_runtime_hours,
    time_since_last_maintenance_hours,
    maintenance_count_to_date,
    last_downtime_hours,
):
    row = default_values.copy()

    row["vibration_level"] = vibration_level
    row["motor_temperature"] = motor_temperature
    row["torque_load"] = torque_load
    row["power_consumption"] = power_consumption

    row["model_type"] = model_type
    row["factory_location"] = factory_location
    row["operating_environment"] = operating_environment

    row["cumulative_runtime_hours"] = cumulative_runtime_hours
    row["time_since_last_maintenance_hours"] = time_since_last_maintenance_hours
    row["maintenance_count_to_date"] = maintenance_count_to_date
    row["last_downtime_hours"] = last_downtime_hours

    if "time_index" in row:
        row["time_index"] = cumulative_runtime_hours / 6

    if "lifecycle_progress" in row:
        row["lifecycle_progress"] = min(cumulative_runtime_hours / 2000, 1.0)

    if "torque_power_stress_index" in row:
        row["torque_power_stress_index"] = torque_load * power_consumption

    if "thermal_power_stress_index" in row:
        row["thermal_power_stress_index"] = motor_temperature * power_consumption

    if "vibration_torque_stress_index" in row:
        row["vibration_torque_stress_index"] = vibration_level * torque_load

    if "load_stress_index" in row:
        row["load_stress_index"] = torque_load * cumulative_runtime_hours

    if "vibration_runtime_wear_index" in row:
        row["vibration_runtime_wear_index"] = vibration_level * cumulative_runtime_hours

    if "thermal_runtime_stress_index" in row:
        row["thermal_runtime_stress_index"] = motor_temperature * cumulative_runtime_hours

    if "cumulative_energy_consumption" in row:
        row["cumulative_energy_consumption"] = power_consumption * max(
            cumulative_runtime_hours / 6, 1
        )

    if "energy_per_runtime_hour" in row:
        row["energy_per_runtime_hour"] = power_consumption

    if "vibration_high_flag" in row:
        row["vibration_high_flag"] = int(vibration_level > 0.6)

    if "temperature_high_flag" in row:
        row["temperature_high_flag"] = int(motor_temperature > 75)

    if "torque_high_flag" in row:
        row["torque_high_flag"] = int(torque_load > 140)

    if "power_high_flag" in row:
        row["power_high_flag"] = int(power_consumption > 1500)

    if "multi_sensor_stress_flag" in row:
        stress_count = (
            int(vibration_level > 0.6)
            + int(motor_temperature > 75)
            + int(torque_load > 140)
            + int(power_consumption > 1500)
        )
        row["multi_sensor_stress_flag"] = int(stress_count >= 2)

    input_df = pd.DataFrame([row])
    input_df = input_df[X_ref.columns]

    return input_df


def predict_rul(
    vibration_level,
    motor_temperature,
    torque_load,
    power_consumption,
    model_type,
    factory_location,
    operating_environment,
    cumulative_runtime_hours,
    time_since_last_maintenance_hours,
    maintenance_count_to_date,
    last_downtime_hours,
):
    input_df = build_input_row(
        vibration_level,
        motor_temperature,
        torque_load,
        power_consumption,
        model_type,
        factory_location,
        operating_environment,
        cumulative_runtime_hours,
        time_since_last_maintenance_hours,
        maintenance_count_to_date,
        last_downtime_hours,
    )

    predicted_rul = float(model.predict(input_df)[0])
    predicted_rul = max(predicted_rul, 0)

    health_status = classify_health(predicted_rul)
    recommendation = recommend_action(predicted_rul)

    mae = 64
    lower_bound = max(predicted_rul - mae, 0)
    upper_bound = predicted_rul + mae

    confidence_range = (
        f"{lower_bound:.0f} – {upper_bound:.0f} hours "
        f"(based on model MAE ≈ {mae} hours)"
    )

    return (
        round(predicted_rul, 2),
        health_status,
        recommendation,
        confidence_range,
    )


# -----------------------------
# Dropdown values
# -----------------------------
model_types = sorted(reference_df["model_type"].dropna().astype(str).unique().tolist())
locations = sorted(
    reference_df["factory_location"].dropna().astype(str).unique().tolist()
)
environments = sorted(
    reference_df["operating_environment"].dropna().astype(str).unique().tolist()
)

# -----------------------------
# Gradio UI
# -----------------------------
with gr.Blocks(theme=gr.themes.Soft()) as app:
    gr.Markdown(
        """
        # Kataoka RUL Predictive Maintenance Dashboard

        Estimate the **Remaining Useful Life (RUL)** of industrial robotic equipment using
        sensor readings, maintenance history, and operational context.

        This tool supports predictive maintenance decisions by identifying whether a robot is
        healthy, at moderate risk, or requires maintenance attention.
        """
    )

    with gr.Row():
        with gr.Column():
            gr.Markdown("## Sensor Inputs")

            vibration_level = gr.Slider(
                minimum=0,
                maximum=1,
                value=0.35,
                step=0.01,
                label="Vibration Level",
            )

            motor_temperature = gr.Slider(
                minimum=20,
                maximum=120,
                value=65,
                step=1,
                label="Motor Temperature (°C)",
            )

            torque_load = gr.Slider(
                minimum=0,
                maximum=250,
                value=120,
                step=1,
                label="Torque Load",
            )

            power_consumption = gr.Slider(
                minimum=0,
                maximum=2500,
                value=1200,
                step=10,
                label="Power Consumption",
            )

            gr.Markdown("## Robot Context")

            model_type = gr.Dropdown(
                choices=model_types,
                value=model_types[0],
                label="Robot Model Type",
            )

            factory_location = gr.Dropdown(
                choices=locations,
                value=locations[0],
                label="Factory Location",
            )

            operating_environment = gr.Dropdown(
                choices=environments,
                value=environments[0],
                label="Operating Environment",
            )

            gr.Markdown("## Maintenance & Runtime")

            cumulative_runtime_hours = gr.Number(
                value=500,
                label="Cumulative Runtime Hours",
            )

            time_since_last_maintenance_hours = gr.Number(
                value=72,
                label="Time Since Last Maintenance (Hours)",
            )

            maintenance_count_to_date = gr.Number(
                value=3,
                label="Maintenance Count to Date",
            )

            last_downtime_hours = gr.Number(
                value=2,
                label="Last Downtime Hours",
            )

            submit_btn = gr.Button("Predict RUL", variant="primary")

        with gr.Column():
            gr.Markdown("## Prediction Output")

            predicted_rul = gr.Number(label="Predicted RUL (Hours)")

            health_status = gr.Textbox(label="Health Status")

            maintenance_recommendation = gr.Textbox(
                label="Maintenance Recommendation",
                lines=3,
            )

            confidence_range = gr.Textbox(
                label="Uncertainty / Confidence Range",
                lines=2,
            )

            gr.Markdown(
                """
                ### How to interpret this output

                - **Healthy**: robot can continue normal operation.
                - **Moderate Risk**: maintenance should be planned.
                - **Warning/Critical**: maintenance should be prioritised.
                - The confidence range shows the approximate prediction error window.
                """
            )

            clear_btn = gr.ClearButton(
                components=[
                    vibration_level,
                    motor_temperature,
                    torque_load,
                    power_consumption,
                    model_type,
                    factory_location,
                    operating_environment,
                    cumulative_runtime_hours,
                    time_since_last_maintenance_hours,
                    maintenance_count_to_date,
                    last_downtime_hours,
                    predicted_rul,
                    health_status,
                    maintenance_recommendation,
                    confidence_range,
                ],
                value="Clear",
            )

    submit_btn.click(
        fn=predict_rul,
        inputs=[
            vibration_level,
            motor_temperature,
            torque_load,
            power_consumption,
            model_type,
            factory_location,
            operating_environment,
            cumulative_runtime_hours,
            time_since_last_maintenance_hours,
            maintenance_count_to_date,
            last_downtime_hours,
        ],
        outputs=[
            predicted_rul,
            health_status,
            maintenance_recommendation,
            confidence_range,
        ],
    )

if __name__ == "__main__":
    app.launch()