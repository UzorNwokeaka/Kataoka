from pathlib import Path

import gradio as gr
import joblib
import numpy as np
import pandas as pd
import plotly.graph_objects as go

# ============================================================
# Paths
# ============================================================
MODEL_PATH = Path("models/best_rul_model.pkl")
DATA_PATH = Path("data/processed/rul_fixed_lifecycle_dataset.csv")

# ============================================================
# Load model and reference data
# ============================================================
model = joblib.load(MODEL_PATH)
reference_df = pd.read_csv(DATA_PATH)

# ============================================================
# Match training feature structure
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


# ============================================================
# Real-world risk scoring
# ============================================================
def score_sensor_risk(
    vibration_level,
    motor_temperature,
    torque_load,
    power_consumption,
    cumulative_runtime_hours,
    time_since_last_maintenance_hours,
):
    """
    Creates risk scores across three operational domains:
    1. Physical degradation: vibration + temperature
    2. Load/energy stress: torque + power consumption
    3. Maintenance/lifecycle pressure: runtime + time since maintenance
    """

    physical_risk = 0
    load_risk = 0
    maintenance_risk = 0
    reasons = []

    # -------------------------
    # Physical degradation risk
    # -------------------------
    if vibration_level >= 0.85:
        physical_risk += 30
        reasons.append("severe vibration")
    elif vibration_level >= 0.70:
        physical_risk += 22
        reasons.append("high vibration")
    elif vibration_level >= 0.60:
        physical_risk += 14
        reasons.append("elevated vibration")

    if motor_temperature >= 95:
        physical_risk += 30
        reasons.append("severe motor temperature")
    elif motor_temperature >= 85:
        physical_risk += 22
        reasons.append("high motor temperature")
    elif motor_temperature >= 75:
        physical_risk += 14
        reasons.append("elevated motor temperature")

    # -------------------------
    # Load / energy stress risk
    # -------------------------
    if torque_load >= 220:
        load_risk += 25
        reasons.append("severe torque load")
    elif torque_load >= 190:
        load_risk += 18
        reasons.append("high torque load")
    elif torque_load >= 160:
        load_risk += 12
        reasons.append("elevated torque load")

    if power_consumption >= 2200:
        load_risk += 25
        reasons.append("severe power consumption")
    elif power_consumption >= 1900:
        load_risk += 18
        reasons.append("high power consumption")
    elif power_consumption >= 1500:
        load_risk += 12
        reasons.append("elevated power consumption")

    # -------------------------
    # Maintenance / lifecycle risk
    # -------------------------
    if cumulative_runtime_hours >= 1900:
        maintenance_risk += 18
        reasons.append("very high cumulative runtime")
    elif cumulative_runtime_hours >= 1500:
        maintenance_risk += 12
        reasons.append("high cumulative runtime")
    elif cumulative_runtime_hours >= 1000:
        maintenance_risk += 6
        reasons.append("moderate cumulative runtime")

    if time_since_last_maintenance_hours >= 400:
        maintenance_risk += 18
        reasons.append("very long time since last maintenance")
    elif time_since_last_maintenance_hours >= 250:
        maintenance_risk += 12
        reasons.append("long time since last maintenance")
    elif time_since_last_maintenance_hours >= 150:
        maintenance_risk += 6
        reasons.append("maintenance interval is increasing")

    # Domain caps
    physical_risk = min(physical_risk, 60)
    load_risk = min(load_risk, 50)
    maintenance_risk = min(maintenance_risk, 36)

    raw_total = physical_risk + load_risk + maintenance_risk

    # Scale to 0–100
    total_risk = min(round(raw_total, 2), 100)

    return total_risk, physical_risk, load_risk, maintenance_risk, reasons


def classify_health(
    rul_hours,
    total_risk,
    physical_risk,
    load_risk,
    maintenance_risk,
):
    """
    Final triage logic.

    This combines:
    - ML-predicted RUL
    - current physical degradation risk
    - current load/energy risk
    - maintenance/runtime risk
    - interaction/amplification logic

    The goal is to behave like a real industrial decision-support system.
    """

    # -------------------------
    # 1. Hard RUL-based overrides
    # -------------------------
    if rul_hours <= 168:
        return "🔴 Critical"

    if rul_hours <= 500:
        return "🟠 Warning"

    # -------------------------
    # 2. Maintenance amplification
    # -------------------------
    # Old/under-maintained machines are more sensitive to otherwise moderate stress.
    amplified_physical = physical_risk + (maintenance_risk * 0.50)
    amplified_load = load_risk + (maintenance_risk * 0.40)
    combined_amplified_stress = amplified_physical + amplified_load

    # -------------------------
    # 3. Critical cases
    # -------------------------
    # Overall risk very high
    if total_risk >= 80:
        return "🔴 Critical"

    # Multiple domains are severe together
    if physical_risk >= 44 and load_risk >= 30:
        return "🔴 Critical"

    if physical_risk >= 44 and maintenance_risk >= 24:
        return "🔴 Critical"

    if load_risk >= 40 and maintenance_risk >= 30:
        return "🔴 Critical"

    # Amplified stress becomes severe
    if combined_amplified_stress >= 78:
        return "🔴 Critical"

    if amplified_physical >= 55:
        return "🔴 Critical"

    if amplified_load >= 58:
        return "🔴 Critical"

    # -------------------------
    # 4. Warning cases
    # -------------------------
    # Overall medium-high risk
    if total_risk >= 45:
        return "🟠 Warning"

    # One domain clearly elevated
    if physical_risk >= 28:
        return "🟠 Warning"

    if load_risk >= 24:
        return "🟠 Warning"

    if maintenance_risk >= 24:
        return "🟠 Warning"

    # Moderate signals amplified by age/maintenance
    if combined_amplified_stress >= 42:
        return "🟠 Warning"

    if maintenance_risk >= 18 and (physical_risk >= 14 or load_risk >= 12):
        return "🟠 Warning"

    # -------------------------
    # 5. Moderate cases
    # -------------------------
    if total_risk >= 25:
        return "🟡 Moderate Risk"

    if rul_hours <= 1500:
        return "🟡 Moderate Risk"

    if maintenance_risk >= 12:
        return "🟡 Moderate Risk"

    return "🟢 Healthy"


def recommend_action(health_status):
    if "Critical" in health_status:
        return (
            "Immediate inspection required. Prioritise this robot for urgent "
            "maintenance review to reduce the risk of unplanned failure."
        )

    if "Warning" in health_status:
        return (
            "Schedule maintenance soon. The robot is showing elevated risk signals "
            "that should be reviewed before they develop into failure."
        )

    if "Moderate" in health_status:
        return (
            "Plan preventive maintenance within the next maintenance cycle and "
            "continue monitoring sensor behaviour."
        )

    return "No immediate action required. Continue normal monitoring."


def calculate_failure_probability(risk_score):
    """
    Business-friendly risk estimate derived from risk score.
    This is not a calibrated statistical probability.
    """
    return round(min(100, risk_score * 1.15), 2)


def create_risk_gauge(risk_score):
    fig = go.Figure(
        go.Indicator(
            mode="gauge+number",
            value=risk_score,
            title={"text": "Real-Time Risk Score"},
            gauge={
                "axis": {"range": [0, 100]},
                "bar": {"color": "black"},
                "steps": [
                    {"range": [0, 25], "color": "#2ECC71"},
                    {"range": [25, 45], "color": "#F4D03F"},
                    {"range": [45, 80], "color": "#F39C12"},
                    {"range": [80, 100], "color": "#E74C3C"},
                ],
            },
        )
    )

    fig.update_layout(height=300, margin=dict(l=20, r=20, t=50, b=20))
    return fig


def explain_prediction(
    reasons,
    physical_risk,
    load_risk,
    maintenance_risk,
    total_risk,
):
    if not reasons:
        return (
            "No major risk drivers detected. The robot is operating within normal "
            "ranges based on the provided inputs."
        )

    return (
        "Key risk drivers detected: "
        + ", ".join(reasons)
        + f". Physical degradation risk: {physical_risk}/60, "
        + f"load/energy risk: {load_risk}/50, "
        + f"maintenance/runtime risk: {maintenance_risk}/36, "
        + f"overall real-time risk score: {total_risk}/100."
    )


# ============================================================
# Build model input
# ============================================================
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

    # Lifecycle proxy features
    if "time_index" in row:
        row["time_index"] = cumulative_runtime_hours / 6

    if "lifecycle_progress" in row:
        row["lifecycle_progress"] = min(cumulative_runtime_hours / 2000, 1.0)

    # Derived stress features
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
        row["thermal_runtime_stress_index"] = (
            motor_temperature * cumulative_runtime_hours
        )

    if "cumulative_energy_consumption" in row:
        row["cumulative_energy_consumption"] = power_consumption * max(
            cumulative_runtime_hours / 6, 1
        )

    if "energy_per_runtime_hour" in row:
        row["energy_per_runtime_hour"] = power_consumption

    # Binary stress flags used by training features if present
    if "vibration_high_flag" in row:
        row["vibration_high_flag"] = int(vibration_level >= 0.60)

    if "temperature_high_flag" in row:
        row["temperature_high_flag"] = int(motor_temperature >= 75)

    if "torque_high_flag" in row:
        row["torque_high_flag"] = int(torque_load >= 160)

    if "power_high_flag" in row:
        row["power_high_flag"] = int(power_consumption >= 1500)

    if "multi_sensor_stress_flag" in row:
        stress_count = (
            int(vibration_level >= 0.60)
            + int(motor_temperature >= 75)
            + int(torque_load >= 160)
            + int(power_consumption >= 1500)
        )
        row["multi_sensor_stress_flag"] = int(stress_count >= 2)

    input_df = pd.DataFrame([row])
    input_df = input_df[X_ref.columns]

    return input_df


# ============================================================
# Prediction function
# ============================================================
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

    (
        total_risk,
        physical_risk,
        load_risk,
        maintenance_risk,
        reasons,
    ) = score_sensor_risk(
        vibration_level,
        motor_temperature,
        torque_load,
        power_consumption,
        cumulative_runtime_hours,
        time_since_last_maintenance_hours,
    )

    health_status = classify_health(
        predicted_rul,
        total_risk,
        physical_risk,
        load_risk,
        maintenance_risk,
    )

    recommendation = recommend_action(health_status)
    failure_probability = calculate_failure_probability(total_risk)
    risk_gauge = create_risk_gauge(total_risk)

    explanation = explain_prediction(
        reasons,
        physical_risk,
        load_risk,
        maintenance_risk,
        total_risk,
    )

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
        total_risk,
        f"{failure_probability}%",
        recommendation,
        confidence_range,
        explanation,
        risk_gauge,
    )


# ============================================================
# Dropdown values
# ============================================================
model_types = sorted(reference_df["model_type"].dropna().astype(str).unique().tolist())
locations = sorted(
    reference_df["factory_location"].dropna().astype(str).unique().tolist()
)
environments = sorted(
    reference_df["operating_environment"].dropna().astype(str).unique().tolist()
)


# ============================================================
# Gradio UI
# ============================================================
with gr.Blocks(theme=gr.themes.Soft()) as app:
    gr.Markdown(
        """
        # Kataoka RUL Predictive Maintenance Dashboard

        Estimate the **Remaining Useful Life (RUL)** of industrial robotic equipment using
        sensor readings, maintenance history, and operational context.

        This dashboard combines:

        - **ML-based RUL prediction**
        - **real-time sensor risk scoring**
        - **maintenance/runtime risk amplification**
        - **rule-based safety overrides**
        - **maintenance recommendations**
        """
    )

    with gr.Row():
        with gr.Column():
            gr.Markdown("## Sensor Inputs")

            vibration_level = gr.Slider(
                0,
                1,
                value=0.35,
                step=0.01,
                label="Vibration Level",
            )

            motor_temperature = gr.Slider(
                20,
                120,
                value=65,
                step=1,
                label="Motor Temperature (°C)",
            )

            torque_load = gr.Slider(
                0,
                250,
                value=120,
                step=1,
                label="Torque Load",
            )

            power_consumption = gr.Slider(
                0,
                2500,
                value=1200,
                step=10,
                label="Power Consumption",
            )

            gr.Markdown("## Robot Context")

            model_type = gr.Dropdown(
                model_types,
                value=model_types[0],
                label="Robot Model Type",
            )

            factory_location = gr.Dropdown(
                locations,
                value=locations[0],
                label="Factory Location",
            )

            operating_environment = gr.Dropdown(
                environments,
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

            health_status = gr.Textbox(label="Final Health Status")

            risk_score = gr.Slider(
                0,
                100,
                value=0,
                step=1,
                label="Real-Time Risk Score",
                interactive=False,
            )

            failure_probability = gr.Textbox(label="Failure Probability (%)")

            risk_gauge = gr.Plot(label="Real-Time Risk Meter")

            maintenance_recommendation = gr.Textbox(
                label="Maintenance Recommendation",
                lines=3,
            )

            confidence_range = gr.Textbox(
                label="Uncertainty / Confidence Range",
                lines=2,
            )

            explanation = gr.Textbox(
                label="Why This Prediction?",
                lines=5,
            )

            gr.Markdown(
                """
                ### How to interpret this output

                - **Predicted RUL** estimates remaining useful operating hours.
                - **Risk Score** shows current operating risk from 0 to 100.
                - **Failure Probability** is a business-friendly estimate derived from risk score.
                - **Healthy**: normal operation.
                - **Moderate Risk**: monitor and plan preventive maintenance.
                - **Warning**: elevated stress or maintenance/runtime risk.
                - **Critical**: urgent inspection recommended.

                The final health status is not based on RUL alone. It also considers
                physical degradation, load/energy stress, maintenance delay, and runtime.
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
                    risk_score,
                    failure_probability,
                    risk_gauge,
                    maintenance_recommendation,
                    confidence_range,
                    explanation,
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
            risk_score,
            failure_probability,
            maintenance_recommendation,
            confidence_range,
            explanation,
            risk_gauge,
        ],
    )

if __name__ == "__main__":
    app.launch()