import pandas as pd
from pathlib import Path

# Paths
INPUT_PATH = Path("data/processed/rul_cleaned.csv")
OUTPUT_DIR = Path("data/sample")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

OUTPUT_PATH = OUTPUT_DIR / "sample_data.csv"

# Load full dataset
df = pd.read_csv(INPUT_PATH)

print("Original dataset shape:", df.shape)

# Create sample
sample_df = df.sample(n=1000, random_state=42)

# Save sample
sample_df.to_csv(OUTPUT_PATH, index=False)

print("Sample dataset created successfully.")
print("Sample shape:", sample_df.shape)
print(f"Saved to: {OUTPUT_PATH}")