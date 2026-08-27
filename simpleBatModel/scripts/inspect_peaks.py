from pathlib import Path
import pandas as pd


# Folder where this .py file is located.
# This makes the script work when it is placed inside the same folder as the CSV files.
try:
    FOLDER = Path(__file__).resolve().parent
except NameError:
    # Fallback for interactive execution
    FOLDER = Path.cwd()


OUTPUT_FILE = FOLDER / "max_values_label_value.csv"


def get_max_value(file_path: Path) -> float:
    """
    Reads a CSV file and returns the maximum numeric value.
    Non-numeric values are ignored.
    """
    df = pd.read_csv(file_path, header=None)
    numeric_df = df.apply(pd.to_numeric, errors="coerce")

    if numeric_df.empty or numeric_df.isna().all().all():
        raise ValueError(f"No numeric values found in {file_path.name}")

    return float(numeric_df.max().max())


results = []

for i in range(1, 9):
    file_name = f"load_cons_{i:02d}.csv"
    file_path = FOLDER / file_name

    if not file_path.exists():
        print(f"Warning: {file_name} not found. Skipping.")
        continue

    max_value = get_max_value(file_path)

    results.append({
        "label": f"load_cons_{i:02d}",
        "value": max_value,
    })


results_df = pd.DataFrame(results, columns=["label", "value"])

results_df.to_csv(OUTPUT_FILE, index=False)

print(results_df)
print(f"\nSaved file:")
print(OUTPUT_FILE)
