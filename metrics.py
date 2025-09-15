import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, precision_score, classification_report
import numpy as np
from pathlib import Path
import sys

# Define the path to the CSV file dynamically
# Assumes the CSV is in a 'dataset/test' subdirectory relative to the script
BASE_DIR = Path(__file__).resolve().parent  # Get the directory of the current script
CSV_PATH = BASE_DIR / "dataset" / "test" / "dataoutput.csv"


# Function to load the CSV with error handling
def load_data(file_path):
    """
    Load the CSV file with error handling.

    Args:
        file_path (Path): Path to the CSV file.

    Returns:
        pd.DataFrame: Loaded DataFrame or None if loading fails.
    """
    try:
        if not file_path.exists():
            raise FileNotFoundError(f"The file {file_path} does not exist.")
        df = pd.read_csv(file_path)
        return df
    except FileNotFoundError as e:
        print(f"Error: {e}")
        sys.exit(1)
    except pd.errors.EmptyDataError:
        print(f"Error: The file {file_path} is empty.")
        sys.exit(1)
    except Exception as e:
        print(f"Error loading file {file_path}: {e}")
        sys.exit(1)


# Functions to get true and predicted labels
def get_true_label(row):
    """
    Return the true label for a DataFrame row.
    If 'has_pattern' is 0, return 'none', else return the value of 'pattern'.

    Args:
        row (pd.Series): DataFrame row.

    Returns:
        str: True label.
    """
    return "none" if row["has_pattern"] == 0 else row["pattern"]


def get_pred_label(row):
    """
    Return the predicted label for a DataFrame row.
    If 'pattern_detected' is 0, return 'none', else return the value of 'detected_pattern'.

    Args:
        row (pd.Series): DataFrame row.

    Returns:
        str: Predicted label.
    """
    return "none" if row["pattern_detected"] == 0 else row["detected_pattern"]


# Load the dataset
df = load_data(CSV_PATH)
if df is None:
    sys.exit(1)  # Exit if data loading failed

# Apply the label mapping functions
df["true_label"] = df.apply(get_true_label, axis=1)
df["pred_label"] = df.apply(get_pred_label, axis=1)

# Define possible classes, including 'none'
classes = ["none", "ceda", "resalto", "pare"]

# Print unique labels present in the data
all_true = df["true_label"].unique()
all_pred = df["pred_label"].unique()
print(f"Unique true labels: {all_true}")
print(f"Unique predicted labels: {all_pred}")

# Calculate the confusion matrix
cm = confusion_matrix(df["true_label"], df["pred_label"], labels=classes)

# Print the confusion matrix
print("Confusion Matrix:")
print(cm)

# Calculate macro precision
precision = precision_score(
    df["true_label"], df["pred_label"], labels=classes, average="macro", zero_division=0
)
print(f"\nMacro Precision: {precision:.4f}")

# Print detailed classification report
print("\nClassification Report:")
print(
    classification_report(
        df["true_label"], df["pred_label"], labels=classes, zero_division=0
    )
)

# Plot the confusion matrix as a heatmap
plt.figure(figsize=(8, 6))
sns.heatmap(
    cm, annot=True, fmt="d", cmap="Blues", xticklabels=classes, yticklabels=classes
)
plt.title("Confusion Matrix")
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.show()
