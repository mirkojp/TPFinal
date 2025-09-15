import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, precision_score, classification_report
import numpy as np

# Carga el archivo CSV con los datos de prueba
df = pd.read_csv("dataset/test/dataoutput.csv")


def get_true_label(row):
    """
    Devuelve la etiqueta verdadera para una fila del DataFrame.
    Si 'has_pattern' es 0, retorna 'none', de lo contrario retorna el valor de 'pattern'.

    Args:
        row (pd.Series): Fila del DataFrame.

    Returns:
        str: Etiqueta verdadera.
    """
    if row["has_pattern"] == 0:
        return "none"
    else:
        return row["pattern"]


def get_pred_label(row):
    """
    Devuelve la etiqueta predicha para una fila del DataFrame.
    Si 'pattern_detected' es 0, retorna 'none', de lo contrario retorna el valor de 'detected_pattern'.

    Args:
        row (pd.Series): Fila del DataFrame.

    Returns:
        str: Etiqueta predicha.
    """
    if row["pattern_detected"] == 0:
        return "none"
    else:
        return row["detected_pattern"]


# Aplica las funciones de mapeo para obtener las etiquetas verdaderas y predichas
df["true_label"] = df.apply(get_true_label, axis=1)
df["pred_label"] = df.apply(get_pred_label, axis=1)

# Define las clases posibles, incluyendo 'none'
classes = ["none", "ceda", "resalto", "pare"]

# Muestra las etiquetas únicas presentes en los datos
all_true = df["true_label"].unique()
all_pred = df["pred_label"].unique()
print(f"Unique true labels: {all_true}")
print(f"Unique predicted labels: {all_pred}")

# Calcula la matriz de confusión usando las clases definidas
cm = confusion_matrix(df["true_label"], df["pred_label"], labels=classes)

# Imprime la matriz de confusión
print("Confusion Matrix:")
print(cm)

# Calcula la precisión macro (promedio entre clases)
precision = precision_score(
    df["true_label"], df["pred_label"], labels=classes, average="macro", zero_division=0
)
print(f"\nMacro Precision: {precision:.4f}")

# Imprime el reporte de clasificación detallado (precision, recall, f1-score por clase)
print("\nClassification Report:")
print(
    classification_report(
        df["true_label"], df["pred_label"], labels=classes, zero_division=0
    )
)

# Grafica la matriz de confusión como un mapa de calor
plt.figure(figsize=(8, 6))
sns.heatmap(
    cm, annot=True, fmt="d", cmap="Blues", xticklabels=classes, yticklabels=classes
)
plt.title("Confusion Matrix")
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.show()