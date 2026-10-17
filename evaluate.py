import pandas as pd
import numpy as np
import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, classification_report, ConfusionMatrixDisplay
from utils import normalize

FILE = input("Test file name (test.csv or friend.csv): ").strip() or "test.csv"

model = joblib.load("model.pkl")
df = pd.read_csv(FILE)
X = np.array([normalize(r) for r in df.drop(columns="label").values])
y = df["label"].values

pred = model.predict(X)
print(f"\nAccuracy on {FILE}: {accuracy_score(y, pred)*100:.1f}%\n")
print(classification_report(y, pred))

ConfusionMatrixDisplay.from_predictions(y, pred, xticks_rotation=45)
plt.title(f"Test: {FILE}")
plt.tight_layout()
plt.savefig("confusion_matrix_test.png", dpi=150)
print("Saved confusion_matrix_test.png")