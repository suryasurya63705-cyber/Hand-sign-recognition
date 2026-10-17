import pandas as pd
import numpy as np
import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, ConfusionMatrixDisplay
from utils import normalize

df = pd.read_csv("data.csv")
print("Rows:", len(df))
print(df["label"].value_counts(), "\n")      # check classes are balanced

X = np.array([normalize(r) for r in df.drop(columns="label").values])
y = df["label"].values

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y)

models = {
    "Logistic Regression (baseline)": make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000)),
    "KNN": make_pipeline(StandardScaler(), KNeighborsClassifier(n_neighbors=5)),
    "Random Forest": RandomForestClassifier(n_estimators=200, random_state=42),
}

best_name, best_model, best_acc = None, None, 0
for name, m in models.items():
    m.fit(X_train, y_train)
    acc = accuracy_score(y_test, m.predict(X_test))
    print(f"{name}: {acc*100:.1f}%")
    if acc > best_acc:
        best_name, best_model, best_acc = name, m, acc

print(f"\nBest: {best_name} ({best_acc*100:.1f}%)")
print(classification_report(y_test, best_model.predict(X_test)))

ConfusionMatrixDisplay.from_estimator(best_model, X_test, y_test, xticks_rotation=45)
plt.title(best_name)
plt.tight_layout()
plt.savefig("confusion_matrix.png", dpi=150)

joblib.dump(best_model, "model.pkl")
print("Saved model.pkl and confusion_matrix.png")
