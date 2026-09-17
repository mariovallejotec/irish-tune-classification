import json
import numpy as np
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, f1_score, classification_report

def load(split):
    d = np.load(f"features_{split}.npz", allow_pickle=True)
    return d["X"], d["y"]

X_train, y_train = load("train")
X_val, y_val = load("val")
X_test, y_test = load("test")

scaler = StandardScaler()
X_train_s = scaler.fit_transform(X_train)
X_val_s = scaler.transform(X_val)
X_test_s = scaler.transform(X_test)

results = {}
for kernel in ["linear", "rbf"]:
    print(f"\n=== SVM {kernel} ===")
    clf = SVC(kernel=kernel, class_weight="balanced", random_state=42)
    clf.fit(X_train_s, y_train)

    val_preds = clf.predict(X_val_s)
    val_acc = accuracy_score(y_val, val_preds)
    val_f1 = f1_score(y_val, val_preds, average="macro")
    print(f"val: acc={val_acc:.4f} macro_f1={val_f1:.4f}")

    test_preds = clf.predict(X_test_s)
    test_acc = accuracy_score(y_test, test_preds)
    test_f1 = f1_score(y_test, test_preds, average="macro")
    report = classification_report(y_test, test_preds, digits=3)
    print(f"test: acc={test_acc:.4f} macro_f1={test_f1:.4f}")
    print(report)

    results[kernel] = {
        "val_acc": val_acc, "val_macro_f1": val_f1,
        "test_acc": test_acc, "test_macro_f1": test_f1,
        "classification_report": report,
    }

with open("svm_results.json", "w") as f:
    json.dump(results, f, indent=2)
print("\nResultados guardados en svm_results.json")
