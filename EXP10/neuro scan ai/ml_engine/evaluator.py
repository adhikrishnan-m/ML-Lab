"""
Model Performance Evaluation Module.
Computes:
- Confusion Matrix (TP, TN, FP, FN)
- Classification Metrics: Accuracy, Precision, Recall/Sensitivity, Specificity, F1-Score
- ROC (Receiver Operating Characteristic) curve coordinates & AUC
- Per-class metrics and formatted report
"""

import numpy as np
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, roc_curve, auc, classification_report
)

def evaluate_model(model, X_test_scaled, y_test):
    """
    Evaluates classifier predictions on the test dataset.
    """
    y_pred = model.predict(X_test_scaled)

    # Probabilities for ROC
    if hasattr(model, "predict_proba"):
        y_prob = model.predict_proba(X_test_scaled)[:, 1]
    elif hasattr(model, "decision_function"):
        df = model.decision_function(X_test_scaled)
        y_prob = 1 / (1 + np.exp(-df))
    else:
        y_prob = y_pred

    # Basic metrics
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, zero_division=0)
    rec = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)

    # Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    # cm format:
    # [[TN, FP],
    #  [FN, TP]]
    tn, fp, fn, tp = int(cm[0, 0]), int(cm[0, 1]), int(cm[1, 0]), int(cm[1, 1])
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0

    # ROC & AUC
    fpr, tpr, thresholds = roc_curve(y_test, y_prob)
    roc_auc = auc(fpr, tpr)

    # Downsample ROC points for clean JSON rendering (max 20 points)
    step = max(1, len(fpr) // 15)
    sampled_fpr = [float(round(val, 4)) for val in fpr[::step]]
    sampled_tpr = [float(round(val, 4)) for val in tpr[::step]]
    if sampled_fpr[-1] != 1.0:
        sampled_fpr.append(1.0)
        sampled_tpr.append(1.0)

    report_dict = classification_report(y_test, y_pred, target_names=["Normal", "Tumor"], output_dict=True)

    return {
        "metrics": {
            "accuracy": round(float(acc) * 100, 2),
            "precision": round(float(prec) * 100, 2),
            "recall": round(float(rec) * 100, 2),
            "specificity": round(float(specificity) * 100, 2),
            "f1_score": round(float(f1) * 100, 2),
            "roc_auc": round(float(roc_auc), 4)
        },
        "confusion_matrix": {
            "matrix": [[tn, fp], [fn, tp]],
            "tn": tn,
            "fp": fp,
            "fn": fn,
            "tp": tp,
            "total_test_samples": len(y_test)
        },
        "roc_curve": {
            "fpr": sampled_fpr,
            "tpr": sampled_tpr,
            "auc": round(float(roc_auc), 4)
        },
        "classification_report": report_dict
    }
