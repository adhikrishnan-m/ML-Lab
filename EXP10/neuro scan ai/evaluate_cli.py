"""
NeuroScan AI - CLI Evaluation Script
Usage:
    python evaluate_cli.py
"""

import os
import sys
from ml_engine.dataset_generator import generate_training_dataset
from ml_engine.model_trainer import ModelManager
from ml_engine.evaluator import evaluate_model
from sklearn.model_selection import train_test_split

def main():
    print("\n" + "="*60)
    print("   NEUROSCAN AI - PERFORMANCE EVALUATION")
    print("="*60)

    mgr = ModelManager(model_dir=os.path.join(os.path.dirname(__file__), "saved_models"))
    if mgr.model is None or mgr.scaler is None:
        print("[ERROR] No trained model found in 'saved_models/'.")
        print("Please run 'python train_cli.py' first.")
        sys.exit(1)

    print(f"Active Model Algorithm: {mgr.active_algorithm}")

    cache_path = os.path.join(os.path.dirname(__file__), "ml_engine", "mri_dataset.npz")
    X, y, feature_names = generate_training_dataset(num_samples=120, cache_file=cache_path)

    _, X_test, _, y_test = train_test_split(X, y, test_size=0.25, random_state=42, stratify=y)
    X_test_scaled = mgr.scaler.transform(X_test)

    eval_res = evaluate_model(mgr.model, X_test_scaled, y_test)
    m = eval_res["metrics"]
    cm = eval_res["confusion_matrix"]

    print("\n" + "-"*45)
    print("        COMPREHENSIVE TEST METRICS")
    print("-"*45)
    print(f"Accuracy                 : {m['accuracy']}%")
    print(f"Precision                : {m['precision']}%")
    print(f"Recall (Sensitivity)     : {m['recall']}%")
    print(f"Specificity              : {m['specificity']}%")
    print(f"F1-Score                 : {m['f1_score']}%")
    print(f"Area Under ROC (AUC)     : {m['roc_auc']}")
    print("-"*45)
    print("CONFUSION MATRIX:")
    print(f"                 Pred Normal   Pred Tumor")
    print(f"  Actual Normal      {cm['tn']:<12} {cm['fp']}")
    print(f"  Actual Tumor       {cm['fn']:<12} {cm['tp']}")
    print("-"*45)

    rep = eval_res["classification_report"]
    print("\nCLASSIFICATION REPORT:")
    print(f"{'Class':<12} {'Precision':<10} {'Recall':<10} {'F1-Score':<10} {'Support':<8}")
    for label in ["Normal", "Tumor"]:
        row = rep[label]
        print(f"{label:<12} {row['precision']*100:<9.1f}% {row['recall']*100:<9.1f}% {row['f1-score']*100:<9.1f}% {int(row['support'])}")
    print("="*60 + "\n")

if __name__ == "__main__":
    main()
