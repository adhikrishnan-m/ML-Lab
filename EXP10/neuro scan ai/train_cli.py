"""
NeuroScan AI - CLI Training Script
Usage:
    python train_cli.py --algo "SVM (RBF)" --c 1.0
    python train_cli.py --algo "Random Forest" --trees 150
    python train_cli.py --algo "K-Nearest Neighbors" --k 5
"""

import argparse
import os
import sys
from ml_engine.dataset_generator import generate_training_dataset
from ml_engine.model_trainer import ModelManager
from ml_engine.evaluator import evaluate_model

def main():
    parser = argparse.ArgumentParser(description="Train Medical Machine Learning Classifier on MRI scans.")
    parser.add_argument("--algo", type=str, default="SVM (RBF)", 
                        choices=["SVM (RBF)", "SVM (Linear)", "Random Forest", "K-Nearest Neighbors", "Logistic Regression"],
                        help="Classification algorithm to train")
    parser.add_argument("--split", type=float, default=0.25, help="Test split ratio (default: 0.25)")
    parser.add_argument("--c", type=float, default=1.0, help="Regularization C for SVM")
    parser.add_argument("--trees", type=int, default=100, help="Number of trees for Random Forest")
    parser.add_argument("--k", type=int, default=5, help="K neighbors for KNN")
    args = parser.parse_args()

    print("\n" + "="*60)
    print("   NEUROSCAN AI - MODEL TRAINING PIPELINE")
    print("="*60)
    print(f"Algorithm Selected : {args.algo}")
    print(f"Test Split Ratio   : {args.split * 100}%")

    # 1. Dataset Generation / Loading
    cache_path = os.path.join(os.path.dirname(__file__), "ml_engine", "mri_dataset.npz")
    print("\n[STEP 1/3] Loading MRI Feature Dataset...")
    X, y, feature_names = generate_training_dataset(num_samples=120, cache_file=cache_path)
    print(f"Total Samples: {len(X)} | Features: {len(feature_names)}")
    print(f"Normal Scans: {sum(y == 0)} | Tumor Scans: {sum(y == 1)}")

    # 2. Model Training
    print(f"\n[STEP 2/3] Training {args.algo} Classifier...")
    mgr = ModelManager(model_dir=os.path.join(os.path.dirname(__file__), "saved_models"))
    train_res = mgr.train(
        X, y, feature_names,
        algorithm=args.algo,
        test_size=args.split,
        c_val=args.c,
        n_estimators=args.trees,
        k_neighbors=args.k
    )
    print(f"Training completed successfully! Saved to: saved_models/mri_classifier.pkl")

    # 3. Model Evaluation
    print("\n[STEP 3/3] Evaluating on Test Split...")
    eval_res = evaluate_model(mgr.model, train_res["X_test_scaled"], train_res["y_test"])
    m = eval_res["metrics"]
    cm = eval_res["confusion_matrix"]

    print("\n" + "-"*40)
    print("       MODEL PERFORMANCE METRICS")
    print("-"*40)
    print(f"Accuracy         : {m['accuracy']}%")
    print(f"Precision        : {m['precision']}%")
    print(f"Recall (Sens.)   : {m['recall']}%")
    print(f"Specificity      : {m['specificity']}%")
    print(f"F1-Score         : {m['f1_score']}%")
    print(f"ROC-AUC Score    : {m['roc_auc']}")
    print("-"*40)
    print("CONFUSION MATRIX:")
    print(f"                 Pred Normal   Pred Tumor")
    print(f"  Actual Normal      {cm['tn']:<12} {cm['fp']}")
    print(f"  Actual Tumor       {cm['fn']:<12} {cm['tp']}")
    print("-"*40)
    print("[SUCCESS] Training and evaluation complete.\n")

if __name__ == "__main__":
    main()
