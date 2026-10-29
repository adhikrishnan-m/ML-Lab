"""
Machine Learning Model Training and Inference Module.
Implements:
- Support Vector Machine (SVM)
- Random Forest Classifier (RF)
- K-Nearest Neighbors (KNN)
- Logistic Regression (LR)

Includes feature standard scaling, hyperparameter tuning, model persistence,
and probability calibration.
"""

import os
import pickle
import warnings
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression

warnings.filterwarnings("ignore", category=FutureWarning)

class ModelManager:
    def __init__(self, model_dir="saved_models"):
        self.model_dir = model_dir
        os.makedirs(self.model_dir, exist_ok=True)
        self.model_path = os.path.join(self.model_dir, "mri_classifier.pkl")
        self.scaler_path = os.path.join(self.model_dir, "scaler.pkl")

        self.model = None
        self.scaler = None
        self.active_algorithm = "SVM (RBF)"
        self.feature_names = []
        self.last_train_results = None

        # Load existing if available
        self.load_model()

    def get_classifier(self, algorithm="SVM (RBF)", **kwargs):
        """Initializes model instance based on chosen algorithm."""
        if algorithm == "SVM (RBF)":
            c_val = float(kwargs.get("c_val", 1.0))
            return SVC(C=c_val, kernel="rbf", probability=True, random_state=42)
        elif algorithm == "SVM (Linear)":
            c_val = float(kwargs.get("c_val", 1.0))
            return SVC(C=c_val, kernel="linear", probability=True, random_state=42)
        elif algorithm == "Random Forest":
            n_est = int(kwargs.get("n_estimators", 100))
            return RandomForestClassifier(n_estimators=n_est, max_depth=6, random_state=42)
        elif algorithm == "K-Nearest Neighbors":
            k_val = int(kwargs.get("k_neighbors", 5))
            return KNeighborsClassifier(n_neighbors=k_val)
        elif algorithm == "Logistic Regression":
            return LogisticRegression(C=1.0, max_iter=500, random_state=42)
        else:
            return SVC(kernel="rbf", probability=True, random_state=42)

    def train(self, X, y, feature_names, algorithm="SVM (RBF)", test_size=0.25, **hyperparams):
        """
        Trains the classifier with data normalization and evaluation split.
        """
        self.feature_names = feature_names
        self.active_algorithm = algorithm

        # Train / Test split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42, stratify=y
        )

        # Standard Scaler
        self.scaler = StandardScaler()
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)

        # Fit Model
        clf = self.get_classifier(algorithm, **hyperparams)
        clf.fit(X_train_scaled, y_train)
        self.model = clf

        # Feature Importances if available
        feature_importance = {}
        if hasattr(clf, "feature_importances_"):
            importances = clf.feature_importances_
            feature_importance = {name: float(round(imp, 4)) for name, imp in zip(feature_names, importances)}
            # Sort descending
            feature_importance = dict(sorted(feature_importance.items(), key=lambda item: item[1], reverse=True))

        # Save model and scaler
        self.save_model()

        return {
            "algorithm": algorithm,
            "train_samples": len(X_train),
            "test_samples": len(X_test),
            "X_test_scaled": X_test_scaled,
            "y_test": y_test,
            "feature_importance": feature_importance
        }

    def predict(self, feature_vector):
        """
        Runs inference on a single 1D feature vector.
        """
        if self.model is None or self.scaler is None:
            raise RuntimeError("Model has not been trained yet. Please train the model first.")

        feats_np = np.array(feature_vector).reshape(1, -1)
        feats_scaled = self.scaler.transform(feats_np)

        pred_class = int(self.model.predict(feats_scaled)[0])
        
        # Probabilities
        if hasattr(self.model, "predict_proba"):
            probs = self.model.predict_proba(feats_scaled)[0]
            confidence = float(probs[pred_class])
            prob_dict = {
                "normal": float(round(probs[0], 4)),
                "tumor": float(round(probs[1], 4))
            }
        else:
            confidence = 1.0
            prob_dict = {
                "normal": 1.0 if pred_class == 0 else 0.0,
                "tumor": 1.0 if pred_class == 1 else 0.0
            }

        label = "Tumor Detected" if pred_class == 1 else "Normal / Healthy"
        status = "Positive" if pred_class == 1 else "Negative"

        return {
            "predicted_class": pred_class,
            "label": label,
            "clinical_status": status,
            "confidence": round(confidence * 100, 2),
            "probabilities": prob_dict,
            "model_algorithm": self.active_algorithm
        }

    def save_model(self):
        """Saves active model, scaler, and metadata to disk."""
        payload = {
            "model": self.model,
            "scaler": self.scaler,
            "active_algorithm": self.active_algorithm,
            "feature_names": self.feature_names
        }
        with open(self.model_path, "wb") as f:
            pickle.dump(payload, f)

    def load_model(self):
        """Loads model from disk if existing."""
        if os.path.exists(self.model_path):
            try:
                with open(self.model_path, "rb") as f:
                    payload = pickle.load(f)
                    self.model = payload.get("model")
                    self.scaler = payload.get("scaler")
                    self.active_algorithm = payload.get("active_algorithm", "SVM (RBF)")
                    self.feature_names = payload.get("feature_names", [])
                return True
            except Exception as e:
                print(f"Error loading model: {e}")
                return False
        return False
