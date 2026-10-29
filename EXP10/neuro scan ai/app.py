"""
NeuroScan AI - Brain Tumor MRI Classification & Diagnostics Web Application.
Unified One-Frame Interactive Medical Machine Learning Dashboard.
"""

import os
import sys
import io
import time
import subprocess
import tempfile
from flask import Flask, render_template, request, jsonify, send_from_directory
from werkzeug.utils import secure_filename

from ml_engine.preprocessor import preprocess_pipeline
from ml_engine.feature_extractor import extract_features
from ml_engine.dataset_generator import ensure_sample_images, generate_training_dataset
from ml_engine.model_trainer import ModelManager
from ml_engine.evaluator import evaluate_model

# Initialize Flask App
app = Flask(__name__, static_folder="static", template_folder="templates")
app.config["SECRET_KEY"] = "neuroscan-ai-medical-key-2026"
app.config["UPLOAD_FOLDER"] = os.path.join(os.path.dirname(__file__), "uploads")
app.config["SAMPLES_FOLDER"] = os.path.join(os.path.dirname(__file__), "static", "samples")
app.config["MAX_CONTENT_LENGTH"] = 16 * 1024 * 1024  # 16 MB max

os.makedirs(app.config["UPLOAD_FOLDER"], exist_ok=True)
os.makedirs(app.config["SAMPLES_FOLDER"], exist_ok=True)

# Ensure sample images exist
ensure_sample_images(app.config["SAMPLES_FOLDER"])

# Initialize ML Model Manager
DATASET_CACHE = os.path.join(os.path.dirname(__file__), "ml_engine", "mri_dataset.npz")
model_mgr = ModelManager(model_dir=os.path.join(os.path.dirname(__file__), "saved_models"))

# Pre-train baseline model if not already trained
def initialize_engine():
    print("[INIT] Initializing NeuroScan Engine...")
    X, y, feature_names = generate_training_dataset(num_samples=120, cache_file=DATASET_CACHE)
    if model_mgr.model is None or model_mgr.scaler is None:
        print("[INIT] Training baseline SVM model on MRI dataset...")
        train_res = model_mgr.train(X, y, feature_names, algorithm="SVM (RBF)", test_size=0.25)
        eval_res = evaluate_model(model_mgr.model, train_res["X_test_scaled"], train_res["y_test"])
        model_mgr.last_train_results = {**train_res, "evaluation": eval_res}
        print(f"[INIT] Model trained successfully! Baseline Accuracy: {eval_res['metrics']['accuracy']}%")
    else:
        print("[INIT] Loaded existing trained model.")

initialize_engine()

# ==================== WEB ROUTES ====================

@app.route("/")
def index():
    """Renders the single-frame unified dashboard."""
    return render_template("index.html")

@app.route("/api/samples", methods=["GET"])
def get_samples():
    """Returns metadata and filenames for available sample MRI scans."""
    samples = [
        {
            "id": "normal_1",
            "filename": "normal_sample_1.png",
            "label": "Normal Brain (Scan A)",
            "ground_truth": "Normal",
            "description": "Healthy cerebral parenchyma with normal ventricular symmetry and no focal lesions."
        },
        {
            "id": "normal_2",
            "filename": "normal_sample_2.png",
            "label": "Normal Brain (Scan B)",
            "ground_truth": "Normal",
            "description": "Symmetrical cortical structures and normal gray/white matter differentiation."
        },
        {
            "id": "tumor_glioma",
            "filename": "tumor_glioma_sample.png",
            "label": "Glioma Tumor",
            "ground_truth": "Tumor",
            "description": "Infiltrative hyper-intense mass in cerebral lobe with peritumoral vasogenic edema."
        },
        {
            "id": "tumor_meningioma",
            "filename": "tumor_meningioma_sample.png",
            "label": "Meningioma Tumor",
            "ground_truth": "Tumor",
            "description": "Well-circumscribed extra-axial mass with characteristic dural tail enhancement."
        },
        {
            "id": "tumor_pituitary",
            "filename": "tumor_pituitary_sample.png",
            "label": "Pituitary Adenoma",
            "ground_truth": "Tumor",
            "description": "Sellar region hyperdense tumor with focal distortion of the skull base."
        }
    ]
    return jsonify({"success": True, "samples": samples})

@app.route("/api/process_image", methods=["POST"])
def process_image():
    """
    Executes complete pipeline on input image:
    1. Load image (Upload or Sample)
    2. Preprocess (Gray, Gaussian Blur, CLAHE, Skull stripping, Otsu mask, Heatmap)
    3. Feature Extraction (GLCM, Intensity, Shape)
    4. Inference & Diagnosis
    """
    try:
        image_bytes = None
        source_name = "Upload"

        if "file" in request.files and request.files["file"].filename != "":
            file = request.files["file"]
            image_bytes = file.read()
            source_name = file.filename
        elif "sample" in request.form:
            sample_file = request.form.get("sample")
            sample_path = os.path.join(app.config["SAMPLES_FOLDER"], secure_filename(sample_file))
            if os.path.exists(sample_path):
                with open(sample_path, "rb") as f:
                    image_bytes = f.read()
                source_name = sample_file
            else:
                return jsonify({"success": False, "error": f"Sample not found: {sample_file}"}), 404
        else:
            return jsonify({"success": False, "error": "No image file or sample provided."}), 400

        # Stage 1 & 2: Preprocess
        prep_results = preprocess_pipeline(image_bytes)

        # Stage 3: Feature Extraction
        feat_results = extract_features(
            prep_results["clean_mri"],
            prep_results["brain_mask_np"],
            prep_results["tumor_mask_np"]
        )

        # Stage 4 & 5: Prediction Inference
        prediction = None
        if model_mgr.model is not None and model_mgr.scaler is not None:
            prediction = model_mgr.predict(feat_results["feature_vector"])

        return jsonify({
            "success": True,
            "source": source_name,
            "stages": prep_results["stages"],
            "metadata": prep_results["metadata"],
            "features": feat_results["feature_dict"],
            "feature_keys": feat_results["feature_keys"],
            "feature_vector": feat_results["feature_vector"],
            "prediction": prediction
        })

    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({"success": False, "error": str(e)}), 500

@app.route("/api/train", methods=["POST"])
def train_model():
    """
    Trains/Retrains the machine learning model with user-selected algorithm and hyperparameters.
    """
    try:
        data = request.get_json() or {}
        algorithm = data.get("algorithm", "SVM (RBF)")
        test_size = float(data.get("test_size", 0.25))
        c_val = float(data.get("c_val", 1.0))
        n_estimators = int(data.get("n_estimators", 100))
        k_neighbors = int(data.get("k_neighbors", 5))

        X, y, feature_names = generate_training_dataset(num_samples=120, cache_file=DATASET_CACHE)

        train_res = model_mgr.train(
            X, y, feature_names,
            algorithm=algorithm,
            test_size=test_size,
            c_val=c_val,
            n_estimators=n_estimators,
            k_neighbors=k_neighbors
        )

        eval_res = evaluate_model(model_mgr.model, train_res["X_test_scaled"], train_res["y_test"])
        model_mgr.last_train_results = {**train_res, "evaluation": eval_res}

        return jsonify({
            "success": True,
            "message": f"Successfully trained {algorithm} model!",
            "algorithm": algorithm,
            "train_samples": train_res["train_samples"],
            "test_samples": train_res["test_samples"],
            "feature_importance": train_res["feature_importance"],
            "evaluation": eval_res
        })

    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500

@app.route("/api/evaluate", methods=["GET"])
def evaluate():
    """Returns current active model's test performance metrics."""
    try:
        if model_mgr.last_train_results and "evaluation" in model_mgr.last_train_results:
            return jsonify({
                "success": True,
                "algorithm": model_mgr.active_algorithm,
                "evaluation": model_mgr.last_train_results["evaluation"],
                "feature_importance": model_mgr.last_train_results.get("feature_importance", {})
            })

        X, y, feature_names = generate_training_dataset(num_samples=120, cache_file=DATASET_CACHE)
        from sklearn.model_selection import train_test_split
        _, X_test, _, y_test = train_test_split(X, y, test_size=0.25, random_state=42, stratify=y)
        X_test_scaled = model_mgr.scaler.transform(X_test)
        eval_res = evaluate_model(model_mgr.model, X_test_scaled, y_test)

        return jsonify({
            "success": True,
            "algorithm": model_mgr.active_algorithm,
            "evaluation": eval_res
        })
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500

@app.route("/api/run_code", methods=["POST"])
def run_code():
    """
    Executes user-entered Python code in a controlled subprocess and returns stdout/stderr.
    Allows user to type program and compile/run code directly in the web application.
    """
    try:
        data = request.get_json() or {}
        code = data.get("code", "")

        if not code.strip():
            return jsonify({"success": False, "error": "Empty code provided."}), 400

        # Create temporary script file
        with tempfile.NamedTemporaryFile("w", suffix=".py", delete=False, encoding="utf-8") as temp_file:
            temp_path = temp_file.name
            temp_file.write(code)

        start_time = time.time()
        # Run python script with 10s timeout
        process = subprocess.run(
            [sys.executable, temp_path],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            timeout=12,
            cwd=os.path.dirname(os.path.abspath(__file__))
        )
        elapsed = round(time.time() - start_time, 3)

        # Cleanup
        try:
            os.remove(temp_path)
        except Exception:
            pass

        return jsonify({
            "success": True,
            "returncode": process.returncode,
            "stdout": process.stdout,
            "stderr": process.stderr,
            "elapsed_seconds": elapsed
        })

    except subprocess.TimeoutExpired:
        try:
            os.remove(temp_path)
        except Exception:
            pass
        return jsonify({
            "success": False,
            "error": "Execution timed out (Limit: 12 seconds). Check for infinite loops."
        }), 408
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500

@app.route("/api/code_templates", methods=["GET"])
def code_templates():
    """Returns pre-loaded code snippets for each stage for the interactive editor."""
    templates = {
        "pipeline": '''# ==========================================
# Complete Medical ML Pipeline (End-to-End)
# ==========================================
import cv2
import numpy as np
from ml_engine.preprocessor import preprocess_pipeline
from ml_engine.feature_extractor import extract_features
from ml_engine.model_trainer import ModelManager

print(">>> Step 1: Loading MRI Image...")
image_path = "static/samples/tumor_glioma_sample.png"
print(f"Reading: {image_path}")

print("\\n>>> Step 2: Running Preprocessing Pipeline...")
prep = preprocess_pipeline(image_path)
print(f"Original size: {prep['metadata']['dimensions']}")
print(f"Skull-stripping & Otsu Threshold applied.")
print(f"Tumor candidate detected: {prep['metadata']['has_tumor_candidate']}")

print("\\n>>> Step 3: Extracting GLCM & Statistical Features...")
feats = extract_features(prep['clean_mri'], prep['brain_mask_np'], prep['tumor_mask_np'])
print("Top Extracted Features:")
for k, v in list(feats['feature_dict'].items())[:6]:
    print(f"  * {k}: {v:.4f}")

print("\\n>>> Step 4 & 5: Model Inference & Diagnosis...")
mgr = ModelManager()
result = mgr.predict(feats['feature_vector'])
print(f"Clinical Diagnosis : {result['label']}")
print(f"Confidence Level   : {result['confidence']}%")
print(f"Active Algorithm   : {result['model_algorithm']}")
print(f"Probabilities      : Normal={result['probabilities']['normal']*100:.1f}%, Tumor={result['probabilities']['tumor']*100:.1f}%")
print("\\n[SUCCESS] Pipeline execution finished.")
''',
        "preprocess": '''# ==========================================
# Stage 2: Medical Image Preprocessing
# ==========================================
import cv2
import numpy as np

img = cv2.imread("static/samples/tumor_glioma_sample.png")
print("1. Original Image Shape:", img.shape)

# Grayscale Conversion
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
print("2. Grayscale Mean Intensity:", np.mean(gray))

# Gaussian Blur for Noise Reduction (5x5 kernel)
blurred = cv2.GaussianBlur(gray, (5, 5), 0)
print("3. Gaussian Filtering completed.")

# Contrast Enhancement via CLAHE
clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))
enhanced = clahe.apply(blurred)
print("4. CLAHE Contrast Enhanced. Range:", enhanced.min(), "to", enhanced.max())

# Otsu Thresholding for Skull/Brain Masking
ret, thresh = cv2.threshold(enhanced, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
print(f"5. Otsu Optimal Threshold Value: {ret}")

print("\\n[SUCCESS] Preprocessing completed.")
''',
        "features": '''# ==========================================
# Stage 3: GLCM Texture & Statistical Features
# ==========================================
import cv2
import numpy as np
from ml_engine.feature_extractor import compute_glcm, extract_glcm_features, extract_intensity_features

img = cv2.imread("static/samples/tumor_glioma_sample.png", cv2.IMREAD_GRAYSCALE)

# Compute GLCM
glcm = compute_glcm(img, distances=[1], angles=[0, np.pi/4, np.pi/2, 3*np.pi/4], levels=32)
print("GLCM Computed. Matrix Shape:", glcm.shape)

# Haralick Texture Metrics
texture_feats = extract_glcm_features(glcm)
print("\\nHaralick GLCM Texture Features:")
for feat, val in texture_feats.items():
    print(f"  {feat:<22}: {val:.5f}")

# First-Order Statistical Features
stats = extract_intensity_features(img)
print("\\nIntensity Statistical Features:")
for feat, val in stats.items():
    print(f"  {feat:<22}: {val}")
''',
        "train": '''# ==========================================
# Stage 4 & 5: Model Training & Evaluation
# ==========================================
from ml_engine.dataset_generator import generate_training_dataset
from ml_engine.model_trainer import ModelManager
from ml_engine.evaluator import evaluate_model

print("Generating/Loading Balanced MRI Feature Dataset...")
X, y, feature_names = generate_training_dataset(num_samples=120)
print(f"Dataset Shape: {X.shape}, Classes: Normal={sum(y==0)}, Tumor={sum(y==1)}")

# Initialize Manager & Train SVM
mgr = ModelManager()
print("\\nTraining Support Vector Machine (RBF Kernel)...")
train_res = mgr.train(X, y, feature_names, algorithm="SVM (RBF)", test_size=0.25, c_val=1.0)

# Evaluate Performance
eval_res = evaluate_model(mgr.model, train_res["X_test_scaled"], train_res["y_test"])
metrics = eval_res["metrics"]
cm = eval_res["confusion_matrix"]

print("\\n================ PERFORMANCE EVALUATION ================")
print(f"Accuracy    : {metrics['accuracy']}%")
print(f"Precision   : {metrics['precision']}%")
print(f"Recall/Sens : {metrics['recall']}%")
print(f"Specificity : {metrics['specificity']}%")
print(f"F1-Score    : {metrics['f1_score']}%")
print(f"ROC-AUC     : {metrics['roc_auc']}")
print("-------------------------------------------------------")
print(f"Confusion Matrix: TN={cm['tn']}, FP={cm['fp']}, FN={cm['fn']}, TP={cm['tp']}")
print("=======================================================")
'''
    }
    return jsonify({"success": True, "templates": templates})

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="NeuroScan AI Web Application")
    parser.add_argument("--port", type=int, default=int(os.environ.get("PORT", 5050)), help="Port to run server on")
    args, _ = parser.parse_known_args()

    port = args.port
    print("\n" + "="*60)
    print("  NEUROSCAN AI - Medical ML Web Application")
    print(f"  Access Web Dashboard at: http://127.0.0.1:{port}")
    print("="*60 + "\n")
    app.run(host="127.0.0.1", port=port, debug=False)
