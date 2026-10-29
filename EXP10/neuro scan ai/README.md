# NeuroScan AI - Brain Tumor MRI Detection & Classification System

An end-to-end Medical Machine Learning Web Application and CLI Toolkit. Covers **Image Loading**, **Medical Preprocessing**, **GLCM & Statistical Feature Extraction**, **Model Training**, and **Performance Evaluation** all within a **unified, interactive single-frame web interface** and command-line environment.

---

## 📌 Project Overview

- **Project Name:** NeuroScan AI - Brain Tumor Detection and Classification System
- **Domain:** Medical Image Analysis & Healthcare Computer-Aided Diagnosis (CAD)
- **Modality:** Axial Brain MRI Scans (T1-Weighted, T2-Weighted, FLAIR)
- **Target Classes:**
  1. `Normal / Healthy (Negative)`: Normal cerebral parenchyma, symmetrical lateral ventricles, absence of focal lesion.
  2. `Tumor Detected (Positive)`: Hyper-intense neoplasm with peritumoral vasogenic edema (e.g., Glioma, Meningioma, Pituitary Adenoma).

---

## 🔄 The 5-Stage Machine Learning Pipeline (In One Frame)

| Stage | Process | Description & Clinical Utility |
|---|---|---|
| **Stage 1** | **Load Input Image** | Accepts raw MRI scan upload (PNG, JPG, DICOM) or selection of clinical benchmark presets. |
| **Stage 2** | **Medical Preprocessing** | Standard resizing (224×224), Grayscale intensity transformation, Gaussian Blur noise filter (5×5 kernel), **CLAHE** (Contrast-Limited Adaptive Histogram Equalization), **Skull-Stripping & Otsu Thresholding** to isolate the brain parenchyma, and Tumor candidate ROI localization. |
| **Stage 3** | **Feature Extraction** | Computes **17 quantitative medical biomarkers** across 3 categories: <br>• **Haralick GLCM Texture Features**: Contrast, Dissimilarity, Homogeneity, Energy, Correlation, ASM.<br>• **First-Order Statistical Intensity**: Mean, Variance, Std Deviation, Skewness, Kurtosis, Shannon Entropy.<br>• **Morphological & Shape Features**: Tumor Candidate Area (px), Perimeter (px), Circularity (compactness), Solidity, Brain Area Ratio (%). |
| **Stage 4** | **Model Training** | Trains supervised classifiers with StandardScaler feature normalization:<br>• **Support Vector Machine (SVM)** with Radial Basis Function (RBF) or Linear kernel.<br>• **Random Forest Classifier** (Ensemble with Gini impurity tree splits).<br>• **K-Nearest Neighbors (KNN)**.<br>• **Logistic Regression** (L2 Regularized). |
| **Stage 5** | **Performance Evaluation** | Real-time diagnostic evaluation metrics:<br>• **Accuracy, Precision, Recall / Sensitivity, Specificity, F1-Score, ROC-AUC**.<br>• **Confusion Matrix Heatmap** (True Positives, True Negatives, False Positives, False Negatives).<br>• **Interactive ROC Curve** coordinate plot.<br>• Diagnostic overlay with bounding box and probability distribution. |

---

## 💻 How to Compile and Run the Project

### Step 1: Install Dependencies
Open **Windows PowerShell** or **Command Prompt** in the project folder (`d:\project\New folder`) and run:
```bash
pip install -r requirements.txt
```

---

### Step 2: Run the One-Frame Web Application
Launch the Flask medical server:
```bash
python app.py
```
Open your web browser and navigate to:
```
http://127.0.0.1:5050
```
*(You can also specify a custom port with `python app.py --port 5000`)*

#### What You Can Do in the Web App:
1. **One-Frame ML Pipeline Tab**:
   - Choose a sample MRI scan (Glioma, Meningioma, Pituitary, Normal A, Normal B) or upload your own scan.
   - Click **"Execute 5-Step ML Pipeline"**.
   - Watch the 5-step stepper tracker light up in real-time.
   - Inspect all 8 intermediate preprocessing images in the visual filmstrip (Original → Grayscale → Gaussian Blur → CLAHE → Otsu Mask → ROI Mask → Heatmap → ROI Overlay).
   - View extracted Haralick GLCM, Intensity, and Shape features with the interactive Radar Profile Chart.
   - View the Clinical Verdict badge with confidence percentage and probability breakdown.
2. **Model Training & Evaluation Tab**:
   - Select algorithm (SVM, Random Forest, KNN, Logistic Regression).
   - Adjust hyperparameters (Regularization C, Number of Trees, K Neighbors, Test Split Ratio).
   - Click **"Train Classifier On MRI Dataset"** to retrain live.
   - Inspect the interactive **Confusion Matrix** (TN, FP, FN, TP) and **ROC Curve** chart.
3. **Code Playground & Compiler Tab**:
   - Select ready-to-run templates or **type your own custom Python program** directly into the code editor.
   - Click **"Compile & Run Code"** to execute the script in the integrated sandbox and see the console output (stdout/stderr) live!

---

### Step 3: Run Standalone CLI Commands (Terminal)

You can also compile, train, evaluate, and predict directly from your terminal:

#### 1. Train the Classifier:
```bash
# Train default SVM (RBF) model:
python train_cli.py --algo "SVM (RBF)" --c 1.0

# Or train Random Forest with 150 trees:
python train_cli.py --algo "Random Forest" --trees 150

# Or train K-Nearest Neighbors:
python train_cli.py --algo "K-Nearest Neighbors" --k 5
```

#### 2. Evaluate Performance on Test Split:
```bash
python evaluate_cli.py
```
*Outputs Accuracy, Precision, Recall, Specificity, F1-Score, ROC-AUC, Confusion Matrix, and Classification Report.*

#### 3. Predict Single MRI Image:
```bash
# Predict a tumor scan:
python predict_cli.py --image "static/samples/tumor_glioma_sample.png"

# Predict a normal scan:
python predict_cli.py --image "static/samples/normal_sample_1.png"
```

---

## 📐 Mathematical Foundations

### 1. Gray-Level Co-occurrence Matrix (GLCM)
For normalized GLCM probability matrix $P(i, j)$ at distance $d$ and angle $\theta$:
- **Contrast:** $\sum_{i} \sum_{j} |i - j|^2 P(i, j)$ *(Measures local gray-level contrast variations)*
- **Homogeneity (IDM):** $\sum_{i} \sum_{j} \frac{P(i, j)}{1 + |i - j|^2}$ *(Measures structural smoothness)*
- **Energy (Uniformity):** $\sqrt{\sum_{i} \sum_{j} P(i, j)^2}$ *(Measures orderliness of tissue texture)*
- **Correlation:** $\sum_{i} \sum_{j} \frac{(i - \mu_i)(j - \mu_j) P(i, j)}{\sigma_i \sigma_j}$ *(Measures linear dependencies)*

### 2. Performance Evaluation Metrics
- **Accuracy:** $\frac{TP + TN}{TP + TN + FP + FN} \times 100\%$
- **Precision (Positive Predictive Value):** $\frac{TP}{TP + FP} \times 100\%$
- **Recall / Sensitivity:** $\frac{TP}{TP + FN} \times 100\%$
- **Specificity (True Negative Rate):** $\frac{TN}{TN + FP} \times 100\%$
- **F1-Score:** $2 \times \frac{\text{Precision} \times \text{Recall}}{\text{Precision} + \text{Recall}}$

---

## 📁 Project Directory Structure

```
d:\project\New folder/
│
├── app.py                      # Flask backend & REST API server
├── train_cli.py                # Standalone CLI training script
├── evaluate_cli.py             # Standalone CLI evaluation script
├── predict_cli.py              # Standalone CLI image inference script
├── requirements.txt            # Python dependencies
├── README.md                   # Complete system documentation
│
├── ml_engine/                  # Modular Machine Learning Engine
│   ├── __init__.py
│   ├── preprocessor.py         # Filtering, CLAHE, Skull stripping, Otsu masking
│   ├── feature_extractor.py    # GLCM Haralick, Intensity, & Shape feature extraction
│   ├── dataset_generator.py    # Synthetic anatomical MRI synthesis & dataset builder
│   ├── model_trainer.py        # SVM, Random Forest, KNN, Logistic Regression models
│   ├── evaluator.py            # Confusion Matrix, ROC-AUC, Classification metrics
│   └── mri_dataset.npz         # Cached 120-sample balanced feature dataset
│
├── saved_models/               # Persisted trained models & standard scalers
│   ├── mri_classifier.pkl      # Active trained classifier model
│   └── scaler.pkl              # Fitted StandardScaler
│
├── static/
│   ├── css/
│   │   └── style.css           # Modern dark medical glassmorphism design system
│   ├── js/
│   │   └── app.js              # One-frame pipeline controller & interactive charts
│   └── samples/                # Benchmark MRI scans (Normal & Tumor types)
│       ├── normal_sample_1.png
│       ├── normal_sample_2.png
│       ├── tumor_glioma_sample.png
│       ├── tumor_meningioma_sample.png
│       └── tumor_pituitary_sample.png
│
├── templates/
│   └── index.html              # Single-frame responsive web application
│
└── uploads/                    # Temporary user uploaded MRI scans
```

---

## 🛡️ License & Academic Usage
This project is built for educational, academic, and research demonstration purposes in Medical Computer Vision and Machine Learning.
