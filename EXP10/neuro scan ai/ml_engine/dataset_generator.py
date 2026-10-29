"""
Medical MRI Image and Dataset Generator.
Synthesizes anatomically realistic brain MRI axial scans for:
- Healthy/Normal brain tissue (Class 0: Negative)
- Tumor pathology scans (Class 1: Positive - Glioma, Meningioma, Pituitary)

Also provides a balanced training & test dataset with extracted feature vectors.
"""

import os
import cv2
import numpy as np
from PIL import Image

def generate_synthetic_mri(scan_type="normal", tumor_type="glioma", seed=None, size=(224, 224)):
    """
    Synthesizes an anatomical axial brain MRI scan.
    """
    if seed is not None:
        np.random.seed(seed)

    h, w = size
    img = np.zeros((h, w), dtype=np.uint8)

    # 1. Outer Skull & Scalp (Elliptical contour)
    center = (w // 2, h // 2)
    axes = (int(w * 0.40), int(h * 0.44))
    
    # Scalp tissue
    cv2.ellipse(img, center, axes, 0, 0, 360, 65, -1)
    # Skull bone (hyper-intense rim in CT/MRI bone window)
    cv2.ellipse(img, center, (int(axes[0]*0.97), int(axes[1]*0.97)), 0, 0, 360, 110, 3)
    # Subdural space / CSF (dark rim)
    cv2.ellipse(img, center, (int(axes[0]*0.93), int(axes[1]*0.93)), 0, 0, 360, 25, -1)
    # Brain Parenchyma (Cerebral Cortex / Gray & White Matter)
    cv2.ellipse(img, center, (int(axes[0]*0.90), int(axes[1]*0.90)), 0, 0, 360, 120, -1)

    # 2. Add anatomical gyri & sulci texture variations
    noise = np.random.normal(0, 10, (h, w)).astype(np.float32)
    brain_mask = np.zeros((h, w), dtype=np.uint8)
    cv2.ellipse(brain_mask, center, (int(axes[0]*0.90), int(axes[1]*0.90)), 0, 0, 360, 255, -1)
    
    img_float = img.astype(np.float32) + noise * (brain_mask / 255.0)
    img = np.clip(img_float, 0, 255).astype(np.uint8)

    # 3. Inter-hemispheric fissure (longitudinal cerebral fissure)
    cv2.line(img, (center[0], center[1] - axes[1] + 15), 
                  (center[0], center[1] + axes[1] - 15), 35, 1)

    # 4. Lateral Ventricles (CSF - hypo-intense dark butterfly structure)
    ventricle_color = 30
    # Left anterior & posterior horns
    pts_left = np.array([
        [center[0] - 8, center[1] - 30],
        [center[0] - 22, center[1] - 10],
        [center[0] - 16, center[1] + 20],
        [center[0] - 6, center[1] + 10]
    ], np.int32)
    cv2.fillPoly(img, [pts_left], ventricle_color)

    # Right anterior & posterior horns
    pts_right = np.array([
        [center[0] + 8, center[1] - 30],
        [center[0] + 22, center[1] - 10],
        [center[0] + 16, center[1] + 20],
        [center[0] + 6, center[1] + 10]
    ], np.int32)
    cv2.fillPoly(img, [pts_right], ventricle_color)

    # 5. Add Tumor Pathology if scan_type is 'tumor'
    if scan_type.lower() == "tumor":
        if tumor_type == "glioma":
            # Infiltrative, hyper-intense irregular mass in right or left frontal/parietal lobe
            t_center = (center[0] + int(axes[0] * 0.42), center[1] - int(axes[1] * 0.25))
            t_radius = int(w * 0.11)
            # Peritumoral vasogenic edema (moderate hyperintensity halo)
            cv2.circle(img, t_center, int(t_radius * 1.55), 160, -1)
            # Active tumor margin (hyper-intense)
            cv2.circle(img, t_center, t_radius, 220, -1)
            # Necrotic core (hypo-intense center)
            cv2.circle(img, t_center, int(t_radius * 0.45), 90, -1)

        elif tumor_type == "meningioma":
            # Well-circumscribed dural-based extra-axial mass with mass effect
            t_center = (center[0] - int(axes[0] * 0.60), center[1] + int(axes[1] * 0.15))
            t_radius = int(w * 0.10)
            cv2.circle(img, t_center, t_radius, 235, -1)
            # Dural tail
            cv2.line(img, (t_center[0] - 5, t_center[1] - t_radius - 10),
                          (t_center[0] + 5, t_center[1] + t_radius + 10), 200, 2)

        elif tumor_type == "pituitary":
            # Sellar/suprasellar mass near brainstem/optic chiasm
            t_center = (center[0], center[1] + int(axes[1] * 0.32))
            t_radius = int(w * 0.085)
            cv2.circle(img, t_center, t_radius, 225, -1)

    # Apply slight Gaussian smoothing to simulate clinical MRI point spread function
    img = cv2.GaussianBlur(img, (3, 3), 0.8)

    # Convert to 3-channel BGR for standard image formats
    img_bgr = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    return img_bgr

def ensure_sample_images(samples_dir):
    """
    Ensures standard sample MRI images exist in the samples folder.
    """
    os.makedirs(samples_dir, exist_ok=True)
    samples = [
        ("normal_sample_1.png", "normal", "none", 101),
        ("normal_sample_2.png", "normal", "none", 202),
        ("tumor_glioma_sample.png", "tumor", "glioma", 303),
        ("tumor_meningioma_sample.png", "tumor", "meningioma", 404),
        ("tumor_pituitary_sample.png", "tumor", "pituitary", 505)
    ]

    for filename, s_type, t_type, s_seed in samples:
        filepath = os.path.join(samples_dir, filename)
        if not os.path.exists(filepath):
            img_bgr = generate_synthetic_mri(scan_type=s_type, tumor_type=t_type, seed=s_seed)
            cv2.imwrite(filepath, img_bgr)

def generate_training_dataset(num_samples=120, cache_file=None):
    """
    Generates balanced feature matrix X and labels y for MRI scans.
    Classes: 0 = Normal / Healthy, 1 = Tumor
    """
    from .preprocessor import preprocess_pipeline
    from .feature_extractor import extract_features

    if cache_file and os.path.exists(cache_file):
        data = np.load(cache_file, allow_pickle=True)
        return data["X"], data["y"], list(data["feature_names"])

    X_list = []
    y_list = []
    feature_names = None

    half = num_samples // 2

    # 1. Normal Scans
    for i in range(half):
        seed = 1000 + i
        img_bgr = generate_synthetic_mri(scan_type="normal", seed=seed)
        prep = preprocess_pipeline(cv2.imencode(".png", img_bgr)[1].tobytes())
        feats = extract_features(prep["clean_mri"], prep["brain_mask_np"], prep["tumor_mask_np"])
        X_list.append(feats["feature_vector"])
        y_list.append(0)
        if feature_names is None:
            feature_names = feats["feature_keys"]

    # 2. Tumor Scans
    tumor_types = ["glioma", "meningioma", "pituitary"]
    for i in range(half):
        seed = 2000 + i
        t_type = tumor_types[i % len(tumor_types)]
        img_bgr = generate_synthetic_mri(scan_type="tumor", tumor_type=t_type, seed=seed)
        prep = preprocess_pipeline(cv2.imencode(".png", img_bgr)[1].tobytes())
        feats = extract_features(prep["clean_mri"], prep["brain_mask_np"], prep["tumor_mask_np"])
        X_list.append(feats["feature_vector"])
        y_list.append(1)

    X = np.array(X_list, dtype=np.float64)
    y = np.array(y_list, dtype=np.int64)

    if cache_file:
        os.makedirs(os.path.dirname(os.path.abspath(cache_file)), exist_ok=True)
        np.savez_compressed(cache_file, X=X, y=y, feature_names=feature_names)

    return X, y, feature_names
