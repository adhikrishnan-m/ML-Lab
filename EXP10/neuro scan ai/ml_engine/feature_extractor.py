"""
Feature Extraction Module for Medical Image Analysis.
Extracts:
1. GLCM (Gray-Level Co-occurrence Matrix) Haralick Texture Features:
   - Contrast
   - Dissimilarity
   - Homogeneity (IDM)
   - Energy
   - Correlation
   - Angular Second Moment (ASM)
2. First-Order Intensity Statistics:
   - Mean Intensity
   - Standard Deviation
   - Variance
   - Skewness
   - Kurtosis
   - Shannon Entropy
3. Morphological & Shape Features:
   - Tumor Area (pixels)
   - Tumor Perimeter
   - Circularity (Compactness)
   - Solidity
   - Area Ratio (Tumor vs Brain Tissue)
"""

import numpy as np
import cv2
from scipy.stats import skew, kurtosis

def compute_glcm(image_gray, distances=[1], angles=[0, np.pi/4, np.pi/2, 3*np.pi/4], levels=32):
    """
    Computes normalized Gray-Level Co-occurrence Matrix (GLCM).
    Quantizes 256 gray levels down to `levels` (e.g. 32) for computational efficiency
    and robustness to sensor noise.
    """
    # Quantize image into bins 0..levels-1
    quantized = (image_gray // (256 // levels)).astype(np.uint8)
    h, w = quantized.shape
    glcm = np.zeros((levels, levels, len(distances), len(angles)), dtype=np.float64)

    for d_idx, d in enumerate(distances):
        for a_idx, angle in enumerate(angles):
            dy = int(round(-d * np.sin(angle)))
            dx = int(round(d * np.cos(angle)))

            # Coordinate bounds
            y_start = max(0, -dy)
            y_end = min(h, h - dy)
            x_start = max(0, -dx)
            x_end = min(w, w - dx)

            y1 = np.arange(y_start, y_end)
            x1 = np.arange(x_start, x_end)
            grid_y1, grid_x1 = np.meshgrid(y1, x1, indexing='ij')

            grid_y2 = grid_y1 + dy
            grid_x2 = grid_x1 + dx

            vals1 = quantized[grid_y1, grid_x1]
            vals2 = quantized[grid_y2, grid_x2]

            # Vectorized 2D histogram
            np.add.at(glcm[:, :, d_idx, a_idx], (vals1, vals2), 1)

            # Symmetrize GLCM
            glcm[:, :, d_idx, a_idx] += glcm[:, :, d_idx, a_idx].T

            # Normalize to probabilities
            total = glcm[:, :, d_idx, a_idx].sum()
            if total > 0:
                glcm[:, :, d_idx, a_idx] /= total

    return glcm

def extract_glcm_features(glcm):
    """
    Computes Haralick texture features from the GLCM:
    Contrast, Dissimilarity, Homogeneity, Energy, Correlation, ASM.
    Averages over all orientations and distances.
    """
    levels = glcm.shape[0]
    i_coords, j_coords = np.indices((levels, levels))
    diff_ij = np.abs(i_coords - j_coords)
    diff_sq = diff_ij ** 2

    contrasts = []
    dissimilarities = []
    homogeneities = []
    energies = []
    correlations = []
    asms = []

    for d in range(glcm.shape[2]):
        for a in range(glcm.shape[3]):
            P = glcm[:, :, d, a]

            # Contrast: sum( |i-j|^2 * P(i,j) )
            cont = np.sum(diff_sq * P)
            contrasts.append(cont)

            # Dissimilarity: sum( |i-j| * P(i,j) )
            diss = np.sum(diff_ij * P)
            dissimilarities.append(diss)

            # Homogeneity: sum( P(i,j) / (1 + (i-j)^2) )
            homo = np.sum(P / (1.0 + diff_sq))
            homogeneities.append(homo)

            # ASM (Angular Second Moment): sum( P(i,j)^2 )
            asm_val = np.sum(P ** 2)
            asms.append(asm_val)

            # Energy: sqrt(ASM)
            energy = np.sqrt(asm_val)
            energies.append(energy)

            # Correlation
            mu_i = np.sum(i_coords * P)
            mu_j = np.sum(j_coords * P)
            var_i = np.sum(((i_coords - mu_i) ** 2) * P)
            var_j = np.sum(((j_coords - mu_j) ** 2) * P)
            std_prod = np.sqrt(var_i * var_j)

            if std_prod > 1e-8:
                corr = np.sum((i_coords - mu_i) * (j_coords - mu_j) * P) / std_prod
            else:
                corr = 1.0
            correlations.append(corr)

    return {
        "glcm_contrast": float(np.mean(contrasts)),
        "glcm_dissimilarity": float(np.mean(dissimilarities)),
        "glcm_homogeneity": float(np.mean(homogeneities)),
        "glcm_energy": float(np.mean(energies)),
        "glcm_correlation": float(np.mean(correlations)),
        "glcm_asm": float(np.mean(asms))
    }

def extract_intensity_features(clean_mri, brain_mask=None):
    """
    Computes first-order statistical properties on brain tissue pixels.
    """
    if brain_mask is not None and np.sum(brain_mask > 0) > 0:
        pixels = clean_mri[brain_mask > 0].astype(np.float64)
    else:
        pixels = clean_mri[clean_mri > 10].astype(np.float64)

    if len(pixels) == 0:
        pixels = clean_mri.flatten().astype(np.float64)

    mean_val = float(np.mean(pixels))
    std_val = float(np.std(pixels))
    var_val = float(np.var(pixels))
    skew_val = float(skew(pixels))
    kurt_val = float(kurtosis(pixels))

    # Shannon Entropy
    hist, _ = np.histogram(pixels, bins=32, range=(0, 256), density=True)
    hist = hist[hist > 0]
    entropy_val = float(-np.sum(hist * np.log2(hist))) if len(hist) > 0 else 0.0

    return {
        "intensity_mean": round(mean_val, 4),
        "intensity_std": round(std_val, 4),
        "intensity_variance": round(var_val, 4),
        "intensity_skewness": round(skew_val, 4),
        "intensity_kurtosis": round(kurt_val, 4),
        "shannon_entropy": round(entropy_val, 4)
    }

def extract_shape_features(tumor_mask, brain_mask=None):
    """
    Extracts geometric and morphological features of tumor candidate regions.
    """
    contours, _ = cv2.findContours(tumor_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    valid_contours = [c for c in contours if cv2.contourArea(c) > 20]

    brain_area = float(np.sum(brain_mask > 0)) if brain_mask is not None else float(tumor_mask.size)
    if brain_area == 0:
        brain_area = float(tumor_mask.size)

    if not valid_contours:
        return {
            "tumor_area_px": 0.0,
            "tumor_perimeter_px": 0.0,
            "circularity": 0.0,
            "solidity": 0.0,
            "area_ratio_pct": 0.0
        }

    # Largest abnormal region
    c = max(valid_contours, key=cv2.contourArea)
    area = float(cv2.contourArea(c))
    perimeter = float(cv2.arcLength(c, True))

    # Circularity = 4 * pi * Area / Perimeter^2 (1.0 for perfect circle)
    circularity = (4.0 * np.pi * area) / (perimeter ** 2) if perimeter > 0 else 0.0
    circularity = min(1.0, max(0.0, circularity))

    # Solidity = Area / Convex Hull Area
    hull = cv2.convexHull(c)
    hull_area = float(cv2.contourArea(hull))
    solidity = (area / hull_area) if hull_area > 0 else 0.0

    # Area ratio percentage relative to whole brain
    area_ratio = (area / brain_area) * 100.0

    return {
        "tumor_area_px": round(area, 2),
        "tumor_perimeter_px": round(perimeter, 2),
        "circularity": round(circularity, 4),
        "solidity": round(solidity, 4),
        "area_ratio_pct": round(area_ratio, 4)
    }

def extract_features(clean_mri, brain_mask=None, tumor_mask=None):
    """
    Full feature extraction orchestrator.
    Returns:
        dict: Combined feature dictionary and an aligned feature vector for ML models.
    """
    # 1. Texture features
    glcm = compute_glcm(clean_mri)
    glcm_feats = extract_glcm_features(glcm)

    # 2. Intensity features
    intensity_feats = extract_intensity_features(clean_mri, brain_mask)

    # 3. Shape features
    shape_feats = extract_shape_features(tumor_mask, brain_mask)

    combined = {}
    combined.update(glcm_feats)
    combined.update(intensity_feats)
    combined.update(shape_feats)

    # Deterministic vector order (17 numeric features)
    feature_keys = [
        "glcm_contrast", "glcm_dissimilarity", "glcm_homogeneity",
        "glcm_energy", "glcm_correlation", "glcm_asm",
        "intensity_mean", "intensity_std", "intensity_variance",
        "intensity_skewness", "intensity_kurtosis", "shannon_entropy",
        "tumor_area_px", "tumor_perimeter_px", "circularity",
        "solidity", "area_ratio_pct"
    ]

    vector = np.array([combined[k] for k in feature_keys], dtype=np.float64)

    return {
        "feature_dict": combined,
        "feature_keys": feature_keys,
        "feature_vector": vector.tolist(),
        "vector_np": vector
    }
