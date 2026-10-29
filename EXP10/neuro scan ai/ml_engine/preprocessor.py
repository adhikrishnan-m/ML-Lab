"""
Image Preprocessing Module for Medical MRI Scans.
Stages:
1. Resizing to standard dimensions (224x224)
2. Grayscale Conversion
3. Gaussian Filtering for Noise Reduction
4. Contrast Enhancement via CLAHE (Contrast-Limited Adaptive Histogram Equalization)
5. Skull-stripping & Otsu Automatic Thresholding
6. Morphological Operations (Opening/Closing) for Brain Tissue Isolation
7. Tumor Candidate Region Detection & Heatmap Overlay
"""

import cv2
import numpy as np
import base64
from io import BytesIO
from PIL import Image

def cv2_to_base64(image_np):
    """Encodes a numpy image (BGR or Grayscale) to a base64 PNG data URL."""
    if len(image_np.shape) == 2:
        # Grayscale
        is_success, buffer = cv2.imencode(".png", image_np)
    else:
        # BGR to RGB for web display
        is_success, buffer = cv2.imencode(".png", image_np)
    if is_success:
        b64_str = base64.b64encode(buffer).decode("utf-8")
        return f"data:image/png;base64,{b64_str}"
    return ""

def preprocess_pipeline(image_path_or_bytes, target_size=(224, 224)):
    """
    Executes the multi-stage medical preprocessing pipeline.
    Returns:
        dict: Processed images at each stage (as base64 strings),
              along with metadata, clean image for feature extraction,
              and tumor mask.
    """
    if isinstance(image_path_or_bytes, str):
        img_bgr = cv2.imread(image_path_or_bytes)
        if img_bgr is None:
            raise ValueError(f"Could not load image from path: {image_path_or_bytes}")
    elif isinstance(image_path_or_bytes, bytes):
        nparr = np.frombuffer(image_path_or_bytes, np.uint8)
        img_bgr = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if img_bgr is None:
            raise ValueError("Could not decode image from provided byte buffer.")
    else:
        raise TypeError("Input must be a file path string or raw image bytes.")

    # 1. Standard Resize
    img_resized = cv2.resize(img_bgr, target_size, interpolation=cv2.INTER_AREA)

    # 2. Grayscale conversion
    gray = cv2.cvtColor(img_resized, cv2.COLOR_BGR2GRAY)

    # 3. Noise reduction using Gaussian Blur (5x5 kernel)
    gaussian_blur = cv2.GaussianBlur(gray, (5, 5), 0)

    # 4. Contrast Enhancement via CLAHE
    clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))
    clahe_enhanced = clahe.apply(gaussian_blur)

    # 5. Skull-stripping / Brain mask isolation via Otsu thresholding
    # MRI scans have a bright skull bone, dark cerebrospinal fluid, and brain tissue.
    # We apply Otsu's thresholding to find the threshold separating brain from background.
    _, otsu_thresh = cv2.threshold(clahe_enhanced, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # Morphological opening and closing to clean small artifacts
    kernel_small = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    kernel_large = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    morph_open = cv2.morphologyEx(otsu_thresh, cv2.MORPH_OPEN, kernel_small, iterations=2)
    brain_mask = cv2.morphologyEx(morph_open, cv2.MORPH_CLOSE, kernel_large, iterations=2)

    # 6. Brain Tissue Isolation (Masking)
    # Erode brain mask slightly to ensure outer skull boundary is completely excluded
    eroded_brain_mask = cv2.erode(brain_mask, kernel_small, iterations=2)
    skull_stripped = cv2.bitwise_and(clahe_enhanced, clahe_enhanced, mask=eroded_brain_mask)

    # 7. High-Intensity Abnormality / Tumor Candidate Region Extraction
    brain_pixels = skull_stripped[eroded_brain_mask > 0]
    if len(brain_pixels) > 0:
        mean_intensity = np.mean(brain_pixels)
        std_intensity = np.std(brain_pixels)
        # Tumor threshold at high intensity relative to brain tissue
        tumor_thresh_val = np.clip(mean_intensity + 1.4 * std_intensity, 130, 240)
    else:
        tumor_thresh_val = 160

    _, raw_tumor_mask = cv2.threshold(skull_stripped, tumor_thresh_val, 255, cv2.THRESH_BINARY)
    clean_tumor_mask = cv2.morphologyEx(raw_tumor_mask, cv2.MORPH_OPEN, kernel_small)

    # 8. Visual Overlay & Contour Localization
    overlay_img = img_resized.copy()
    contours, _ = cv2.findContours(clean_tumor_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Filter contours by minimum clinical tumor area (ignore minor noise speckles < 180 px)
    tumor_contours = [c for c in contours if cv2.contourArea(c) > 180]
    
    has_tumor_candidate = len(tumor_contours) > 0
    max_area = 0
    bounding_box = None

    # Mask with only valid tumor contours
    tumor_mask = np.zeros_like(clean_tumor_mask)
    if has_tumor_candidate:
        cv2.drawContours(tumor_mask, tumor_contours, -1, 255, -1)
        largest_c = max(tumor_contours, key=cv2.contourArea)
        max_area = float(cv2.contourArea(largest_c))
        x, y, w, h = cv2.boundingRect(largest_c)
        bounding_box = {"x": int(x), "y": int(y), "w": int(w), "h": int(h)}
        # Draw contour on overlay (Glowing cyan/red bounding box)
        cv2.drawContours(overlay_img, tumor_contours, -1, (0, 0, 255), 2)
        cv2.rectangle(overlay_img, (x, y), (x + w, y + h), (0, 255, 255), 2)
        cv2.putText(overlay_img, "ROI Detected", (x, max(15, y - 5)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 255), 1)

    # Colormap Heatmap overlay
    norm_heatmap = cv2.normalize(clahe_enhanced, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)
    heatmap_colored = cv2.applyColorMap(norm_heatmap, cv2.COLORMAP_JET)
    blended_heatmap = cv2.addWeighted(img_resized, 0.55, heatmap_colored, 0.45, 0)

    return {
        "stages": {
            "original": cv2_to_base64(img_resized),
            "grayscale": cv2_to_base64(gray),
            "gaussian_blur": cv2_to_base64(gaussian_blur),
            "clahe": cv2_to_base64(clahe_enhanced),
            "brain_mask": cv2_to_base64(brain_mask),
            "skull_stripped": cv2_to_base64(skull_stripped),
            "tumor_mask": cv2_to_base64(tumor_mask),
            "heatmap": cv2_to_base64(blended_heatmap),
            "overlay": cv2_to_base64(overlay_img)
        },
        "clean_mri": clahe_enhanced,
        "brain_mask_np": brain_mask,
        "tumor_mask_np": tumor_mask,
        "metadata": {
            "dimensions": f"{target_size[0]}x{target_size[1]}",
            "has_tumor_candidate": has_tumor_candidate,
            "candidate_count": len(tumor_contours),
            "max_contour_area_px": max_area,
            "bounding_box": bounding_box
        }
    }
