"""
NeuroScan AI - CLI Single Image Inference Script
Usage:
    python predict_cli.py --image "static/samples/tumor_glioma_sample.png"
    python predict_cli.py --image "static/samples/normal_sample_1.png"
"""

import argparse
import os
import sys
from ml_engine.preprocessor import preprocess_pipeline
from ml_engine.feature_extractor import extract_features
from ml_engine.model_trainer import ModelManager

def main():
    parser = argparse.ArgumentParser(description="Predict Brain Tumor from an MRI scan image.")
    parser.add_argument("--image", type=str, required=True, help="Path to input MRI scan (.png, .jpg)")
    args = parser.parse_args()

    if not os.path.exists(args.image):
        print(f"[ERROR] Image not found: {args.image}")
        sys.exit(1)

    print("\n" + "="*60)
    print("   NEUROSCAN AI - CLINICAL INFERENCE")
    print("="*60)
    print(f"Input Image : {args.image}")

    # 1. Preprocessing
    print("\n[STEP 1/3] Executing Preprocessing Pipeline...")
    prep = preprocess_pipeline(args.image)
    meta = prep["metadata"]
    print(f"Dimensions       : {meta['dimensions']}")
    print(f"ROI Detected     : {'Yes' if meta['has_tumor_candidate'] else 'None'}")
    print(f"Candidate Area   : {meta['max_contour_area_px']} px")
    if meta['bounding_box']:
        b = meta['bounding_box']
        print(f"Bounding Box     : X={b['x']}, Y={b['y']}, W={b['w']}, H={b['h']}")

    # 2. Feature Extraction
    print("\n[STEP 2/3] Extracting GLCM & Statistical Features...")
    feats = extract_features(prep["clean_mri"], prep["brain_mask_np"], prep["tumor_mask_np"])
    f = feats["feature_dict"]
    print(f"GLCM Contrast    : {f['glcm_contrast']:.4f}")
    print(f"GLCM Homogeneity : {f['glcm_homogeneity']:.4f}")
    print(f"GLCM Energy      : {f['glcm_energy']:.4f}")
    print(f"Mean Intensity   : {f['intensity_mean']:.2f}")
    print(f"Shannon Entropy  : {f['shannon_entropy']:.4f}")

    # 3. Model Inference
    print("\n[STEP 3/3] Running Machine Learning Classification...")
    mgr = ModelManager(model_dir=os.path.join(os.path.dirname(__file__), "saved_models"))
    result = mgr.predict(feats["feature_vector"])

    print("\n" + "*"*45)
    print("           DIAGNOSTIC VERDICT")
    print("*"*45)
    print(f"Diagnosis Result : {result['label']}")
    print(f"Clinical Status  : {result['clinical_status']}")
    print(f"Confidence Level : {result['confidence']}%")
    print(f"Model Algorithm  : {result['model_algorithm']}")
    print(f"Probabilities    : Normal={result['probabilities']['normal']*100:.1f}%, Tumor={result['probabilities']['tumor']*100:.1f}%")
    print("*"*45 + "\n")

if __name__ == "__main__":
    main()
