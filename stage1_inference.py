# ---------------------------------------------------------
# stage1_inference.py
# Stage-1 cleanup inference with scanned-rough preprocessing
# ---------------------------------------------------------

import os
import sys
import argparse
import torch
import cv2
import numpy as np

# Add current directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from models import PhaseEmbedder, UNetGenerator
from utils.io_utils import save_tensor_as_png
from utils.preprocess_utils import resize_and_pad_rgba, normalize_rgba


# ---------------------------------------------------------
# FIXED PREPROCESSOR FOR REAL SCANNED ROUGHS
# ---------------------------------------------------------
# ---------------------------------------------------------
# FIXED PREPROCESSOR FOR REAL SCANNED ROUGHS (FINAL PATCH)
# ---------------------------------------------------------
def preprocess_for_model(path, size=512, strong=False):
    img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    if img is None:
        raise ValueError(f"Unable to load image: {path}")

    # If missing alpha → add opaque alpha
    if img.shape[-1] == 3:
        img = np.concatenate([img, np.ones((*img.shape[:2], 1), dtype=np.uint8) * 255], axis=-1)

    rgb = img[..., :3]

    gray = cv2.cvtColor(rgb, cv2.COLOR_BGR2GRAY)

    th = cv2.adaptiveThreshold(
        gray, 255,
        cv2.ADAPTIVE_THRESH_MEAN_C,
        cv2.THRESH_BINARY_INV,
        blockSize=51 if strong else 35,
        C=10,
    )

    th = cv2.medianBlur(th, 3)

    alpha = th.copy()
    if strong:
        alpha = cv2.dilate(alpha, np.ones((3, 3), np.uint8), iterations=1)

    alpha = alpha.astype(np.uint8)

    line_rgb = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)

    # ✅ FIX: concatenate along channel axis (axis=2)
    rgba = np.concatenate([line_rgb, alpha[..., None]], axis=2)

    # Resize to 512×512 padded
    rgba = resize_and_pad_rgba(rgba, size)

    # Normalize like training
    rgba = normalize_rgba(rgba)

    return torch.from_numpy(rgba.transpose(2, 0, 1)).float()


# ---------------------------------------------------------
# RUN SINGLE IMAGE INFERENCE
# ---------------------------------------------------------
PHASES = ["rough", "tiedown", "line", "clean", "color", "skeleton"]

def run_inference_single(model, embedder, img_path, input_phase, target_phase, device):
    x = preprocess_for_model(img_path).unsqueeze(0).to(device)
    B, _, H, W = x.shape

    cond = embedder([input_phase], [target_phase], B, H, W, device)
    x_cond = torch.cat([x, cond], dim=1)

    with torch.no_grad():
        pred = model(x_cond)[0]

    return pred.cpu()


# ---------------------------------------------------------
# MAIN ENTRY POINT
# ---------------------------------------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, required=True)
    parser.add_argument("--phase", type=str, required=True)
    parser.add_argument("--target", type=str, default="clean")
    parser.add_argument("--out", type=str, required=True)
    parser.add_argument("--ckpt", type=str,
                        default=os.path.join(os.path.dirname(__file__), "epoch_014.pth"),
                        help="Path to checkpoint file")
    args = parser.parse_args()

    if args.phase not in PHASES:
        raise ValueError(f"Invalid input phase: {args.phase}. Must be one of {PHASES}")

    if args.target not in PHASES:
        raise ValueError(f"Invalid target phase: {args.target}. Must be one of {PHASES}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Validate checkpoint path
    if not os.path.exists(args.ckpt):
        raise FileNotFoundError(f"Checkpoint file not found: {args.ckpt}")

    print(f"Loading checkpoint from: {args.ckpt}")
    try:
        ckpt = torch.load(args.ckpt, map_location=device)
    except Exception as e:
        raise RuntimeError(f"Failed to load checkpoint: {e}")

    E = 16
    in_ch = 4 + 2 * E

    embedder = PhaseEmbedder(PHASES, embed_dim=E).to(device)
    model = UNetGenerator(in_ch=in_ch, out_ch=4).to(device)
    
    # Load model weights
    if "G" in ckpt:
        model.load_state_dict(ckpt["G"])
    elif "model" in ckpt:
        model.load_state_dict(ckpt["model"])
    else:
        model.load_state_dict(ckpt)
    
    model.eval()
    print("Model loaded successfully!")

    # Single image
    if os.path.isfile(args.input):
        pred = run_inference_single(model, embedder, args.input, args.phase, args.target, device)
        save_tensor_as_png(pred, args.out)
        print(f"Saved → {args.out}")
        return

    # Folder
    elif os.path.isdir(args.input):
        os.makedirs(args.out, exist_ok=True)
        files = [f for f in os.listdir(args.input) if f.lower().endswith(".png")]
        for fname in files:
            inp = os.path.join(args.input, fname)
            outp = os.path.join(args.out, fname)

            pred = run_inference_single(model, embedder, inp, args.phase, args.target, device)
            save_tensor_as_png(pred, outp)
            print(f"[OK] {fname}")

        print("Done.")
        return

    else:
        raise ValueError("Input must be a file or directory.")


# ---------------------------------------------------------
if __name__ == "__main__":
    main()
