import os
import sys
import torch
import cv2
import numpy as np
import torch.nn.functional as F

# Set working directory so project-internal modules can be imported
# Must set path before importing the network module
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

from network.models.segface_celeb import SegFaceCeleb

# Class definitions (CelebAMask-HQ 19 classes)
LABELS = [
    'background', 'neck', 'skin', 'cloth', 'l_ear', 'r_ear', 
    'l_brow', 'r_brow','l_eye', 'r_eye', 'nose', 'mouth', 
    'l_lip', 'u_lip', 'hair','eye_g', 'hat', 'ear_r', 'neck_l'
]

# Global model variable (singleton)
_segface_model = None
_model_loaded = False


def load_segface_model(checkpoint_path=None, device='cuda'):
    """
    Load the SegFace model (loaded only once).

    Args:
        checkpoint_path: Path to model weights; if None, use the default path
        device: Device ('cuda' or 'cpu')

    Returns:
        Loaded model, or None on failure
    """
    global _segface_model, _model_loaded
    
    # If the model is already loaded, return it directly
    if _model_loaded and _segface_model is not None:
        return _segface_model
    
    if checkpoint_path is None:
        # Use default path (relative to current file location)
        current_dir = os.path.dirname(os.path.abspath(__file__))
        checkpoint_path = os.path.join(current_dir, 'weights', 'convnext_celeba_512', 'model_299.pt')
    
    if not os.path.exists(checkpoint_path):
        print(f"Error: weight file not found {checkpoint_path}")
        return None
    
    try:
        # 1. Initialize model structure
        model = SegFaceCeleb(input_resolution=512, model='convnext_base')
        
        # 2. Load weights
        checkpoint = torch.load(checkpoint_path, map_location=device)
        
        # Handle nested dict key names
        if isinstance(checkpoint, dict) and 'state_dict_backbone' in checkpoint:
            state_dict = checkpoint['state_dict_backbone']
        else:
            state_dict = checkpoint
        
        # Remove 'module.' prefix
        new_state_dict = { (k[7:] if k.startswith('module.') else k): v for k, v in state_dict.items() }
        
        # Load weights
        model.load_state_dict(new_state_dict, strict=False)
        model.to(device).eval()
        
        # Save as global variable
        _segface_model = model
        _model_loaded = True
        
        print(f"SegFace model loaded on {device}")
        return model
    except Exception as e:
        print(f"Failed to load SegFace model: {e}")
        return None


def generate_mask(image_path, target_parts=['nose', 'u_lip', 'l_lip'], model=None):
    """
    Generate a facial mask.

    Args:
        image_path: Image path
        target_parts: List of target facial regions
        model: SegFace model; if None, use the global model

    Returns:
        Mask as a numpy array, or None on failure
    """
    global _segface_model
    
    # If no model is passed in, use the global model
    if model is None:
        if _segface_model is None:
            # Try to load the model
            model = load_segface_model()
            if model is None:
                return None
        else:
            model = _segface_model

    # Read image
    if not os.path.exists(image_path):
        print(f"Error: image not found '{image_path}'. Please check the filename and extension!")
        return None
        
    img = cv2.imread(image_path)
    if img is None:
        print(f"Error: unable to read image '{image_path}'")
        return None
        
    img_resized = cv2.resize(img, (512, 512))
    
    # Convert to Tensor (RGB order, normalized)
    img_rgb = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)
    img_tensor = torch.from_numpy(img_rgb).permute(2, 0, 1).float() / 255.0
    device = next(model.parameters()).device  # Get the device the model is on
    img_tensor = img_tensor.unsqueeze(0).to(device)

    # Run inference
    with torch.no_grad():
        # The last two args (labels and dataset) are set to None at inference time
        seg_output = model(img_tensor, None, None) 
        mask_probs = F.interpolate(seg_output, size=(512, 512), mode='bilinear', align_corners=False)
        preds = torch.argmax(mask_probs, dim=1).cpu().numpy()[0]

    # Build mask for the specified parts
    final_mask = np.zeros_like(preds, dtype=np.uint8)
    found_any = False
    for part in target_parts:
        if part in LABELS:
            idx = LABELS.index(part)
            if np.any(preds == idx):
                final_mask[preds == idx] = 255
                found_any = True
    
    return final_mask

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Generate a SegFace accessory mask for one image.")
    parser.add_argument("--input", type=str, required=True, help="Input face image path.")
    parser.add_argument(
        "--parts",
        type=str,
        nargs="+",
        default=["hat"],
        help="Target parts to mask (hat eye_g). Default: hat",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="SegFace/output_mask.png",
        help="Output mask path. Default: SegFace/output_mask.png",
    )
    args = parser.parse_args()

    mask = generate_mask(args.input, target_parts=args.parts)

    if mask is not None:
        save_path = args.output
        cv2.imwrite(save_path, mask)
        print(f"Success: mask saved to {os.path.abspath(save_path)}")
