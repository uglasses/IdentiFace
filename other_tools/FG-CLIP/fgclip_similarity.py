#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
FG-CLIP image-text similarity evaluation program.
Imports the FG-CLIP module standalone and evaluates similarity between one input image and given text.
"""

import os
import sys
import argparse
from pathlib import Path
import torch
from PIL import Image
from transformers import (
    AutoImageProcessor,
    AutoTokenizer,
    AutoModelForCausalLM,
)


# Default model path (models/fg-clip2-large; override with --fgclip-model / --model-path)
DEFAULT_MODEL_PATH = str(
    Path(__file__).resolve().parent.parent.parent / "models" / "fg-clip2-large"
)


def _materialize_fgclip2_text_embedding_masks(model):
    """
    mask1/mask2 are not registered as Parameter/buffer, so HF meta init does not load their
    weights and they stay meta when walk_type=long. Rebuild from text_config (same as
    modeling_fgclip2.Fgclip2TextEmbeddings.__init__).
    """
    tm = getattr(model, "text_model", None)
    emb = getattr(tm, "embeddings", None) if tm is not None else None
    if emb is None or not hasattr(emb, "mask1"):
        return
    if not getattr(emb.mask1, "is_meta", False):
        return
    tc = getattr(model.config, "text_config", None)
    if tc is None:
        return
    keep_len, longtext_len = int(tc.keep_len), int(tc.longtext_len)
    device = next(model.parameters()).device
    mask1 = torch.zeros(longtext_len, 1, device=device, dtype=torch.float32)
    mask1[:keep_len, :] = 1
    mask2 = torch.zeros(longtext_len, 1, device=device, dtype=torch.float32)
    mask2[keep_len:, :] = 1
    emb.mask1 = mask1
    emb.mask2 = mask2


def _fix_fgclip2_text_embedding_position_ids(model):
    """If position_ids in the checkpoint are abnormal, rebuild arange from longtext_len (same as end of __init__)."""
    tm = getattr(model, "text_model", None)
    emb = getattr(tm, "embeddings", None) if tm is not None else None
    if emb is None or not hasattr(emb, "position_ids"):
        return
    tc = getattr(model.config, "text_config", None)
    if tc is None:
        return
    longtext_len = int(tc.longtext_len)
    device = emb.position_ids.device
    correct = torch.arange(longtext_len, device=device, dtype=torch.long).view(1, -1)
    emb.register_buffer("position_ids", correct, persistent=False)


class FGCLIPSimilarityEvaluator:
    """FG-CLIP image-text similarity evaluator"""
    
    def __init__(self, model_path=None, device=None):
        """
        Initialize the evaluator.
        
        Args:
            model_path: FG-CLIP model path, defaults to DEFAULT_MODEL_PATH
            device: Device ('cuda' or 'cpu'), defaults to auto-detect
        """
        self.model_path = model_path or DEFAULT_MODEL_PATH
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        
        # Model components
        self.model = None
        self.tokenizer = None
        self.image_processor = None
        
        # Load model
        self._load_model()
    
    def _load_model(self):
        """Load FG-CLIP model and related components"""
        print(f"Loading FG-CLIP model...")
        print(f"Model path: {self.model_path}")
        print(f"Device: {self.device}")
        
        # Check model path
        if not os.path.exists(self.model_path):
            raise FileNotFoundError(
                f"Model path does not exist: {self.model_path}\n"
                f"Please ensure the model has been downloaded and placed at the specified path"
            )
        
        try:
            # Load model
            print("Loading model weights (this may take a few minutes)...")
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_path,
                trust_remote_code=True
            )
            
            # Move to specified device
            if self.device == "cuda" and torch.cuda.is_available():
                self.model = self.model.cuda()
                print(f"Model loaded onto GPU: {torch.cuda.get_device_name(0)}")
            else:
                self.model = self.model.cpu()
                self.device = "cpu"
                print("Model loaded onto CPU")

            _materialize_fgclip2_text_embedding_masks(self.model)
            _fix_fgclip2_text_embedding_position_ids(self.model)
            
            # Load tokenizer and image processor
            print("Loading Tokenizer and Image Processor...")
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_path, trust_remote_code=True
            )
            self.image_processor = AutoImageProcessor.from_pretrained(
                self.model_path, trust_remote_code=True
            )
            
            # Set to evaluation mode
            self.model.eval()
            
            print("✓ Model loading complete!\n")
            
        except Exception as e:
            print(f"❌ Model loading failed: {str(e)}")
            import traceback
            traceback.print_exc()
            raise
    
    def _determine_max_patches(self, image):
        """
        Determine the maximum number of patches from image size.
        
        Args:
            image: PIL Image object
            
        Returns:
            max_num_patches: Maximum number of patches
        """
        w, h = image.size
        max_val = (w // 16) * (h // 16)
        
        if max_val > 784:
            return 1024
        elif max_val > 576:
            return 784
        elif max_val > 256:
            return 576
        elif max_val > 128:
            return 256
        else:
            return 128
    
    def compute_similarity(self, image_path, text, walk_type="long"):
        """
        Compute similarity between an image and text.
        
        Args:
            image_path: Image file path
            text: Text description
            walk_type: Text processing type, either "short" (short text, max_length=64)
                       or "long" (long text, max_length=196), defaults to "long"
        
        Returns:
            dict: Dictionary containing similarity scores
                - similarity: Similarity score after applying logit scale and bias
                - cosine_similarity: Cosine similarity (range: -1 to 1)
                - probability: Sigmoid probability (range: 0 to 1)
        """
        # Load image
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Image file does not exist: {image_path}")
        
        try:
            image = Image.open(image_path).convert("RGB")
        except Exception as e:
            raise ValueError(f"Unable to load image: {str(e)}")
        
        # Determine maximum number of patches
        max_patches = self._determine_max_patches(image)
        
        # Process image
        image_input = self.image_processor(
            images=image,
            max_num_patches=max_patches,
            return_tensors="pt"
        ).to(self.device)
        
        # Process text
        max_length = 196 if walk_type == "long" else 64
        text_lower = text.lower()
        
        # Count tokens in the raw text (no truncation)
        raw_tokens = self.tokenizer.encode(text_lower, add_special_tokens=False)
        actual_token_count = len(raw_tokens)
        
        # Tokenize text (will truncate to max_length)
        text_input = self.tokenizer(
            [text_lower],
            padding="max_length",
            max_length=max_length,
            truncation=True,
            return_tensors="pt"
        ).to(self.device)
        
        # Get the number of tokens actually used (accounting for truncation)
        used_token_count = text_input['input_ids'].shape[1]  # sequence length
        # Count non-padding tokens
        if 'attention_mask' in text_input:
            non_padding_tokens = text_input['attention_mask'].sum().item()
        else:
            # If no attention_mask, count non-padding tokens in input_ids
            non_padding_tokens = (text_input['input_ids'] != self.tokenizer.pad_token_id).sum().item()
        
        # Store token statistics
        token_stats = {
            "raw_token_count": actual_token_count,  # token count of raw text
            "max_length": max_length,  # maximum allowed length
            "used_token_count": non_padding_tokens,  # tokens actually used (excluding padding)
            "was_truncated": actual_token_count > max_length  # whether truncated
        }
        
        # Compute features and similarity
        with torch.no_grad():
            # Get image and text features
            image_feature = self.model.get_image_features(**image_input)
            text_feature = self.model.get_text_features(**text_input, walk_type=walk_type)
            
            # Normalize features
            image_feature = image_feature / image_feature.norm(p=2, dim=-1, keepdim=True)
            text_feature = text_feature / text_feature.norm(p=2, dim=-1, keepdim=True)
            
            # Cosine similarity (dot product of normalized features)
            cosine_sim = (image_feature @ text_feature.T).item()
            
            # Compute logits (raw similarity before applying logit scale and bias)
            # Note: image @ text.T here is equivalent to (text @ image.T).T in the official implementation
            # See: FG-CLIP README.md and modeling_fgclip2.py
            logits_per_image = image_feature @ text_feature.T
            
            # Apply logit scale and bias
            # logit_scale is a learned temperature parameter (usually stored in log form; needs exp)
            # logit_bias is a learned bias term
            # Formula: logit = cosine_similarity * exp(logit_scale) + logit_bias
            logit_scale = self.model.logit_scale.to(text_feature.device)
            logit_bias = self.model.logit_bias.to(text_feature.device)
            logits_per_image = logits_per_image * logit_scale.exp() + logit_bias
            
            similarity = logits_per_image.item()
            
            # Compute sigmoid probability
            probability = torch.sigmoid(logits_per_image).item()
        
        return {
            "similarity": similarity,
            "cosine_similarity": cosine_sim,
            "probability": probability,
            "token_stats": token_stats
        }
    
    def evaluate(self, image_path, text, walk_type="long", verbose=True):
        """
        Evaluate image-text similarity (with output).
        
        Args:
            image_path: Image file path
            text: Text description
            walk_type: Text processing type ("short" or "long")
            verbose: Whether to print detailed information
        
        Returns:
            dict: Similarity results
        """
        if verbose:
            print("=" * 60)
            print("FG-CLIP Image-Text Similarity Evaluation")
            print("=" * 60)
            print(f"Image path: {image_path}")
            print(f"Text description: {text}")
            print(f"Text type: {walk_type} (max_length={196 if walk_type=='long' else 64})")
            print("-" * 60)
        
        # Precompute token count for display
        try:
            max_length = 196 if walk_type == "long" else 64
            text_lower = text.lower()
            raw_tokens = self.tokenizer.encode(text_lower, add_special_tokens=False)
            actual_token_count = len(raw_tokens)
            was_truncated = actual_token_count > max_length
            
            if verbose:
                print(f"\nText Token Statistics:")
                print(f"  - Raw text token count: {actual_token_count}")
                print(f"  - Max allowed token count: {max_length}")
                if was_truncated:
                    print(f"  - ⚠️  Text was truncated; exceeded by {actual_token_count - max_length} tokens")
                else:
                    print(f"  - ✓ Text within limit")
                print("-" * 60)
        except Exception as e:
            if verbose:
                print(f"  - ⚠️  Unable to compute token statistics: {str(e)}")
                print("-" * 60)
        
        try:
            results = self.compute_similarity(image_path, text, walk_type)
            
            if verbose:
                print(f"\nSimilarity Results:")
                print(f"  - Similarity score (logit): {results['similarity']:.4f}")
                print(f"  - Cosine similarity: {results['cosine_similarity']:.4f}")
                print(f"  - Probability (sigmoid): {results['probability']:.4f} ({results['probability']*100:.2f}%)")
                
                # Display token statistics
                if 'token_stats' in results:
                    token_stats = results['token_stats']
                    print(f"\nToken Statistics Details:")
                    print(f"  - Raw token count: {token_stats['raw_token_count']}")
                    print(f"  - Tokens actually used: {token_stats['used_token_count']}")
                    print(f"  - Max length limit: {token_stats['max_length']}")
                    if token_stats['was_truncated']:
                        print(f"  - ⚠️  Truncated: {token_stats['raw_token_count'] - token_stats['max_length']} tokens discarded")
                
                print("=" * 60)
            
            return results
            
        except Exception as e:
            if verbose:
                print(f"❌ Evaluation failed: {str(e)}")
                import traceback
                traceback.print_exc()
            raise


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(
        description="FG-CLIP image-text similarity evaluation program",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Example usage:
  # Basic usage
  python fgclip_similarity.py --image path/to/image.jpg --text "a cat sitting on a mat"
  
  # Short text mode
  python fgclip_similarity.py --image path/to/image.jpg --text "a cat" --walk-type short
  
  # Specify model path
  python fgclip_similarity.py --image path/to/image.jpg --text "a cat" --model-path /path/to/model
  
  # Use CPU
  python fgclip_similarity.py --image path/to/image.jpg --text "a cat" --device cpu
        """
    )
    
    parser.add_argument(
        "--image",
        type=str,
        required=True,
        help="Input image file path"
    )
    
    parser.add_argument(
        "--text",
        type=str,
        required=True,
        help="Text description"
    )
    
    parser.add_argument(
        "--model-path",
        type=str,
        default=None,
        help=f"FG-CLIP model path (default: {DEFAULT_MODEL_PATH})"
    )
    
    parser.add_argument(
        "--walk-type",
        type=str,
        choices=["short", "long"],
        default="long",
        help="Text processing type: 'short' (short text, max_length=64) or 'long' (long text, max_length=196, default)"
    )
    
    parser.add_argument(
        "--device",
        type=str,
        choices=["cuda", "cpu"],
        default=None,
        help="Compute device (default: auto-detect, prefers CUDA)"
    )
    
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Quiet mode; only output the final result"
    )
    
    args = parser.parse_args()
    
    # Create evaluator
    try:
        evaluator = FGCLIPSimilarityEvaluator(
            model_path=args.model_path,
            device=args.device
        )
    except Exception as e:
        print(f"Initialization failed: {str(e)}")
        sys.exit(1)
    
    # Run evaluation
    try:
        results = evaluator.evaluate(
            image_path=args.image,
            text=args.text,
            walk_type=args.walk_type,
            verbose=not args.quiet
        )
        
        # In quiet mode, only print the key result
        if args.quiet:
            print(f"{results['similarity']:.4f}")
        
        sys.exit(0)
        
    except Exception as e:
        print(f"Evaluation failed: {str(e)}")
        sys.exit(1)


if __name__ == "__main__":
    main()
