"""
ParamΔ: Post-Train Large Language Model at Zero Cost

This module implements the ParamΔ method for transferring knowledge from post-trained
models to newly updated base models without additional training.

Based on the paper: "ParamΔ for Direct Weight Mixing: Post-Train Large Language Model at Zero Cost"
"""

import torch
import numpy as np
from typing import Dict, Optional, Tuple, List, Union
from pathlib import Path
import logging
from dataclasses import dataclass
import json
from tqdm import tqdm

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class ModelConfig:
    """Configuration for a model checkpoint"""
    name: str
    path: str
    model_type: str  # base, post, continual
    architecture: str  # llama, qwen, etc.
    size: str  # 8B, 70B, etc.


class ParamDelta:
    """
    Main class for ParamΔ operations.
    
    Implements the core formula: Θ'_post = Θ'_base + (Θ_post - Θ_base)
    """
    
    def __init__(self, device: str = "cpu"):
        """
        Initialize ParamDelta.
        
        Args:
            device: Device to use for computations (cpu/cuda)
        """
        self.device = device
        self.dtype = torch.float16  # Use fp16 for memory efficiency
        
    def calculate_delta(
        self,
        theta_post: Dict[str, torch.Tensor],
        theta_base: Dict[str, torch.Tensor]
    ) -> Dict[str, torch.Tensor]:
        """
        Calculate parameter delta: ΔΘ = Θ_post - Θ_base
        
        Args:
            theta_post: Post-trained model parameters
            theta_base: Base model parameters
            
        Returns:
            Dictionary of parameter deltas
        """
        logger.info("Calculating parameter delta...")
        delta = {}
        
        # Verify architectures match
        if set(theta_post.keys()) != set(theta_base.keys()):
            raise ValueError("Model architectures do not match!")
        
        for key in tqdm(theta_post.keys(), desc="Computing deltas"):
            if theta_post[key].shape != theta_base[key].shape:
                raise ValueError(f"Shape mismatch for parameter {key}")
            
            delta[key] = theta_post[key] - theta_base[key]
            
        logger.info(f"Calculated delta for {len(delta)} parameters")
        return delta
    
    def apply_delta(
        self,
        theta_base_new: Dict[str, torch.Tensor],
        delta: Dict[str, torch.Tensor],
        scale: float = 1.0
    ) -> Dict[str, torch.Tensor]:
        """
        Apply parameter delta to new base model: Θ' = Θ'_base + α*ΔΘ
        
        Args:
            theta_base_new: New base model parameters
            delta: Parameter delta to apply
            scale: Scaling factor α for the delta
            
        Returns:
            New model with applied delta
        """
        logger.info(f"Applying parameter delta with scale={scale}...")
        theta_new = {}
        
        for key in tqdm(theta_base_new.keys(), desc="Applying deltas"):
            if key not in delta:
                logger.warning(f"Key {key} not found in delta, using base value")
                theta_new[key] = theta_base_new[key]
            else:
                theta_new[key] = theta_base_new[key] + scale * delta[key]
                
        logger.info("Parameter delta applied successfully")
        return theta_new
    
    def combine_multiple_deltas(
        self,
        theta_base: Dict[str, torch.Tensor],
        deltas: List[Tuple[Dict[str, torch.Tensor], float]]
    ) -> Dict[str, torch.Tensor]:
        """
        Combine multiple deltas: Θ' = Θ_base + Σ(α_i * ΔΘ_i)
        
        Args:
            theta_base: Base model parameters
            deltas: List of (delta_dict, scale) tuples
            
        Returns:
            Combined model parameters
        """
        logger.info(f"Combining {len(deltas)} parameter deltas...")
        theta_combined = {}
        
        for key in tqdm(theta_base.keys(), desc="Combining deltas"):
            theta_combined[key] = theta_base[key].clone()
            
            for delta, scale in deltas:
                if key in delta:
                    theta_combined[key] += scale * delta[key]
                    
        logger.info("Multiple deltas combined successfully")
        return theta_combined
    
    def compute_cosine_similarity(
        self,
        delta1: Dict[str, torch.Tensor],
        delta2: Dict[str, torch.Tensor],
        layer_types: Optional[List[str]] = None
    ) -> Dict[str, float]:
        """
        Compute cosine similarity between two parameter deltas.
        
        Args:
            delta1: First parameter delta
            delta2: Second parameter delta
            layer_types: List of layer types to analyze separately
            
        Returns:
            Dictionary of cosine similarities by layer type
        """
        if layer_types is None:
            layer_types = ["attention", "mlp", "overall"]
            
        similarities = {}
        
        for layer_type in layer_types:
            cos_sims = []
            
            for key in delta1.keys():
                if key not in delta2:
                    continue
                    
                # Filter by layer type
                if layer_type == "attention" and "attn" not in key:
                    continue
                elif layer_type == "mlp" and "mlp" not in key:
                    continue
                elif layer_type == "overall":
                    pass  # Include all layers
                else:
                    continue
                
                # Flatten tensors and compute cosine similarity
                vec1 = delta1[key].flatten().float()
                vec2 = delta2[key].flatten().float()
                
                cos_sim = torch.nn.functional.cosine_similarity(
                    vec1.unsqueeze(0), vec2.unsqueeze(0)
                ).item()
                
                cos_sims.append(cos_sim)
            
            if cos_sims:
                similarities[layer_type] = np.mean(cos_sims)
                
        return similarities
    
    def compute_weight_norms(
        self,
        delta: Dict[str, torch.Tensor],
        layer_types: Optional[List[str]] = None
    ) -> Dict[str, List[float]]:
        """
        Compute L2 norms of parameter deltas by layer.
        
        Args:
            delta: Parameter delta
            layer_types: List of layer types to analyze
            
        Returns:
            Dictionary of norms by layer type
        """
        if layer_types is None:
            layer_types = ["attention", "mlp"]
            
        norms = {lt: [] for lt in layer_types}
        
        for key, tensor in delta.items():
            norm = torch.norm(tensor.float()).item()
            
            if "attn" in key:
                norms["attention"].append(norm)
            elif "mlp" in key:
                norms["mlp"].append(norm)
                
        return norms


class ModelLoader:
    """Utility class for loading and saving model checkpoints"""
    
    @staticmethod
    def load_state_dict(path: Union[str, Path], device: str = "cpu") -> Dict[str, torch.Tensor]:
        """
        Load model state dict from file.
        
        Args:
            path: Path to checkpoint file
            device: Device to load tensors to
            
        Returns:
            Model state dictionary
        """
        logger.info(f"Loading model from {path}")
        checkpoint = torch.load(path, map_location=device)
        
        # Handle different checkpoint formats
        if "state_dict" in checkpoint:
            state_dict = checkpoint["state_dict"]
        elif "model" in checkpoint:
            state_dict = checkpoint["model"]
        else:
            state_dict = checkpoint
            
        return state_dict
    
    @staticmethod
    def save_state_dict(
        state_dict: Dict[str, torch.Tensor],
        path: Union[str, Path],
        metadata: Optional[Dict] = None
    ):
        """
        Save model state dict to file.
        
        Args:
            state_dict: Model parameters to save
            path: Output path
            metadata: Optional metadata to include
        """
        logger.info(f"Saving model to {path}")
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        
        checkpoint = {
            "state_dict": state_dict,
            "metadata": metadata or {}
        }
        
        torch.save(checkpoint, path)
        logger.info("Model saved successfully")


def create_param_delta_model(
    base_model_path: str,
    post_model_path: str,
    new_base_model_path: str,
    output_path: str,
    scale: float = 1.0,
    device: str = "cpu"
) -> None:
    """
    Create a ParamΔ model using the standard recipe.
    
    Args:
        base_model_path: Path to original base model
        post_model_path: Path to post-trained model
        new_base_model_path: Path to new base model
        output_path: Where to save the ParamΔ model
        scale: Scaling factor for delta
        device: Device for computation
    """
    param_delta = ParamDelta(device=device)
    loader = ModelLoader()
    
    # Load models
    logger.info("Loading models...")
    theta_base = loader.load_state_dict(base_model_path, device)
    theta_post = loader.load_state_dict(post_model_path, device)
    theta_base_new = loader.load_state_dict(new_base_model_path, device)
    
    # Calculate delta
    delta = param_delta.calculate_delta(theta_post, theta_base)
    
    # Apply delta to new base
    theta_param_delta = param_delta.apply_delta(theta_base_new, delta, scale)
    
    # Save result
    metadata = {
        "method": "param_delta",
        "base_model": base_model_path,
        "post_model": post_model_path,
        "new_base_model": new_base_model_path,
        "scale": scale
    }
    
    loader.save_state_dict(theta_param_delta, output_path, metadata)
    logger.info(f"ParamΔ model saved to {output_path}")


if __name__ == "__main__":
    # Example usage
    print("ParamΔ implementation loaded successfully")