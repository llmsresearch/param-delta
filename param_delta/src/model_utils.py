"""
Model utilities for loading and handling different model formats.
Supports HuggingFace, PyTorch checkpoints, and safetensors.
"""

import torch
import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Union, Tuple
import numpy as np
from tqdm import tqdm
import gc
import os

logger = logging.getLogger(__name__)

# Try to import optional dependencies
try:
    from safetensors import safe_open
    from safetensors.torch import save_file
    SAFETENSORS_AVAILABLE = True
except ImportError:
    SAFETENSORS_AVAILABLE = False
    logger.warning("safetensors not available. Install with: pip install safetensors")

try:
    from transformers import AutoModel, AutoTokenizer
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False
    logger.warning("transformers not available. Install with: pip install transformers")


class ModelFormatHandler:
    """Handles different model checkpoint formats"""
    
    @staticmethod
    def detect_format(path: Union[str, Path]) -> str:
        """
        Detect the format of a model checkpoint.
        
        Args:
            path: Path to model file or directory
            
        Returns:
            Format string: "pytorch", "safetensors", "huggingface", or "unknown"
        """
        path = Path(path)
        
        if path.is_dir():
            # Check for HuggingFace model
            if (path / "config.json").exists():
                return "huggingface"
            # Check for sharded checkpoints
            if list(path.glob("*.safetensors")):
                return "safetensors_sharded"
            if list(path.glob("pytorch_model*.bin")):
                return "pytorch_sharded"
        else:
            # Single file
            if path.suffix == ".safetensors":
                return "safetensors"
            elif path.suffix in [".pt", ".pth", ".bin"]:
                return "pytorch"
        
        return "unknown"
    
    @staticmethod
    def load_state_dict(
        path: Union[str, Path],
        device: str = "cpu",
        dtype: Optional[torch.dtype] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Load model state dict from various formats.
        
        Args:
            path: Path to model
            device: Device to load to
            dtype: Optional dtype to convert to
            
        Returns:
            State dictionary
        """
        format_type = ModelFormatHandler.detect_format(path)
        logger.info(f"Detected format: {format_type}")
        
        if format_type == "pytorch":
            return ModelFormatHandler._load_pytorch(path, device, dtype)
        elif format_type == "safetensors":
            return ModelFormatHandler._load_safetensors(path, device, dtype)
        elif format_type == "huggingface":
            return ModelFormatHandler._load_huggingface(path, device, dtype)
        elif format_type == "pytorch_sharded":
            return ModelFormatHandler._load_pytorch_sharded(path, device, dtype)
        elif format_type == "safetensors_sharded":
            return ModelFormatHandler._load_safetensors_sharded(path, device, dtype)
        else:
            raise ValueError(f"Unknown model format at {path}")
    
    @staticmethod
    def _load_pytorch(
        path: Path,
        device: str,
        dtype: Optional[torch.dtype]
    ) -> Dict[str, torch.Tensor]:
        """Load PyTorch checkpoint"""
        checkpoint = torch.load(path, map_location=device)
        
        if isinstance(checkpoint, dict):
            if "state_dict" in checkpoint:
                state_dict = checkpoint["state_dict"]
            elif "model" in checkpoint:
                state_dict = checkpoint["model"]
            else:
                state_dict = checkpoint
        else:
            raise ValueError("Invalid checkpoint format")
        
        if dtype:
            state_dict = {k: v.to(dtype) for k, v in state_dict.items()}
        
        return state_dict
    
    @staticmethod
    def _load_safetensors(
        path: Path,
        device: str,
        dtype: Optional[torch.dtype]
    ) -> Dict[str, torch.Tensor]:
        """Load safetensors checkpoint"""
        if not SAFETENSORS_AVAILABLE:
            raise ImportError("safetensors is required to load this format")
        
        state_dict = {}
        with safe_open(path, framework="pt", device=device) as f:
            for key in f.keys():
                tensor = f.get_tensor(key)
                if dtype:
                    tensor = tensor.to(dtype)
                state_dict[key] = tensor
        
        return state_dict
    
    @staticmethod
    def _load_pytorch_sharded(
        path: Path,
        device: str,
        dtype: Optional[torch.dtype]
    ) -> Dict[str, torch.Tensor]:
        """Load sharded PyTorch checkpoint"""
        state_dict = {}
        
        # Load index file if exists
        index_file = path / "pytorch_model.bin.index.json"
        if index_file.exists():
            with open(index_file) as f:
                index = json.load(f)
            shard_files = set(index["weight_map"].values())
        else:
            # Find all shard files
            shard_files = [f.name for f in path.glob("pytorch_model*.bin")]
        
        # Load each shard
        for shard_file in tqdm(shard_files, desc="Loading shards"):
            shard_path = path / shard_file
            shard_dict = ModelFormatHandler._load_pytorch(shard_path, device, dtype)
            state_dict.update(shard_dict)
        
        return state_dict
    
    @staticmethod
    def _load_safetensors_sharded(
        path: Path,
        device: str,
        dtype: Optional[torch.dtype]
    ) -> Dict[str, torch.Tensor]:
        """Load sharded safetensors checkpoint"""
        if not SAFETENSORS_AVAILABLE:
            raise ImportError("safetensors is required to load this format")
        
        state_dict = {}
        shard_files = list(path.glob("*.safetensors"))
        
        for shard_file in tqdm(shard_files, desc="Loading shards"):
            shard_dict = ModelFormatHandler._load_safetensors(shard_file, device, dtype)
            state_dict.update(shard_dict)
        
        return state_dict
    
    @staticmethod
    def _load_huggingface(
        path: Path,
        device: str,
        dtype: Optional[torch.dtype]
    ) -> Dict[str, torch.Tensor]:
        """Load HuggingFace model"""
        if not TRANSFORMERS_AVAILABLE:
            # Fall back to loading raw files
            if list(path.glob("*.safetensors")):
                return ModelFormatHandler._load_safetensors_sharded(path, device, dtype)
            else:
                return ModelFormatHandler._load_pytorch_sharded(path, device, dtype)
        
        # Load with transformers
        model = AutoModel.from_pretrained(
            path,
            torch_dtype=dtype or torch.float16,
            device_map=device
        )
        
        return model.state_dict()
    
    @staticmethod
    def save_state_dict(
        state_dict: Dict[str, torch.Tensor],
        path: Union[str, Path],
        format: str = "safetensors",
        metadata: Optional[Dict] = None
    ):
        """
        Save state dict in specified format.
        
        Args:
            state_dict: Model parameters
            path: Output path
            format: Output format ("pytorch" or "safetensors")
            metadata: Optional metadata
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        if format == "safetensors" and SAFETENSORS_AVAILABLE:
            save_file(state_dict, path, metadata=metadata)
        else:
            checkpoint = {
                "state_dict": state_dict,
                "metadata": metadata or {}
            }
            torch.save(checkpoint, path)


class MemoryEfficientDeltaComputer:
    """Compute parameter deltas with memory efficiency for large models"""
    
    def __init__(self, chunk_size: int = 1000):
        """
        Initialize delta computer.
        
        Args:
            chunk_size: Number of parameters to process at once
        """
        self.chunk_size = chunk_size
    
    def compute_delta_chunked(
        self,
        model1_path: Union[str, Path],
        model2_path: Union[str, Path],
        output_path: Union[str, Path],
        device: str = "cpu",
        dtype: torch.dtype = torch.float16
    ):
        """
        Compute delta between two models in a memory-efficient way.
        
        Args:
            model1_path: Path to post-trained model
            model2_path: Path to base model
            output_path: Where to save delta
            device: Computation device
            dtype: Data type for computation
        """
        logger.info("Computing delta in memory-efficient mode")
        
        # Get parameter names from both models
        format1 = ModelFormatHandler.detect_format(model1_path)
        format2 = ModelFormatHandler.detect_format(model2_path)
        
        # For this example, we'll load parameter names first
        with torch.no_grad():
            # This is a simplified version - in practice, you'd stream through files
            state1 = ModelFormatHandler.load_state_dict(model1_path, device="cpu", dtype=dtype)
            state2 = ModelFormatHandler.load_state_dict(model2_path, device="cpu", dtype=dtype)
            
            delta = {}
            
            for key in tqdm(state1.keys(), desc="Computing deltas"):
                if key in state2:
                    # Move to device for computation
                    param1 = state1[key].to(device)
                    param2 = state2[key].to(device)
                    
                    # Compute delta
                    delta[key] = (param1 - param2).cpu()
                    
                    # Clear GPU memory
                    del param1, param2
                    if device != "cpu":
                        torch.cuda.empty_cache()
            
            # Save delta
            ModelFormatHandler.save_state_dict(delta, output_path)
            
            # Clean up
            del state1, state2, delta
            gc.collect()
            if device != "cpu":
                torch.cuda.empty_cache()


class LayerAnalyzer:
    """Analyze model layers for ParamΔ operations"""
    
    @staticmethod
    def categorize_layers(state_dict: Dict[str, torch.Tensor]) -> Dict[str, List[str]]:
        """
        Categorize model parameters by layer type.
        
        Args:
            state_dict: Model state dictionary
            
        Returns:
            Dictionary mapping layer types to parameter names
        """
        categories = {
            "attention": [],
            "mlp": [],
            "embedding": [],
            "norm": [],
            "other": []
        }
        
        for name in state_dict.keys():
            name_lower = name.lower()
            
            if any(x in name_lower for x in ["attn", "attention", "self_attn"]):
                categories["attention"].append(name)
            elif any(x in name_lower for x in ["mlp", "fc", "dense", "feedforward"]):
                categories["mlp"].append(name)
            elif any(x in name_lower for x in ["embed", "wte", "wpe"]):
                categories["embedding"].append(name)
            elif any(x in name_lower for x in ["norm", "ln", "layernorm"]):
                categories["norm"].append(name)
            else:
                categories["other"].append(name)
        
        return categories
    
    @staticmethod
    def get_layer_statistics(
        state_dict: Dict[str, torch.Tensor],
        categories: Optional[Dict[str, List[str]]] = None
    ) -> Dict[str, Dict[str, float]]:
        """
        Get statistics for each layer category.
        
        Args:
            state_dict: Model state dictionary
            categories: Pre-computed layer categories
            
        Returns:
            Statistics by layer type
        """
        if categories is None:
            categories = LayerAnalyzer.categorize_layers(state_dict)
        
        stats = {}
        
        for category, param_names in categories.items():
            if not param_names:
                continue
            
            norms = []
            total_params = 0
            
            for name in param_names:
                if name in state_dict:
                    tensor = state_dict[name]
                    norms.append(torch.norm(tensor).item())
                    total_params += tensor.numel()
            
            if norms:
                stats[category] = {
                    "mean_norm": np.mean(norms),
                    "std_norm": np.std(norms),
                    "max_norm": np.max(norms),
                    "min_norm": np.min(norms),
                    "total_params": total_params,
                    "num_layers": len(param_names)
                }
        
        return stats


class ModelValidator:
    """Validate model compatibility for ParamΔ operations"""
    
    @staticmethod
    def check_architecture_compatibility(
        state_dict1: Dict[str, torch.Tensor],
        state_dict2: Dict[str, torch.Tensor]
    ) -> Tuple[bool, List[str]]:
        """
        Check if two models have compatible architectures.
        
        Args:
            state_dict1: First model state dict
            state_dict2: Second model state dict
            
        Returns:
            Tuple of (is_compatible, list_of_issues)
        """
        issues = []
        
        # Check parameter names
        keys1 = set(state_dict1.keys())
        keys2 = set(state_dict2.keys())
        
        missing_in_2 = keys1 - keys2
        missing_in_1 = keys2 - keys1
        
        if missing_in_2:
            issues.append(f"Parameters missing in model 2: {list(missing_in_2)[:5]}...")
        if missing_in_1:
            issues.append(f"Parameters missing in model 1: {list(missing_in_1)[:5]}...")
        
        # Check shapes for common parameters
        common_keys = keys1 & keys2
        shape_mismatches = []
        
        for key in common_keys:
            if state_dict1[key].shape != state_dict2[key].shape:
                shape_mismatches.append(
                    f"{key}: {state_dict1[key].shape} vs {state_dict2[key].shape}"
                )
        
        if shape_mismatches:
            issues.append(f"Shape mismatches: {shape_mismatches[:5]}...")
        
        is_compatible = len(issues) == 0
        return is_compatible, issues
    
    @staticmethod
    def estimate_model_size(state_dict: Dict[str, torch.Tensor]) -> Dict[str, float]:
        """
        Estimate model size and memory requirements.
        
        Args:
            state_dict: Model state dictionary
            
        Returns:
            Size information
        """
        total_params = 0
        total_bytes = 0
        
        for tensor in state_dict.values():
            total_params += tensor.numel()
            total_bytes += tensor.numel() * tensor.element_size()
        
        return {
            "total_params": total_params,
            "total_params_millions": total_params / 1e6,
            "total_params_billions": total_params / 1e9,
            "size_mb": total_bytes / (1024 * 1024),
            "size_gb": total_bytes / (1024 * 1024 * 1024)
        }


if __name__ == "__main__":
    print("Model utilities loaded successfully")
    print(f"Safetensors available: {SAFETENSORS_AVAILABLE}")
    print(f"Transformers available: {TRANSFORMERS_AVAILABLE}")