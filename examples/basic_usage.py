#!/usr/bin/env python3
"""
Basic usage examples for ParamΔ implementation.
"""

import torch
import sys
from pathlib import Path

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from src.param_delta import ParamDelta, create_param_delta_model
from src.model_utils import ModelFormatHandler, ModelValidator
from src.evaluation import ParamDeltaEvaluator
from src.visualization import ParamDeltaVisualizer, DeltaAnalyzer


def example_basic_param_delta():
    """Example 1: Basic ParamΔ computation and application"""
    print("=== Example 1: Basic ParamΔ Usage ===")
    
    # Initialize ParamDelta
    param_delta = ParamDelta(device="cpu")
    
    # Create synthetic models for demonstration
    print("Creating synthetic models...")
    base_model = {
        "layer1.weight": torch.randn(100, 100),
        "layer2.weight": torch.randn(50, 100),
        "layer3.weight": torch.randn(10, 50)
    }
    
    # Simulate post-training (add small changes)
    post_model = {}
    for key, tensor in base_model.items():
        post_model[key] = tensor + torch.randn_like(tensor) * 0.1
    
    # New base model (different initialization)
    new_base_model = {
        "layer1.weight": torch.randn(100, 100),
        "layer2.weight": torch.randn(50, 100),
        "layer3.weight": torch.randn(10, 50)
    }
    
    # Step 1: Calculate delta
    print("\nCalculating parameter delta...")
    delta = param_delta.calculate_delta(post_model, base_model)
    print(f"Delta computed for {len(delta)} parameters")
    
    # Step 2: Apply delta to new base model
    print("\nApplying delta to new base model...")
    param_delta_model = param_delta.apply_delta(new_base_model, delta)
    print("ParamΔ model created successfully!")
    
    # Verify the formula: Θ' = Θ'_base + (Θ_post - Θ_base)
    for key in base_model:
        expected = new_base_model[key] + (post_model[key] - base_model[key])
        actual = param_delta_model[key]
        assert torch.allclose(expected, actual, rtol=1e-5)
    
    print("✓ ParamΔ formula verified!")


def example_scaled_delta():
    """Example 2: Applying scaled deltas"""
    print("\n=== Example 2: Scaled Delta Application ===")
    
    param_delta = ParamDelta(device="cpu")
    
    # Create models
    base_model = {"layer1.weight": torch.randn(10, 10)}
    post_model = {"layer1.weight": base_model["layer1.weight"] + torch.ones(10, 10)}
    new_base = {"layer1.weight": torch.randn(10, 10)}
    
    # Calculate delta
    delta = param_delta.calculate_delta(post_model, base_model)
    
    # Apply with different scales
    scales = [0.0, 0.5, 1.0, 1.5]
    
    print("Applying delta with different scales:")
    for scale in scales:
        result = param_delta.apply_delta(new_base, delta, scale=scale)
        
        # Calculate how much the model changed
        change_norm = torch.norm(result["layer1.weight"] - new_base["layer1.weight"])
        print(f"  Scale α={scale:.1f}: Change norm = {change_norm:.3f}")


def example_multiple_deltas():
    """Example 3: Combining multiple deltas"""
    print("\n=== Example 3: Combining Multiple Deltas ===")
    
    param_delta = ParamDelta(device="cpu")
    
    # Base model
    base_model = {
        "layer1.weight": torch.randn(20, 20),
        "layer2.weight": torch.randn(10, 20)
    }
    
    # Create two different post-trained models
    # Model 1: General instruction following
    general_model = {}
    for key, tensor in base_model.items():
        general_model[key] = tensor + torch.randn_like(tensor) * 0.1
    
    # Model 2: Task-specific (e.g., medical)
    specialized_model = {}
    for key, tensor in base_model.items():
        specialized_model[key] = tensor + torch.randn_like(tensor) * 0.15
    
    # Calculate deltas
    print("Calculating deltas...")
    general_delta = param_delta.calculate_delta(general_model, base_model)
    specialized_delta = param_delta.calculate_delta(specialized_model, base_model)
    
    # Combine deltas with weights
    print("\nCombining deltas (70% general, 30% specialized)...")
    deltas = [(general_delta, 0.7), (specialized_delta, 0.3)]
    combined_model = param_delta.combine_multiple_deltas(base_model, deltas)
    
    print("✓ Combined model created successfully!")


def example_delta_analysis():
    """Example 4: Analyzing parameter deltas"""
    print("\n=== Example 4: Delta Analysis ===")
    
    param_delta = ParamDelta(device="cpu")
    
    # Create transformer-like models for analysis
    def create_transformer_model(seed):
        torch.manual_seed(seed)
        model = {}
        for i in range(4):  # 4 layers
            # Attention weights
            model[f"transformer.h.{i}.attn.weight"] = torch.randn(64, 64)
            # MLP weights
            model[f"transformer.h.{i}.mlp.weight"] = torch.randn(256, 64)
        return model
    
    # Create different model variants
    base = create_transformer_model(seed=0)
    instruct = create_transformer_model(seed=1)
    medical = create_transformer_model(seed=2)
    
    # Calculate deltas
    instruct_delta = param_delta.calculate_delta(instruct, base)
    medical_delta = param_delta.calculate_delta(medical, base)
    
    # Compute cosine similarities
    print("Computing cosine similarities between deltas...")
    similarities = param_delta.compute_cosine_similarity(
        instruct_delta,
        medical_delta,
        layer_types=["attention", "mlp", "overall"]
    )
    
    print("\nCosine Similarities:")
    for layer_type, sim in similarities.items():
        print(f"  {layer_type}: {sim:.4f}")
    
    # Compute weight norms
    print("\nComputing weight norms...")
    instruct_norms = param_delta.compute_weight_norms(instruct_delta)
    
    print("Instruction Delta Norms:")
    print(f"  Attention layers: mean={np.mean(instruct_norms['attention']):.3f}")
    print(f"  MLP layers: mean={np.mean(instruct_norms['mlp']):.3f}")


def example_evaluation():
    """Example 5: Model evaluation"""
    print("\n=== Example 5: Model Evaluation ===")
    
    # Create evaluator
    evaluator = ParamDeltaEvaluator(device="cpu")
    
    # Evaluate on mock benchmarks
    print("Running evaluation on mock benchmarks...")
    
    # Note: This uses mock benchmarks for demonstration
    # In real usage, you would point to actual model files
    results = evaluator.evaluate_model(
        model_path="mock_model",  # Mock model
        benchmarks=["MMLU", "HumanEval", "GSM8K"],
        model_type="llama"
    )
    
    print("\nEvaluation Results:")
    for benchmark, result in results.items():
        print(f"  {benchmark}: {result.score:.3f}")
    
    # Calculate average performance
    avg_score = np.mean([r.score for r in results.values()])
    print(f"\nAverage Score: {avg_score:.3f}")


def example_visualization():
    """Example 6: Visualization of delta analysis"""
    print("\n=== Example 6: Visualization ===")
    
    # Create analyzer
    analyzer = DeltaAnalyzer()
    
    # Create sample deltas with known patterns
    delta1 = {}
    delta2 = {}
    
    for i in range(12):  # 12 layers
        # Create deltas with varying similarity
        base_vec = torch.randn(100)
        
        # Delta 1
        delta1[f"transformer.h.{i}.weight"] = base_vec + torch.randn(100) * 0.1
        
        # Delta 2 - similar in early layers, different in later layers
        if i < 6:
            delta2[f"transformer.h.{i}.weight"] = base_vec + torch.randn(100) * 0.1
        else:
            delta2[f"transformer.h.{i}.weight"] = torch.randn(100)
    
    # Compute layer similarities
    similarities = analyzer.compute_layer_similarities(delta1, delta2)
    
    print("Layer-wise Cosine Similarities:")
    for layer, sim in sorted(similarities.items()):
        print(f"  Layer {layer}: {sim:.3f}")
    
    print("\nNote: Similarity decreases in later layers as expected")


def main():
    """Run all examples"""
    examples = [
        example_basic_param_delta,
        example_scaled_delta,
        example_multiple_deltas,
        example_delta_analysis,
        example_evaluation,
        example_visualization
    ]
    
    for example in examples:
        example()
        print("\n" + "="*50 + "\n")
    
    print("All examples completed successfully!")


if __name__ == "__main__":
    # For numpy in examples
    import numpy as np
    
    main()