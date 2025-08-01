#!/usr/bin/env python3
"""
ParamΔ CLI - Command-line interface for ParamΔ operations.

Usage:
    python cli.py compute-delta --base MODEL --post MODEL --output DELTA
    python cli.py apply-delta --base MODEL --delta DELTA --output MODEL
    python cli.py merge-deltas --base MODEL --deltas DELTA1:0.5,DELTA2:0.5 --output MODEL
    python cli.py evaluate --model MODEL --benchmarks MMLU,HumanEval
    python cli.py analyze --delta1 DELTA1 --delta2 DELTA2 --output DIR
"""

import argparse
import logging
import sys
from pathlib import Path
from typing import List, Tuple
import json
import os

# Add src to path
sys.path.append(str(Path(__file__).parent))

from src.param_delta import ParamDelta, create_param_delta_model
from src.model_utils import ModelFormatHandler, ModelValidator, LayerAnalyzer
from src.evaluation import ParamDeltaEvaluator, ResultsAnalyzer
from src.visualization import create_paper_figures, ParamDeltaVisualizer
from src.azure_integration import AzureModelStorage, AzureComputeManager, ModelRegistry

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def parse_delta_specs(delta_str: str) -> List[Tuple[str, float]]:
    """Parse delta specifications from string format."""
    deltas = []
    for spec in delta_str.split(','):
        parts = spec.split(':')
        if len(parts) == 2:
            path, scale = parts
            deltas.append((path, float(scale)))
        else:
            deltas.append((spec, 1.0))
    return deltas


def cmd_compute_delta(args):
    """Compute parameter delta between two models."""
    logger.info(f"Computing delta: {args.post} - {args.base}")
    
    param_delta = ParamDelta(device=args.device)
    
    # Load models
    logger.info("Loading base model...")
    theta_base = ModelFormatHandler.load_state_dict(
        args.base,
        device=args.device,
        dtype=args.dtype
    )
    
    logger.info("Loading post-trained model...")
    theta_post = ModelFormatHandler.load_state_dict(
        args.post,
        device=args.device,
        dtype=args.dtype
    )
    
    # Validate compatibility
    is_compatible, issues = ModelValidator.check_architecture_compatibility(
        theta_base, theta_post
    )
    
    if not is_compatible:
        logger.error("Models are not compatible:")
        for issue in issues:
            logger.error(f"  - {issue}")
        return 1
    
    # Compute delta
    delta = param_delta.calculate_delta(theta_post, theta_base)
    
    # Save delta
    ModelFormatHandler.save_state_dict(
        delta,
        args.output,
        format=args.format,
        metadata={
            "operation": "compute_delta",
            "base_model": str(args.base),
            "post_model": str(args.post)
        }
    )
    
    logger.info(f"Delta saved to {args.output}")
    
    # Show statistics
    if args.stats:
        stats = LayerAnalyzer.get_layer_statistics(delta)
        print("\nDelta Statistics:")
        for layer_type, layer_stats in stats.items():
            print(f"\n{layer_type}:")
            for key, value in layer_stats.items():
                print(f"  {key}: {value:.4f}")
    
    return 0


def cmd_apply_delta(args):
    """Apply parameter delta to a base model."""
    logger.info(f"Applying delta to create ParamΔ model")
    
    param_delta = ParamDelta(device=args.device)
    
    # Load base model
    logger.info("Loading base model...")
    theta_base = ModelFormatHandler.load_state_dict(
        args.base,
        device=args.device,
        dtype=args.dtype
    )
    
    # Load delta
    logger.info("Loading delta...")
    delta = ModelFormatHandler.load_state_dict(
        args.delta,
        device=args.device,
        dtype=args.dtype
    )
    
    # Apply delta
    theta_new = param_delta.apply_delta(theta_base, delta, scale=args.scale)
    
    # Save result
    ModelFormatHandler.save_state_dict(
        theta_new,
        args.output,
        format=args.format,
        metadata={
            "operation": "apply_delta",
            "base_model": str(args.base),
            "delta": str(args.delta),
            "scale": args.scale
        }
    )
    
    logger.info(f"ParamΔ model saved to {args.output}")
    
    return 0


def cmd_merge_deltas(args):
    """Merge multiple deltas into a base model."""
    logger.info("Merging multiple deltas")
    
    param_delta = ParamDelta(device=args.device)
    
    # Load base model
    logger.info("Loading base model...")
    theta_base = ModelFormatHandler.load_state_dict(
        args.base,
        device=args.device,
        dtype=args.dtype
    )
    
    # Parse and load deltas
    delta_specs = parse_delta_specs(args.deltas)
    deltas = []
    
    for delta_path, scale in delta_specs:
        logger.info(f"Loading delta: {delta_path} (scale={scale})")
        delta = ModelFormatHandler.load_state_dict(
            delta_path,
            device=args.device,
            dtype=args.dtype
        )
        deltas.append((delta, scale))
    
    # Combine deltas
    theta_combined = param_delta.combine_multiple_deltas(theta_base, deltas)
    
    # Save result
    ModelFormatHandler.save_state_dict(
        theta_combined,
        args.output,
        format=args.format,
        metadata={
            "operation": "merge_deltas",
            "base_model": str(args.base),
            "deltas": [{"path": str(p), "scale": s} for p, s in delta_specs]
        }
    )
    
    logger.info(f"Combined model saved to {args.output}")
    
    return 0


def cmd_evaluate(args):
    """Evaluate a model on benchmarks."""
    logger.info(f"Evaluating model: {args.model}")
    
    evaluator = ParamDeltaEvaluator(device=args.device)
    
    # Parse benchmarks
    benchmarks = args.benchmarks.split(',') if args.benchmarks else None
    
    # Evaluate
    results = evaluator.evaluate_model(
        args.model,
        benchmarks=benchmarks,
        model_type=args.model_type
    )
    
    # Display results
    print("\nEvaluation Results:")
    print("-" * 50)
    
    for benchmark, result in results.items():
        print(f"{benchmark}: {result.score:.4f}")
        if args.detailed:
            for metric, value in result.metrics.items():
                print(f"  {metric}: {value:.4f}")
    
    # Calculate average
    avg_score = ResultsAnalyzer.calculate_average_performance(results)
    print(f"\nAverage Score: {avg_score:.4f}")
    
    # Save results if requested
    if args.output:
        ResultsAnalyzer.export_results(
            {args.model: results},
            args.output
        )
        logger.info(f"Results saved to {args.output}")
    
    return 0


def cmd_analyze(args):
    """Analyze parameter deltas (cosine similarity, norms, etc.)."""
    logger.info("Analyzing parameter deltas")
    
    # Load deltas
    logger.info("Loading first delta...")
    delta1 = ModelFormatHandler.load_state_dict(
        args.delta1,
        device="cpu",  # Analysis doesn't need GPU
        dtype=args.dtype
    )
    
    logger.info("Loading second delta...")
    delta2 = ModelFormatHandler.load_state_dict(
        args.delta2,
        device="cpu",
        dtype=args.dtype
    )
    
    # Create output directory
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Generate figures
    delta_names = (
        Path(args.delta1).stem,
        Path(args.delta2).stem
    )
    
    create_paper_figures(delta1, delta2, delta_names, output_dir)
    
    # Compute and save statistics
    param_delta = ParamDelta(device="cpu")
    similarities = param_delta.compute_cosine_similarity(delta1, delta2)
    
    stats = {
        "delta1": str(args.delta1),
        "delta2": str(args.delta2),
        "cosine_similarities": similarities,
        "delta1_stats": LayerAnalyzer.get_layer_statistics(delta1),
        "delta2_stats": LayerAnalyzer.get_layer_statistics(delta2)
    }
    
    with open(output_dir / "analysis_stats.json", "w") as f:
        json.dump(stats, f, indent=2)
    
    logger.info(f"Analysis complete. Results saved to {output_dir}")
    
    return 0


def cmd_azure_upload(args):
    """Upload model to Azure storage."""
    logger.info(f"Uploading {args.model} to Azure")
    
    try:
        storage = AzureModelStorage()
        
        # Prepare metadata
        metadata = {
            "model_type": args.model_type,
            "architecture": args.architecture,
            "size": args.size
        }
        
        # Upload
        blob_name = args.blob_name or Path(args.model).name
        url = storage.upload_model(args.model, blob_name, metadata)
        
        logger.info(f"Model uploaded successfully: {url}")
        
        # Register if requested
        if args.register:
            registry = ModelRegistry(storage)
            registry.register_model(
                model_id=args.model_id or blob_name,
                model_type=args.model_type,
                architecture=args.architecture,
                size=args.size,
                blob_name=blob_name,
                metadata=metadata
            )
            logger.info("Model registered in registry")
        
        return 0
        
    except Exception as e:
        logger.error(f"Azure upload failed: {e}")
        return 1


def cmd_info(args):
    """Show information about a model or delta."""
    logger.info(f"Getting info for: {args.path}")
    
    # Detect format
    format_type = ModelFormatHandler.detect_format(args.path)
    print(f"Format: {format_type}")
    
    # Load state dict
    state_dict = ModelFormatHandler.load_state_dict(
        args.path,
        device="cpu",
        dtype=args.dtype
    )
    
    # Get size info
    size_info = ModelValidator.estimate_model_size(state_dict)
    print("\nModel Size:")
    for key, value in size_info.items():
        print(f"  {key}: {value:.2f}")
    
    # Get layer info
    categories = LayerAnalyzer.categorize_layers(state_dict)
    print("\nLayer Categories:")
    for category, params in categories.items():
        if params:
            print(f"  {category}: {len(params)} parameters")
    
    # Get statistics
    if args.stats:
        stats = LayerAnalyzer.get_layer_statistics(state_dict, categories)
        print("\nLayer Statistics:")
        for layer_type, layer_stats in stats.items():
            print(f"\n  {layer_type}:")
            for key, value in layer_stats.items():
                if isinstance(value, float):
                    print(f"    {key}: {value:.4f}")
                else:
                    print(f"    {key}: {value}")
    
    return 0


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="ParamΔ CLI - Zero-cost post-training for LLMs",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    # Global arguments
    parser.add_argument(
        "--device",
        default="cpu",
        help="Device to use (cpu, cuda, cuda:0, etc.)"
    )
    parser.add_argument(
        "--dtype",
        default="float16",
        choices=["float16", "float32", "bfloat16"],
        help="Data type for computations"
    )
    
    # Subcommands
    subparsers = parser.add_subparsers(dest="command", help="Command to run")
    
    # compute-delta command
    compute_parser = subparsers.add_parser(
        "compute-delta",
        help="Compute parameter delta between models"
    )
    compute_parser.add_argument("--base", required=True, help="Base model path")
    compute_parser.add_argument("--post", required=True, help="Post-trained model path")
    compute_parser.add_argument("--output", required=True, help="Output delta path")
    compute_parser.add_argument("--format", default="safetensors", help="Output format")
    compute_parser.add_argument("--stats", action="store_true", help="Show statistics")
    
    # apply-delta command
    apply_parser = subparsers.add_parser(
        "apply-delta",
        help="Apply parameter delta to base model"
    )
    apply_parser.add_argument("--base", required=True, help="Base model path")
    apply_parser.add_argument("--delta", required=True, help="Delta path")
    apply_parser.add_argument("--output", required=True, help="Output model path")
    apply_parser.add_argument("--scale", type=float, default=1.0, help="Delta scaling factor")
    apply_parser.add_argument("--format", default="safetensors", help="Output format")
    
    # merge-deltas command
    merge_parser = subparsers.add_parser(
        "merge-deltas",
        help="Merge multiple deltas into base model"
    )
    merge_parser.add_argument("--base", required=True, help="Base model path")
    merge_parser.add_argument("--deltas", required=True, help="Deltas (path:scale,path:scale)")
    merge_parser.add_argument("--output", required=True, help="Output model path")
    merge_parser.add_argument("--format", default="safetensors", help="Output format")
    
    # evaluate command
    eval_parser = subparsers.add_parser(
        "evaluate",
        help="Evaluate model on benchmarks"
    )
    eval_parser.add_argument("--model", required=True, help="Model path")
    eval_parser.add_argument("--benchmarks", help="Comma-separated benchmark names")
    eval_parser.add_argument("--model-type", default="llama", help="Model architecture type")
    eval_parser.add_argument("--output", help="Save results to file")
    eval_parser.add_argument("--detailed", action="store_true", help="Show detailed metrics")
    
    # analyze command
    analyze_parser = subparsers.add_parser(
        "analyze",
        help="Analyze parameter deltas"
    )
    analyze_parser.add_argument("--delta1", required=True, help="First delta path")
    analyze_parser.add_argument("--delta2", required=True, help="Second delta path")
    analyze_parser.add_argument("--output", required=True, help="Output directory")
    
    # azure-upload command
    azure_parser = subparsers.add_parser(
        "azure-upload",
        help="Upload model to Azure storage"
    )
    azure_parser.add_argument("--model", required=True, help="Model path")
    azure_parser.add_argument("--blob-name", help="Blob name (default: filename)")
    azure_parser.add_argument("--model-type", required=True, help="Model type (base/post/delta)")
    azure_parser.add_argument("--architecture", required=True, help="Architecture (llama/qwen)")
    azure_parser.add_argument("--size", required=True, help="Model size (8B/70B)")
    azure_parser.add_argument("--register", action="store_true", help="Register in model registry")
    azure_parser.add_argument("--model-id", help="Model ID for registry")
    
    # info command
    info_parser = subparsers.add_parser(
        "info",
        help="Show information about model or delta"
    )
    info_parser.add_argument("path", help="Model or delta path")
    info_parser.add_argument("--stats", action="store_true", help="Show detailed statistics")
    
    # Parse arguments
    args = parser.parse_args()
    
    # Convert dtype string to torch dtype
    if hasattr(args, 'dtype'):
        import torch
        dtype_map = {
            "float16": torch.float16,
            "float32": torch.float32,
            "bfloat16": torch.bfloat16
        }
        args.dtype = dtype_map[args.dtype]
    
    # Execute command
    if not args.command:
        parser.print_help()
        return 1
    
    command_map = {
        "compute-delta": cmd_compute_delta,
        "apply-delta": cmd_apply_delta,
        "merge-deltas": cmd_merge_deltas,
        "evaluate": cmd_evaluate,
        "analyze": cmd_analyze,
        "azure-upload": cmd_azure_upload,
        "info": cmd_info
    }
    
    try:
        return command_map[args.command](args)
    except Exception as e:
        logger.error(f"Command failed: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())