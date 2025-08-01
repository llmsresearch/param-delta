#!/usr/bin/env python3
"""
Example workflow using Azure integration for ParamΔ.
"""

import os
import sys
from pathlib import Path

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from src.azure_integration import AzureModelStorage, AzureComputeManager, ModelRegistry
from src.param_delta import ParamDelta


def setup_azure_environment():
    """Set up Azure environment variables"""
    print("=== Azure Environment Setup ===")
    
    # Check for Azure credentials
    has_connection_string = "AZURE_STORAGE_CONNECTION_STRING" in os.environ
    has_account_url = "AZURE_STORAGE_ACCOUNT_URL" in os.environ
    
    if not has_connection_string and not has_account_url:
        print("WARNING: No Azure credentials found!")
        print("\nTo use Azure integration, set one of:")
        print("  - AZURE_STORAGE_CONNECTION_STRING")
        print("  - AZURE_STORAGE_ACCOUNT_URL (with Azure CLI auth)")
        print("\nExample:")
        print('  export AZURE_STORAGE_CONNECTION_STRING="DefaultEndpointsProtocol=..."')
        return False
    
    print("SUCCESS: Azure credentials detected")
    return True


def example_azure_model_management():
    """Example: Managing models in Azure storage"""
    print("\n=== Azure Model Management Example ===")
    
    if not setup_azure_environment():
        print("Skipping Azure examples - no credentials")
        return
    
    try:
        # Initialize storage client
        storage = AzureModelStorage(container_name="param-delta-demo")
        
        # Example 1: List existing models
        print("\n1. Listing models in Azure:")
        models = storage.list_models()
        
        if models:
            for model in models[:5]:  # Show first 5
                print(f"  - {model['name']} ({model['size'] / 1024 / 1024:.1f} MB)")
        else:
            print("  No models found in container")
        
        # Example 2: Get GPU information
        print("\n2. Checking compute resources:")
        compute = AzureComputeManager()
        gpu_info = compute.get_gpu_info()
        
        print(f"  GPUs available: {gpu_info['available']}")
        print(f"  GPU count: {gpu_info['device_count']}")
        
        # Example 3: Estimate memory requirements
        print("\n3. Memory requirements estimation:")
        for model_size in ["8B", "70B"]:
            for operation in ["delta", "merge", "evaluate"]:
                req = compute.estimate_memory_requirements(model_size, operation)
                print(f"  {model_size} {operation}: {req['total']:.1f} GB")
        
    except Exception as e:
        print(f"Azure example failed: {e}")


def example_model_registry():
    """Example: Using the model registry"""
    print("\n=== Model Registry Example ===")
    
    if not setup_azure_environment():
        return
    
    try:
        storage = AzureModelStorage(container_name="param-delta-demo")
        registry = ModelRegistry(storage)
        
        # Register a base model
        print("\n1. Registering models:")
        registry.register_model(
            model_id="llama3-base-demo",
            model_type="base",
            architecture="llama",
            size="8B",
            blob_name="models/llama3-base.safetensors",
            metadata={"demo": True}
        )
        print("  SUCCESS: Registered base model")
        
        # Register a post-trained model
        registry.register_model(
            model_id="llama3-instruct-demo",
            model_type="post",
            architecture="llama",
            size="8B",
            blob_name="models/llama3-instruct.safetensors",
            metadata={"demo": True}
        )
        print("  SUCCESS: Registered post-trained model")
        
        # Register a delta
        registry.register_delta(
            delta_id="llama3-instruct-delta-demo",
            base_model_id="llama3-base-demo",
            post_model_id="llama3-instruct-demo",
            blob_name="deltas/llama3-instruct.delta",
            metadata={"demo": True}
        )
        print("  SUCCESS: Registered parameter delta")
        
        # Find compatible deltas
        print("\n2. Finding compatible deltas:")
        compatible = registry.get_compatible_deltas("llama3-base-demo")
        
        for delta_id in compatible:
            print(f"  - {delta_id}")
        
    except Exception as e:
        print(f"Registry example failed: {e}")


def example_azure_workflow():
    """Example: Complete Azure-based ParamΔ workflow"""
    print("\n=== Complete Azure Workflow Example ===")
    
    if not setup_azure_environment():
        return
    
    # This demonstrates the workflow without actual model files
    print("\nWorkflow steps:")
    print("1. Upload base and post-trained models to Azure")
    print("2. Compute delta in Azure (using Azure compute)")
    print("3. Store delta in blob storage")
    print("4. Register all models in registry")
    print("5. When new base model arrives:")
    print("   a. Upload new base model")
    print("   b. Find compatible deltas from registry")
    print("   c. Apply delta to create ParamΔ model")
    print("   d. Store and register result")
    
    print("\nExample CLI commands:")
    print("\n# Upload models")
    print("python cli.py azure-upload --model llama3-base \\")
    print("  --model-type base --architecture llama --size 8B --register")
    
    print("\n# Compute delta")
    print("python cli.py compute-delta \\")
    print("  --base az://llama3-base --post az://llama3-instruct \\")
    print("  --output az://llama3-delta")
    
    print("\n# Apply to new model")
    print("python cli.py apply-delta \\")
    print("  --base az://llama3.1-base --delta az://llama3-delta \\")
    print("  --output az://llama3.1-param-delta")


def example_distributed_param_delta():
    """Example: Distributed ParamΔ computation"""
    print("\n=== Distributed Computation Example ===")
    
    print("For large models (70B+), use distributed computation:")
    
    print("\n1. Shard-wise delta computation:")
    print("   - Load model shards separately")
    print("   - Compute delta for each shard")
    print("   - Combine shard deltas")
    
    print("\n2. Memory-efficient approach:")
    print("   - Stream parameters from Azure")
    print("   - Compute delta in chunks")
    print("   - Write results directly to blob")
    
    print("\nExample pseudo-code:")
    print("""
    # Memory-efficient delta computation
    for shard_idx in range(num_shards):
        # Load shard from Azure
        base_shard = load_from_azure(f"base_shard_{shard_idx}")
        post_shard = load_from_azure(f"post_shard_{shard_idx}")
        
        # Compute delta for shard
        delta_shard = post_shard - base_shard
        
        # Upload delta shard
        upload_to_azure(delta_shard, f"delta_shard_{shard_idx}")
        
        # Free memory
        del base_shard, post_shard, delta_shard
    """)


def main():
    """Run all Azure examples"""
    print("ParamΔ Azure Integration Examples")
    print("=" * 50)
    
    examples = [
        example_azure_model_management,
        example_model_registry,
        example_azure_workflow,
        example_distributed_param_delta
    ]
    
    for example in examples:
        try:
            example()
            print("\n" + "=" * 50)
        except Exception as e:
            print(f"Example failed: {e}")
            print("\n" + "=" * 50)
    
    print("\nAzure examples completed!")
    print("\nNote: These examples demonstrate the Azure integration API.")
    print("For production use, ensure proper Azure credentials and permissions.")


if __name__ == "__main__":
    main()