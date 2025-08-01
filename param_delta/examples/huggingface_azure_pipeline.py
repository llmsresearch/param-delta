#!/usr/bin/env python3
"""
HuggingFace to Azure Pipeline for ParamΔ

This example demonstrates a complete end-to-end pipeline for:
1. Loading models directly from HuggingFace Hub
2. Computing parameter deltas
3. Storing deltas in Azure Blob Storage
4. Applying deltas to new models
5. Batch processing multiple model pairs

This is useful for organizations that want to:
- Compute deltas from publicly available models
- Store and manage deltas in the cloud
- Apply deltas to new model releases automatically

Usage:
    # Set up environment variables first (see .env.example)
    
    # Compute delta from HuggingFace models
    python huggingface_azure_pipeline.py compute \
        --base meta-llama/Llama-3.2-1B \
        --post meta-llama/Llama-3.2-1B-Instruct \
        --output deltas/llama-instruct.safetensors
        
    # Apply delta to new model
    python huggingface_azure_pipeline.py apply \
        --base meta-llama/Llama-3.2.1-1B \
        --delta deltas/llama-instruct.safetensors \
        --output ./llama-3.2.1-instruct
"""

import os
import sys
import torch
import logging
from pathlib import Path
from typing import Optional, Dict, List
import argparse
from datetime import datetime
import json

from azure.storage.blob import BlobServiceClient
from azure.identity import DefaultAzureCredential
from transformers import AutoModelForCausalLM, AutoTokenizer
from safetensors.torch import save_file, load_file
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class AzureParamDelta:
    """Production ParamΔ implementation for Azure"""
    
    def __init__(self, storage_account_name: str = None, container_name: str = None):
        """Initialize with Azure credentials from environment or parameters"""
        # Get from environment if not provided
        self.storage_account_name = storage_account_name or os.getenv("AZURE_STORAGE_ACCOUNT_NAME")
        self.container_name = container_name or os.getenv("AZURE_STORAGE_CONTAINER_NAME", "models")
        
        if not self.storage_account_name:
            raise ValueError("Storage account name must be provided or set in AZURE_STORAGE_ACCOUNT_NAME environment variable")
        
        # Check for connection string first (simpler auth)
        conn_str = os.getenv("AZURE_STORAGE_CONNECTION_STRING")
        if conn_str:
            self.blob_service = BlobServiceClient.from_connection_string(conn_str)
        else:
            # Use DefaultAzureCredential (works with Azure CLI, managed identity, etc.)
            self.credential = DefaultAzureCredential()
            self.blob_service = BlobServiceClient(
                account_url=f"https://{self.storage_account_name}.blob.core.windows.net",
                credential=self.credential
            )
        
        self.container_client = self.blob_service.get_container_client(self.container_name)
        
        # Ensure container exists
        try:
            self.container_client.create_container()
        except:
            pass  # Container already exists
    
    def compute_delta_from_huggingface(
        self,
        base_model_id: str,
        post_model_id: str,
        output_blob_name: str,
        device: str = "cuda",
        dtype: torch.dtype = torch.float16
    ) -> str:
        """
        Compute delta between HuggingFace models and upload to Azure.
        
        Args:
            base_model_id: HuggingFace model ID for base model
            post_model_id: HuggingFace model ID for post-trained model
            output_blob_name: Blob name for the delta
            device: Computation device
            dtype: Model dtype
            
        Returns:
            Azure blob URL
        """
        logger.info(f"Computing delta: {post_model_id} - {base_model_id}")
        
        # Load models from HuggingFace
        logger.info("Loading base model...")
        base_model = AutoModelForCausalLM.from_pretrained(
            base_model_id,
            torch_dtype=dtype,
            device_map=device if device != "cpu" else None,
            low_cpu_mem_usage=True
        )
        
        logger.info("Loading post-trained model...")
        post_model = AutoModelForCausalLM.from_pretrained(
            post_model_id,
            torch_dtype=dtype,
            device_map=device if device != "cpu" else None,
            low_cpu_mem_usage=True
        )
        
        # Compute delta
        logger.info("Computing parameter delta...")
        delta = {}
        
        base_state = base_model.state_dict()
        post_state = post_model.state_dict()
        
        # Verify architectures match
        if set(base_state.keys()) != set(post_state.keys()):
            raise ValueError("Model architectures do not match!")
        
        # Compute differences
        for name in base_state.keys():
            delta[name] = post_state[name] - base_state[name]
        
        # Save delta locally first
        temp_path = f"/tmp/{output_blob_name}"
        os.makedirs(os.path.dirname(temp_path), exist_ok=True)
        
        if output_blob_name.endswith('.safetensors'):
            save_file(delta, temp_path)
        else:
            torch.save(delta, temp_path)
        
        # Upload to Azure
        logger.info(f"Uploading delta to Azure: {output_blob_name}")
        blob_client = self.container_client.get_blob_client(output_blob_name)
        
        with open(temp_path, "rb") as data:
            blob_client.upload_blob(data, overwrite=True)
        
        # Clean up
        os.remove(temp_path)
        del base_model, post_model, delta
        torch.cuda.empty_cache()
        
        blob_url = f"https://{self.storage_account_name}.blob.core.windows.net/{self.container_name}/{output_blob_name}"
        logger.info(f"Delta uploaded successfully: {blob_url}")
        
        return blob_url
    
    def apply_delta_from_azure(
        self,
        base_model_id: str,
        delta_blob_name: str,
        output_path: str,
        scale: float = 1.0,
        device: str = "cuda",
        dtype: torch.dtype = torch.float16
    ):
        """
        Apply delta from Azure to a model.
        
        Args:
            base_model_id: HuggingFace model ID or local path
            delta_blob_name: Blob name of the delta
            output_path: Where to save the result
            scale: Scaling factor for delta
            device: Computation device
            dtype: Model dtype
        """
        logger.info(f"Applying delta {delta_blob_name} to {base_model_id}")
        
        # Download delta from Azure
        temp_delta_path = f"/tmp/{delta_blob_name}"
        os.makedirs(os.path.dirname(temp_delta_path), exist_ok=True)
        
        blob_client = self.container_client.get_blob_client(delta_blob_name)
        with open(temp_delta_path, "wb") as f:
            f.write(blob_client.download_blob().readall())
        
        # Load delta
        if delta_blob_name.endswith('.safetensors'):
            delta = load_file(temp_delta_path)
        else:
            delta = torch.load(temp_delta_path, map_location=device)
        
        # Load base model
        logger.info("Loading base model...")
        base_model = AutoModelForCausalLM.from_pretrained(
            base_model_id,
            torch_dtype=dtype,
            device_map=device if device != "cpu" else None,
            low_cpu_mem_usage=True
        )
        
        # Apply delta with scaling
        logger.info(f"Applying delta with scale={scale}")
        with torch.no_grad():
            for name, param in base_model.named_parameters():
                if name in delta:
                    param.data += scale * delta[name].to(param.device)
        
        # Save result
        logger.info(f"Saving ParamΔ model to {output_path}")
        base_model.save_pretrained(output_path)
        
        # Also save tokenizer
        tokenizer = AutoTokenizer.from_pretrained(base_model_id)
        tokenizer.save_pretrained(output_path)
        
        # Clean up
        os.remove(temp_delta_path)
        del base_model, delta
        torch.cuda.empty_cache()
        
        logger.info("ParamΔ model saved successfully")
    
    def batch_compute_deltas(self, model_pairs: List[Dict[str, str]]):
        """
        Compute deltas for multiple model pairs.
        
        Args:
            model_pairs: List of dicts with 'base', 'post', 'output' keys
        """
        results = []
        
        for pair in model_pairs:
            try:
                url = self.compute_delta_from_huggingface(
                    base_model_id=pair['base'],
                    post_model_id=pair['post'],
                    output_blob_name=pair['output']
                )
                results.append({
                    'pair': pair,
                    'success': True,
                    'url': url
                })
            except Exception as e:
                logger.error(f"Failed to compute delta for {pair}: {e}")
                results.append({
                    'pair': pair,
                    'success': False,
                    'error': str(e)
                })
        
        return results


def main():
    """Main entry point for CLI usage"""
    parser = argparse.ArgumentParser(description="Production ParamΔ on Azure")
    parser.add_argument("--storage-account", help="Azure storage account name (or set AZURE_STORAGE_ACCOUNT_NAME)")
    parser.add_argument("--container", help="Container name (or set AZURE_STORAGE_CONTAINER_NAME)")
    
    subparsers = parser.add_subparsers(dest="command", help="Command to run")
    
    # Compute delta command
    compute_parser = subparsers.add_parser("compute", help="Compute delta")
    compute_parser.add_argument("--base", required=True, help="Base model ID")
    compute_parser.add_argument("--post", required=True, help="Post-trained model ID")
    compute_parser.add_argument("--output", required=True, help="Output blob name")
    compute_parser.add_argument("--device", default="cuda", help="Device")
    
    # Apply delta command
    apply_parser = subparsers.add_parser("apply", help="Apply delta")
    apply_parser.add_argument("--base", required=True, help="Base model ID")
    apply_parser.add_argument("--delta", required=True, help="Delta blob name")
    apply_parser.add_argument("--output", required=True, help="Output path")
    apply_parser.add_argument("--scale", type=float, default=1.0, help="Delta scale")
    apply_parser.add_argument("--device", default="cuda", help="Device")
    
    # Batch command
    batch_parser = subparsers.add_parser("batch", help="Batch process")
    batch_parser.add_argument("--config", required=True, help="Config JSON file")
    
    args = parser.parse_args()
    
    # Initialize Azure client
    azure_pd = AzureParamDelta(args.storage_account, args.container)
    
    if args.command == "compute":
        azure_pd.compute_delta_from_huggingface(
            base_model_id=args.base,
            post_model_id=args.post,
            output_blob_name=args.output,
            device=args.device
        )
    
    elif args.command == "apply":
        azure_pd.apply_delta_from_azure(
            base_model_id=args.base,
            delta_blob_name=args.delta,
            output_path=args.output,
            scale=args.scale,
            device=args.device
        )
    
    elif args.command == "batch":
        with open(args.config, 'r') as f:
            config = json.load(f)
        
        results = azure_pd.batch_compute_deltas(config['model_pairs'])
        
        # Save results
        output_file = f"batch_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2)
        
        logger.info(f"Batch results saved to {output_file}")
    
    else:
        parser.print_help()


if __name__ == "__main__":
    main()