"""
Azure integration for ParamΔ - handles model storage and compute resources.
"""

import os
import logging
from typing import Optional, Dict, List, Union
from pathlib import Path
from azure.storage.blob import BlobServiceClient, ContainerClient
from azure.identity import DefaultAzureCredential
import json
from datetime import datetime
import torch

logger = logging.getLogger(__name__)


class AzureModelStorage:
    """Manages model storage in Azure Blob Storage"""
    
    def __init__(
        self,
        connection_string: Optional[str] = None,
        container_name: str = "param-delta-models",
        use_managed_identity: bool = True
    ):
        """
        Initialize Azure storage client.
        
        Args:
            connection_string: Azure storage connection string
            container_name: Container name for models
            use_managed_identity: Use managed identity for authentication
        """
        self.container_name = container_name
        
        if use_managed_identity and not connection_string:
            # Use Azure CLI credentials
            credential = DefaultAzureCredential()
            account_url = os.getenv("AZURE_STORAGE_ACCOUNT_URL")
            if not account_url:
                raise ValueError("AZURE_STORAGE_ACCOUNT_URL environment variable not set")
            self.blob_service_client = BlobServiceClient(
                account_url=account_url,
                credential=credential
            )
        else:
            # Use connection string
            if not connection_string:
                connection_string = os.getenv("AZURE_STORAGE_CONNECTION_STRING")
            if not connection_string:
                raise ValueError("No Azure storage connection string provided")
            self.blob_service_client = BlobServiceClient.from_connection_string(connection_string)
        
        self._ensure_container_exists()
    
    def _ensure_container_exists(self):
        """Create container if it doesn't exist"""
        try:
            container_client = self.blob_service_client.get_container_client(self.container_name)
            if not container_client.exists():
                container_client.create_container()
                logger.info(f"Created container: {self.container_name}")
        except Exception as e:
            logger.warning(f"Could not create container: {e}")
    
    def upload_model(
        self,
        local_path: Union[str, Path],
        blob_name: str,
        metadata: Optional[Dict] = None
    ) -> str:
        """
        Upload model to Azure Blob Storage.
        
        Args:
            local_path: Local path to model file
            blob_name: Name for the blob in storage
            metadata: Optional metadata to attach
            
        Returns:
            Blob URL
        """
        logger.info(f"Uploading {local_path} to blob {blob_name}")
        
        blob_client = self.blob_service_client.get_blob_client(
            container=self.container_name,
            blob=blob_name
        )
        
        # Add metadata
        if metadata is None:
            metadata = {}
        metadata["upload_time"] = datetime.utcnow().isoformat()
        metadata["source_file"] = str(local_path)
        
        with open(local_path, "rb") as data:
            blob_client.upload_blob(
                data,
                overwrite=True,
                metadata=metadata
            )
        
        logger.info(f"Model uploaded successfully to {blob_name}")
        return blob_client.url
    
    def download_model(
        self,
        blob_name: str,
        local_path: Union[str, Path]
    ) -> Path:
        """
        Download model from Azure Blob Storage.
        
        Args:
            blob_name: Blob name in storage
            local_path: Where to save the model
            
        Returns:
            Path to downloaded file
        """
        logger.info(f"Downloading blob {blob_name} to {local_path}")
        
        blob_client = self.blob_service_client.get_blob_client(
            container=self.container_name,
            blob=blob_name
        )
        
        # Ensure directory exists
        Path(local_path).parent.mkdir(parents=True, exist_ok=True)
        
        with open(local_path, "wb") as download_file:
            download_file.write(blob_client.download_blob().readall())
        
        logger.info(f"Model downloaded successfully to {local_path}")
        return Path(local_path)
    
    def list_models(self, prefix: Optional[str] = None) -> List[Dict]:
        """
        List all models in the container.
        
        Args:
            prefix: Optional prefix to filter blobs
            
        Returns:
            List of model information dictionaries
        """
        container_client = self.blob_service_client.get_container_client(self.container_name)
        
        models = []
        for blob in container_client.list_blobs(name_starts_with=prefix):
            model_info = {
                "name": blob.name,
                "size": blob.size,
                "last_modified": blob.last_modified.isoformat() if blob.last_modified else None,
                "metadata": blob.metadata
            }
            models.append(model_info)
        
        return models
    
    def get_model_metadata(self, blob_name: str) -> Dict:
        """
        Get metadata for a specific model.
        
        Args:
            blob_name: Blob name
            
        Returns:
            Metadata dictionary
        """
        blob_client = self.blob_service_client.get_blob_client(
            container=self.container_name,
            blob=blob_name
        )
        
        properties = blob_client.get_blob_properties()
        return properties.metadata


class AzureComputeManager:
    """Manages Azure compute resources for ParamΔ operations"""
    
    def __init__(self):
        """Initialize compute manager"""
        self.credential = DefaultAzureCredential()
        
    def get_gpu_info(self) -> Dict:
        """Get information about available GPU resources"""
        gpu_info = {
            "available": torch.cuda.is_available(),
            "device_count": torch.cuda.device_count() if torch.cuda.is_available() else 0,
            "devices": []
        }
        
        if gpu_info["available"]:
            for i in range(gpu_info["device_count"]):
                device_props = torch.cuda.get_device_properties(i)
                gpu_info["devices"].append({
                    "name": device_props.name,
                    "memory_total": device_props.total_memory,
                    "memory_allocated": torch.cuda.memory_allocated(i),
                    "memory_reserved": torch.cuda.memory_reserved(i)
                })
        
        return gpu_info
    
    def estimate_memory_requirements(
        self,
        model_size: str,
        operation: str = "delta"
    ) -> Dict[str, float]:
        """
        Estimate memory requirements for ParamΔ operations.
        
        Args:
            model_size: Model size (e.g., "8B", "70B")
            operation: Type of operation ("delta", "merge", "evaluate")
            
        Returns:
            Memory requirements in GB
        """
        # Extract number of parameters
        size_num = float(model_size.rstrip("B"))
        
        # Base memory for model parameters (fp16)
        param_memory_gb = size_num * 2 / 1024  # 2 bytes per param in fp16
        
        requirements = {
            "model_memory": param_memory_gb,
            "operation_overhead": 0,
            "total": 0
        }
        
        if operation == "delta":
            # Need memory for 2 models + delta
            requirements["operation_overhead"] = param_memory_gb * 2
        elif operation == "merge":
            # Need memory for base + multiple deltas
            requirements["operation_overhead"] = param_memory_gb * 1.5
        elif operation == "evaluate":
            # Need memory for model + evaluation buffers
            requirements["operation_overhead"] = param_memory_gb * 0.5
        
        requirements["total"] = requirements["model_memory"] + requirements["operation_overhead"]
        
        return requirements


class ModelRegistry:
    """Registry for tracking available models and their relationships"""
    
    def __init__(self, storage_client: AzureModelStorage):
        """
        Initialize model registry.
        
        Args:
            storage_client: Azure storage client
        """
        self.storage = storage_client
        self.registry_blob = "model_registry.json"
        self._load_registry()
    
    def _load_registry(self):
        """Load registry from storage"""
        try:
            local_path = Path("/tmp/model_registry.json")
            self.storage.download_model(self.registry_blob, local_path)
            with open(local_path, "r") as f:
                self.registry = json.load(f)
        except Exception:
            logger.info("Creating new model registry")
            self.registry = {
                "models": {},
                "deltas": {},
                "relationships": []
            }
    
    def _save_registry(self):
        """Save registry to storage"""
        local_path = Path("/tmp/model_registry.json")
        with open(local_path, "w") as f:
            json.dump(self.registry, f, indent=2)
        
        self.storage.upload_model(
            local_path,
            self.registry_blob,
            metadata={"type": "registry"}
        )
    
    def register_model(
        self,
        model_id: str,
        model_type: str,
        architecture: str,
        size: str,
        blob_name: str,
        metadata: Optional[Dict] = None
    ):
        """Register a new model"""
        self.registry["models"][model_id] = {
            "type": model_type,
            "architecture": architecture,
            "size": size,
            "blob_name": blob_name,
            "metadata": metadata or {},
            "registered_at": datetime.utcnow().isoformat()
        }
        self._save_registry()
    
    def register_delta(
        self,
        delta_id: str,
        base_model_id: str,
        post_model_id: str,
        blob_name: str,
        metadata: Optional[Dict] = None
    ):
        """Register a parameter delta"""
        self.registry["deltas"][delta_id] = {
            "base_model": base_model_id,
            "post_model": post_model_id,
            "blob_name": blob_name,
            "metadata": metadata or {},
            "created_at": datetime.utcnow().isoformat()
        }
        
        # Add relationship
        self.registry["relationships"].append({
            "type": "delta",
            "from": [base_model_id, post_model_id],
            "to": delta_id,
            "timestamp": datetime.utcnow().isoformat()
        })
        
        self._save_registry()
    
    def get_compatible_deltas(self, base_model_id: str) -> List[str]:
        """Get all deltas compatible with a base model"""
        compatible = []
        
        base_info = self.registry["models"].get(base_model_id, {})
        if not base_info:
            return compatible
        
        for delta_id, delta_info in self.registry["deltas"].items():
            # Check if architectures match
            base_model_info = self.registry["models"].get(delta_info["base_model"], {})
            if (base_model_info.get("architecture") == base_info["architecture"] and
                base_model_info.get("size") == base_info["size"]):
                compatible.append(delta_id)
        
        return compatible


if __name__ == "__main__":
    # Test Azure connectivity
    print("Testing Azure integration...")
    
    # Check for Azure credentials
    if os.getenv("AZURE_STORAGE_CONNECTION_STRING") or os.getenv("AZURE_STORAGE_ACCOUNT_URL"):
        print("Azure credentials found")
        
        # Test GPU availability
        compute_manager = AzureComputeManager()
        gpu_info = compute_manager.get_gpu_info()
        print(f"GPU Available: {gpu_info['available']}")
        print(f"GPU Count: {gpu_info['device_count']}")
        
        # Test memory estimation
        mem_req = compute_manager.estimate_memory_requirements("8B", "delta")
        print(f"Memory required for 8B model delta operation: {mem_req['total']:.2f} GB")
    else:
        print("No Azure credentials found. Set AZURE_STORAGE_CONNECTION_STRING or AZURE_STORAGE_ACCOUNT_URL")