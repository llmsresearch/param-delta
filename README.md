# ParamΔ (Parameter Delta) Implementation

![Python](https://img.shields.io/badge/python-3.8%2B-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-red)
![License](https://img.shields.io/badge/license-MIT-green)
![Code Style](https://img.shields.io/badge/code%20style-black-black)
![Azure](https://img.shields.io/badge/cloud-Azure-0078D4)

## About This Repository
> This implementation is part of [LLMs Research](https://llmsresearch.substack.com), where I break down ~100 papers weekly and build the ones worth building.

This repository provides a practical implementation of the **ParamΔ** methodology introduced in the research paper:

> **"ParamΔ: Enabling Continuous Model Post-Training at No Training Cost"**  
> Authors: Zhiyuan Li, Dongyang Li, Weidong Zhang, Qing Li, Shuguo Ma, and Mao Yang  
> Paper: [arXiv:2504.21023](https://arxiv.org/abs/2504.21023)

**Important**: This is an independent implementation of the ParamΔ methodology. All credit for the original research and methodology goes to the paper authors. We are implementing their approach to make it accessible for practical use.

## Detailed Methodology

### The Core Concept

The ParamΔ methodology addresses a critical challenge in large language model deployment: when base models are updated (e.g., Llama 3.2 to Llama 3.2.1), organizations must wait for new post-trained versions or spend significant resources on re-training. ParamΔ solves this by treating post-training improvements as transferable "deltas."

### Mathematical Foundation

According to the original paper, the ParamΔ approach is based on the following mathematical framework:

1. **Delta Computation**: Given a base model Θ_base and its post-trained version Θ_post, the parameter delta is computed as:
   ```
   ΔΘ = Θ_post - Θ_base
   ```
   This delta captures all the knowledge gained during post-training, including instruction following, safety alignment, and task-specific capabilities.

2. **Delta Application**: When a new base model Θ'_base is released, the delta can be directly applied:
   ```
   Θ'_post = Θ'_base + ΔΘ
   ```
   This creates a post-trained version of the new base model without any training.

3. **Theoretical Justification**: The paper demonstrates that this approach works because:
   - Post-training typically involves small parameter changes relative to pre-training
   - These changes are largely independent of minor base model updates
   - The delta captures the "direction" of improvement that remains valid across versions

### Key Findings from the Paper

The authors validated ParamΔ across multiple model families:

- **Llama Family**: Tested on Llama3-8B, Llama3.1-8B, showing consistent performance preservation
- **Qwen Models**: Validated on Qwen1.5 and Qwen2 variants
- **Cross-Architecture**: Even worked between different model architectures with similar design

Performance results showed:
- **Instruction Following**: Maintained 95-98% of original post-training performance
- **Task Performance**: Negligible degradation on benchmarks like MMLU, GSM8K, HumanEval
- **Safety Alignment**: Preserved safety features from original post-training

### Advanced Applications

The paper also explores:
1. **Multi-Delta Fusion**: Combining multiple post-training deltas (e.g., general + domain-specific)
2. **Delta Scaling**: Adjusting delta magnitude for fine-tuned control
3. **Continual Updates**: Chaining deltas across multiple base model updates

## Repository Structure

```
param_delta/
├── src/                      # Core implementation
│   ├── param_delta.py       # Main ParamΔ algorithm
│   ├── model_utils.py       # Model loading/saving utilities
│   ├── azure_integration.py # Azure storage integration
│   ├── evaluation.py        # Model evaluation tools
│   └── visualization.py     # Visualization utilities
├── examples/                 # Usage examples
│   ├── basic_usage.py       # Simple local example
│   ├── azure_workflow.py    # Azure cloud workflow
│   └── huggingface_azure_pipeline.py  # HuggingFace to Azure pipeline
├── cli.py                   # Command-line interface
├── requirements.txt         # Python dependencies
├── .env.example            # Environment variables template
└── .gitignore              # Git ignore rules
```

## Getting Started

### Installation

```bash
# Clone the repository
git clone https://github.com/llmsresearch/param-delta.git
cd param-delta

# Install dependencies
pip install -r requirements.txt

# For Azure integration (optional)
pip install azure-storage-blob azure-identity

# Copy environment template (for Azure usage)
cp .env.example .env
# Edit .env with your credentials
```

### Basic Usage

#### 1. Using the CLI

```bash
# Compute delta between models
python cli.py compute-delta \
    --base-model "TinyLlama/TinyLlama-1.1B-intermediate-step-1431k-3T" \
    --post-model "TinyLlama/TinyLlama-1.1B-Chat-v1.0" \
    --output "tinyllama-chat-delta.safetensors"

# Apply delta to new model
python cli.py apply-delta \
    --base-model "path/to/new-base-model" \
    --delta-path "tinyllama-chat-delta.safetensors" \
    --output "path/to/output-model"
```

#### 2. Using Python API

```python
from src.param_delta import ParamDelta
from src.model_utils import ModelFormatHandler

# Initialize
pd = ParamDelta(device="cuda")  # or "cpu"

# Load models
base_model = ModelFormatHandler.load_state_dict("path/to/base_model")
post_model = ModelFormatHandler.load_state_dict("path/to/post_model")

# Compute delta
delta = pd.calculate_delta(post_model, base_model)

# Save delta
ModelFormatHandler.save_state_dict(delta, "model_delta.safetensors")

# Apply to new base model
new_base = ModelFormatHandler.load_state_dict("path/to/new_base_model")
new_post = pd.apply_delta(new_base, delta)

# Save result
ModelFormatHandler.save_state_dict(new_post, "new_post_model.safetensors")
```

#### 3. Example Scripts

We provide several example scripts in the `examples/` directory:

**Basic Usage** (`examples/basic_usage.py`):
```python
# Simple demonstration with mock models
from examples.basic_usage import simple_param_delta_demo
simple_param_delta_demo()
```

**Azure Workflow** (`examples/azure_workflow.py`):
```python
# Demonstrates Azure storage integration
from examples.azure_workflow import run_azure_workflow
run_azure_workflow()
```

**HuggingFace Pipeline** (`examples/huggingface_azure_pipeline.py`):
```bash
# Complete pipeline from HuggingFace to Azure
python examples/huggingface_azure_pipeline.py compute \
    --base "TinyLlama/TinyLlama-1.1B-intermediate-step-1431k-3T" \
    --post "TinyLlama/TinyLlama-1.1B-Chat-v1.0" \
    --output "deltas/tinyllama-chat.safetensors"
```

## Azure Integration

For production use with large models, we provide Azure Blob Storage integration:

### Setup Azure Environment

1. Create a `.env` file with your Azure credentials:
```env
AZURE_STORAGE_ACCOUNT_NAME=your_storage_account
AZURE_STORAGE_ACCOUNT_KEY=your_storage_key
AZURE_STORAGE_CONTAINER_NAME=models

# Or use connection string
AZURE_STORAGE_CONNECTION_STRING=DefaultEndpointsProtocol=https;AccountName=...
```

2. Use the production Azure script:

```bash
# Compute and upload delta to Azure (see examples/huggingface_azure_pipeline.py)
python examples/huggingface_azure_pipeline.py compute \
    --base "meta-llama/Llama-3.2-1B" \
    --post "meta-llama/Llama-3.2-1B-Instruct" \
    --output "deltas/llama-3.2-instruct.safetensors"

# Apply delta from Azure
python examples/huggingface_azure_pipeline.py apply \
    --base "meta-llama/Llama-3.2.1-1B" \
    --delta "deltas/llama-3.2-instruct.safetensors" \
    --output "./llama-3.2.1-instruct"

# Batch processing
python examples/huggingface_azure_pipeline.py batch \
    --config batch_config.json
```

## Supported Models

The implementation supports models compatible with HuggingFace Transformers:
- **Llama Family**: Llama 3, Llama 3.1, Llama 3.2
- **Qwen Models**: Qwen1.5, Qwen2
- **Other Models**: Gemma, Phi, Mistral, and any `AutoModelForCausalLM` compatible model

## Advanced Features

### 1. Scaled Delta Application
Apply deltas with different scaling factors:
```python
# Apply delta at 50% strength
new_model = pd.apply_delta(base_model, delta, scale=0.5)
```

### 2. Multi-Delta Combination
Combine multiple deltas:
```python
# Average multiple deltas
combined_delta = pd.combine_deltas([delta1, delta2], weights=[0.7, 0.3])
```

### 3. Delta Analysis
Analyze delta characteristics:
```python
from src.visualization import create_analysis_plots

# Create visualization plots
create_analysis_plots(
    delta_path="model_delta.safetensors",
    output_dir="analysis_results/"
)
```

### 4. Model Evaluation
Evaluate model performance:
```python
from src.evaluation import ParamDeltaEvaluator

evaluator = ParamDeltaEvaluator()
results = evaluator.compare_models(
    original_model="path/to/original",
    param_delta_model="path/to/param_delta",
    tasks=["perplexity", "accuracy"]
)
```

## Performance Considerations

| Operation | Memory Required | Time Complexity |
|-----------|----------------|-----------------|
| Delta Computation | 2x model size | O(n) parameters |
| Delta Application | 1.5x model size | O(n) parameters |
| Delta Storage | 1x model size | - |

### Optimization Tips

1. **Use CPU for small models** (<7B parameters)
2. **Use GPU with sufficient VRAM** for larger models
3. **Enable memory mapping** for very large models:
   ```python
   pd = ParamDelta(device="cuda", use_memory_map=True)
   ```

## Implementation Details

### Parameter Matching
Our implementation ensures exact parameter matching between models:
- Verifies layer names match exactly
- Checks tensor shapes for compatibility
- Raises clear errors for architecture mismatches

### Numerical Precision
- Supports both FP16 and FP32 computations
- Maintains numerical stability during delta operations
- Optional mixed-precision for memory efficiency

### Storage Formats
- **SafeTensors**: Recommended for safety and efficiency
- **PyTorch**: Native .pt/.pth format support
- **Sharded**: Support for multi-file checkpoints

## Testing

Run the test suite:
```bash
python -m pytest tests/
```

## Contributing

We welcome contributions! Please:
1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## License

This implementation is provided under the MIT License. See [LICENSE](LICENSE) file for details.

## Acknowledgments

**All credit for the ParamΔ methodology goes to the original authors:**
- Zhiyuan Li, Dongyang Li, Weidong Zhang, Qing Li, Shuguo Ma, and Mao Yang

Their groundbreaking research demonstrated that ParamΔ achieves remarkable results across multiple model families, maintaining post-training improvements while enabling instant updates to new base models.

## Citation

If you use this implementation in your research, please cite the original paper:

```bibtex
@article{li2025paramdelta,
  title={Param$\Delta$: Enabling Continuous Model Post-Training at No Training Cost},
  author={Li, Zhiyuan and Li, Dongyang and Zhang, Weidong and Li, Qing and Ma, Shuguo and Yang, Mao},
  journal={arXiv preprint arXiv:2504.21023},
  year={2025}
}
```

## Resources

- [Original Paper on arXiv](https://arxiv.org/abs/2504.21023)

---

**Note**: This is an independent implementation of the research methodology. For theoretical questions, please refer to the original paper. For implementation-specific issues, please use the issue tracker.
