"""
Test suite for ParamΔ implementation.
Uses small synthetic models to validate functionality.
"""

import sys
import torch
import numpy as np
from pathlib import Path
import tempfile
import shutil
import unittest
import logging

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from src.param_delta import ParamDelta, ModelLoader
from src.model_utils import ModelFormatHandler, ModelValidator, LayerAnalyzer
from src.evaluation import ParamDeltaEvaluator, MockBenchmark
from src.visualization import DeltaAnalyzer

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class TestParamDelta(unittest.TestCase):
    """Test ParamDelta core functionality"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.temp_dir = tempfile.mkdtemp()
        self.param_delta = ParamDelta(device="cpu")
        
        # Create small synthetic models
        self.model_size = 128  # Small hidden size for testing
        self.num_layers = 4
        
    def tearDown(self):
        """Clean up"""
        shutil.rmtree(self.temp_dir)
    
    def create_synthetic_model(self, seed: int = 42) -> dict:
        """Create a small synthetic model for testing"""
        torch.manual_seed(seed)
        
        model = {}
        
        # Create transformer-like structure
        for i in range(self.num_layers):
            # Attention layers
            model[f"transformer.h.{i}.attn.q_proj.weight"] = torch.randn(
                self.model_size, self.model_size
            )
            model[f"transformer.h.{i}.attn.k_proj.weight"] = torch.randn(
                self.model_size, self.model_size
            )
            model[f"transformer.h.{i}.attn.v_proj.weight"] = torch.randn(
                self.model_size, self.model_size
            )
            model[f"transformer.h.{i}.attn.out_proj.weight"] = torch.randn(
                self.model_size, self.model_size
            )
            
            # MLP layers
            model[f"transformer.h.{i}.mlp.fc1.weight"] = torch.randn(
                self.model_size * 4, self.model_size
            )
            model[f"transformer.h.{i}.mlp.fc2.weight"] = torch.randn(
                self.model_size, self.model_size * 4
            )
            
            # Layer norm
            model[f"transformer.h.{i}.ln1.weight"] = torch.randn(self.model_size)
            model[f"transformer.h.{i}.ln2.weight"] = torch.randn(self.model_size)
        
        # Embeddings
        model["transformer.wte.weight"] = torch.randn(1000, self.model_size)
        model["transformer.wpe.weight"] = torch.randn(512, self.model_size)
        
        return model
    
    def test_delta_calculation(self):
        """Test basic delta calculation"""
        # Create base and post-trained models
        base_model = self.create_synthetic_model(seed=42)
        post_model = self.create_synthetic_model(seed=43)
        
        # Calculate delta
        delta = self.param_delta.calculate_delta(post_model, base_model)
        
        # Verify delta
        self.assertEqual(len(delta), len(base_model))
        
        for key in base_model:
            self.assertIn(key, delta)
            expected_delta = post_model[key] - base_model[key]
            torch.testing.assert_close(delta[key], expected_delta)
    
    def test_delta_application(self):
        """Test applying delta to new base model"""
        # Create models
        base_model = self.create_synthetic_model(seed=42)
        post_model = self.create_synthetic_model(seed=43)
        new_base_model = self.create_synthetic_model(seed=44)
        
        # Calculate delta
        delta = self.param_delta.calculate_delta(post_model, base_model)
        
        # Apply delta
        param_delta_model = self.param_delta.apply_delta(new_base_model, delta)
        
        # Verify result
        self.assertEqual(len(param_delta_model), len(new_base_model))
        
        for key in new_base_model:
            expected = new_base_model[key] + delta[key]
            torch.testing.assert_close(param_delta_model[key], expected)
    
    def test_scaled_delta(self):
        """Test applying scaled delta"""
        base_model = self.create_synthetic_model(seed=42)
        post_model = self.create_synthetic_model(seed=43)
        new_base_model = self.create_synthetic_model(seed=44)
        
        # Calculate delta
        delta = self.param_delta.calculate_delta(post_model, base_model)
        
        # Apply with different scales
        scales = [0.5, 1.0, 1.5]
        
        for scale in scales:
            result = self.param_delta.apply_delta(new_base_model, delta, scale=scale)
            
            for key in new_base_model:
                expected = new_base_model[key] + scale * delta[key]
                torch.testing.assert_close(result[key], expected)
    
    def test_multiple_deltas(self):
        """Test combining multiple deltas"""
        base_model = self.create_synthetic_model(seed=42)
        
        # Create multiple post-trained models
        post1 = self.create_synthetic_model(seed=43)
        post2 = self.create_synthetic_model(seed=44)
        
        # Calculate deltas
        delta1 = self.param_delta.calculate_delta(post1, base_model)
        delta2 = self.param_delta.calculate_delta(post2, base_model)
        
        # Combine with weights
        deltas = [(delta1, 0.5), (delta2, 0.5)]
        combined = self.param_delta.combine_multiple_deltas(base_model, deltas)
        
        # Verify
        for key in base_model:
            expected = base_model[key] + 0.5 * delta1[key] + 0.5 * delta2[key]
            torch.testing.assert_close(combined[key], expected)
    
    def test_cosine_similarity(self):
        """Test cosine similarity computation"""
        # Create two deltas with known similarity
        delta1 = {
            "layer1": torch.tensor([1.0, 0.0, 0.0]),
            "layer2": torch.tensor([1.0, 1.0, 0.0])
        }
        
        delta2 = {
            "layer1": torch.tensor([0.0, 1.0, 0.0]),  # Orthogonal to delta1
            "layer2": torch.tensor([1.0, 1.0, 0.0])   # Same as delta1
        }
        
        similarities = self.param_delta.compute_cosine_similarity(delta1, delta2)
        
        # Overall similarity should be average of individual similarities
        self.assertAlmostEqual(similarities["overall"], 0.5, places=3)
    
    def test_weight_norms(self):
        """Test weight norm computation"""
        # Create delta with known norms
        delta = {
            "transformer.h.0.attn.q_proj.weight": torch.ones(3, 3),  # norm = 3
            "transformer.h.0.mlp.fc1.weight": torch.ones(4, 4) * 2,  # norm = 8
        }
        
        norms = self.param_delta.compute_weight_norms(delta)
        
        self.assertAlmostEqual(norms["attention"][0], 3.0, places=3)
        self.assertAlmostEqual(norms["mlp"][0], 8.0, places=3)


class TestModelUtils(unittest.TestCase):
    """Test model utilities"""
    
    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
    
    def tearDown(self):
        shutil.rmtree(self.temp_dir)
    
    def test_format_detection(self):
        """Test model format detection"""
        # Create test files
        pytorch_file = Path(self.temp_dir) / "model.pt"
        torch.save({"state_dict": {}}, pytorch_file)
        
        # Test detection
        format_type = ModelFormatHandler.detect_format(pytorch_file)
        self.assertEqual(format_type, "pytorch")
    
    def test_architecture_compatibility(self):
        """Test architecture compatibility checking"""
        # Compatible models
        model1 = {
            "layer1.weight": torch.randn(10, 10),
            "layer2.weight": torch.randn(5, 10)
        }
        
        model2 = {
            "layer1.weight": torch.randn(10, 10),
            "layer2.weight": torch.randn(5, 10)
        }
        
        is_compatible, issues = ModelValidator.check_architecture_compatibility(
            model1, model2
        )
        self.assertTrue(is_compatible)
        self.assertEqual(len(issues), 0)
        
        # Incompatible models (different shapes)
        model3 = {
            "layer1.weight": torch.randn(20, 20),  # Different shape
            "layer2.weight": torch.randn(5, 10)
        }
        
        is_compatible, issues = ModelValidator.check_architecture_compatibility(
            model1, model3
        )
        self.assertFalse(is_compatible)
        self.assertGreater(len(issues), 0)
    
    def test_layer_categorization(self):
        """Test layer categorization"""
        state_dict = {
            "transformer.h.0.attn.q_proj.weight": torch.randn(10, 10),
            "transformer.h.0.mlp.fc1.weight": torch.randn(10, 10),
            "transformer.wte.weight": torch.randn(100, 10),
            "transformer.ln_f.weight": torch.randn(10),
            "unknown_layer": torch.randn(5, 5)
        }
        
        categories = LayerAnalyzer.categorize_layers(state_dict)
        
        self.assertIn("transformer.h.0.attn.q_proj.weight", categories["attention"])
        self.assertIn("transformer.h.0.mlp.fc1.weight", categories["mlp"])
        self.assertIn("transformer.wte.weight", categories["embedding"])
        self.assertIn("transformer.ln_f.weight", categories["norm"])
        self.assertIn("unknown_layer", categories["other"])


class TestEvaluation(unittest.TestCase):
    """Test evaluation framework"""
    
    def test_mock_benchmark(self):
        """Test mock benchmark functionality"""
        benchmark = MockBenchmark("test_benchmark", (0.5, 0.9))
        
        # Mock model and tokenizer
        model = type('Model', (), {'config': type('Config', (), {'hidden_size': 1024})()})()
        tokenizer = None
        
        result = benchmark.evaluate(model, tokenizer)
        
        self.assertEqual(result.benchmark, "test_benchmark")
        self.assertGreaterEqual(result.score, 0.5)
        self.assertLessEqual(result.score, 1.0)
        self.assertIn("accuracy", result.metrics)
    
    def test_transfer_efficiency(self):
        """Test transfer efficiency calculation"""
        evaluator = ParamDeltaEvaluator()
        
        # Create mock results
        base_results = {
            "MMLU": type('Result', (), {'score': 0.6})(),
            "HumanEval": type('Result', (), {'score': 0.3})()
        }
        
        param_delta_results = {
            "MMLU": type('Result', (), {'score': 0.75})(),
            "HumanEval": type('Result', (), {'score': 0.6})()
        }
        
        reference_results = {
            "MMLU": type('Result', (), {'score': 0.8})(),
            "HumanEval": type('Result', (), {'score': 0.7})()
        }
        
        efficiency = evaluator.calculate_transfer_efficiency(
            base_results,
            param_delta_results,
            reference_results
        )
        
        # Check gamma calculation
        # MMLU: (0.75-0.6)/(0.8-0.6) = 0.75
        # HumanEval: (0.6-0.3)/(0.7-0.3) = 0.75
        self.assertAlmostEqual(efficiency["gamma"], 0.75, places=2)


class TestVisualization(unittest.TestCase):
    """Test visualization analysis"""
    
    def test_layer_similarity_computation(self):
        """Test layer-wise similarity computation"""
        delta1 = {
            "transformer.h.0.attn.weight": torch.randn(10, 10),
            "transformer.h.1.attn.weight": torch.randn(10, 10),
        }
        
        delta2 = {
            "transformer.h.0.attn.weight": torch.randn(10, 10),
            "transformer.h.1.attn.weight": torch.randn(10, 10),
        }
        
        similarities = DeltaAnalyzer.compute_layer_similarities(delta1, delta2)
        
        self.assertIn(0, similarities)
        self.assertIn(1, similarities)
        
        # Similarities should be between -1 and 1
        for sim in similarities.values():
            self.assertGreaterEqual(sim, -1.0)
            self.assertLessEqual(sim, 1.0)


def run_integration_test():
    """Run a full integration test of the ParamΔ pipeline"""
    logger.info("Running integration test...")
    
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        
        # Create synthetic models
        param_delta = ParamDelta(device="cpu")
        
        # Generate test models
        torch.manual_seed(42)
        base_model = {}
        post_model = {}
        new_base_model = {}
        
        for i in range(4):
            base_weight = torch.randn(128, 128)
            post_weight = base_weight + torch.randn(128, 128) * 0.1
            new_base_weight = torch.randn(128, 128)
            
            key = f"layer.{i}.weight"
            base_model[key] = base_weight
            post_model[key] = post_weight
            new_base_model[key] = new_base_weight
        
        # Save models
        base_path = temp_path / "base_model.pt"
        post_path = temp_path / "post_model.pt"
        new_base_path = temp_path / "new_base_model.pt"
        
        torch.save(base_model, base_path)
        torch.save(post_model, post_path)
        torch.save(new_base_model, new_base_path)
        
        # Test full pipeline
        logger.info("1. Computing delta...")
        delta = param_delta.calculate_delta(post_model, base_model)
        
        logger.info("2. Applying delta...")
        param_delta_model = param_delta.apply_delta(new_base_model, delta)
        
        logger.info("3. Verifying results...")
        for key in base_model:
            expected = new_base_model[key] + (post_model[key] - base_model[key])
            actual = param_delta_model[key]
            
            if not torch.allclose(expected, actual, rtol=1e-5):
                logger.error(f"Mismatch in {key}")
                return False
        
        logger.info("Integration test passed!")
        return True


if __name__ == "__main__":
    # Run unit tests
    unittest.main(argv=[''], exit=False, verbosity=2)
    
    # Run integration test
    print("\n" + "="*50)
    print("Running integration test...")
    print("="*50)
    
    success = run_integration_test()
    if success:
        print("\n✓ All tests passed!")
    else:
        print("\n✗ Integration test failed!")
        sys.exit(1)