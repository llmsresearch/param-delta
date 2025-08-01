"""
Visualization tools for ParamΔ analysis.
Creates plots for cosine similarity, weight norms, and performance metrics.
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Optional, Tuple, Union
from pathlib import Path
import logging
import torch
from dataclasses import dataclass

logger = logging.getLogger(__name__)

# Set style for publication-quality figures
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("husl")


@dataclass
class LayerInfo:
    """Information about a model layer"""
    name: str
    layer_num: int
    layer_type: str  # attention, mlp, etc.


class ParamDeltaVisualizer:
    """Main visualization class for ParamΔ analysis"""
    
    def __init__(self, figure_dir: Union[str, Path] = "figures"):
        """
        Initialize visualizer.
        
        Args:
            figure_dir: Directory to save figures
        """
        self.figure_dir = Path(figure_dir)
        self.figure_dir.mkdir(parents=True, exist_ok=True)
    
    def plot_cosine_similarity(
        self,
        similarities: Dict[int, float],
        delta_names: Tuple[str, str],
        save_name: Optional[str] = None,
        layer_types: Optional[Dict[int, str]] = None
    ) -> plt.Figure:
        """
        Plot cosine similarity between parameter deltas by layer.
        
        Args:
            similarities: Dict mapping layer index to cosine similarity
            delta_names: Names of the two deltas being compared
            save_name: Optional filename to save figure
            layer_types: Optional dict mapping layer index to type
            
        Returns:
            Matplotlib figure
        """
        fig, ax = plt.subplots(figsize=(10, 6))
        
        layers = sorted(similarities.keys())
        values = [similarities[l] for l in layers]
        
        # Color by layer type if provided
        if layer_types:
            colors = []
            for l in layers:
                if l in layer_types:
                    color = 'blue' if 'attn' in layer_types[l] else 'orange'
                else:
                    color = 'gray'
                colors.append(color)
            bars = ax.bar(layers, values, color=colors)
            
            # Add legend
            from matplotlib.patches import Patch
            legend_elements = [
                Patch(facecolor='blue', label='Attention'),
                Patch(facecolor='orange', label='MLP')
            ]
            ax.legend(handles=legend_elements)
        else:
            bars = ax.bar(layers, values)
        
        ax.set_xlabel('Layer Index')
        ax.set_ylabel('Cosine Similarity')
        ax.set_title(f'Cosine Similarity: {delta_names[0]} vs {delta_names[1]}')
        ax.set_ylim(-0.1, 1.1)
        ax.axhline(y=0, color='black', linestyle='--', alpha=0.3)
        
        # Add value labels on bars
        for bar, value in zip(bars, values):
            if abs(value) > 0.1:  # Only label significant values
                ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                       f'{value:.2f}', ha='center', va='bottom', fontsize=8)
        
        plt.tight_layout()
        
        if save_name:
            save_path = self.figure_dir / f"{save_name}.png"
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Saved figure to {save_path}")
        
        return fig
    
    def plot_weight_norms(
        self,
        norms_dict: Dict[str, List[Tuple[int, float]]],
        model_names: List[str],
        save_name: Optional[str] = None
    ) -> plt.Figure:
        """
        Plot weight norm distributions by layer type.
        
        Args:
            norms_dict: Dict mapping layer type to list of (layer_idx, norm) tuples
            model_names: Names of models being compared
            save_name: Optional filename to save figure
            
        Returns:
            Matplotlib figure
        """
        fig, axes = plt.subplots(1, 2, figsize=(15, 6))
        
        for idx, (layer_type, ax) in enumerate(zip(['attention', 'mlp'], axes)):
            if layer_type not in norms_dict:
                continue
            
            # Prepare data
            data = norms_dict[layer_type]
            if not data:
                continue
            
            # Group by model
            for model_idx, model_name in enumerate(model_names):
                model_data = [(l, n) for l, n in data if l % len(model_names) == model_idx]
                if model_data:
                    layers, norms = zip(*model_data)
                    ax.plot(layers, norms, label=model_name, marker='o', markersize=4)
            
            ax.set_xlabel('Layer Index')
            ax.set_ylabel('L2 Norm')
            ax.set_title(f'{layer_type.capitalize()} Layer Norms')
            ax.legend()
            ax.grid(True, alpha=0.3)
        
        plt.suptitle('Weight Norm Distribution by Layer Type')
        plt.tight_layout()
        
        if save_name:
            save_path = self.figure_dir / f"{save_name}.png"
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Saved figure to {save_path}")
        
        return fig
    
    def plot_performance_comparison(
        self,
        results: Dict[str, Dict[str, float]],
        baseline_model: str,
        save_name: Optional[str] = None
    ) -> plt.Figure:
        """
        Plot performance comparison across benchmarks.
        
        Args:
            results: Nested dict {model_name: {benchmark: score}}
            baseline_model: Name of baseline model for comparison
            save_name: Optional filename to save figure
            
        Returns:
            Matplotlib figure
        """
        # Prepare data
        benchmarks = list(next(iter(results.values())).keys())
        models = list(results.keys())
        
        # Create figure
        fig, ax = plt.subplots(figsize=(12, 8))
        
        # Set up bar positions
        x = np.arange(len(benchmarks))
        width = 0.8 / len(models)
        
        # Plot bars for each model
        for i, model in enumerate(models):
            scores = [results[model].get(b, 0) for b in benchmarks]
            offset = (i - len(models)/2 + 0.5) * width
            bars = ax.bar(x + offset, scores, width, label=model)
            
            # Add value labels
            for bar, score in zip(bars, scores):
                ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                       f'{score:.2f}', ha='center', va='bottom', fontsize=8)
        
        # Customize plot
        ax.set_xlabel('Benchmark')
        ax.set_ylabel('Score')
        ax.set_title('Model Performance Comparison')
        ax.set_xticks(x)
        ax.set_xticklabels(benchmarks, rotation=45, ha='right')
        ax.legend()
        ax.set_ylim(0, 1.1)
        
        plt.tight_layout()
        
        if save_name:
            save_path = self.figure_dir / f"{save_name}.png"
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Saved figure to {save_path}")
        
        return fig
    
    def plot_scaling_analysis(
        self,
        scale_factors: List[float],
        performances: Dict[str, List[float]],
        save_name: Optional[str] = None
    ) -> plt.Figure:
        """
        Plot performance vs scaling factor (α) analysis.
        
        Args:
            scale_factors: List of α values
            performances: Dict mapping benchmark to list of scores
            save_name: Optional filename to save figure
            
        Returns:
            Matplotlib figure
        """
        fig, ax = plt.subplots(figsize=(10, 6))
        
        for benchmark, scores in performances.items():
            ax.plot(scale_factors, scores, marker='o', label=benchmark)
        
        # Mark optimal point (α=1.0)
        ax.axvline(x=1.0, color='red', linestyle='--', alpha=0.5, label='α=1.0')
        
        ax.set_xlabel('Scaling Factor (α)')
        ax.set_ylabel('Performance Score')
        ax.set_title('Performance vs Parameter Delta Scaling Factor')
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_name:
            save_path = self.figure_dir / f"{save_name}.png"
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Saved figure to {save_path}")
        
        return fig
    
    def plot_transfer_efficiency(
        self,
        hypothetical_scores: List[float],
        actual_scores: List[float],
        benchmark_names: List[str],
        save_name: Optional[str] = None
    ) -> plt.Figure:
        """
        Plot transfer efficiency analysis (actual vs hypothetical performance).
        
        Args:
            hypothetical_scores: Expected scores from linear interpolation
            actual_scores: Actual ParamΔ model scores
            benchmark_names: Names of benchmarks
            save_name: Optional filename to save figure
            
        Returns:
            Matplotlib figure
        """
        fig, ax = plt.subplots(figsize=(10, 10))
        
        # Scatter plot
        scatter = ax.scatter(hypothetical_scores, actual_scores, s=100, alpha=0.6)
        
        # Add benchmark labels
        for i, name in enumerate(benchmark_names):
            ax.annotate(name, (hypothetical_scores[i], actual_scores[i]),
                       xytext=(5, 5), textcoords='offset points', fontsize=8)
        
        # Add ideal line (y=x)
        min_val = min(min(hypothetical_scores), min(actual_scores))
        max_val = max(max(hypothetical_scores), max(actual_scores))
        ax.plot([min_val, max_val], [min_val, max_val], 'r--', alpha=0.5, label='y=x')
        
        # Fit regression line
        from scipy import stats
        slope, intercept, r_value, _, _ = stats.linregress(hypothetical_scores, actual_scores)
        x_fit = np.linspace(min_val, max_val, 100)
        y_fit = slope * x_fit + intercept
        ax.plot(x_fit, y_fit, 'b-', alpha=0.8,
               label=f'γ={slope:.4f}, R²={r_value**2:.4f}')
        
        ax.set_xlabel('Hypothetical Performance')
        ax.set_ylabel('Actual Performance')
        ax.set_title('Transfer Efficiency Analysis')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Equal aspect ratio
        ax.set_aspect('equal', adjustable='box')
        
        plt.tight_layout()
        
        if save_name:
            save_path = self.figure_dir / f"{save_name}.png"
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Saved figure to {save_path}")
        
        return fig


class DeltaAnalyzer:
    """Analyze parameter deltas for visualization"""
    
    @staticmethod
    def compute_layer_similarities(
        delta1: Dict[str, torch.Tensor],
        delta2: Dict[str, torch.Tensor]
    ) -> Dict[int, float]:
        """
        Compute cosine similarities by layer.
        
        Args:
            delta1: First parameter delta
            delta2: Second parameter delta
            
        Returns:
            Dict mapping layer index to similarity
        """
        similarities = {}
        
        # Extract layer numbers from parameter names
        for key in delta1.keys():
            if key not in delta2:
                continue
            
            # Try to extract layer number
            layer_num = DeltaAnalyzer._extract_layer_number(key)
            if layer_num is not None:
                # Compute cosine similarity
                vec1 = delta1[key].flatten().float()
                vec2 = delta2[key].flatten().float()
                
                cos_sim = torch.nn.functional.cosine_similarity(
                    vec1.unsqueeze(0), vec2.unsqueeze(0)
                ).item()
                
                similarities[layer_num] = cos_sim
        
        return similarities
    
    @staticmethod
    def compute_layer_norms(
        delta: Dict[str, torch.Tensor]
    ) -> Dict[str, List[Tuple[int, float]]]:
        """
        Compute norms by layer type.
        
        Args:
            delta: Parameter delta
            
        Returns:
            Dict mapping layer type to list of (layer_num, norm) tuples
        """
        norms = {"attention": [], "mlp": []}
        
        for key, tensor in delta.items():
            layer_num = DeltaAnalyzer._extract_layer_number(key)
            if layer_num is None:
                continue
            
            norm = torch.norm(tensor).item()
            
            if any(x in key.lower() for x in ["attn", "attention"]):
                norms["attention"].append((layer_num, norm))
            elif any(x in key.lower() for x in ["mlp", "fc", "feedforward"]):
                norms["mlp"].append((layer_num, norm))
        
        # Sort by layer number
        for layer_type in norms:
            norms[layer_type].sort(key=lambda x: x[0])
        
        return norms
    
    @staticmethod
    def _extract_layer_number(param_name: str) -> Optional[int]:
        """Extract layer number from parameter name"""
        import re
        
        # Common patterns for layer numbers
        patterns = [
            r'layer[._]?(\d+)',
            r'layers[._]?(\d+)',
            r'block[._]?(\d+)',
            r'transformer[._]?h[._]?(\d+)',
        ]
        
        for pattern in patterns:
            match = re.search(pattern, param_name, re.IGNORECASE)
            if match:
                return int(match.group(1))
        
        return None


def create_paper_figures(
    delta1: Dict[str, torch.Tensor],
    delta2: Dict[str, torch.Tensor],
    delta_names: Tuple[str, str],
    output_dir: Union[str, Path] = "figures"
) -> None:
    """
    Create all figures similar to those in the paper.
    
    Args:
        delta1: First parameter delta
        delta2: Second parameter delta  
        delta_names: Names for the deltas
        output_dir: Where to save figures
    """
    visualizer = ParamDeltaVisualizer(output_dir)
    analyzer = DeltaAnalyzer()
    
    # Compute similarities
    similarities = analyzer.compute_layer_similarities(delta1, delta2)
    
    # Plot cosine similarity
    visualizer.plot_cosine_similarity(
        similarities,
        delta_names,
        save_name=f"cosine_similarity_{delta_names[0]}_{delta_names[1]}"
    )
    
    # Compute and plot norms
    norms1 = analyzer.compute_layer_norms(delta1)
    norms2 = analyzer.compute_layer_norms(delta2)
    
    # Combine norms for plotting
    combined_norms = {}
    for layer_type in ["attention", "mlp"]:
        combined_norms[layer_type] = []
        combined_norms[layer_type].extend([(l, n) for l, n in norms1.get(layer_type, [])])
        combined_norms[layer_type].extend([(l, n) for l, n in norms2.get(layer_type, [])])
    
    visualizer.plot_weight_norms(
        combined_norms,
        list(delta_names),
        save_name=f"norm_distribution_{delta_names[0]}_{delta_names[1]}"
    )
    
    logger.info(f"Created figures in {output_dir}")


if __name__ == "__main__":
    # Test visualization
    print("Testing visualization tools...")
    
    # Create mock data
    layers = list(range(32))
    mock_similarities = {l: np.random.uniform(-0.1, 0.3) for l in layers}
    
    # Create visualizer
    viz = ParamDeltaVisualizer("test_figures")
    
    # Test cosine similarity plot
    fig = viz.plot_cosine_similarity(
        mock_similarities,
        ("Model1-Delta", "Model2-Delta"),
        save_name="test_cosine_similarity"
    )
    plt.close(fig)
    
    print("Visualization test complete. Check test_figures/ directory.")