import gc
import os
import time
import torch
import psutil
import warnings
import numpy as np
import pandas as pd
import seaborn as sns
from tqdm import tqdm
import matplotlib.pyplot as plt
from datasets import load_dataset
from typing import Dict, List, Tuple, Any
from transformers import AutoModelForCausalLM, AutoTokenizer

warnings.filterwarnings('ignore')  # Suppress warnings for cleaner output
warnings.filterwarnings('ignore', category=UserWarning)
warnings.filterwarnings('ignore', category=FutureWarning)

def load_model_and_tokenizer(model_name: str) -> Tuple[AutoModelForCausalLM, AutoTokenizer]:
    """
    Load the model and tokenizer with proper padding token configuration.
    
    Args:
        model_name: Name of the model to load from HuggingFace
    Returns:
        tuple: (model, tokenizer)
    """
    print(f"Loading model {model_name}...")
    model = AutoModelForCausalLM.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        clean_up_tokenization_spaces=True,
        padding_side='left',
    )
    
    # Properly handle padding token
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        model.config.pad_token_id = model.config.eos_token_id
        
    return model, tokenizer

def load_evaluation_data(max_samples: int = 3000) -> List[str]:
    """
    Load and prepare evaluation dataset from WikiText-2.
    
    Args:
        max_samples: Maximum number of samples to load
    Returns:
        list: List of text samples for evaluation
    """
    print("Loading evaluation dataset...")
    try:
        dataset = load_dataset('wikitext', 'wikitext-2-raw-v1', trust_remote_code=True)
        eval_data = [
            text for text in dataset['validation']['text'] 
            if isinstance(text, str) and len(text.strip()) > 0
        ][:max_samples]
        
        if not eval_data:
            raise ValueError("No valid evaluation data found")
            
    except Exception as e:
        print(f"Error loading dataset: {str(e)}")
        print("Falling back to synthetic data...")
        # Create synthetic data if dataset loading fails
        eval_data = [
            "The quick brown fox jumps over the lazy dog.",
            "A journey of a thousand miles begins with a single step.",
            "To be or not to be, that is the question."
        ] * (max_samples // 3 + 1)
        eval_data = eval_data[:max_samples]
    
    print(f"Loaded {len(eval_data)} evaluation samples")
    return eval_data

def calculate_model_size(model: AutoModelForCausalLM, quantized_params: set = None, bits: int = None) -> float:
    """
    Calculate the model size in MB, accounting for both quantized and non-quantized parameters.
    
    Args:
        model: The model to analyze
        quantized_params: Set of parameter names that have been quantized
        bits: Number of bits used for quantized parameters
    Returns:
        float: Model size in MB
    """
    total_size = 0
    for name, param in model.named_parameters():
        if quantized_params and name in quantized_params:
            # For quantized parameters, calculate size based on bit width
            param_size = (param.nelement() * bits) / 8
            param_size += 8  # Add overhead for scale and zero point
        else:
            # For non-quantized parameters, use actual size
            param_size = param.nelement() * param.element_size()
        total_size += param_size
    return total_size / (1024 * 1024)  # Convert bytes to MB

def quantize_tensor(tensor: torch.Tensor, bits: int) -> torch.Tensor:
    """
    Quantize a single tensor with improved stability and performance preservation.
    
    Args:
        tensor: Input tensor to quantize
        bits: Target bit width
    Returns:
        torch.Tensor: Quantized tensor in original dtype
    """
    qmin = -(2 ** (bits - 1))
    qmax = 2 ** (bits - 1) - 1
    
    tensor_flat = tensor.detach().flatten()
    
    # Calculate more conservative bounds
    abs_max = abs(tensor_flat).max().item()
    scale = abs_max / qmax
    
    if scale == 0:
        return tensor.clone()
    
    # Quantize
    tensor_q = torch.clamp(torch.round(tensor.detach() / scale), qmin, qmax)
    
    # Dequantize
    tensor_d = tensor_q * scale
    
    return tensor_d.to(tensor.dtype)

def quantize_whole_model(model: AutoModelForCausalLM, bits: int) -> Tuple[AutoModelForCausalLM, set]:
    """
    Quantize model with improved stability.
    
    Args:
        model: Model to quantize
        bits: Target bit width
    Returns:
        tuple: (quantized model, set of quantized parameter names)
    """
    print(f"Starting {bits}-bit whole model quantization...")
    quantized_params = set()
    
    with torch.no_grad():
        for name, param in model.named_parameters():
            if param.dtype in [torch.float32, torch.float16]:
                # Only quantize weights, not biases
                if len(param.shape) >= 2:  # Skip bias terms
                    if bits == 4:
                        # For 4-bit, be more selective
                        if param.numel() > 1000 and 'weight' in name:
                            quantized_data = quantize_tensor(param.data, bits)
                            param.data.copy_(quantized_data)
                            quantized_params.add(name)
                    else:
                        quantized_data = quantize_tensor(param.data, bits)
                        param.data.copy_(quantized_data)
                        quantized_params.add(name)
    
    return model, quantized_params

def quantize_selective_components(model: AutoModelForCausalLM, bits: int, num_layers: int = 5) -> Tuple[AutoModelForCausalLM, set]:
    """
    Selectively quantize with improved stability.
    
    Args:
        model: Model to quantize
        bits: Target bit width
        num_layers: Number of layers to quantize
    Returns:
        tuple: (quantized model, set of quantized parameter names)
    """
    print(f"Starting selective {bits}-bit quantization for first {num_layers} layers...")
    quantized_params = set()
    
    with torch.no_grad():
        for i in range(num_layers):
            try:
                # Quantize only the weight matrices, not the biases
                attn_name = f"transformer.h.{i}.attn.c_attn.weight"
                if hasattr(model, 'transformer') and hasattr(model.transformer, 'h'):
                    attn_wts = model.transformer.h[i].attn.c_attn.weight
                    quantized_attn_wts = quantize_tensor(attn_wts, bits)
                    attn_wts.data.copy_(quantized_attn_wts)
                    quantized_params.add(attn_name)
                    print(f"Quantized attention weights for layer {i}")
            except Exception as e:
                print(f"Failed to quantize attention weights for layer {i}: {str(e)}")

            try:
                ffn_name = f"transformer.h.{i}.mlp.c_fc.weight"
                if hasattr(model, 'transformer') and hasattr(model.transformer, 'h'):
                    ffn_wts = model.transformer.h[i].mlp.c_fc.weight
                    quantized_ffn_wts = quantize_tensor(ffn_wts, bits)
                    ffn_wts.data.copy_(quantized_ffn_wts)
                    quantized_params.add(ffn_name)
                    print(f"Quantized FFN weights for layer {i}")
            except Exception as e:
                print(f"Failed to quantize FFN weights for layer {i}: {str(e)}")
    
    return model, quantized_params

def measure_latency(model: AutoModelForCausalLM, tokenizer: AutoTokenizer, 
                   text: str, num_runs: int = 5) -> Tuple[float, float]:
    """
    Measure model inference latency and memory usage.
    
    Args:
        model: Model to evaluate
        tokenizer: Tokenizer for input processing
        text: Input text for inference
        num_runs: Number of runs for averaging
    Returns:
        tuple: (average latency, average memory usage)
    """
    inputs = tokenizer(text, return_tensors="pt", padding=True)
    
    # Warm-up run
    with torch.no_grad():
        model.generate(**inputs, max_new_tokens=20)
    
    latencies = []
    memory_usage = []
    
    for _ in range(num_runs):
        torch.cuda.empty_cache() if torch.cuda.is_available() else gc.collect()
        
        memory_before = psutil.Process().memory_info().rss / 1024 / 1024
        
        start_time = time.time()
        with torch.no_grad():
            model.generate(**inputs, max_new_tokens=20)
        latencies.append(time.time() - start_time)
        
        memory_after = psutil.Process().memory_info().rss / 1024 / 1024
        memory_usage.append(memory_after - memory_before)
    
    return np.mean(latencies), np.mean(memory_usage)

def compute_perplexity(model: AutoModelForCausalLM, tokenizer: AutoTokenizer, 
                      eval_data: List[str]) -> float:
    """
    Compute model perplexity on evaluation data.
    
    Args:
        model: Model to evaluate
        tokenizer: Tokenizer for input processing
        eval_data: List of evaluation texts
    Returns:
        float: Perplexity score
    """
    total_loss = 0
    total_tokens = 0
    
    model.eval()
    with torch.no_grad():
        for text in tqdm(eval_data, desc="Computing perplexity"):
            inputs = tokenizer(text, return_tensors="pt", padding=True)
            outputs = model(**inputs, labels=inputs["input_ids"])
            total_loss += outputs.loss.item() * inputs["input_ids"].size(1)
            total_tokens += inputs["input_ids"].size(1)
    
    return float(torch.exp(torch.tensor(total_loss / total_tokens)))

def evaluate_model_version(model: AutoModelForCausalLM, tokenizer: AutoTokenizer,
                         eval_data: List[str], sample_text: str, 
                         quantized_params: set = None, bits: int = None) -> Dict[str, float]:
    """
    Evaluate a model version on all metrics.
    
    Args:
        model: Model to evaluate
        tokenizer: Tokenizer for input processing
        eval_data: Evaluation dataset
        sample_text: Text for latency testing
        quantized_params: Set of quantized parameter names
        bits: Bit width used for quantization
    Returns:
        dict: Dictionary of evaluation metrics
    """
    return {
        'size': calculate_model_size(model, quantized_params, bits),
        'latency': measure_latency(model, tokenizer, sample_text)[0],
        'memory_usage': measure_latency(model, tokenizer, sample_text)[1],
        'perplexity': compute_perplexity(model, tokenizer, eval_data)
    }

def create_visualizations(results: Dict[str, Dict[str, float]]):
    """
    Create and save visualization plots for all metrics.
    
    Args:
        results: Dictionary of evaluation results
    """
    metrics = ['size', 'latency', 'memory_usage', 'perplexity']
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('Model Quantization Performance Comparison', fontsize=16, y=1.02)
    
    # Prepare data for plotting
    df = pd.DataFrame(results).T.reset_index()
    df.columns = ['Model'] + metrics
    
    # Create subplots
    for idx, metric in enumerate(metrics):
        ax = axes[idx // 2, idx % 2]
        
        # Create barplot
        sns.barplot(
            data=df,
            x='Model',
            y=metric,
            hue='Model',
            ax=ax,
            legend=False
        )
        
        # Customize plot
        ax.set_title(f'{metric.replace("_", " ").title()}')
        plt.setp(ax.get_xticklabels(), rotation=45, ha='right')
        
        # Add value labels
        for i, bar in enumerate(ax.patches):
            ax.text(
                bar.get_x() + bar.get_width()/2,
                bar.get_height(),
                f'{bar.get_height():.2f}',
                ha='center',
                va='bottom',
                fontsize=8
            )
    
    plt.tight_layout()
    plt.savefig('quantization_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()

def print_evaluation_summary(results: Dict[str, Dict[str, float]]):
    print("\n=== Quantization Evaluation Summary ===")
    metrics = ['size', 'latency', 'memory_usage', 'perplexity']
    
    for metric in metrics:
        print(f"\n{metric.replace('_', ' ').title()}:")
        baseline = results['original'][metric]
        for model_type, values in results.items():
            # Handle zero division cases
            if baseline == 0:
                change = 0 if values[metric] == 0 else float('inf')
            else:
                change = ((values[metric] - baseline) / baseline) * 100
            
            # Format output based on metric type
            if metric == 'size':
                print(f"{model_type:15s}: {values[metric]:8.2f} MB ({change:+.1f}% change)")
            elif metric == 'latency':
                print(f"{model_type:15s}: {values[metric]:8.3f} s ({change:+.1f}% change)")
            elif metric == 'memory_usage':
                print(f"{model_type:15s}: {values[metric]:8.2f} MB ({change:+.1f}% change)")
            else:  # perplexity
                print(f"{model_type:15s}: {values[metric]:8.2f} ({change:+.1f}% change)")

def save_model(model: AutoModelForCausalLM, tokenizer: AutoTokenizer, path: str):
    """
    Save model and tokenizer to disk.
    
    Args:
        model: Model to save
        tokenizer: Tokenizer to save
        path: Directory path to save to
    """
    try:
        print(f"Saving model to {path}...")
        model.save_pretrained(path)
        tokenizer.save_pretrained(path)
        print(f"Model and tokenizer saved successfully to {path}")
    except Exception as e:
        print(f"Error saving model: {str(e)}")


output_dir = "quantized_models"
os.makedirs(output_dir, exist_ok=True)

# Set random seeds for reproducibility
torch.manual_seed(42)
np.random.seed(42)

# Configure warnings
warnings.filterwarnings('ignore', category=UserWarning)
warnings.filterwarnings('ignore', category=FutureWarning)

# Initialize
print("\nInitializing quantization analysis...")
output_dir = "quantized_models"
os.makedirs(output_dir, exist_ok=True)

model, tokenizer = load_model_and_tokenizer("gpt2")
eval_data = load_evaluation_data(max_samples=3000)
sample_text = "The quick brown fox jumps over the lazy dog"

results = {}
original_state = {name: param.clone() for name, param in model.named_parameters()}

# Evaluate and save original model
print("\nEvaluating original model...")
results['original'] = evaluate_model_version(model, tokenizer, eval_data, sample_text)
save_model(model, tokenizer, os.path.join(output_dir, "original"))

# Evaluate and save selective quantization
print("\nEvaluating selective quantization...")
model_selective, params_selective = quantize_selective_components(model, bits=8)
results['selective_8bit'] = evaluate_model_version(
    model_selective, tokenizer, eval_data, sample_text, params_selective, 8
)
save_model(model_selective, tokenizer, os.path.join(output_dir, "selective_8bit"))

# Restore original state
with torch.no_grad():
    for name, param in model.named_parameters():
        param.data.copy_(original_state[name])

# Evaluate and save 8-bit whole model quantization
print("\nEvaluating 8-bit quantization...")
model_8bit, params_8bit = quantize_whole_model(model, bits=8)
results['quantized_8bit'] = evaluate_model_version(
    model_8bit, tokenizer, eval_data, sample_text, params_8bit, 8
)
save_model(model_8bit, tokenizer, os.path.join(output_dir, "quantized_8bit"))

# Create visualizations and save results
print("\nGenerating analysis results...")
create_visualizations(results)
print_evaluation_summary(results)

# Save the quantization results
results_df = pd.DataFrame(results).round(4)
results_df.to_csv(os.path.join(output_dir, "quantization_results.csv"))

print("\nQuantization analysis completed successfully!")
print(f"\nAll models and results have been saved to: {output_dir}")