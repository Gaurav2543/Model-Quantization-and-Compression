import gc
import os
import time
import torch
import psutil
import warnings
import numpy as np
import pandas as pd
from tqdm import tqdm
import seaborn as sns
import bitsandbytes as bnb
import matplotlib.pyplot as plt
from datasets import load_dataset
from typing import Dict, List, Tuple, Any
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

warnings.filterwarnings('ignore')  # Suppress warnings for cleaner output
warnings.filterwarnings('ignore', category=UserWarning)
warnings.filterwarnings('ignore', category=FutureWarning)

def load_model_and_tokenizer(model_name: str, quantization_config: BitsAndBytesConfig = None) -> Tuple[AutoModelForCausalLM, AutoTokenizer]:
    """
    Load model with optional quantization configuration.
    """
    print(f"Loading model {model_name}...")
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    if quantization_config:
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            quantization_config=quantization_config,
            device_map={"": device}  # Modified device mapping
        )
    else:
        model = AutoModelForCausalLM.from_pretrained(model_name).to(device)
    
    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        padding_side='left',
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        model.config.pad_token_id = model.config.eos_token_id
    
    return model, tokenizer

def compute_perplexity(model: AutoModelForCausalLM, tokenizer: AutoTokenizer, 
                      eval_data: List[str]) -> float:
    """
    Compute model perplexity.
    """
    total_loss = 0
    total_tokens = 0
    device = next(model.parameters()).device  # Get model's device
    
    model.eval()
    with torch.no_grad():
        for text in tqdm(eval_data, desc="Computing perplexity"):
            inputs = tokenizer(text, return_tensors="pt", padding=True)
            inputs = {k: v.to(device) for k, v in inputs.items()}  # Move inputs to correct device
            outputs = model(**inputs, labels=inputs["input_ids"])
            total_loss += outputs.loss.item() * inputs["input_ids"].size(1)
            total_tokens += inputs["input_ids"].size(1)
    
    return float(torch.exp(torch.tensor(total_loss / total_tokens)))

def measure_latency(model: AutoModelForCausalLM, tokenizer: AutoTokenizer, 
                   text: str, num_runs: int = 5) -> Tuple[float, float]:
    """
    Measure inference latency and memory usage.
    """
    device = next(model.parameters()).device  # Get model's device
    inputs = tokenizer(text, return_tensors="pt", padding=True)
    inputs = {k: v.to(device) for k, v in inputs.items()}  # Move inputs to correct device
    
    # Warm-up run
    with torch.no_grad():
        model.generate(**inputs, max_new_tokens=20)
    
    latencies = []
    memory_usage = []
    
    for _ in range(num_runs):
        if device.type == "cuda":
            torch.cuda.empty_cache()
        else:
            gc.collect()
        
        memory_before = psutil.Process().memory_info().rss / 1024 / 1024
        
        start_time = time.time()
        with torch.no_grad():
            model.generate(**inputs, max_new_tokens=20)
        latencies.append(time.time() - start_time)
        
        memory_after = psutil.Process().memory_info().rss / 1024 / 1024
        memory_usage.append(memory_after - memory_before)
    
    return np.mean(latencies), np.mean(memory_usage)


def load_evaluation_data(max_samples: int = 3000) -> List[str]:
    """
    Load evaluation dataset.
    """
    print("Loading evaluation dataset...")
    try:
        dataset = load_dataset('wikitext', 'wikitext-2-raw-v1', trust_remote_code=True)
        eval_data = [
            text for text in dataset['validation']['text'] 
            if isinstance(text, str) and len(text.strip()) > 0
        ][:max_samples]
    except Exception as e:
        print(f"Error loading dataset: {str(e)}")
        eval_data = [
            "The quick brown fox jumps over the lazy dog.",
            "A journey of a thousand miles begins with a single step.",
            "To be or not to be, that is the question."
        ] * (max_samples // 3 + 1)
        eval_data = eval_data[:max_samples]
    
    print(f"Loaded {len(eval_data)} evaluation samples")
    return eval_data

def get_model_size(model: AutoModelForCausalLM) -> float:
    """
    Calculate model size in MB.
    """
    param_size = 0
    for param in model.parameters():
        param_size += param.nelement() * param.element_size()
    return param_size / (1024 * 1024)

def evaluate_model_version(model: AutoModelForCausalLM, tokenizer: AutoTokenizer,
                         eval_data: List[str], sample_text: str) -> Dict[str, float]:
    """
    Evaluate model on all metrics.
    """
    return {
        'size': get_model_size(model),
        'latency': measure_latency(model, tokenizer, sample_text)[0],
        'memory_usage': measure_latency(model, tokenizer, sample_text)[1],
        'perplexity': compute_perplexity(model, tokenizer, eval_data)
    }

def create_visualizations(results: Dict[str, Dict[str, float]]):
    """
    Create comparison visualizations.
    """
    metrics = ['size', 'latency', 'memory_usage', 'perplexity']
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('Quantization Performance Comparison (Bitsandbytes)', fontsize=16, y=1.02)
    
    df = pd.DataFrame(results).T.reset_index()
    df.columns = ['Model'] + metrics
    
    for idx, metric in enumerate(metrics):
        ax = axes[idx // 2, idx % 2]
        
        sns.barplot(
            data=df,
            x='Model',
            y=metric,
            hue='Model',
            ax=ax,
            legend=False
        )
        
        ax.set_title(f'{metric.replace("_", " ").title()}')
        plt.setp(ax.get_xticklabels(), rotation=45, ha='right')
        
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
    plt.savefig('bitsandbytes_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()

def print_evaluation_summary(results: Dict[str, Dict[str, float]]):
    """
    Print evaluation summary.
    """
    print("\n=== Bitsandbytes Quantization Evaluation Summary ===")
    metrics = ['size', 'latency', 'memory_usage', 'perplexity']
    
    for metric in metrics:
        print(f"\n{metric.replace('_', ' ').title()}:")
        baseline = results['original'][metric]
        for model_type, values in results.items():
            if baseline == 0:
                change = 0 if values[metric] == 0 else float('inf')
            else:
                change = ((values[metric] - baseline) / baseline) * 100
            print(f"{model_type:15s}: {values[metric]:8.2f} ({change:+.1f}% change)")

def save_model(model: AutoModelForCausalLM, tokenizer: AutoTokenizer, path: str):
    """
    Save model and tokenizer to disk with quantization config.
    
    Args:
        model: Model to save
        tokenizer: Tokenizer to save
        path: Directory path to save to
    """
    try:
        print(f"Saving model to {path}...")
        model.save_pretrained(path)
        tokenizer.save_pretrained(path)
        
        # Save quantization configuration if it exists
        if hasattr(model, 'config') and hasattr(model.config, 'quantization_config'):
            model.config.save_pretrained(path)
            
        print(f"Model and tokenizer saved successfully to {path}")
    except Exception as e:
        print(f"Error saving model: {str(e)}")

# Initialize
model_name = "gpt2"
output_dir = "bitsandbytes_models"
os.makedirs(output_dir, exist_ok=True)

# Set random seeds for reproducibility
torch.manual_seed(42)
np.random.seed(42)
results = {}

# Original model
print("\nEvaluating original model...")
model, tokenizer = load_model_and_tokenizer(model_name)
eval_data = load_evaluation_data()
sample_text = "The quick brown fox jumps over the lazy dog"

results['original'] = evaluate_model_version(model, tokenizer, eval_data, sample_text)
save_model(model, tokenizer, os.path.join(output_dir, "original"))
del model
gc.collect()

# 8-bit quantization
print("\nEvaluating 8-bit quantization...")
quantization_config = BitsAndBytesConfig(
    load_in_8bit=True,
    llm_int8_enable_fp32_cpu_offload=True
)
model_8bit, tokenizer = load_model_and_tokenizer(model_name, quantization_config)
results['bitsandbytes_8bit'] = evaluate_model_version(model_8bit, tokenizer, eval_data, sample_text)
save_model(model_8bit, tokenizer, os.path.join(output_dir, "8bit"))
del model_8bit
gc.collect()

# NF4 quantization
print("\nEvaluating NF4 quantization...")
quantization_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_quant_type="nf4"
)
model_nf4, tokenizer = load_model_and_tokenizer(model_name, quantization_config)
results['bitsandbytes_nf4'] = evaluate_model_version(model_nf4, tokenizer, eval_data, sample_text)
save_model(model_nf4, tokenizer, os.path.join(output_dir, "nf4"))

# Create visualizations and print summary
create_visualizations(results)
print_evaluation_summary(results)

# Save the quantization results
results_df = pd.DataFrame(results).round(4)
results_df.to_csv(os.path.join(output_dir, "bitsandbytes_results.csv"))

print(f"\nAll models and results have been saved to: {output_dir}")