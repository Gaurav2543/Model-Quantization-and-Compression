# Quantization and Model Compression

## Overview

This repository contains the implementation of various model quantization techniques for Large Language Models (LLMs). It includes both custom quantization implementations and Bitsandbytes integration.

## File Structure

```
.
├── README.md
├── src/                     
│    ├── Part1.py                    # Custom quantization implementation
│    └── Part2.py                    # Bitsandbytes integration
├── results/                     
│    ├── quantization_comparison.png        # Results and visualizations
│    └──  bitsandbytes_comparison.png        # Results and visualizations
└── models/                     # Saved quantized models (One drive link)
    ├── quantized_models
    │   ├── original/
    │   ├── selective_8bit/
    │   └── quantized_8bit/
    └── bitsandbytes_models
        ├── original/
        ├── 8bit/
        └── nf4/                 
```

## Requirements

### Dependencies

```bash
pip install torch transformers datasets bitsandbytes accelerate tqdm seaborn matplotlib pandas psutil
```

### Hardware Requirements

- Minimum 8GB RAM
- CUDA-capable GPU recommended (for faster inference)
- Storage: ~2GB for models and data

## Usage

### Part 1: Custom Quantization

```bash
python src/Part1.py
```

This script will:

1. Load the GPT-2 model
2. Perform custom 8-bit quantization
3. Implement selective quantization
4. Generate performance comparisons
5. Save quantized models and results

### Part 2: Bitsandbytes Integration

```bash
python src/Part2.py
```

This script will:

1. Load the GPT-2 model
2. Apply 8-bit quantization using Bitsandbytes
3. Implement NF4 quantization
4. Generate performance comparisons
5. Save quantized models and results

## Model Checkpoints

Due to file size limitations, the quantized models are available at the following link:

- [Models](https://iiitaphyd-my.sharepoint.com/:f:/g/personal/gaurav_bhole_research_iiit_ac_in/EgVS_bhEMYxBjOduHIYR7BQBzLUSLlCPCOn-XncuYXblvg?e=sgZ34U)

## Results

The scripts generate:

1. Model size comparisons
2. Inference latency measurements
3. Memory usage statistics
4. Perplexity evaluations

Results are saved in:

- Visualization plots: `results/bitsandbytes_comparison.png` and `results/quantization_comparison.png`

## Notes

- For optimal performance, use a GPU
- Memory measurements might vary based on system configuration
- Perplexity computation can be time-consuming
