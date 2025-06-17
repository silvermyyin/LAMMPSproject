# Scripts Directory Organization

This directory contains all executable scripts organized by functionality for the LAMMPS input generation project. The structure follows research best practices for clear workflow management and reproducibility.

## 📁 Directory Structure Overview

### 🏗️ Data Creation
Scripts for generating and processing datasets:

- **`dataset_generation/`**: Scripts to create training/validation datasets
  - `create_training_data.py`: Generate training datasets from real-world scripts
  - `synthetic_data_generator.py`: Create synthetic LAMMPS examples
  - `data_augmentation.py`: Augment existing datasets
- **`preprocessing/`**: Data preprocessing and cleaning scripts
  - `clean_lammps_scripts.py`: Validate and clean LAMMPS input files
  - `extract_documentation.py`: Extract knowledge from LAMMPS manuals
  - `format_converter.py`: Convert between different data formats
- **`validation/`**: Data validation and quality assurance
  - `validate_lammps_syntax.py`: Check LAMMPS script syntax
  - `test_generated_scripts.py`: Validate generated scripts with LAMMPS
  - `quality_metrics.py`: Calculate data quality metrics

### 🧠 Inference
Scripts for running trained models and generating outputs:

- **`model_inference/`**: Core inference functionality
  - `generate_lammps.py`: Generate LAMMPS scripts using trained models
  - `batch_generation.py`: Process multiple requests in batch
  - `interactive_generator.py`: Interactive LAMMPS generation interface
- **`batch_processing/`**: Large-scale processing scripts
  - `process_dataset.py`: Process entire datasets through models
  - `parallel_inference.py`: Parallel processing for large batches
  - `queue_manager.py`: Manage inference job queues
- **`evaluation/`**: Model evaluation and testing
  - `evaluate_outputs.py`: Evaluate generated LAMMPS scripts
  - `benchmark_models.py`: Compare different model approaches
  - `error_analysis.py`: Analyze generation errors and patterns

### 🎓 Training
Scripts for model training and fine-tuning:

- **`fine_tuning/`**: Model fine-tuning scripts
  - `finetune_llm.py`: Fine-tune language models on LAMMPS data
  - `train_embeddings.py`: Train custom embeddings for LAMMPS
  - `curriculum_learning.py`: Implement curriculum learning strategies
- **`hyperparameter_search/`**: Optimization scripts
  - `grid_search.py`: Grid search for optimal hyperparameters
  - `bayesian_optimization.py`: Bayesian hyperparameter optimization
  - `early_stopping.py`: Implement early stopping strategies
- **`model_validation/`**: Training validation scripts
  - `cross_validation.py`: K-fold cross-validation
  - `performance_tracking.py`: Track training metrics
  - `model_comparison.py`: Compare different model architectures

### 📊 Evaluation
Scripts for comprehensive evaluation and benchmarking:

- **`performance_metrics/`**: Calculate various performance metrics
  - `syntax_accuracy.py`: Measure LAMMPS syntax correctness
  - `semantic_similarity.py`: Evaluate semantic similarity to targets
  - `execution_success.py`: Test actual LAMMPS execution success
- **`comparison/`**: Comparative analysis scripts
  - `baseline_comparison.py`: Compare against baseline methods
  - `human_evaluation.py`: Facilitate human evaluation processes
  - `ablation_studies.py`: Conduct ablation studies
- **`benchmarking/`**: Standardized benchmarking
  - `standard_benchmarks.py`: Run standard LAMMPS benchmarks
  - `custom_benchmarks.py`: Project-specific benchmark tests
  - `performance_profiling.py`: Profile model and system performance

### ⚙️ Preprocessing
Data preprocessing and preparation scripts:

- **`data_cleaning/`**: Data cleaning and normalization
  - `remove_duplicates.py`: Remove duplicate LAMMPS scripts
  - `normalize_format.py`: Standardize script formatting
  - `filter_invalid.py`: Filter out invalid or problematic scripts
- **`format_conversion/`**: Format conversion utilities
  - `json_to_lammps.py`: Convert JSON training data to LAMMPS format
  - `lammps_to_json.py`: Convert LAMMPS scripts to training format
  - `text_to_structured.py`: Convert text descriptions to structured data
- **`augmentation/`**: Data augmentation techniques
  - `parameter_variation.py`: Create variants with different parameters
  - `comment_augmentation.py`: Add/remove comments for robustness
  - `style_variation.py`: Generate different formatting styles

### 🛠️ Utilities
General utility scripts and tools:

- **`file_management/`**: File and directory management
  - `organize_files.py`: Organize files by type and date
  - `backup_data.py`: Create backups of important data
  - `cleanup_temp.py`: Clean up temporary files
- **`logging/`**: Logging and monitoring utilities
  - `setup_logging.py`: Configure project logging
  - `monitor_experiments.py`: Monitor long-running experiments
  - `error_reporting.py`: Automated error reporting
- **`monitoring/`**: System and process monitoring
  - `resource_monitor.py`: Monitor system resource usage
  - `progress_tracker.py`: Track experiment progress
  - `alert_system.py`: Send alerts for important events

## 🚀 Quick Start Scripts

### Essential Scripts for Getting Started

1. **Data Setup**:
   ```bash
   python scripts/preprocessing/data_cleaning/validate_lammps_syntax.py
   python scripts/data_creation/preprocessing/extract_documentation.py
   ```

2. **Training**:
   ```bash
   python scripts/training/fine_tuning/finetune_llm.py --config configs/model_configs/base_config.yaml
   ```

3. **Inference**:
   ```bash
   python scripts/inference/model_inference/generate_lammps.py --prompt "Create MD simulation for water"
   ```

4. **Evaluation**:
   ```bash
   python scripts/evaluation/performance_metrics/syntax_accuracy.py --input results/generated_inputs/
   ```

## 📋 Script Dependencies

### Environment Setup
```bash
# Install required packages
pip install -r requirements.txt

# Set up environment variables
export LAMMPS_PATH=/path/to/lammps
export OPENAI_API_KEY=your_api_key
```

### Common Parameters
Most scripts accept these common parameters:
- `--config`: Path to configuration file
- `--input`: Input data directory
- `--output`: Output directory
- `--verbose`: Enable verbose logging
- `--dry-run`: Test run without making changes

## 🔧 Configuration

### Script Configuration
Each script category has corresponding configuration files in `configs/`:
- `model_configs/`: Model training and inference configurations
- `simulation_configs/`: LAMMPS simulation parameters
- `experiment_configs/`: Experiment-specific settings

### Logging Configuration
All scripts use centralized logging configuration:
- Log files stored in `logs/`
- Configurable log levels (DEBUG, INFO, WARNING, ERROR)
- Automatic log rotation and archival

## 📊 Workflow Examples

### Complete Training Pipeline
```bash
# 1. Preprocess data
python scripts/preprocessing/data_cleaning/normalize_format.py
python scripts/data_creation/validation/validate_lammps_syntax.py

# 2. Create training datasets
python scripts/data_creation/dataset_generation/create_training_data.py

# 3. Train model
python scripts/training/fine_tuning/finetune_llm.py

# 4. Evaluate model
python scripts/evaluation/performance_metrics/syntax_accuracy.py
```

### RAG Pipeline
```bash
# 1. Process documents
python scripts/preprocessing/format_conversion/text_to_structured.py

# 2. Create embeddings
python scripts/training/fine_tuning/train_embeddings.py

# 3. Test RAG system
python scripts/inference/model_inference/generate_lammps.py --use-rag
```

## 🔍 Troubleshooting

### Common Issues
1. **LAMMPS Path Issues**: Ensure LAMMPS_PATH environment variable is set
2. **Memory Issues**: Use batch processing scripts for large datasets
3. **API Limits**: Implement rate limiting in inference scripts

### Debug Mode
Most scripts support `--debug` flag for detailed troubleshooting:
```bash
python script_name.py --debug --verbose
```

## 🚀 Adding New Scripts

### Script Template
```python
#!/usr/bin/env python3
"""
Script description here.
"""

import argparse
import logging
from pathlib import Path

def main():
    parser = argparse.ArgumentParser(description="Script description")
    parser.add_argument("--input", required=True, help="Input directory")
    parser.add_argument("--output", required=True, help="Output directory")
    parser.add_argument("--verbose", action="store_true", help="Verbose output")
    
    args = parser.parse_args()
    
    # Set up logging
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    
    # Script logic here
    
if __name__ == "__main__":
    main()
```

### Integration Guidelines
1. Follow the established directory structure
2. Include comprehensive argument parsing
3. Use consistent logging patterns
4. Add appropriate error handling
5. Include docstrings and comments
6. Add to this README when creating new scripts 