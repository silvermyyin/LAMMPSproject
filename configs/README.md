# Configuration Management

This directory contains all configuration files for the LAMMPS input generation system. The organization follows a hierarchical structure that separates different types of configurations for maintainability and ease of use.

## 📁 Directory Structure Overview

### 🤖 Model Configurations
Machine learning model and training configurations:

- **`model_configs/`**: Core model configuration files
  - `base_config.yaml`: Base configuration for all models
  - `gpt_config.yaml`: GPT model specific settings
  - `fine_tuning_config.yaml`: Fine-tuning hyperparameters
  - `rag_config.yaml`: RAG system configuration
  - `embedding_config.yaml`: Embedding model settings

### 🔬 Simulation Configurations
LAMMPS simulation parameter configurations:

- **`simulation_configs/`**: LAMMPS simulation parameters
  - `md_config.yaml`: Molecular dynamics simulation settings
  - `minimization_config.yaml`: Energy minimization parameters
  - `force_field_config.yaml`: Force field specifications
  - `ensemble_config.yaml`: Thermodynamic ensemble settings
  - `analysis_config.yaml`: Post-processing analysis parameters

### 🧪 Experiment Configurations
Experiment design and execution configurations:

- **`experiment_configs/`**: Experiment-specific settings
  - `baseline_experiment.yaml`: Baseline comparison experiments
  - `rag_experiment.yaml`: RAG system experiments
  - `ablation_study.yaml`: Ablation study configurations
  - `benchmark_config.yaml`: Benchmarking experiment settings
  - `evaluation_config.yaml`: Model evaluation parameters

## 📋 Configuration File Formats

### Model Configuration Example
```yaml
# model_configs/base_config.yaml
model:
  name: "gpt-4"
  api_key_env: "OPENAI_API_KEY"
  temperature: 0.7
  max_tokens: 2048
  top_p: 0.9
  frequency_penalty: 0.0
  presence_penalty: 0.0

generation:
  batch_size: 10
  timeout: 300
  retry_attempts: 3
  validation:
    syntax_check: true
    physics_check: true
    execution_test: false

logging:
  level: "INFO"
  format: "%(asctime)s - %(levelname)s - %(message)s"
  file: "logs/model_generation.log"
```

### RAG Configuration Example
```yaml
# model_configs/rag_config.yaml
rag:
  document_store:
    type: "faiss"
    index_path: "data/documents/embedded/faiss_index"
    embedding_dim: 1536
  
  retrieval:
    top_k: 5
    similarity_threshold: 0.7
    max_context_length: 4000
    rerank: true
  
  embedding:
    model: "text-embedding-ada-002"
    batch_size: 100
    chunk_size: 512
    chunk_overlap: 50

  context_formatting:
    template: "Based on the following LAMMPS documentation:\n{context}\n\nGenerate a LAMMPS script for: {query}"
    max_retrieved_chars: 8000
```

### Simulation Configuration Example
```yaml
# simulation_configs/md_config.yaml
molecular_dynamics:
  ensembles:
    nvt:
      thermostat: "nose"
      temperature: 300.0
      damping: 0.1
    npt:
      thermostat: "nose"
      barostat: "parrinello_rahman"
      temperature: 300.0
      pressure: 1.0
      damping_temp: 0.1
      damping_press: 1.0
  
  timestep: 0.001  # ps
  run_length: 1000000  # steps
  
  output:
    trajectory_freq: 1000
    energy_freq: 100
    restart_freq: 10000
    
  force_fields:
    default: "lj"
    available: ["lj", "eam", "tersoff", "reax"]
```

### Experiment Configuration Example
```yaml
# experiment_configs/rag_experiment.yaml
experiment:
  name: "rag_vs_baseline_comparison"
  description: "Compare RAG-enhanced generation with baseline methods"
  
  dataset:
    train: "data/training/fine_tuning/"
    validation: "data/training/validation/"
    test: "data/training/test/"
  
  models:
    - name: "baseline_gpt4"
      config: "model_configs/base_config.yaml"
      use_rag: false
    - name: "rag_gpt4"
      config: "model_configs/rag_config.yaml"
      use_rag: true
  
  evaluation:
    metrics: ["syntax_accuracy", "execution_success", "bleu_score"]
    sample_size: 100
    validation_timeout: 600
  
  output:
    results_dir: "results/experiments/rag_comparison/"
    save_generated_scripts: true
    save_metrics: true
```

## 🔧 Configuration Management

### Environment Variables
Configuration files can reference environment variables:
```yaml
model:
  api_key: ${OPENAI_API_KEY}
  base_url: ${OPENAI_BASE_URL:-https://api.openai.com/v1}
```

### Configuration Inheritance
Configurations support inheritance for code reuse:
```yaml
# Inherit from base configuration
base: "model_configs/base_config.yaml"

# Override specific settings
model:
  temperature: 0.5  # Override base temperature
  
# Add new settings
custom_setting:
  value: "specific_to_this_config"
```

### Validation Schema
Each configuration type has a validation schema:
```python
# Example validation for model config
from pydantic import BaseModel, Field

class ModelConfig(BaseModel):
    name: str = Field(..., description="Model name")
    temperature: float = Field(0.7, ge=0.0, le=2.0)
    max_tokens: int = Field(2048, gt=0)
    
class GenerationConfig(BaseModel):
    batch_size: int = Field(10, gt=0)
    timeout: int = Field(300, gt=0)
```

## 📊 Configuration Categories

### 1. Model Parameters
- **LLM Settings**: Temperature, tokens, penalties
- **Generation Parameters**: Batch size, timeout, retries
- **API Configuration**: Keys, endpoints, rate limits

### 2. RAG System
- **Document Processing**: Chunking, embedding parameters
- **Retrieval**: Top-k, thresholds, reranking
- **Context Formatting**: Templates, length limits

### 3. LAMMPS Simulation
- **Physical Parameters**: Temperature, pressure, timestep
- **Force Fields**: Type, parameters, mixing rules
- **Output Control**: Frequencies, file formats

### 4. Experiment Design
- **Dataset Configuration**: Train/test splits, sampling
- **Evaluation Metrics**: Accuracy measures, benchmarks
- **Resource Allocation**: Compute, memory, storage

## 🚀 Usage Examples

### Loading Configuration
```python
from src.utilities.config import ConfigLoader

# Load model configuration
config = ConfigLoader("configs/model_configs/rag_config.yaml")

# Access nested values
temperature = config.get("model.temperature", 0.7)
top_k = config.get("rag.retrieval.top_k", 5)
```

### Merging Configurations
```python
# Merge experiment config with model config
base_config = ConfigLoader("configs/model_configs/base_config.yaml")
exp_config = ConfigLoader("configs/experiment_configs/rag_experiment.yaml")

merged_config = base_config.merge(exp_config)
```

### Runtime Configuration Override
```python
# Override configuration at runtime
config = ConfigLoader("configs/model_configs/base_config.yaml")
config.set("model.temperature", 0.5)
config.set("generation.batch_size", 20)
```

## 🔍 Best Practices

### 1. Organization
- Group related configurations together
- Use descriptive file names
- Maintain consistent structure across files

### 2. Documentation
- Include comments explaining parameters
- Document default values and ranges
- Provide usage examples

### 3. Validation
- Use schema validation for all configs
- Implement range checks for numerical values
- Validate file paths and dependencies

### 4. Version Control
- Track configuration changes in git
- Use meaningful commit messages for config updates
- Tag configurations for important experiments

### 5. Security
- Never commit API keys or secrets
- Use environment variables for sensitive data
- Implement proper access controls

## 🔧 Configuration Tools

### Validation Script
```bash
# Validate all configurations
python scripts/utilities/validate_configs.py --config-dir configs/

# Validate specific configuration
python scripts/utilities/validate_configs.py --file configs/model_configs/rag_config.yaml
```

### Configuration Generator
```bash
# Generate new experiment configuration
python scripts/utilities/generate_config.py \
  --template experiment \
  --name new_experiment \
  --output configs/experiment_configs/new_experiment.yaml
```

### Configuration Comparison
```bash
# Compare two configurations
python scripts/utilities/compare_configs.py \
  --config1 configs/model_configs/base_config.yaml \
  --config2 configs/model_configs/rag_config.yaml
```

## 📈 Advanced Features

### 1. Dynamic Configuration
- Runtime parameter updates
- Adaptive parameter tuning
- A/B testing configurations

### 2. Configuration Templates
- Reusable configuration templates
- Parameter substitution
- Conditional configurations

### 3. Integration
- IDE support with schema validation
- Automated configuration testing
- Configuration-driven CI/CD

## 🔍 Troubleshooting

### Common Issues
1. **Missing Environment Variables**: Check environment setup
2. **Invalid YAML Syntax**: Use YAML validator
3. **Schema Validation Errors**: Check parameter types and ranges
4. **File Path Issues**: Verify relative path references

### Debug Tools
- Configuration validation scripts
- Schema checking utilities
- Environment variable verification
- Path resolution debugging 