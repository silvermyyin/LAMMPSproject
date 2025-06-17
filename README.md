# LAMMPS Input File Generation Project

This project implements a comprehensive AI-powered system for generating LAMMPS input files using multiple approaches including RAG, fine-tuning, and rule-based generation. The project has been organized following proven research project principles for scalability, maintainability, and reproducibility.

## 🎯 Project Overview

Generate high-quality LAMMPS input files using Large Language Models with:
- **Multiple Generation Approaches**: RAG-enhanced, fine-tuned models, and rule-based systems
- **Comprehensive Validation**: Syntax checking, execution testing, and physics validation
- **Advanced RAG System**: FAISS-based retrieval with LAMMPS documentation
- **Professional Organization**: Research-grade project structure and workflows
- **Experiment Management**: Complete tracking and reproducibility support

## 🏗️ Complete Project Structure

```
FinalProject/
├── 📁 data/                          # Comprehensive data management
│   ├── documents/                    # RAG documentation and embeddings
│   │   ├── raw/                     # Original LAMMPS manuals and docs
│   │   ├── embedded/                # FAISS embeddings for RAG (planned)
│   │   └── extracted/               # Processed knowledge (planned)
│   ├── training/                     # ML training datasets
│   │   ├── fine_tuning/             # Fine-tuning datasets
│   │   ├── validation/              # Validation data
│   │   └── test/                    # Test datasets
│   ├── generated/                    # AI-generated outputs
│   ├── real_world/                   # Research scripts and benchmarks
│   ├── prompts/                      # Prompt templates and management
│   └── scriptsfromweb/              # Legacy web-sourced scripts
│
├── 📁 src/                           # Modular source code organization
│   ├── lammps_generators/            # Enhanced LAMMPS generators
│   ├── model/                        # ML model implementations
│   ├── rag/                          # RAG system components
│   ├── prompt_creation/              # Prompt engineering
│   ├── calculations/                 # LAMMPS calculations and validation
│   ├── utilities/                    # Common utilities
│   ├── generators/                   # Legacy generators (to be migrated)
│   └── fine_tuning/                  # Legacy fine-tuning (to be migrated)
│
├── 📁 scripts/                       # Purpose-driven executable scripts
│   ├── data_creation/                # Dataset generation and processing
│   ├── inference/                    # Model inference and generation
│   ├── training/                     # Model training and fine-tuning
│   ├── evaluation/                   # Performance evaluation
│   ├── preprocessing/                # Data preprocessing
│   ├── utilities/                    # Script utilities
│   └── reorganize_enhanced.sh        # Project reorganization script
│
├── 📁 results/                       # Comprehensive results tracking
│   ├── generated_inputs/             # Generated LAMMPS scripts by status
│   ├── simulation_outputs/           # LAMMPS execution results
│   ├── model_performance/            # ML model evaluation metrics
│   ├── benchmarks/                   # Benchmark comparisons
│   ├── experiments/                  # Experiment tracking
│   ├── baseline/                     # Legacy baseline results
│   ├── plots/                        # Legacy visualization outputs
│   ├── baseline_results.png          # Historical baseline visualization
│   └── results_baseline.csv          # Historical baseline data
│
├── 📁 configs/                       # Configuration management
│   ├── model_configs/                # Model and training configurations
│   ├── simulation_configs/           # LAMMPS simulation parameters
│   ├── experiment_configs/           # Experiment design configurations
│   └── model_configs.json            # Legacy model configuration
│
├── 📁 docs/                          # Documentation and guides
│   ├── project_overview_enhanced.md  # Complete project overview
│   ├── project_history_and_structure.md # Project evolution history
│   ├── api_documentation/            # API documentation
│   ├── user_guides/                  # User guides and tutorials
│   ├── tutorials/                    # Step-by-step tutorials
│   └── README.md                     # Documentation index
│
├── 📁 tests/                         # Testing framework
│   ├── unit/                         # Unit tests
│   ├── integration/                  # Integration tests (planned)
│   ├── system/                       # System tests (planned)
│   └── test_data/                    # Test datasets (planned)
│
├── 📁 experiments/                   # Experiment management
│   ├── ongoing/                      # Active experiments
│   ├── completed/                    # Finished experiments
│   ├── archived/                     # Historical experiments
│   └── .ipynb_checkpoints/          # Jupyter notebook checkpoints
│
├── 📁 .venv/                         # Python virtual environment
├── 📁 data_backup_20250612_003400/   # Backup of original data structure
├── 📁 LAMMPSrun/                     # LAMMPS execution directory
├── 📁 InputFileSynthesis-master/     # Reference project for organization
│
├── 📄 README.md                      # This file - main project documentation
├── 📄 requirements.txt               # Python dependencies
├── 📄 setup.py                       # Package setup configuration
├── 📄 log.lammps                     # LAMMPS execution logs
├── 📄 RAG_diagram.svg.png           # RAG system architecture diagram
└── 📄 .DS_Store                     # macOS system file
```

## 🚀 Key Features

### **Multiple Generation Approaches**
- **RAG-Enhanced Generation**: Context-aware generation using LAMMPS documentation
- **Fine-Tuned Models**: Custom-trained models on domain-specific data
- **Rule-Based Systems**: Template and pattern-based generation
- **Hybrid Approaches**: Combinations of the above methods

### **Advanced RAG System**
- FAISS-based vector storage for efficient retrieval
- Hierarchical document processing
- Context-aware prompt enhancement
- Relevance ranking and filtering

### **Comprehensive Validation**
- **Syntax Validation**: LAMMPS command syntax checking
- **Execution Testing**: Actual LAMMPS run validation
- **Physics Validation**: Physical reasonableness checks
- **Performance Benchmarking**: Speed and quality metrics

### **Professional Organization**
- Research-grade project structure inspired by InputFileSynthesis-master
- Modular and extensible architecture
- Comprehensive configuration management
- Complete experiment tracking and reproducibility

## 🔧 Quick Start

### 1. Environment Setup
```bash
# Clone and setup environment
cd FinalProject
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Configuration
```bash
# Set up environment variables
export OPENAI_API_KEY="your_api_key_here"
export LAMMPS_PATH="/path/to/lammps"

# Validate configurations
python scripts/utilities/validate_configs.py
```

### 3. Data Preparation
```bash
# Process documents for RAG
python scripts/preprocessing/data_cleaning/process_documents.py

# Create training datasets
python scripts/data_creation/dataset_generation/create_training_data.py
```

### 4. Model Training (Optional)
```bash
# Train embeddings for RAG
python scripts/training/fine_tuning/train_embeddings.py

# Fine-tune model on LAMMPS data
python scripts/training/fine_tuning/finetune_llm.py \
  --config configs/model_configs/fine_tuning_config.yaml
```

### 5. Generate LAMMPS Scripts
```bash
# Generate with RAG enhancement
python scripts/inference/model_inference/generate_lammps.py \
  --prompt "Create NVT molecular dynamics simulation for water at 300K" \
  --use-rag

# Batch generation
python scripts/inference/batch_processing/process_dataset.py \
  --input data/prompts/user_prompts/ \
  --output results/generated_inputs/pending_validation/
```

### 6. Evaluation and Validation
```bash
# Validate generated scripts
python scripts/evaluation/performance_metrics/syntax_accuracy.py \
  --input results/generated_inputs/pending_validation/

# Run comprehensive evaluation
python scripts/evaluation/performance_metrics/evaluate_outputs.py \
  --config configs/experiment_configs/evaluation_config.yaml
```

## 📊 Data Pipeline

### Document Processing
```
Raw LAMMPS Docs → Text Extraction → Knowledge Structure → Vector Embeddings → RAG System
```

### Training Data Creation
```
Real LAMMPS Scripts → Analysis & Validation → Augmentation → Format Conversion → Training Datasets
```

### Generation Pipeline
```
User Prompt → RAG Enhancement → Model Inference → LAMMPS Generation → Multi-level Validation
```

## 🎓 Research Capabilities

### **Experiment Management**
- Complete experiment tracking and reproducibility
- Configuration-driven experiment design
- Performance comparison and analysis tools
- Results archival and retrieval systems

### **Advanced Analytics**
- Comprehensive performance metrics
- Ablation studies and comparative analysis
- Cross-experiment meta-analysis
- Automated report generation

### **Quality Assurance**
- Multi-level validation pipeline
- Automated testing framework
- Continuous integration support
- Error tracking and analysis

## 📚 Documentation

Comprehensive documentation is available:
- **`data/README.md`**: Data organization and management
- **`src/README.md`**: Source code architecture and APIs
- **`scripts/README.md`**: Script usage and workflows
- **`results/README.md`**: Results tracking and analysis
- **`configs/README.md`**: Configuration management
- **`tests/README.md`**: Testing framework and guidelines
- **`experiments/README.md`**: Experiment management
- **`docs/README.md`**: Documentation index and guides
- **`docs/project_overview_enhanced.md`**: Complete project overview

## 🔄 Project Evolution

### **Current Status**
The project has been enhanced following the organizational principles of InputFileSynthesis-master:

- ✅ **Enhanced Structure Implemented**: Modular organization with clear separation of concerns
- ✅ **Legacy Support Maintained**: Original components preserved during transition
- ✅ **Documentation Complete**: Comprehensive README files for all directories
- ✅ **Backup Created**: Original structure preserved in `data_backup_20250612_003400/`

### **Migration Status**
- **Data Organization**: ✅ Completed with enhanced structure
- **Source Code**: 🔧 In progress - enhanced modules created, legacy being integrated
- **Scripts**: ✅ Organized by functionality with comprehensive workflows
- **Results**: 🔧 Enhanced structure created, legacy results preserved
- **Configuration**: ✅ Hierarchical management implemented

## 🔬 Advanced Usage

### Custom Generator Development
```python
from src.lammps_generators.base_generator import BaseGenerator

class CustomGenerator(BaseGenerator):
    def generate(self, prompt: str) -> str:
        # Implement custom generation logic
        pass
```

### RAG System Customization
```python
from src.rag.context_creation import ContextFormatter

formatter = ContextFormatter(
    template="Custom template: {context}\nUser query: {query}",
    max_context_length=8000
)
```

### Experiment Configuration
```yaml
# configs/experiment_configs/custom_experiment.yaml
experiment:
  name: "custom_comparison"
  models:
    - name: "baseline"
      config: "model_configs/base_config.yaml"
    - name: "enhanced"
      config: "model_configs/rag_config.yaml"
  evaluation:
    metrics: ["syntax_accuracy", "execution_success", "bleu_score"]
```

## 🤝 Contributing

1. **Fork the repository**
2. **Create a feature branch**: `git checkout -b feature/amazing-feature`
3. **Follow the project structure**: Add code to appropriate directories
4. **Write tests**: Include comprehensive test coverage
5. **Update documentation**: Update relevant README files
6. **Commit changes**: `git commit -m 'Add amazing feature'`
7. **Push to branch**: `git push origin feature/amazing-feature`
8. **Create Pull Request**

### Development Guidelines
- Follow the established directory structure
- Include comprehensive documentation
- Write unit tests for new functionality
- Use type hints and docstrings
- Update configuration files as needed

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- Inspired by the InputFileSynthesis-master project organization principles
- Built on proven research project management methodologies
- Leverages state-of-the-art LLM and RAG technologies
- Designed for the computational physics and materials science community

---

**Note**: This project represents a significant enhancement over traditional LAMMPS input generation approaches, providing a comprehensive, scalable, and research-grade platform for AI-powered molecular dynamics simulation setup. The enhanced organization ensures maintainability, reproducibility, and collaboration-friendly development. 