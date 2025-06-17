# Source Code Organization

This directory contains the core implementation of the LAMMPS input generation system. The organization follows modular design principles inspired by successful research projects for maintainability and extensibility.

## 📁 Current Directory Structure

### 🚀 LAMMPS Generators (✅ Enhanced Structure)
Specialized generators for different types of LAMMPS simulations:

- **`lammps_generators/`** (✅ Exists): Enhanced modular generator system
  - `md_generators/` (🔧 Planned): Molecular dynamics simulation generators
    - `nvt_generator.py`: NVT ensemble MD simulations
    - `npt_generator.py`: NPT ensemble MD simulations
    - `nve_generator.py`: NVE ensemble MD simulations
    - `custom_md_generator.py`: Custom MD simulation configurations
  - `minimization/` (🔧 Planned): Energy minimization generators
    - `steepest_descent.py`: Steepest descent minimization
    - `conjugate_gradient.py`: Conjugate gradient minimization
    - `fire_minimization.py`: FIRE minimization algorithm
  - `analysis/` (🔧 Planned): Post-processing and analysis generators
    - `rdf_analysis.py`: Radial distribution function analysis
    - `msd_analysis.py`: Mean square displacement analysis
    - `trajectory_analysis.py`: Trajectory analysis tools
  - `force_field_setup/` (🔧 Planned): Force field configuration generators
    - `lj_setup.py`: Lennard-Jones force field setup
    - `eam_setup.py`: Embedded Atom Method setup
    - `tersoff_setup.py`: Tersoff potential setup
    - `reax_setup.py`: ReaxFF force field setup

- **`generators/`** (🔧 Legacy - To Be Migrated): Original generator implementations
  - Contains baseline generation scripts
  - Will be integrated into the enhanced `lammps_generators/` structure

### 🤖 Model (✅ Enhanced Structure)
Core machine learning model implementations:

- **`model/`** (✅ Exists): Enhanced model infrastructure
  - `embeddings/` (🔧 Planned): Embedding models and utilities
    - `lammps_embeddings.py`: LAMMPS-specific embeddings
    - `command_embeddings.py`: Command-specific embeddings
    - `parameter_embeddings.py`: Parameter embeddings
  - `fine_tuning/` (🔧 Planned): Model fine-tuning implementations
    - `llm_fine_tuner.py`: Language model fine-tuning
    - `curriculum_trainer.py`: Curriculum learning implementation
    - `adapter_tuning.py`: Parameter-efficient fine-tuning
  - `inference/` (🔧 Planned): Model inference engines
    - `generator_engine.py`: Core generation engine
    - `batch_inference.py`: Batch processing inference
    - `streaming_inference.py`: Streaming inference for real-time use
  - `evaluation/` (🔧 Planned): Model evaluation frameworks
    - `syntax_evaluator.py`: LAMMPS syntax validation
    - `semantic_evaluator.py`: Semantic similarity evaluation
    - `execution_evaluator.py`: Actual LAMMPS execution testing

- **`fine_tuning/`** (🔧 Legacy - To Be Migrated): Original fine-tuning scripts
  - Will be integrated into `model/fine_tuning/`

### 🧠 RAG (Retrieval-Augmented Generation) (✅ Enhanced Structure)
RAG system implementation for knowledge-enhanced generation:

- **`rag/`** (✅ Exists): RAG system components
  - `document_processing/` (🔧 Planned): Document processing pipeline
    - `pdf_processor.py`: PDF document processing
    - `text_extractor.py`: Text extraction utilities
    - `knowledge_extractor.py`: Knowledge structure extraction
  - `retrieval/` (🔧 Planned): Information retrieval components
    - `vector_store.py`: Vector database management
    - `similarity_search.py`: Semantic similarity search
    - `context_retrieval.py`: Context retrieval for prompts
  - `context_creation/` (🔧 Planned): Context creation for prompts
    - `context_formatter.py`: Format retrieved context
    - `relevance_ranker.py`: Rank context by relevance
    - `context_compressor.py`: Compress context for efficiency

### 💬 Prompt Creation (✅ Enhanced Structure)
Prompt engineering and template management:

- **`prompt_creation/`** (✅ Exists): Enhanced prompt engineering system
  - `system_prompts/` (🔧 Planned): System prompt templates
    - `base_system_prompt.py`: Base system prompt templates
    - `task_specific_prompts.py`: Task-specific system prompts
    - `rag_system_prompts.py`: RAG-enhanced system prompts
  - `user_prompts/` (🔧 Planned): User prompt processing
    - `prompt_parser.py`: Parse and understand user prompts
    - `intent_classifier.py`: Classify user intent
    - `parameter_extractor.py`: Extract parameters from prompts
  - `templates/` (🔧 Planned): Reusable prompt templates
    - `md_templates.py`: Molecular dynamics prompt templates
    - `minimization_templates.py`: Minimization prompt templates
    - `analysis_templates.py`: Analysis prompt templates

### 🔬 Calculations (✅ Enhanced Structure)
LAMMPS calculation and validation utilities:

- **`calculations/`** (✅ Exists): Enhanced calculation framework
  - `energy/` (🔧 Planned): Energy calculation utilities
    - `potential_energy.py`: Potential energy calculations
    - `kinetic_energy.py`: Kinetic energy calculations
    - `total_energy.py`: Total energy validation
  - `trajectory/` (🔧 Planned): Trajectory processing
    - `trajectory_reader.py`: Read LAMMPS trajectory files
    - `trajectory_analyzer.py`: Analyze trajectory data
    - `trajectory_validator.py`: Validate trajectory outputs
  - `properties/` (🔧 Planned): Physical property calculations
    - `thermodynamic_properties.py`: Temperature, pressure, etc.
    - `structural_properties.py`: RDF, coordination numbers, etc.
    - `transport_properties.py`: Diffusion coefficients, viscosity, etc.
  - `validation/` (🔧 Planned): Validation frameworks
    - `syntax_validator.py`: LAMMPS syntax validation
    - `physics_validator.py`: Physical consistency validation
    - `convergence_checker.py`: Check simulation convergence

### 🛠️ Utilities (✅ Enhanced Structure)
General utilities and helper functions:

- **`utilities/`** (✅ Exists): Enhanced utility framework
  - `file_io/` (🔧 Planned): File input/output utilities
    - `lammps_reader.py`: Read LAMMPS input files
    - `lammps_writer.py`: Write LAMMPS input files
    - `data_file_handler.py`: Handle LAMMPS data files
  - `logging/` (🔧 Planned): Logging configuration and utilities
    - `logger_config.py`: Configure project logging
    - `experiment_logger.py`: Log experiment details
    - `performance_logger.py`: Log performance metrics
  - `config/` (🔧 Planned): Configuration management
    - `config_loader.py`: Load configuration files
    - `parameter_manager.py`: Manage model parameters
    - `environment_setup.py`: Set up environment variables
  - `validation/` (🔧 Planned): General validation utilities
    - `input_validator.py`: Validate user inputs
    - `output_validator.py`: Validate generated outputs
    - `format_validator.py`: Validate file formats

## 🔄 Migration Status

The source code directory is currently transitioning from the original structure to the enhanced modular organization:

### ✅ **Completed**:
- Enhanced directory structure created
- Core modules (`lammps_generators`, `model`, `rag`, `prompt_creation`, `calculations`, `utilities`) established
- Legacy directories preserved for transition

### 🔧 **In Progress**:
- Migration of code from legacy directories (`generators/`, `fine_tuning/`) to enhanced structure
- Implementation of planned subdirectories and modules
- Integration testing of new modular components

### 📋 **Planned**:
- Complete subdirectory implementation with actual Python modules
- Legacy directory removal after successful migration
- Full API documentation for all components

## 🏗️ Architecture Overview

### Design Principles
1. **Modular Design**: Each component has a single responsibility
2. **Extensibility**: Easy to add new generators or models
3. **Testability**: Clear interfaces for unit testing
4. **Configurability**: Extensive configuration options
5. **Scalability**: Designed for batch processing and scaling

### Data Flow
```
User Input → Prompt Processing → RAG Enhancement → Model Inference → LAMMPS Generation → Validation → Output
```

### Component Interactions
- **Generators** use **Model** components for inference
- **RAG** enhances prompts with relevant documentation
- **Calculations** validate generated outputs
- **Utilities** provide common functionality across components

## 🚀 Getting Started

### Basic Usage Example (When Implemented)
```python
from src.lammps_generators.md_generators import NVTGenerator
from src.model.inference import GeneratorEngine
from src.rag.context_creation import ContextFormatter

# Initialize components
generator = NVTGenerator()
engine = GeneratorEngine()
rag = ContextFormatter()

# Generate LAMMPS script
user_prompt = "Create NVT simulation for water at 300K"
enhanced_prompt = rag.enhance_prompt(user_prompt)
lammps_script = generator.generate(enhanced_prompt)
```

### Current Usage (Legacy Support)
```python
# Using existing generators during transition
from src.generators import ScriptGenerator
from src.model import RAGModel

# Current functionality
generator = ScriptGenerator()
model = RAGModel()
```

### Configuration
```python
from src.utilities.config import ConfigLoader

config = ConfigLoader('configs/model_configs/base_config.yaml')
generator = NVTGenerator(config=config)
```

## 🧪 Testing

### Unit Tests (Planned)
Each module will include comprehensive unit tests:
```bash
python -m pytest src/lammps_generators/tests/
python -m pytest src/model/tests/
python -m pytest src/rag/tests/
```

### Integration Tests (Planned)
```bash
python -m pytest src/tests/integration/
```

## 📊 Performance Considerations

### Optimization
- **Caching**: Embeddings and processed documents are cached
- **Batch Processing**: Support for batch inference
- **Memory Management**: Efficient memory usage for large documents
- **Parallel Processing**: Multi-threading for CPU-intensive tasks

### Monitoring
- **Performance Metrics**: Track generation speed and quality
- **Resource Usage**: Monitor memory and CPU usage
- **Error Tracking**: Comprehensive error logging and tracking

## 🔧 Development Guidelines

### Adding New Components

1. **New Generator**:
   ```python
   from src.lammps_generators.base_generator import BaseGenerator
   
   class NewGenerator(BaseGenerator):
       def generate(self, prompt: str) -> str:
           # Implementation here
           pass
   ```

2. **New Model Component**:
   - Follow the established interface patterns
   - Include comprehensive error handling
   - Add appropriate logging
   - Write unit tests

3. **New RAG Component**:
   - Implement document processing interface
   - Include embedding generation
   - Add context formatting capabilities

### Code Standards
- **Type Hints**: Use type hints for all function signatures
- **Docstrings**: Comprehensive docstrings for all public methods
- **Error Handling**: Graceful error handling with informative messages
- **Logging**: Appropriate logging at different levels
- **Testing**: Unit tests for all new functionality

### Documentation
- Update relevant README files
- Add inline documentation for complex logic
- Include usage examples
- Document configuration options

## 🔄 Integration with Other Components

### With Scripts
- Scripts in `scripts/` use source code components
- Clear separation between executable scripts and library code
- Consistent interfaces across all components

### With Data
- Source code processes data from `data/` directory
- Generated outputs go to `results/` directory
- Configuration files in `configs/` directory

### With Experiments
- Experiment tracking through logging utilities
- Performance monitoring and metrics collection
- Integration with experiment management systems

## 📈 Future Enhancements

### Planned Features
1. **Multi-modal Input**: Support for molecular structure inputs
2. **Advanced RAG**: Hierarchical document retrieval
3. **Model Optimization**: Quantization and pruning support
4. **Real-time Generation**: Streaming generation capabilities
5. **Collaborative Features**: Multi-user experiment tracking

### Extension Points
- Plugin architecture for new force fields
- Custom embedding models
- External validation services
- Cloud deployment capabilities

## 🚨 Important Notes

1. **Legacy Support**: The `generators/` and `fine_tuning/` directories contain the original implementation and remain functional during the transition period.

2. **Gradual Migration**: New development should use the enhanced structure (`lammps_generators/`, enhanced `model/`, etc.) while maintaining backward compatibility.

3. **Testing**: Both legacy and enhanced components should be thoroughly tested before the final migration is complete. 