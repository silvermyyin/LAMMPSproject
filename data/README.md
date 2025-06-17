# Data Directory Organization

This directory contains all datasets, documents, and data artifacts used in the LAMMPS input generation project. The organization follows proven research project principles for clear data management and reproducibility.

## 📁 Current Directory Structure

### 🗂️ Documents (✅ Exists)
Contains all documentation and reference materials used for RAG (Retrieval-Augmented Generation):

- **`raw/`** (✅ Exists): Original LAMMPS documentation, manuals, and reference materials
  - LAMMPS manual PDFs
  - Tutorial documents
  - Command reference materials
- **`embedded/`** (🔧 Planned): FAISS embeddings and processed documents for RAG
  - Vector embeddings of documentation
  - Indexed knowledge base
- **`extracted/`** (🔧 Planned): Processed and extracted knowledge from documentation
  - Structured command descriptions
  - Parameter explanations
  - Usage examples

### 🎯 Training (✅ Exists)
Training and evaluation datasets for machine learning models:

- **`fine_tuning/`** (✅ Exists): Datasets formatted for model fine-tuning
  - Input-output pairs for LAMMPS generation
  - Structured as JSON with system/user/assistant prompts
- **`validation/`** (✅ Exists): Validation datasets for model evaluation
- **`test/`** (✅ Exists): Test datasets for final performance assessment

### 🤖 Generated (✅ Exists)
AI-generated LAMMPS inputs from different approaches:

- **`rule_based/`** (✅ Exists): Outputs from rule-based generation systems
- **`manual/`** (✅ Exists): Manually created or validated LAMMPS scripts
- **`llm_generated/`** (✅ Exists): LLM-generated LAMMPS input files
- **`evaluation/`** (✅ Exists): Generated files ready for LAMMPS validation

### 🏭 Real World (✅ Exists)
Real-world LAMMPS scripts and validated inputs:

- **`research_scripts/`** (✅ Exists): LAMMPS scripts from actual research projects
- **`validated_inputs/`** (✅ Exists): Proven working LAMMPS configurations
- **`benchmarks/`** (✅ Exists): Standard benchmark simulations

### 💬 Prompts (✅ Exists)
Prompt engineering and template management:

- **`system_prompts/`** (🔧 Planned): System prompts for different model approaches
- **`user_prompts/`** (✅ Exists): User prompt templates and examples
- **`templates/`** (🔧 Planned): Reusable prompt templates

### 🌐 Scripts from Web (✅ Exists - Legacy)
Scripts collected from various web sources:
- **`scriptsfromweb/`**: Collection of LAMMPS scripts from online sources
- Note: This is legacy data that will be migrated to `real_world/benchmarks/`

## 🔧 Planned Directory Structure (Future Enhancement)

The following directories are planned for future implementation:

### 📚 Extracted Documentation (🔧 Planned)
Systematically organized LAMMPS knowledge extracted from manuals:

- **`force_fields/`**: Force field descriptions and parameters
- **`commands/`**: LAMMPS command syntax and usage
- **`parameters/`**: Parameter descriptions and valid ranges
- **`examples/`**: Example scripts and common patterns
- **`simulation_types/`**: Different simulation type explanations

### 🧬 Molecules (🔧 Planned)
Molecular structure data and configurations:

- **`structures/`**: Molecular coordinate files and structures
- **`coordinates/`**: XYZ coordinates and initial configurations
- **`topologies/`**: Molecular topology information

### 🔬 Simulation Types (🔧 Planned)
LAMMPS simulation categorization:

- **`md/`**: Molecular dynamics simulations
- **`minimization/`**: Energy minimization scripts
- **`analysis/`**: Post-processing and analysis scripts
- **`nvt/`**, **`npt/`**, **`nve/`**: Ensemble-specific configurations

### ⚛️ Force Fields (🔧 Planned)
Force field specific data and parameters:

- **`lj/`**: Lennard-Jones force field data
- **`eam/`**: Embedded Atom Method parameters
- **`tersoff/`**: Tersoff potential data
- **`reax/`**: ReaxFF force field information
- **`custom/`**: Custom force field implementations

## 📊 Data Formats

### Training Data Format
```json
{
  "system": "System prompt for LAMMPS generation",
  "user": "User request for specific simulation",
  "assistant": "Generated LAMMPS input file content"
}
```

### Molecular Structure Format
- **XYZ files**: Standard molecular coordinate format
- **LAMMPS data files**: Native LAMMPS structure format
- **PDB files**: Protein Data Bank format for complex molecules

### Documentation Format
- **PDF**: Original documentation files
- **JSON**: Structured extracted knowledge
- **TXT**: Plain text processed documents

## 🔄 Data Pipeline

1. **Raw Documents** → **Document Processing** → **Embedded Knowledge**
2. **Real-world Scripts** → **Data Extraction** → **Training Datasets**
3. **Generated Outputs** → **Validation** → **Benchmarking**

## 📝 Usage Guidelines

### Adding New Data
1. Place raw documents in `documents/raw/` directory
2. Process documents through embedding pipeline (when implemented)
3. Update training datasets with new examples in `training/fine_tuning/`
4. Validate generated outputs before inclusion

### Data Quality
- All training data should be validated LAMMPS scripts
- Generated outputs require validation before research use
- Real-world scripts should include metadata about source and validation

### File Naming Conventions
- Training files: `train_XXXX.in` or `train_XXXX.json`
- Test files: `test_XXXX.in` or `test_XXXX.json`
- Generated files: `generated_YYYY-MM-DD_XXXX.in`
- Documentation: Descriptive names with source information

## 🚀 Getting Started

1. **For RAG**: Place documents in `documents/raw/` (embeddings system coming soon)
2. **For Training**: Use datasets in `training/fine_tuning/`
3. **For Validation**: Use scripts in `real_world/research_scripts/`
4. **For Examples**: Check existing scripts in various subdirectories

## 🔧 Maintenance

- Regularly update embeddings when new documentation is added
- Validate generated outputs before adding to training data
- Maintain clear metadata for all data sources
- Archive old datasets while preserving reproducibility

## 🚀 Migration Status

The data directory is currently in transition. The enhanced structure from the reorganization script has been partially implemented:

✅ **Completed**:
- Basic directory structure created
- Existing data migrated to appropriate locations
- Training/validation/test splits organized

🔧 **In Progress**:
- Document embedding system setup
- Knowledge extraction from LAMMPS manuals
- Advanced prompt template system

📋 **Planned**:
- Complete molecular structure support
- Advanced force field categorization
- Automated data quality validation 