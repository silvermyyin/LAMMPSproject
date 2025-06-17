#!/bin/bash

# Enhanced LAMMPS Project Reorganization Script
# Based on InputFileSynthesis-master organizational principles

echo "🚀 Starting enhanced LAMMPS project reorganization..."

# Create backup
echo "📦 Creating backup of current structure..."
cp -r data data_backup_$(date +%Y%m%d_%H%M%S)

# Phase 1: Enhanced Data Directory Structure
echo "📁 Creating enhanced data directory structure..."

# Create new data organization
mkdir -p data/documents/{raw,embedded,extracted}
mkdir -p data/extracted_documentation/{force_fields,commands,parameters,examples,simulation_types}
mkdir -p data/training/{fine_tuning,validation,test}
mkdir -p data/molecules/{structures,coordinates,topologies}
mkdir -p data/generated/{rule_based,manual,llm_generated,evaluation}
mkdir -p data/real_world/{research_scripts,validated_inputs,benchmarks}
mkdir -p data/prompts/{system_prompts,user_prompts,templates}
mkdir -p data/simulation_types/{md,minimization,analysis,nvt,npt,nve}
mkdir -p data/force_fields/{lj,eam,tersoff,reax,custom}

# Phase 2: Enhanced Scripts Organization
echo "🔧 Creating enhanced scripts structure..."
mkdir -p scripts/data_creation/{dataset_generation,preprocessing,validation}
mkdir -p scripts/inference/{model_inference,batch_processing,evaluation}
mkdir -p scripts/training/{fine_tuning,hyperparameter_search,model_validation}
mkdir -p scripts/evaluation/{performance_metrics,comparison,benchmarking}
mkdir -p scripts/preprocessing/{data_cleaning,format_conversion,augmentation}
mkdir -p scripts/utilities/{file_management,logging,monitoring}

# Phase 3: Enhanced Source Code Organization
echo "💻 Enhancing source code structure..."
mkdir -p src/lammps_generators/{md_generators,minimization,analysis,force_field_setup}
mkdir -p src/model/{embeddings,fine_tuning,inference,evaluation}
mkdir -p src/rag/{document_processing,retrieval,context_creation}
mkdir -p src/prompt_creation/{system_prompts,user_prompts,templates}
mkdir -p src/calculations/{energy,trajectory,properties,validation}
mkdir -p src/utilities/{file_io,logging,config,validation}

# Phase 4: Enhanced Results Organization
echo "📊 Creating enhanced results structure..."
mkdir -p results/generated_inputs/{validated,pending_validation,failed}
mkdir -p results/simulation_outputs/{trajectories,logs,energies,properties}
mkdir -p results/model_performance/{metrics,comparisons,visualizations}
mkdir -p results/benchmarks/{baseline_comparison,performance_tests}
mkdir -p results/experiments/{dated_runs,archived,analysis}

# Phase 5: Enhanced Configuration and Documentation
echo "📚 Setting up configuration and documentation..."
mkdir -p configs/{model_configs,simulation_configs,experiment_configs}
mkdir -p docs/{api_documentation,user_guides,tutorials,examples}
mkdir -p experiments/{ongoing,completed,archived}

echo "✅ Enhanced directory structure created!"

# Phase 6: Data Migration
echo "🔄 Migrating existing data..."

# Move RAG documents
if [ -d "data/RAGdocs" ]; then
    echo "  📄 Moving RAG documents..."
    mv data/RAGdocs/* data/documents/raw/ 2>/dev/null || true
    rmdir data/RAGdocs 2>/dev/null || true
fi

# Move training data
if [ -d "data/train" ]; then
    echo "  🎯 Moving training data..."
    mv data/train/* data/training/fine_tuning/ 2>/dev/null || true
    rmdir data/train 2>/dev/null || true
fi

# Move test data
if [ -d "data/test" ]; then
    echo "  🧪 Moving test data..."
    mv data/test/* data/training/test/ 2>/dev/null || true
    rmdir data/test 2>/dev/null || true
fi

# Move generated data
if [ -d "data/generated" ]; then
    echo "  🔄 Reorganizing generated data..."
    if [ -d "data/generated/rule_based" ]; then
        mv data/generated/rule_based/* data/generated/rule_based/ 2>/dev/null || true
    fi
    if [ -d "data/generated/manual" ]; then
        mv data/generated/manual/* data/generated/manual/ 2>/dev/null || true
    fi
    if [ -d "data/generated/bruteforce" ]; then
        mv data/generated/bruteforce/* data/generated/llm_generated/ 2>/dev/null || true
        rmdir data/generated/bruteforce 2>/dev/null || true
    fi
fi

# Move reference scripts
if [ -d "data/reference_scripts" ]; then
    echo "  📜 Moving reference scripts..."
    mv data/reference_scripts/* data/real_world/research_scripts/ 2>/dev/null || true
    rmdir data/reference_scripts 2>/dev/null || true
fi

if [ -d "data/reference_scripts_flat" ]; then
    mv data/reference_scripts_flat/* data/real_world/research_scripts/ 2>/dev/null || true
    rmdir data/reference_scripts_flat 2>/dev/null || true
fi

# Move scripts from web
if [ -d "data/scriptsfromweb" ]; then
    echo "  🌐 Moving web scripts..."
    mv data/scriptsfromweb/* data/real_world/benchmarks/ 2>/dev/null || true
    rmdir data/scriptsfromweb 2>/dev/null || true
fi

# Move processed data
if [ -d "data/processed" ]; then
    echo "  ⚙️ Moving processed data..."
    mv data/processed/* data/training/validation/ 2>/dev/null || true
    rmdir data/processed 2>/dev/null || true
fi

# Move baseline data
if [ -d "data/baseline" ]; then
    echo "  📊 Moving baseline data..."
    mv data/baseline/* data/real_world/benchmarks/ 2>/dev/null || true
    rmdir data/baseline 2>/dev/null || true
fi

# Move prompts
if [ -d "data/prompts" ]; then
    echo "  💬 Moving prompts..."
    mv data/prompts/* data/prompts/user_prompts/ 2>/dev/null || true
fi

# Move output data
if [ -d "data/output" ]; then
    echo "  📤 Moving output data..."
    mv data/output/* results/simulation_outputs/ 2>/dev/null || true
    rmdir data/output 2>/dev/null || true
fi

# Clean up empty directories
echo "🧹 Cleaning up old directory structure..."
find data -type d -empty -delete 2>/dev/null || true

echo "✅ Data migration completed!"

# Phase 7: Create comprehensive README files
echo "📝 Creating comprehensive documentation..."

# We'll create README files in the next steps

echo "🎉 Enhanced reorganization completed!"
echo ""
echo "📋 Summary of changes:"
echo "  ✓ Enhanced data organization with clear separation of concerns"
echo "  ✓ Improved scripts organization by functionality"
echo "  ✓ Better source code modularization"
echo "  ✓ Comprehensive results tracking"
echo "  ✓ Data migration from old structure"
echo ""
echo "🚀 Next steps:"
echo "  1. Review the new structure"
echo "  2. Update import paths in your code"
echo "  3. Populate README files with documentation"
echo "  4. Configure your development environment" 