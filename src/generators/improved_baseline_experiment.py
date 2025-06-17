"""
Improved Baseline Experiment with Natural Language Prompts
Tests true NL → LAMMPS generation capability
"""

import os
import json
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
from typing import List, Dict, Any
import sys
import logging
from sacrebleu.metrics import BLEU
import random
import re

# Add the project root to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.calculations.llm_interface import LLMInterface
from src.calculations.evaluate import LAMMPSEvaluator

def extract_simulation_description(script_path: str, script_content: str) -> str:
    """
    Extract natural language description from LAMMPS script
    """
    
    # Pattern-based descriptions for different simulation types
    descriptions = {
        'reaxff': """Create a reactive molecular dynamics simulation using ReaxFF potential.
Requirements:
- Use ReaxFF force field for reactive chemistry
- Set up 3D periodic box with appropriate dimensions  
- Include multiple atom types (C, O, Si, F)
- Apply NVT ensemble with temperature control
- Use small timestep (0.1 fs) for reactive MD
- Include charge equilibration (QEQ)
- Run for sufficient steps for chemical reactions
- Output thermodynamics and trajectories""",

        'rigid': """Set up a rigid body molecular dynamics simulation.
Requirements:
- Define rigid body groups and constraints
- Use appropriate force field (LJ or other)
- Apply NVE ensemble with rigid body dynamics
- Configure inter-body and intra-body interactions
- Set up velocity initialization
- Use suitable timestep for rigid bodies
- Output rigid body motion data""",

        'lj': """Create a Lennard-Jones molecular dynamics simulation.
Requirements:
- Use LJ/cut pair potential with cutoff radius
- Set up simulation box with periodic boundaries
- Initialize particle positions and velocities
- Choose ensemble (NVE, NVT, or NPT)
- Set appropriate timestep and run length
- Include temperature and energy monitoring""",

        'nvt': """Set up an NVT (canonical) ensemble molecular dynamics simulation.
Requirements:
- Apply temperature thermostat (Nosé-Hoover or similar)
- Define target temperature and damping parameters
- Choose appropriate force field for the system
- Set up proper system size and density
- Monitor temperature equilibration
- Output thermodynamic properties""",

        'npt': """Create an NPT (isothermal-isobaric) ensemble simulation.
Requirements:
- Apply both temperature and pressure control
- Set target temperature and pressure values
- Choose appropriate barostat and thermostat
- Allow box size fluctuations
- Monitor density equilibration
- Include pressure tensor output""",

        'equilibrat': """Set up an equilibration molecular dynamics run.
Requirements:
- Choose appropriate ensemble for equilibration
- Start from initial configuration
- Apply gradual heating or cooling if needed
- Monitor system energy and temperature
- Run sufficient steps for equilibration
- Prepare system for production run""",

        'deposit': """Create a molecular deposition simulation.
Requirements:
- Define substrate and depositing molecules
- Set up deposition region and insertion protocol
- Control deposition rate and energy
- Include surface-molecule interactions
- Monitor film growth and structure
- Output deposition statistics""",

        'fep': """Set up a free energy perturbation (FEP) calculation.
Requirements:
- Define initial and final states for perturbation
- Set up lambda coupling parameters
- Use appropriate ensemble (usually NVT)
- Include dual topology if needed
- Configure sampling for FEP analysis
- Output energy differences for each lambda""",

        'water': """Create a water molecular dynamics simulation.
Requirements:
- Use appropriate water model (SPC, TIP3P, etc.)
- Set up water box with proper density
- Apply periodic boundary conditions
- Choose ensemble based on simulation goals
- Include long-range electrostatics
- Monitor hydrogen bonding and structure"""
    }
    
    # Extract filename and script content for pattern matching
    filename = os.path.basename(script_path).lower()
    content_lower = script_content.lower()
    
    # Check for specific patterns
    for pattern, description in descriptions.items():
        if pattern in filename or pattern in content_lower:
            return description
    
    # Default description for unknown types
    return """Create a molecular dynamics simulation.
Requirements:
- Set up appropriate simulation box and boundary conditions
- Define atom types, masses, and initial configuration
- Choose suitable force field and interaction parameters
- Select appropriate ensemble (NVE, NVT, or NPT)
- Set reasonable timestep and simulation length
- Include thermodynamic output and trajectory saving
- Ensure physical correctness and stability"""

def create_natural_language_prompt(script_path: str, script_content: str) -> str:
    """
    Create natural language prompt that tests NL→LAMMPS generation
    """
    description = extract_simulation_description(script_path, script_content)
    
    prompt = f"""You are a LAMMPS molecular dynamics expert. Generate a complete and valid LAMMPS input script based on the following natural language description:

SIMULATION DESCRIPTION:
{description}

Please write a complete LAMMPS input script that fulfills these requirements. Your script must include:

1. SYSTEM SETUP:
   - units, dimension, boundary conditions
   - atom_style appropriate for the simulation
   - box definition or data file reading

2. FORCE FIELD:
   - pair_style and pair_coeff commands
   - bond, angle, dihedral styles if needed
   - mass definitions for all atom types

3. SIMULATION CONTROL:
   - timestep appropriate for the system
   - ensemble specification (fix commands)
   - temperature and pressure settings if applicable

4. OUTPUT:
   - thermo_style and thermo frequency
   - dump commands for trajectory output
   - any special output for analysis

5. EXECUTION:
   - run command with appropriate number of steps

Make sure your script is syntactically correct, physically reasonable, and ready to execute."""

    return prompt

def run_improved_baseline_experiment(
    num_samples: int = 10,
    experiment_type: str = "natural_language_baseline",
    model: str = "gpt-4",
    temperature: float = 0.7,
    reference_dir: str = "data/real_world/research_scripts"
) -> Dict[str, Any]:
    """
    Run improved baseline experiment with natural language prompts
    """
    
    # Set up logging
    logger = logging.getLogger(experiment_type)
    logger.setLevel(logging.INFO)
    
    # Initialize components
    evaluator = LAMMPSEvaluator()
    llm = LLMInterface()
    
    # Get reference scripts
    test_files = []
    for root, _, files in os.walk(reference_dir):
        for file in files:
            if file.endswith(('.in', '.lammps', '.lmp', '.input', '.nvt', '.nve', '.min')):
                test_files.append(os.path.join(root, file))
    
    # Random sampling
    random.seed(42)
    test_files = random.sample(test_files, min(num_samples, len(test_files)))
    
    # Process each script
    results = []
    for i, test_file in enumerate(test_files):
        print(f"\nProcessing {i+1}/{len(test_files)}: {os.path.basename(test_file)}")
        
        # Read reference script
        with open(test_file, 'r', encoding='utf-8', errors='ignore') as f:
            reference_script = f.read()
        
        # Create natural language prompt (NO reference script shown!)
        nl_prompt = create_natural_language_prompt(test_file, reference_script)
        
        # Generate script using natural language description only
        generated_script = llm.call_llm(nl_prompt, model)
        
        # Evaluate against reference
        is_valid, error_messages = evaluator.validate_lammps_script(generated_script, experiment_type)
        syntax_valid, syntax_errors = evaluator.check_syntax_validity(generated_script)
        f1_score = evaluator.calculate_f1_score(reference_script, generated_script)
        bleu_score = evaluator.calculate_bleu_score(reference_script, generated_script)
        semantic_similarity = evaluator.calculate_semantic_similarity(reference_script, generated_script)
        
        # Store results
        result = {
            'test_file': os.path.basename(test_file),
            'is_valid': is_valid,
            'syntax_valid': syntax_valid,
            'f1_score': f1_score,
            'bleu_score': bleu_score,
            'semantic_similarity': semantic_similarity,
            'error_messages': error_messages,
            'syntax_errors': syntax_errors,
            'prompt_type': 'natural_language',
            'generated_script_length': len(generated_script),
            'reference_script_length': len(reference_script)
        }
        results.append(result)
        
        print(f"  Results: Valid={is_valid}, Syntax={syntax_valid}, F1={f1_score:.3f}, BLEU={bleu_score:.3f}")
    
    # Save results
    results_df = pd.DataFrame(results)
    results_file = f"results/experiments/{experiment_type}_results.csv"
    os.makedirs(os.path.dirname(results_file), exist_ok=True)
    results_df.to_csv(results_file, index=False)
    
    print(f"\nResults saved to: {results_file}")
    
    # Calculate summary statistics
    summary = {
        'avg_f1_score': results_df['f1_score'].mean(),
        'avg_bleu_score': results_df['bleu_score'].mean(),
        'avg_semantic_similarity': results_df['semantic_similarity'].mean(),
        'executability_rate': results_df['is_valid'].mean(),
        'syntax_validity_rate': results_df['syntax_valid'].mean(),
        'total_samples': len(results)
    }
    
    print(f"\nSUMMARY STATISTICS:")
    print(f"Average F1 Score: {summary['avg_f1_score']:.4f}")
    print(f"Average BLEU Score: {summary['avg_bleu_score']:.4f}")
    print(f"Average Semantic Similarity: {summary['avg_semantic_similarity']:.4f}")
    print(f"Executability Rate: {summary['executability_rate']:.2%}")
    print(f"Syntax Validity Rate: {summary['syntax_validity_rate']:.2%}")
    
    return {
        'results': results,
        'summary': summary,
        'experiment_type': experiment_type
    }

if __name__ == "__main__":
    # Run improved baseline with natural language prompts
    run_improved_baseline_experiment(num_samples=5)  # Start with small test 