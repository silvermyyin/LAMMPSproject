"""
Chain-of-Thought (CoT) Experiment with Natural Language Prompts
Tests CoT prompting for NL → LAMMPS generation.
"""

import os
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
from typing import Dict, Any
import sys
import logging
from sacrebleu.metrics import BLEU
import re

# Add the project root to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.calculations.llm_interface import LLMInterface
from src.calculations.evaluate import LAMMPSEvaluator

def create_cot_prompt(description: str) -> str:
    """
    Create a Chain-of-Thought (CoT) prompt for LAMMPS script generation.
    """
    prompt = f"""You are an expert in LAMMPS. Given the following description, first, provide a step-by-step plan (chain of thought) outlining the necessary commands. Then, based on that plan, generate the complete LAMMPS input script.

SIMULATION DESCRIPTION:
{description}

First, provide your chain of thought. Then, provide the full script.
"""
    return prompt

def run_cot_experiment(
    descriptions_csv: str,
    num_samples: int = -1,
    experiment_type: str = "cot_experiment",
    model: str = "gpt-4",
    temperature: float = 0.7,
) -> Dict[str, Any]:
    """
    Run Chain-of-Thought (CoT) experiment with natural language prompts from a CSV file.
    """
    # Create a directory for the generated scripts
    date_str = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = f"results/experiments/cot_{date_str}"
    os.makedirs(output_dir, exist_ok=True)
    
    # Set up logging
    logger = logging.getLogger(experiment_type)
    logger.setLevel(logging.INFO)
    
    # Initialize components
    evaluator = LAMMPSEvaluator()
    llm = LLMInterface()
    
    # Read descriptions
    descriptions_df = pd.read_csv(descriptions_csv)
    if num_samples > 0:
        descriptions_df = descriptions_df.sample(n=num_samples, random_state=42)

    # Process each description
    results = []
    for index, row in descriptions_df.iterrows():
        filename = row['filename']
        description = row['description']
        
        print(f"\nProcessing {index+1}/{len(descriptions_df)}: {filename}")
        
        # Define reference script path
        reference_script_path = os.path.join("data/cleaned_real_world_scripts", filename)
        
        # Read reference script
        try:
            with open(reference_script_path, 'r', encoding='utf-8', errors='ignore') as f:
                reference_script = f.read()
        except FileNotFoundError:
            print(f"  Reference script not found: {reference_script_path}. Skipping.")
            continue
            
        # Create CoT prompt
        cot_prompt = create_cot_prompt(description)
        
        # Generate script using CoT prompt
        generated_script = llm.call_llm(cot_prompt, model)
        
        # Save the generated script
        generated_script_path = os.path.join(output_dir, filename)
        with open(generated_script_path, 'w', encoding='utf-8') as f:
            f.write(generated_script)
        
        # Evaluate against reference
        is_valid, error_messages = evaluator.validate_lammps_script(generated_script, experiment_type)
        syntax_valid, syntax_errors = evaluator.check_syntax_validity(generated_script)
        f1_score = evaluator.calculate_f1_score(reference_script, generated_script)
        bleu_score = evaluator.calculate_bleu_score(reference_script, generated_script)
        semantic_similarity = evaluator.calculate_semantic_similarity(reference_script, generated_script)
        
        # Store results
        result = {
            'test_file': filename,
            'is_valid': is_valid,
            'syntax_valid': syntax_valid,
            'f1_score': f1_score,
            'bleu_score': bleu_score,
            'semantic_similarity': semantic_similarity,
            'error_messages': error_messages,
            'syntax_errors': syntax_errors,
            'prompt_type': 'cot',
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
    
    # Print summary
    print(f"\nExperiment '{experiment_type}' finished.")
    print(f"Results saved to {results_file}")
    print(f"Generated scripts saved in {output_dir}")
    
    # Optional: Plotting key metrics
    if not results_df.empty:
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))
        results_df['is_valid'].value_counts().plot(kind='pie', ax=axes[0], title='Runnable Scripts', autopct='%1.1f%%')
        results_df['f1_score'].plot(kind='hist', ax=axes[1], title='F1 Score Distribution', bins=10)
        results_df['bleu_score'].plot(kind='hist', ax=axes[2], title='BLEU Score Distribution', bins=10)
        plt.tight_layout()
        plot_file = f"results/experiments/{experiment_type}_summary.png"
        plt.savefig(plot_file)
        print(f"Summary plot saved to {plot_file}")
    else:
        plot_file = None
        print("No results to plot.")

    return {
        "results_file": results_file,
        "plot_file": plot_file,
        "output_dir": output_dir,
        "results_df": results_df.to_dict('records')
    }

if __name__ == '__main__':
    # Example of how to run the experiment
    # Process all 917 descriptions
    run_cot_experiment(
        descriptions_csv="data/cleaned_real_world_scripts/descriptions.csv",
        num_samples=5  # Set to -1 or remove to process all
    ) 