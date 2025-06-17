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

# Add the project root to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.calculations.llm_interface import LLMInterface
from src.calculations.evaluate import LAMMPSEvaluator

def setup_logging(experiment_type: str = "baseline") -> logging.Logger:
    """Set up logging for the experiment."""
    # Create logs directory if it doesn't exist
    log_dir = os.path.join("results", "logs")
    os.makedirs(log_dir, exist_ok=True)
    
    # Create a logger
    logger = logging.getLogger(f"{experiment_type}_experiment")
    logger.setLevel(logging.INFO)
    
    # Create a file handler
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(log_dir, f"{experiment_type}_{timestamp}.log")
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(logging.INFO)
    
    # Create a console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    
    # Create a formatter
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)
    
    # Add the handlers to the logger
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    
    return logger

def get_prompt_engineering_technique(technique: str) -> str:
    """Get the system prompt for a specific prompt engineering technique."""
    prompts = {
        'none': "You are a LAMMPS expert. Generate valid LAMMPS input scripts based on the given requirements.",
        'basic': """You are a LAMMPS expert. Generate valid LAMMPS input scripts based on the given requirements.
Follow these steps:
1. Identify the simulation type and requirements
2. Set up the basic simulation parameters
3. Define the system and its properties
4. Configure the simulation settings
5. Add necessary output commands""",
        'cot': """You are a LAMMPS expert. Generate valid LAMMPS input scripts based on the given requirements.
Let's solve this step by step:
1. First, analyze the requirements and determine the simulation type
2. Then, think about what parameters and settings are needed
3. Consider the system properties and interactions
4. Plan the simulation workflow
5. Finally, write the complete LAMMPS script""",
        'cove': """You are a LAMMPS expert. Generate valid LAMMPS input scripts based on the given requirements.
For each step, verify your choices:
1. Check if the simulation type matches the requirements
2. Verify that all necessary parameters are included
3. Ensure system properties are correctly defined
4. Validate simulation settings
5. Confirm all required output commands are present"""
    }
    return prompts.get(technique, prompts['none'])

def run_baseline_experiment(
    num_samples: int = 10,
    experiment_type: str = "baseline",
    model: str = "gpt-4",
    temperature: float = 0.7,
    reference_dir: str = "data/real_world/research_scripts"
) -> Dict[str, Any]:
    """
    Run the baseline experiment for LAMMPS script generation.
    
    Args:
        num_samples (int): Number of samples to generate
        experiment_type (str): Type of experiment
        model (str): LLM model to use
        temperature (float): Temperature for generation
        reference_dir (str): Directory containing reference scripts
        
    Returns:
        Dict[str, Any]: Results of the experiment
    """
    # Set up logging
    logger = setup_logging(experiment_type)
    logger.info(f"Starting {experiment_type} experiment with {num_samples} samples")
    
    # Initialize components
    evaluator = LAMMPSEvaluator()
    llm = LLMInterface()
    naive_bleu = BLEU(smooth_method="floor", effective_order=True)
    
    # Prepare test data from reference directory
    test_dir = reference_dir
    test_files = []
    for root, _, files in os.walk(test_dir):
        for file in files:
            if file.endswith(('.in', '.lammps', '.lmp', '.input', '.nvt', '.nve', '.min')):
                test_files.append(os.path.join(root, file))
    
    # Randomly select num_samples files
    random.seed(42)
    test_files = random.sample(test_files, min(num_samples, len(test_files)))
    
    # Create experiment directory under new results hierarchy
    experiment_dir = os.path.join("results", "experiments", "dated_runs", datetime.now().strftime("%Y-%m-%d") + f"_{experiment_type}")
    os.makedirs(experiment_dir, exist_ok=True)
    
    # Log experiment parameters
    logger.info(f"Experiment parameters:")
    logger.info(f"- Number of samples: {len(test_files)}")
    logger.info(f"- Model: {model}")
    logger.info(f"- Temperature: {temperature}")
    
    # Process each test file
    results = []
    for test_file in test_files:
        logger.info(f"\nProcessing {test_file}")
        
        # Read reference script
        with open(test_file, 'r', encoding='utf-8', errors='ignore') as f:
            reference_script = f.read()
        
        # Generate prompt based on the script content
        prompt = f"""Generate a LAMMPS input script for a molecular dynamics simulation.
The script should be similar to the following reference script:

{reference_script}

Please generate a complete and valid LAMMPS input script."""
        
        # Generate script
        generated_script = llm.call_llm(prompt, model)
        
        # Validate script
        is_valid, error_messages = evaluator.validate_lammps_script(generated_script, experiment_type)
        syntax_valid, syntax_errors = evaluator.check_syntax_validity(generated_script)
        
        # Calculate metrics
        f1_score = evaluator.calculate_f1_score(reference_script, generated_script)
        bleu_score = evaluator.calculate_bleu_score(reference_script, generated_script)
        semantic_similarity = evaluator.calculate_semantic_similarity(reference_script, generated_script)
        
        # Calculate additional metrics
        naive_bleu_score = naive_bleu.sentence_score(
            hypothesis=generated_script.strip(),
            references=[reference_script.strip()]
        ).score
        
        # Extract keywords and settings
        ref_keywords = evaluator._extract_parameters(reference_script)
        gen_keywords = evaluator._extract_parameters(generated_script)
        
        # Calculate keyword-level metrics
        kw_f1, kw_precision, kw_recall = evaluator.calculate_f1(
            list(gen_keywords.keys()),
            list(ref_keywords.keys())
        )
        
        # Store results
        result = {
            'test_file': os.path.basename(test_file),
            'is_valid': is_valid,
            'syntax_valid': syntax_valid,
            'f1_score': f1_score,
            'bleu_score': bleu_score,
            'naive_bleu_score': naive_bleu_score,
            'semantic_similarity': semantic_similarity,
            'kw_f1': kw_f1,
            'kw_precision': kw_precision,
            'kw_recall': kw_recall,
            'error_messages': error_messages,
            'syntax_errors': syntax_errors
        }
        results.append(result)
        
        # Log results
        logger.info(f"Results for {os.path.basename(test_file)}:")
        logger.info(f"- Valid: {is_valid}")
        logger.info(f"- Syntax valid: {syntax_valid}")
        logger.info(f"- F1 score: {f1_score:.4f}")
        logger.info(f"- BLEU score: {bleu_score:.4f}")
        logger.info(f"- Naive BLEU score: {naive_bleu_score:.4f}")
        logger.info(f"- Semantic similarity: {semantic_similarity:.4f}")
        logger.info(f"- Keyword F1: {kw_f1:.4f}")
        if error_messages:
            logger.info(f"- Error messages: {error_messages}")
        if syntax_errors:
            logger.info(f"- Syntax errors: {syntax_errors}")
    
    # Calculate aggregate metrics
    avg_f1 = sum(r['f1_score'] for r in results) / len(results)
    avg_bleu = sum(r['bleu_score'] for r in results) / len(results)
    avg_naive_bleu = sum(r['naive_bleu_score'] for r in results) / len(results)
    avg_semantic = sum(r['semantic_similarity'] for r in results) / len(results)
    avg_kw_f1 = sum(r['kw_f1'] for r in results) / len(results)
    executability_rate = sum(1 for r in results if r['is_valid']) / len(results)
    syntax_validity_rate = sum(1 for r in results if r['syntax_valid']) / len(results)
    
    # Log aggregate results
    logger.info(f"\nAggregate Results:")
    logger.info(f"Average F1 score: {avg_f1:.4f}")
    logger.info(f"Average BLEU score: {avg_bleu:.4f}")
    logger.info(f"Average Naive BLEU score: {avg_naive_bleu:.4f}")
    logger.info(f"Average semantic similarity: {avg_semantic:.4f}")
    logger.info(f"Average Keyword F1: {avg_kw_f1:.4f}")
    logger.info(f"Executability rate: {executability_rate:.2%}")
    logger.info(f"Syntax validity rate: {syntax_validity_rate:.2%}")
    
    # Save results to CSV
    results_df = pd.DataFrame(results)
    results_dir = os.path.join("results", "experiments")
    os.makedirs(results_dir, exist_ok=True)
    results_file = os.path.join(results_dir, f"{experiment_type}_results.csv")
    results_df.to_csv(results_file, index=False)
    logger.info(f"\nResults saved to {results_file}")
    
    # Generate visualizations
    plots_dir = os.path.join("results", "plots")
    os.makedirs(plots_dir, exist_ok=True)
    
    # Plot metrics distribution
    metrics = ['f1_score', 'bleu_score', 'naive_bleu_score', 'semantic_similarity', 'kw_f1']
    for metric in metrics:
        plt.figure(figsize=(10, 6))
        plt.hist(results_df[metric], bins=10)
        plt.title(f'Distribution of {metric.replace("_", " ").title()}')
        plt.xlabel(metric.replace("_", " ").title())
        plt.ylabel('Frequency')
        plt.tight_layout()
        plt.savefig(os.path.join(plots_dir, f"{experiment_type}_{metric}_distribution.png"))
        plt.close()
    
    # Plot validity rates
    plt.figure(figsize=(8, 6))
    validity_data = [executability_rate, syntax_validity_rate]
    plt.bar(['Executability', 'Syntax Validity'], validity_data)
    plt.title('Script Validity Rates')
    plt.ylabel('Rate')
    plt.ylim(0, 1)
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, f"{experiment_type}_validity_rates.png"))
    plt.close()
    
    return {
        'results': results,
        'metrics': metrics
    }

if __name__ == "__main__":
    run_baseline_experiment() 