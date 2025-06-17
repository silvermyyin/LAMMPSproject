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

def setup_logging(experiment_type: str = "description_baseline") -> logging.Logger:
    """Set up logging for the experiment."""
    # Create logs directory if it doesn't exist
    log_dir = os.path.join("results", "logs")
    os.makedirs(log_dir, exist_ok=True)
    
    # Create a logger
    logger = logging.getLogger(f"{experiment_type}_experiment")
    logger.setLevel(logging.INFO)
    
    # Avoid adding handlers if they already exist
    if logger.hasHandlers():
        logger.handlers.clear()

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

def run_description_baseline(
    num_samples: int = 10,
    experiment_type: str = "description_baseline",
    model: str = "gpt-4o",
    temperature: float = 0.7,
    descriptions_file: str = "data/cleaned_real_world_scripts/descriptions.json",
    reference_script_dir: str = "data/cleaned_real_world_scripts"
) -> Dict[str, Any]:
    """
    Run a baseline experiment for LAMMPS script generation from descriptions.
    
    Args:
        num_samples (int): Number of samples to generate.
        experiment_type (str): Type of experiment.
        model (str): LLM model to use.
        temperature (float): Temperature for generation.
        descriptions_file (str): Path to the JSON file with descriptions.
        reference_script_dir (str): Directory containing the original reference scripts.
        
    Returns:
        Dict[str, Any]: Results of the experiment.
    """
    # Set up logging
    logger = setup_logging(experiment_type)
    logger.info(f"Starting {experiment_type} experiment with {num_samples} samples")
    
    # Initialize components
    evaluator = LAMMPSEvaluator()
    llm = LLMInterface()
    naive_bleu = BLEU(smooth_method="floor", effective_order=True)
    
    # Load descriptions
    try:
        with open(descriptions_file, 'r', encoding='utf-8') as f:
            descriptions_data = json.load(f)
    except FileNotFoundError:
        logger.error(f"Descriptions file not found: {descriptions_file}")
        return {}

    # Convert dictionary to list of items and select the first num_samples
    descriptions_list = list(descriptions_data.items()) if isinstance(descriptions_data, dict) else descriptions_data
    if num_samples < len(descriptions_list) and num_samples > 0:
        test_data = descriptions_list[:num_samples]
    else:
        test_data = descriptions_list  # Use all descriptions if num_samples is 0 or larger than available
    
    # Create experiment directory
    experiment_dir = os.path.join("results", "experiments", "dated_runs", datetime.now().strftime("%Y-%m-%d") + f"_{experiment_type}")
    os.makedirs(experiment_dir, exist_ok=True)
    
    # Log experiment parameters
    logger.info(f"Experiment parameters:")
    logger.info(f"- Number of samples: {len(test_data)}")
    logger.info(f"- Model: {model}")
    logger.info(f"- Temperature: {temperature}")
    
    # Process each item
    results = []
    for item in test_data:
        if isinstance(item, tuple):
            reference_filename, description = item
        else:
            description = item.get('description') if isinstance(item, dict) else None
            reference_filename = item.get('filename') if isinstance(item, dict) else None
        
        if not description or not reference_filename:
            logger.warning(f"Skipping item due to missing description or filename: {item}")
            continue

        logger.info(f"\nProcessing {reference_filename}")
        
        reference_filepath = os.path.join(reference_script_dir, reference_filename)
        try:
            with open(reference_filepath, 'r', encoding='utf-8', errors='ignore') as f:
                reference_script = f.read()
        except FileNotFoundError:
            logger.warning(f"Reference file not found: {reference_filepath}. Skipping.")
            continue
        
        # Generate prompt based on the description
        prompt = f"""You are a LAMMPS expert. Generate a complete, valid LAMMPS input script that matches the following description.

DESCRIPTION:
{description}

OUTPUT: (only the script)"""
        
        # Generate script
        generated_script = llm.call_llm(prompt, model=model)
        
        # Save generated script
        generated_script_path = os.path.join(experiment_dir, f"generated_{reference_filename}")
        with open(generated_script_path, 'w', encoding='utf-8') as f:
            f.write(generated_script)

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
            'test_file': reference_filename,
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
        logger.info(f"Results for {reference_filename}:")
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

    if not results:
        logger.error("No results were generated. Exiting.")
        return {}

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
    metrics_to_plot = ['f1_score', 'bleu_score', 'naive_bleu_score', 'semantic_similarity', 'kw_f1']
    for metric in metrics_to_plot:
        plt.figure(figsize=(10, 6))
        results_df[metric].hist(bins=10)
        plt.title(f'Distribution of {metric.replace("_", " ").title()}')
        plt.xlabel(metric.replace("_", " ").title())
        plt.ylabel('Frequency')
        plt.tight_layout()
        plt.savefig(os.path.join(plots_dir, f"{experiment_type}_{metric}_distribution.png"))
        plt.close()
    
    # Plot validity rates
    plt.figure(figsize=(8, 6))
    validity_data = {'Executability': executability_rate, 'Syntax Validity': syntax_validity_rate}
    plt.bar(validity_data.keys(), validity_data.values())
    plt.title('Script Validity Rates')
    plt.ylabel('Rate')
    plt.ylim(0, 1)
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, f"{experiment_type}_validity_rates.png"))
    plt.close()
    
    return {
        'results': results_df.to_dict('records'),
        'metrics': {
            'avg_f1': avg_f1,
            'avg_bleu': avg_bleu,
            'executability_rate': executability_rate,
            'syntax_validity_rate': syntax_validity_rate
        }
    }

if __name__ == "__main__":
    # To run this script, ensure your API keys are set up as environment variables
    # (e.g., OPENAI_API_KEY)
    # Set num_samples=0 to process ALL descriptions
    run_description_baseline(num_samples=0) 