#!/usr/bin/env python3
"""
Improved Baseline Experiment with Natural Language Prompts

This script tests true Natural Language → LAMMPS generation capability
by using natural language descriptions instead of showing reference scripts.
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

# Add the project root to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.calculations.llm_interface import LLMInterface  
from src.calculations.evaluate import LAMMPSEvaluator

def load_prompt_script_pairs(mapping_file: str) -> pd.DataFrame:
    """
    Load the prompt-script mapping table
    """
    print(f"📊 Loading prompt-script pairs from: {mapping_file}")
    df = pd.read_csv(mapping_file)
    print(f"✅ Loaded {len(df)} prompt-script pairs")
    
    # Print summary
    print(f"📈 Distribution by simulation type:")
    for sim_type, count in df['simulation_type'].value_counts().items():
        print(f"   - {sim_type}: {count}")
    
    return df

def select_test_samples(df: pd.DataFrame, num_samples: int, 
                       selection_strategy: str = 'balanced') -> pd.DataFrame:
    """
    Select test samples from the dataset
    
    Args:
        df: Full prompt-script mapping dataframe
        num_samples: Number of samples to select
        selection_strategy: 'balanced', 'simple_first', 'complex_first', 'random'
    """
    print(f"🎯 Selecting {num_samples} samples using '{selection_strategy}' strategy")
    
    if selection_strategy == 'balanced':
        # Select samples to get balanced representation of simulation types
        selected_samples = []
        sim_types = df['simulation_type'].unique()
        samples_per_type = max(1, num_samples // len(sim_types))
        
        for sim_type in sim_types:
            type_df = df[df['simulation_type'] == sim_type]
            # Sort by complexity score for this type
            type_df = type_df.sort_values('complexity_score')
            # Take samples from different complexity levels
            n_samples = min(samples_per_type, len(type_df))
            if n_samples > 0:
                # Select evenly spaced complexity levels
                indices = [i * len(type_df) // n_samples for i in range(n_samples)]
                selected_samples.extend(type_df.iloc[indices].to_dict('records'))
        
        # If we need more samples, fill with remaining
        remaining_needed = num_samples - len(selected_samples)
        if remaining_needed > 0:
            remaining_df = df[~df.index.isin([s['script_path'] for s in selected_samples])]
            additional_samples = remaining_df.sample(n=min(remaining_needed, len(remaining_df)))
            selected_samples.extend(additional_samples.to_dict('records'))
            
        selected_df = pd.DataFrame(selected_samples[:num_samples])
        
    elif selection_strategy == 'simple_first':
        # Select simplest scripts first
        selected_df = df.sort_values('complexity_score').head(num_samples)
        
    elif selection_strategy == 'complex_first':
        # Select most complex scripts first  
        selected_df = df.sort_values('complexity_score', ascending=False).head(num_samples)
        
    elif selection_strategy == 'random':
        # Random selection
        selected_df = df.sample(n=min(num_samples, len(df)))
        
    else:
        raise ValueError(f"Unknown selection strategy: {selection_strategy}")
    
    print(f"✅ Selected {len(selected_df)} samples:")
    print(f"   - Complexity range: {selected_df['complexity_score'].min():.1f} - {selected_df['complexity_score'].max():.1f}")
    print(f"   - Simulation types: {selected_df['simulation_type'].value_counts().to_dict()}")
    
    return selected_df

def run_improved_baseline_experiment(
    num_samples: int = 50,
    mapping_file: str = "data/prompts/user_prompts/script_prompt_mapping.csv",
    selection_strategy: str = 'balanced',
    output_dir: str = "results/experiments",
    model_name: str = "gpt-4"
) -> Dict[str, Any]:
    """
    Run improved baseline experiment with natural language prompts
    """
    
    print(f"🚀 STARTING IMPROVED BASELINE EXPERIMENT")
    print(f"   - Samples: {num_samples}")
    print(f"   - Model: {model_name}")
    print(f"   - Selection strategy: {selection_strategy}")
    print(f"   - Using NL prompts (NOT showing reference scripts)")
    
    # Setup logging
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = f"results/logs/improved_baseline_{timestamp}.log"
    os.makedirs(os.path.dirname(log_file), exist_ok=True)
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
    logger = logging.getLogger('improved_baseline_experiment')
    logger.info(f"Starting improved baseline experiment with {num_samples} samples")
    
    # Load prompt-script mapping
    try:
        df = load_prompt_script_pairs(mapping_file)
    except Exception as e:
        logger.error(f"Failed to load mapping file: {e}")
        raise
        
    # Select test samples
    selected_samples = select_test_samples(df, num_samples, selection_strategy)
    
    # Initialize components
    try:
        llm = LLMInterface()
        evaluator = LAMMPSEvaluator()
        logger.info("✅ Initialized LLM interface and evaluator")
    except Exception as e:
        logger.error(f"Failed to initialize components: {e}")
        raise
    
    # Storage for results
    results = []
    
    # Process each sample
    for i, row in selected_samples.iterrows():
        sample_num = len(results) + 1
        script_name = row['script_name']
        script_path = row['script_path']
        nl_prompt = row['natural_language_prompt']
        simulation_type = row['simulation_type']
        complexity_score = row['complexity_score']
        
        print(f"\n🔄 Processing sample {sample_num}/{num_samples}: {script_name}")
        print(f"   Type: {simulation_type}, Complexity: {complexity_score}")
        
        logger.info(f"Processing sample {sample_num}: {script_name}")
        
        try:
            # Read reference script for evaluation
            with open(script_path, 'r', encoding='utf-8', errors='ignore') as f:
                reference_script = f.read()
            
            # Generate script using natural language prompt (NO reference shown!)
            print(f"   🤖 Generating script from NL prompt...")
            generated_script = llm.call_llm(
                prompt=nl_prompt,  # Only natural language description!
                model=model_name
            )
            
            if not generated_script:
                logger.warning(f"Empty generation for sample {sample_num}")
                continue
                
            print(f"   ✅ Generated {len(generated_script)} characters")
            
            # Evaluate generated script against reference
            print(f"   📊 Evaluating against reference...")
            metrics = evaluator.evaluate_script(reference_script, generated_script)
            
            # Store results
            result = {
                'sample_id': sample_num,
                'script_name': script_name,
                'script_path': script_path,
                'simulation_type': simulation_type,
                'complexity_score': complexity_score,
                'natural_language_prompt': nl_prompt,
                'reference_script': reference_script,
                'generated_script': generated_script,
                'timestamp': datetime.now().isoformat(),
                **metrics  # All evaluation metrics
            }
            
            results.append(result)
            
            # Print metrics for this sample
            print(f"   📈 Metrics: F1={metrics.get('f1_score', 0):.3f}, "
                  f"BLEU={metrics.get('bleu_score', 0):.3f}, "
                  f"Similarity={metrics.get('semantic_similarity', 0):.3f}, "
                  f"Valid={metrics.get('syntax_valid', False)}")
            
            logger.info(f"Sample {sample_num} completed - F1: {metrics.get('f1_score', 0):.3f}")
            
        except Exception as e:
            logger.error(f"Error processing sample {sample_num}: {e}")
            print(f"   ❌ Error: {e}")
            continue
    
    if not results:
        logger.error("No successful generations!")
        raise RuntimeError("No successful generations!")
    
    # Calculate aggregate metrics
    print(f"\n📊 CALCULATING AGGREGATE METRICS...")
    
    aggregate_metrics = {}
    metric_keys = ['f1_score', 'bleu_score', 'semantic_similarity', 'keyword_f1_score']
    
    for key in metric_keys:
        values = [r[key] for r in results if key in r and r[key] is not None]
        if values:
            aggregate_metrics[f'avg_{key}'] = sum(values) / len(values)
            aggregate_metrics[f'min_{key}'] = min(values)
            aggregate_metrics[f'max_{key}'] = max(values)
    
    # Calculate validity rates
    syntax_valid_count = sum(1 for r in results if r.get('syntax_valid', False))
    executable_count = sum(1 for r in results if r.get('executable', False))
    
    aggregate_metrics['syntax_validity_rate'] = syntax_valid_count / len(results)
    aggregate_metrics['executability_rate'] = executable_count / len(results)
    aggregate_metrics['total_samples'] = len(results)
    
    # Save detailed results to CSV
    results_df = pd.DataFrame(results)
    csv_file = f"{output_dir}/improved_baseline_results_{timestamp}.csv"
    os.makedirs(output_dir, exist_ok=True)
    results_df.to_csv(csv_file, index=False)
    
    # Save aggregate metrics to JSON
    json_file = f"{output_dir}/improved_baseline_summary_{timestamp}.json"
    with open(json_file, 'w') as f:
        json.dump(aggregate_metrics, f, indent=2)
    
    # Print final results
    print(f"\n🎉 IMPROVED BASELINE EXPERIMENT COMPLETE!")
    print(f"📁 Results saved to: {csv_file}")
    print(f"📊 Summary saved to: {json_file}")
    
    print(f"\n📈 FINAL METRICS (Natural Language → LAMMPS):")
    print(f"   🎯 Average F1 Score: {aggregate_metrics.get('avg_f1_score', 0):.3f}")
    print(f"   🔤 Average BLEU Score: {aggregate_metrics.get('avg_bleu_score', 0):.3f}")
    print(f"   🧠 Average Semantic Similarity: {aggregate_metrics.get('avg_semantic_similarity', 0):.3f}")
    print(f"   🔑 Average Keyword F1: {aggregate_metrics.get('avg_keyword_f1_score', 0):.3f}")
    print(f"   ✅ Syntax Validity Rate: {aggregate_metrics['syntax_validity_rate']*100:.1f}%")
    print(f"   🚀 Executability Rate: {aggregate_metrics['executability_rate']*100:.1f}%")
    
    # Compare with previous baseline if available
    print(f"\n🔄 This is TRUE Natural Language → LAMMPS generation!")
    print(f"   - NO reference scripts were shown to the model")
    print(f"   - Only natural language descriptions were provided")
    print(f"   - This tests real-world usage scenarios")
    
    logger.info("Improved baseline experiment completed successfully")
    
    return aggregate_metrics

def main():
    """
    Main execution function
    """
    import argparse
    
    parser = argparse.ArgumentParser(description='Run improved baseline experiment with natural language prompts')
    parser.add_argument('--num_samples', type=int, default=50, help='Number of samples to test')
    parser.add_argument('--mapping_file', default='data/prompts/user_prompts/script_prompt_mapping.csv',
                        help='Prompt-script mapping file')
    parser.add_argument('--selection_strategy', choices=['balanced', 'simple_first', 'complex_first', 'random'],
                        default='balanced', help='Sample selection strategy')
    parser.add_argument('--model', default='gpt-4', help='LLM model name')
    
    args = parser.parse_args()
    
    # Run experiment
    results = run_improved_baseline_experiment(
        num_samples=args.num_samples,
        mapping_file=args.mapping_file,
        selection_strategy=args.selection_strategy,
        model_name=args.model
    )
    
    print(f"\n✅ Experiment completed successfully!")

if __name__ == "__main__":
    main() 