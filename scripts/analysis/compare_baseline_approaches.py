#!/usr/bin/env python3
"""
Compare Baseline Approaches: Script-to-Script vs Natural Language-to-Script

This script analyzes and compares the performance differences between:
1. Old baseline: Shows reference script → generates similar script
2. New improved baseline: Natural language description → generates script
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from typing import Dict, List
import os
import json

def load_baseline_results(results_dir: str = "results/experiments") -> Dict[str, pd.DataFrame]:
    """
    Load both old and new baseline results
    """
    results = {}
    
    # Find CSV files
    csv_files = []
    for file in os.listdir(results_dir):
        if file.endswith('.csv'):
            csv_files.append(os.path.join(results_dir, file))
    
    print(f"Found {len(csv_files)} result files:")
    for file in csv_files:
        print(f"  - {os.path.basename(file)}")
    
    # Load datasets
    for file_path in csv_files:
        filename = os.path.basename(file_path)
        if 'improved_baseline' in filename:
            results['nl_to_lammps'] = pd.read_csv(file_path)
            print(f"✅ Loaded Natural Language → LAMMPS results: {len(results['nl_to_lammps'])} samples")
        elif 'baseline_results' in filename and 'improved' not in filename:
            results['script_to_script'] = pd.read_csv(file_path)
            print(f"✅ Loaded Script → Script results: {len(results['script_to_script'])} samples")
    
    return results

def compare_metrics(results: Dict[str, pd.DataFrame]) -> Dict[str, Dict]:
    """
    Compare metrics between the two approaches
    """
    comparison = {}
    
    for approach_name, df in results.items():
        if df is None or len(df) == 0:
            continue
            
        metrics = {}
        
        # Calculate averages for key metrics
        if 'f1_score' in df.columns:
            metrics['avg_f1_score'] = df['f1_score'].mean()
            metrics['std_f1_score'] = df['f1_score'].std()
            
        if 'bleu_score' in df.columns:
            metrics['avg_bleu_score'] = df['bleu_score'].mean()
            metrics['std_bleu_score'] = df['bleu_score'].std()
            
        if 'semantic_similarity' in df.columns:
            metrics['avg_semantic_similarity'] = df['semantic_similarity'].mean()
            metrics['std_semantic_similarity'] = df['semantic_similarity'].std()
            
        if 'keyword_f1_score' in df.columns:
            metrics['avg_keyword_f1'] = df['keyword_f1_score'].mean()
            metrics['std_keyword_f1'] = df['keyword_f1_score'].std()
            
        # Calculate validity rates
        if 'syntax_valid' in df.columns:
            metrics['syntax_validity_rate'] = df['syntax_valid'].mean()
        elif 'syntax_validity' in df.columns:
            metrics['syntax_validity_rate'] = df['syntax_validity'].mean()
            
        if 'executable' in df.columns:
            metrics['executability_rate'] = df['executable'].mean()
        elif 'executability' in df.columns:
            metrics['executability_rate'] = df['executability'].mean()
            
        metrics['total_samples'] = len(df)
        
        comparison[approach_name] = metrics
    
    return comparison

def create_comparison_plots(results: Dict[str, pd.DataFrame], output_dir: str = "results/plots"):
    """
    Create comparison plots
    """
    os.makedirs(output_dir, exist_ok=True)
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('Baseline Approach Comparison: Script-to-Script vs Natural Language-to-LAMMPS', fontsize=16)
    
    # Plot 1: F1 Score Distribution
    ax1 = axes[0, 0]
    for approach_name, df in results.items():
        if 'f1_score' in df.columns:
            ax1.hist(df['f1_score'], alpha=0.7, label=f'{approach_name.replace("_", " ").title()}', bins=10)
    ax1.set_xlabel('F1 Score')
    ax1.set_ylabel('Frequency')
    ax1.set_title('F1 Score Distribution')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: BLEU Score Distribution
    ax2 = axes[0, 1]
    for approach_name, df in results.items():
        if 'bleu_score' in df.columns:
            ax2.hist(df['bleu_score'], alpha=0.7, label=f'{approach_name.replace("_", " ").title()}', bins=10)
    ax2.set_xlabel('BLEU Score')
    ax2.set_ylabel('Frequency')
    ax2.set_title('BLEU Score Distribution')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Plot 3: Semantic Similarity Distribution
    ax3 = axes[1, 0]
    for approach_name, df in results.items():
        if 'semantic_similarity' in df.columns:
            ax3.hist(df['semantic_similarity'], alpha=0.7, label=f'{approach_name.replace("_", " ").title()}', bins=10)
    ax3.set_xlabel('Semantic Similarity')
    ax3.set_ylabel('Frequency')
    ax3.set_title('Semantic Similarity Distribution')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # Plot 4: Validity Rates Comparison
    ax4 = axes[1, 1]
    
    # Collect validity rates
    approaches = []
    syntax_rates = []
    exec_rates = []
    
    for approach_name, df in results.items():
        approaches.append(approach_name.replace('_', ' ').title())
        
        if 'syntax_valid' in df.columns:
            syntax_rates.append(df['syntax_valid'].mean())
        elif 'syntax_validity' in df.columns:
            syntax_rates.append(df['syntax_validity'].mean())
        else:
            syntax_rates.append(0)
            
        if 'executable' in df.columns:
            exec_rates.append(df['executable'].mean())
        elif 'executability' in df.columns:
            exec_rates.append(df['executability'].mean())
        else:
            exec_rates.append(0)
    
    x = np.arange(len(approaches))
    width = 0.35
    
    ax4.bar(x - width/2, syntax_rates, width, label='Syntax Validity Rate', alpha=0.8)
    ax4.bar(x + width/2, exec_rates, width, label='Executability Rate', alpha=0.8)
    
    ax4.set_xlabel('Approach')
    ax4.set_ylabel('Rate')
    ax4.set_title('Validity and Executability Rates')
    ax4.set_xticks(x)
    ax4.set_xticklabels(approaches, rotation=45, ha='right')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save plot
    plot_file = os.path.join(output_dir, 'baseline_approach_comparison.png')
    plt.savefig(plot_file, dpi=300, bbox_inches='tight')
    print(f"📊 Comparison plots saved to: {plot_file}")
    
    plt.show()

def generate_comparison_report(comparison: Dict[str, Dict], output_file: str = "results/baseline_comparison_report.md"):
    """
    Generate a detailed comparison report
    """
    
    report = f"""# Baseline Approach Comparison Report

## Overview

This report compares two different baseline approaches for LAMMPS script generation:

1. **Script-to-Script Generation**: Shows reference LAMMPS script → generates similar script
   - Tests: Script modification and copying abilities
   - Baseline metric: How well can LLM modify existing code
   - Problem: Not realistic for real-world usage

2. **Natural Language-to-LAMMPS Generation**: Natural language description → generates script  
   - Tests: True natural language understanding and code generation
   - Real-world metric: Can users describe what they want and get working code
   - Advantage: Tests actual user scenarios

## Methodology

- **Input Approach 1**: Reference script directly provided to LLM
- **Input Approach 2**: Only natural language description provided (NO reference script shown)
- **Evaluation**: Both approaches evaluated against the same reference scripts
- **Metrics**: F1 score, BLEU score, semantic similarity, syntax validity, executability

---

## Results Comparison

"""
    
    # Add metrics comparison table
    if len(comparison) >= 2:
        approaches = list(comparison.keys())
        metrics = ['avg_f1_score', 'avg_bleu_score', 'avg_semantic_similarity', 'syntax_validity_rate', 'executability_rate']
        
        report += "| Metric | Script-to-Script | Natural Language-to-LAMMPS | Difference |\n"
        report += "|--------|------------------|----------------------------|------------|\n"
        
        for metric in metrics:
            if metric in comparison[approaches[0]] and metric in comparison[approaches[1]]:
                val1 = comparison[approaches[0]][metric]
                val2 = comparison[approaches[1]][metric]
                diff = val2 - val1
                diff_str = f"+{diff:.3f}" if diff > 0 else f"{diff:.3f}"
                
                metric_name = metric.replace('_', ' ').replace('avg ', '').title()
                report += f"| {metric_name} | {val1:.3f} | {val2:.3f} | {diff_str} |\n"
        
        report += "\n"
    
    # Add detailed analysis for each approach
    for approach_name, metrics in comparison.items():
        approach_display = approach_name.replace('_', ' ').title()
        report += f"### {approach_display}\n\n"
        
        report += f"- **Total Samples**: {metrics.get('total_samples', 'N/A')}\n"
        report += f"- **Average F1 Score**: {metrics.get('avg_f1_score', 0):.3f} ± {metrics.get('std_f1_score', 0):.3f}\n"
        report += f"- **Average BLEU Score**: {metrics.get('avg_bleu_score', 0):.3f} ± {metrics.get('std_bleu_score', 0):.3f}\n"
        report += f"- **Average Semantic Similarity**: {metrics.get('avg_semantic_similarity', 0):.3f} ± {metrics.get('std_semantic_similarity', 0):.3f}\n"
        report += f"- **Syntax Validity Rate**: {metrics.get('syntax_validity_rate', 0)*100:.1f}%\n"
        report += f"- **Executability Rate**: {metrics.get('executability_rate', 0)*100:.1f}%\n\n"
    
    # Add conclusions
    report += """## Key Insights

### 🎯 **Natural Language-to-LAMMPS is the Real Test**

The **Natural Language-to-LAMMPS** approach provides a much more realistic assessment of LLM capabilities for scientific computing applications:

1. **Real-world Usage**: Users typically describe what they want in natural language, not provide existing code to modify
2. **True Generation**: Tests actual code generation from scratch, not code modification
3. **Practical Value**: Results directly translate to real user experience

### 📊 **Expected Performance Differences**

- **Lower Metrics for NL→LAMMPS is Normal**: Natural language generation is inherently harder than script modification
- **Syntax Issues Expected**: Without seeing reference syntax, LLMs must generate from memory
- **Focus on Semantic Similarity**: More important than exact token matching for NL generation

### 🔬 **Scientific Significance**

This comparison demonstrates the importance of using appropriate baselines:
- Script-to-script may overestimate LLM capabilities
- Natural language generation provides more honest assessment
- Results guide development of more effective prompt engineering and RAG systems

---

*Generated automatically by the LAMMPS LLM Benchmark System*
"""
    
    # Save report
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    with open(output_file, 'w') as f:
        f.write(report)
    
    print(f"📄 Comparison report saved to: {output_file}")
    
    return report

def main():
    """
    Main execution function
    """
    print("🔍 COMPARING BASELINE APPROACHES")
    print("="*50)
    
    # Load results
    results = load_baseline_results()
    
    if len(results) == 0:
        print("❌ No baseline results found!")
        return
    
    print(f"\n📊 Found {len(results)} baseline approaches to compare")
    
    # Compare metrics
    comparison = compare_metrics(results)
    
    print(f"\n📈 COMPARISON SUMMARY:")
    for approach, metrics in comparison.items():
        approach_display = approach.replace('_', ' ').title()
        print(f"\n{approach_display}:")
        print(f"  F1 Score: {metrics.get('avg_f1_score', 0):.3f}")
        print(f"  BLEU Score: {metrics.get('avg_bleu_score', 0):.3f}")
        print(f"  Semantic Similarity: {metrics.get('avg_semantic_similarity', 0):.3f}")
        print(f"  Syntax Validity: {metrics.get('syntax_validity_rate', 0)*100:.1f}%")
    
    # Generate visualizations
    if len(results) > 0:
        create_comparison_plots(results)
    
    # Generate detailed report
    generate_comparison_report(comparison)
    
    print(f"\n✅ Baseline comparison analysis complete!")

if __name__ == "__main__":
    main() 