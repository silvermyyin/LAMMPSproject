#!/usr/bin/env python3
"""
Full evaluation script to process ALL 1,090 consolidated real-world LAMMPS scripts.
This will generate prompts, create new scripts, and provide comprehensive evaluation metrics.
"""

import os
import json
import time
import numpy as np
from pathlib import Path
from comprehensive_generation_and_comparison import ComprehensiveGenerationAndComparison

def run_full_evaluation():
    """Run the complete evaluation pipeline on all scripts."""
    
    print("=" * 100)
    print("FULL COMPREHENSIVE LAMMPS EVALUATION - ALL 1,090 SCRIPTS")
    print("=" * 100)
    print()
    print("This is a REALISTIC benchmark using real-world LAMMPS scripts!")
    print("We've moved beyond artificial script-copying to true NL understanding.")
    print()
    
    # Initialize processor
    processor = ComprehensiveGenerationAndComparison()
    
    # Load all prompts
    processor.load_prompts()
    total_scripts = len(processor.prompts_data['prompts'])
    
    print(f"📊 DATASET STATISTICS:")
    print(f"   • Total unique scripts: {total_scripts}")
    print(f"   • Original total files: 2,529")
    print(f"   • Duplicates removed: {2529 - total_scripts}")
    print(f"   • Deduplication rate: {((2529 - total_scripts) / 2529) * 100:.1f}%")
    print()
    
    # Show simulation type distribution
    sim_types = processor.prompts_data['metadata']['simulation_type_distribution']
    print(f"📈 SIMULATION TYPE DISTRIBUTION:")
    for sim_type, count in sorted(sim_types.items(), key=lambda x: x[1], reverse=True):
        percentage = (count / total_scripts) * 100
        print(f"   • {sim_type}: {count} scripts ({percentage:.1f}%)")
    print()
    
    # Generate all scripts
    print(f"🚀 GENERATING SCRIPTS FOR ALL {total_scripts} PROMPTS...")
    print("   This demonstrates true natural language → LAMMPS generation capability")
    print()
    
    start_time = time.time()
    processor.generate_all_scripts(max_scripts=None, use_threading=False)
    generation_time = time.time() - start_time
    
    print(f"✅ Generation completed in {generation_time:.2f} seconds")
    print(f"   Average time per script: {generation_time/total_scripts:.3f} seconds")
    print()
    
    # Compare all scripts
    print(f"🔍 COMPARING ALL {total_scripts} GENERATED SCRIPTS WITH ORIGINALS...")
    print("   One-on-one comparison with F1 and BLEU metrics")
    print()
    
    start_time = time.time()
    comparison_results, metrics_summary = processor.compare_scripts()
    comparison_time = time.time() - start_time
    
    print(f"✅ Comparison completed in {comparison_time:.2f} seconds")
    print()
    
    # Save results
    processor.save_results()
    
    # Generate comprehensive report
    generate_final_report(metrics_summary, processor, generation_time, comparison_time)
    
    return metrics_summary, processor

def generate_final_report(metrics_summary, processor, generation_time, comparison_time):
    """Generate a comprehensive final report."""
    
    print("=" * 100)
    print("🏆 FINAL COMPREHENSIVE EVALUATION REPORT")
    print("=" * 100)
    print()
    
    # Overall Performance
    print("📊 OVERALL PERFORMANCE METRICS:")
    print(f"   • Total Scripts Processed: {metrics_summary['total_comparisons']:,}")
    print(f"   • Average F1 Score: {metrics_summary['f1_score_avg']:.4f}")
    print(f"   • Total F1 Score: {metrics_summary['f1_score_total']:.2f}")
    print(f"   • Average BLEU Score: {metrics_summary['bleu_score_avg']:.4f}")
    print(f"   • Total BLEU Score: {metrics_summary['bleu_score_total']:.2f}")
    print(f"   • F1 Standard Deviation: {metrics_summary['f1_score_std']:.4f}")
    print(f"   • BLEU Standard Deviation: {metrics_summary['bleu_score_std']:.4f}")
    print()
    
    # Performance Analysis
    print("📈 PERFORMANCE ANALYSIS:")
    f1_categories = categorize_performance(metrics_summary['f1_score_avg'])
    bleu_categories = categorize_performance(metrics_summary['bleu_score_avg'])
    
    print(f"   • F1 Score Performance: {f1_categories}")
    print(f"   • BLEU Score Performance: {bleu_categories}")
    print()
    
    # Timing Performance
    print("⏱️  TIMING PERFORMANCE:")
    print(f"   • Total Generation Time: {generation_time:.2f} seconds")
    print(f"   • Total Comparison Time: {comparison_time:.2f} seconds")
    print(f"   • Average Time per Script: {(generation_time + comparison_time)/metrics_summary['total_comparisons']:.3f} seconds")
    print()
    
    # Benchmark Significance
    print("🎯 BENCHMARK SIGNIFICANCE:")
    print("   ✓ This is a REAL-WORLD benchmark using actual LAMMPS scripts")
    print("   ✓ Moved beyond artificial script-copying tasks")
    print("   ✓ Tests true natural language understanding capability")
    print("   ✓ Comprehensive coverage of diverse simulation types")
    print("   ✓ Rigorous one-on-one comparison methodology")
    print(f"   ✓ Processed {metrics_summary['total_comparisons']:,} unique real-world scripts")
    print()
    
    # Detailed Breakdown by Simulation Type
    analyze_by_simulation_type(processor)
    
    # Recommendations
    print("💡 RECOMMENDATIONS FOR IMPROVEMENT:")
    if metrics_summary['f1_score_avg'] < 0.5:
        print("   • F1 scores suggest need for better command structure generation")
    if metrics_summary['bleu_score_avg'] < 0.3:
        print("   • BLEU scores suggest need for improved sequence similarity")
    
    print("   • Consider fine-tuning on LAMMPS-specific datasets")
    print("   • Implement domain-specific prompt engineering")
    print("   • Add physics-aware validation layers")
    print()
    
    print("=" * 100)
    print("🎉 EVALUATION COMPLETE - COMPREHENSIVE BENCHMARK ESTABLISHED!")
    print("=" * 100)

def categorize_performance(score):
    """Categorize performance scores."""
    if score >= 0.8:
        return "Excellent"
    elif score >= 0.6:
        return "Good"
    elif score >= 0.4:
        return "Fair"
    elif score >= 0.2:
        return "Poor"
    else:
        return "Very Poor"

def analyze_by_simulation_type(processor):
    """Analyze performance breakdown by simulation type."""
    
    print("🔬 PERFORMANCE BY SIMULATION TYPE:")
    
    # Group results by simulation type
    type_analysis = {}
    for result in processor.comparison_results:
        sim_type = result['simulation_type']
        if sim_type not in type_analysis:
            type_analysis[sim_type] = {'f1_scores': [], 'bleu_scores': [], 'count': 0}
        
        type_analysis[sim_type]['f1_scores'].append(result['f1_score'])
        type_analysis[sim_type]['bleu_scores'].append(result['bleu_score'])
        type_analysis[sim_type]['count'] += 1
    
    # Print analysis for each type
    for sim_type, data in sorted(type_analysis.items(), key=lambda x: x[1]['count'], reverse=True):
        avg_f1 = np.mean(data['f1_scores'])
        avg_bleu = np.mean(data['bleu_scores'])
        count = data['count']
        
        print(f"   • {sim_type}: {count} scripts")
        print(f"     - Avg F1: {avg_f1:.3f}, Avg BLEU: {avg_bleu:.3f}")
    print()

def main():
    """Main execution function."""
    
    print("Starting FULL evaluation of ALL consolidated real-world LAMMPS scripts...")
    print("This will take some time but will provide comprehensive benchmark results.")
    print()
    
    response = input("Proceed with full evaluation of all 1,090 scripts? (y/N): ")
    if response.lower() not in ['y', 'yes']:
        print("Running smaller demo with 100 scripts instead...")
        
        # Run smaller demo
        processor = ComprehensiveGenerationAndComparison()
        processor.load_prompts()
        processor.generate_all_scripts(max_scripts=100, use_threading=False)
        comparison_results, metrics_summary = processor.compare_scripts()
        processor.save_results()
        
        print(f"\n🎯 DEMO RESULTS (100 scripts):")
        print(f"   • Average F1: {metrics_summary['f1_score_avg']:.4f}")
        print(f"   • Average BLEU: {metrics_summary['bleu_score_avg']:.4f}")
        print(f"   • This demonstrates the comprehensive benchmark framework!")
        
        return
    
    # Run full evaluation
    metrics_summary, processor = run_full_evaluation()
    
    print("\n🎊 SUCCESS! Comprehensive benchmark complete with all real-world scripts!")

if __name__ == "__main__":
    main() 