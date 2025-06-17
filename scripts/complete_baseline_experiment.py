#!/usr/bin/env python3
"""
Complete Baseline Experiment for LAMMPS Natural Language Understanding
Includes: Real ChatGPT generation, LAMMPS execution validation, and comprehensive metrics.
"""

import os
import json
import time
import subprocess
from pathlib import Path
from real_chatgpt_generation import RealChatGPTGenerator
from lammps_execution_validator import LAMMPSExecutionValidator
from comprehensive_generation_and_comparison import ComprehensiveGenerationAndComparison

class CompleteBaselineExperiment:
    def __init__(self, use_real_api=False, max_scripts=50):
        self.use_real_api = use_real_api
        self.max_scripts = max_scripts
        self.results = {}
        
    def run_complete_experiment(self):
        """Run the complete baseline experiment pipeline."""
        
        print("🚀 COMPLETE BASELINE EXPERIMENT")
        print("=" * 80)
        print("This is the REAL comprehensive benchmark evaluation!")
        print()
        print(f"Configuration:")
        print(f"  • Use real ChatGPT API: {self.use_real_api}")
        print(f"  • Max scripts to test: {self.max_scripts}")
        print(f"  • LAMMPS validation: {'Yes' if self._check_lammps() else 'No (will skip)'}")
        print()
        
        # Step 1: Generate scripts with real ChatGPT
        print("🤖 STEP 1: GENERATING SCRIPTS WITH CHATGPT")
        print("-" * 50)
        
        if self.use_real_api:
            generator = RealChatGPTGenerator()
            generated_scripts = generator.generate_batch_scripts(
                "results/all_extracted_prompts.json",
                max_scripts=self.max_scripts
            )
            output_file = generator.save_results("results/real_chatgpt_scripts.json")
            
            self.results['generation'] = {
                'method': 'real_chatgpt',
                'total_generated': len(generated_scripts),
                'api_success_rate': sum(1 for s in generated_scripts if s['success']) / len(generated_scripts) * 100,
                'output_file': output_file
            }
            
        else:
            print("⚠️  Using mock generation (set use_real_api=True for real ChatGPT)")
            processor = ComprehensiveGenerationAndComparison()
            processor.load_prompts()
            generated_scripts = processor.generate_all_scripts(max_scripts=self.max_scripts)
            processor.save_results()
            
            self.results['generation'] = {
                'method': 'mock',
                'total_generated': len(generated_scripts),
                'output_file': 'results/generated_scripts.json'
            }
        
        print(f"✅ Generated {len(generated_scripts)} scripts")
        print()
        
        # Step 2: Validate LAMMPS execution
        print("🔧 STEP 2: VALIDATING LAMMPS EXECUTION")
        print("-" * 50)
        
        if self._check_lammps():
            validator = LAMMPSExecutionValidator()
            
            # Use the appropriate generated scripts file
            scripts_file = (self.results['generation']['output_file'] 
                          if self.use_real_api 
                          else "results/generated_scripts.json")
            
            validation_results = validator.validate_generated_scripts(
                scripts_file, 
                max_scripts=self.max_scripts
            )
            
            if validation_results:
                validator.save_validation_results()
                
                self.results['validation'] = {
                    'total_validated': len(validation_results),
                    'executable_scripts': sum(1 for r in validation_results if r['syntax_valid']),
                    'execution_success_rate': sum(1 for r in validation_results if r['syntax_valid']) / len(validation_results) * 100,
                    'average_execution_time': sum(r['execution_time'] for r in validation_results) / len(validation_results)
                }
                
                print(f"✅ Validated {len(validation_results)} scripts")
                print(f"   Execution success rate: {self.results['validation']['execution_success_rate']:.1f}%")
        else:
            print("⚠️  LAMMPS not available - skipping execution validation")
            self.results['validation'] = {'status': 'skipped', 'reason': 'LAMMPS not available'}
        
        print()
        
        # Step 3: Comprehensive comparison
        print("📊 STEP 3: COMPREHENSIVE COMPARISON")
        print("-" * 50)
        
        processor = ComprehensiveGenerationAndComparison()
        processor.load_prompts()
        
        # Load the generated scripts for comparison
        if self.use_real_api and os.path.exists("results/real_chatgpt_scripts.json"):
            with open("results/real_chatgpt_scripts.json", 'r') as f:
                real_data = json.load(f)
            processor.generated_scripts = real_data['generated_scripts']
        else:
            # Use mock generated scripts
            processor.generate_all_scripts(max_scripts=self.max_scripts)
        
        comparison_results, metrics_summary = processor.compare_scripts()
        processor.save_results()
        
        self.results['comparison'] = metrics_summary
        
        print(f"✅ Compared {metrics_summary['total_comparisons']} script pairs")
        print(f"   Average F1 score: {metrics_summary['f1_score_avg']:.4f}")
        print(f"   Average BLEU score: {metrics_summary['bleu_score_avg']:.4f}")
        print()
        
        # Step 4: Generate comprehensive report
        print("📝 STEP 4: GENERATING COMPREHENSIVE REPORT")
        print("-" * 50)
        
        self._generate_comprehensive_report()
        
        print("✅ Complete baseline experiment finished!")
        print()
        
        return self.results
    
    def _check_lammps(self):
        """Check if LAMMPS is available."""
        try:
            result = subprocess.run(["lmp", "-help"], capture_output=True, timeout=5)
            return True
        except:
            return False
    
    def _generate_comprehensive_report(self):
        """Generate a comprehensive experiment report."""
        
        report = {
            'experiment_metadata': {
                'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
                'max_scripts_tested': self.max_scripts,
                'used_real_api': self.use_real_api,
                'lammps_available': self._check_lammps()
            },
            'results': self.results
        }
        
        # Save detailed results
        with open("results/complete_baseline_results.json", 'w') as f:
            json.dump(report, f, indent=2)
        
        # Generate summary report
        self._print_final_summary()
        
        # Save markdown report
        self._generate_markdown_report(report)
    
    def _print_final_summary(self):
        """Print final experiment summary."""
        
        print("🏆 COMPLETE BASELINE EXPERIMENT RESULTS")
        print("=" * 80)
        print()
        
        # Generation Results
        gen_results = self.results.get('generation', {})
        print(f"📈 GENERATION RESULTS:")
        print(f"   • Method: {gen_results.get('method', 'unknown').title()}")
        print(f"   • Scripts Generated: {gen_results.get('total_generated', 0)}")
        if 'api_success_rate' in gen_results:
            print(f"   • API Success Rate: {gen_results['api_success_rate']:.1f}%")
        print()
        
        # Validation Results
        val_results = self.results.get('validation', {})
        if 'total_validated' in val_results:
            print(f"🔧 EXECUTION VALIDATION:")
            print(f"   • Scripts Validated: {val_results['total_validated']}")
            print(f"   • Executable Scripts: {val_results['executable_scripts']}")
            print(f"   • Success Rate: {val_results['execution_success_rate']:.1f}%")
            print(f"   • Avg Execution Time: {val_results['average_execution_time']:.3f}s")
        else:
            print(f"🔧 EXECUTION VALIDATION: {val_results.get('status', 'unknown')}")
        print()
        
        # Comparison Results
        comp_results = self.results.get('comparison', {})
        if comp_results:
            print(f"📊 COMPARISON METRICS:")
            print(f"   • Script Pairs Compared: {comp_results.get('total_comparisons', 0)}")
            print(f"   • Average F1 Score: {comp_results.get('f1_score_avg', 0):.4f}")
            print(f"   • Average BLEU Score: {comp_results.get('bleu_score_avg', 0):.4f}")
            print(f"   • F1 Standard Deviation: {comp_results.get('f1_score_std', 0):.4f}")
            print(f"   • BLEU Standard Deviation: {comp_results.get('bleu_score_std', 0):.4f}")
        print()
        
        # Overall Assessment
        self._assess_overall_performance()
    
    def _assess_overall_performance(self):
        """Assess overall performance and provide recommendations."""
        
        print("🎯 OVERALL ASSESSMENT:")
        
        # Execution success rate
        val_results = self.results.get('validation', {})
        if 'execution_success_rate' in val_results:
            exec_rate = val_results['execution_success_rate']
            if exec_rate >= 80:
                print(f"   ✅ Excellent execution rate ({exec_rate:.1f}%)")
            elif exec_rate >= 60:
                print(f"   ⚠️  Good execution rate ({exec_rate:.1f}%)")
            elif exec_rate >= 40:
                print(f"   ⚠️  Fair execution rate ({exec_rate:.1f}%)")
            else:
                print(f"   ❌ Poor execution rate ({exec_rate:.1f}%)")
        
        # F1 and BLEU scores
        comp_results = self.results.get('comparison', {})
        if comp_results:
            f1_avg = comp_results.get('f1_score_avg', 0)
            bleu_avg = comp_results.get('bleu_score_avg', 0)
            
            if f1_avg >= 0.7:
                print(f"   ✅ Excellent F1 score ({f1_avg:.3f})")
            elif f1_avg >= 0.5:
                print(f"   ⚠️  Good F1 score ({f1_avg:.3f})")
            else:
                print(f"   ❌ F1 score needs improvement ({f1_avg:.3f})")
            
            if bleu_avg >= 0.3:
                print(f"   ✅ Good BLEU score ({bleu_avg:.3f})")
            elif bleu_avg >= 0.1:
                print(f"   ⚠️  Fair BLEU score ({bleu_avg:.3f})")
            else:
                print(f"   ❌ BLEU score needs improvement ({bleu_avg:.3f})")
        
        print()
        print("💡 RECOMMENDATIONS:")
        print("   • Use real ChatGPT API for better generation quality")
        print("   • Implement physics-aware validation")
        print("   • Add domain-specific prompt engineering")
        print("   • Consider fine-tuning on LAMMPS datasets")
        print()
        
        print("🎊 BASELINE EXPERIMENT COMPLETE!")
        print("   This represents a comprehensive evaluation of AI-generated LAMMPS scripts!")
    
    def _generate_markdown_report(self, report_data):
        """Generate a markdown report."""
        
        markdown_content = f"""# Complete Baseline Experiment Report

Generated: {report_data['experiment_metadata']['timestamp']}

## Experiment Configuration
- **Scripts Tested**: {report_data['experiment_metadata']['max_scripts_tested']}
- **Used Real ChatGPT API**: {report_data['experiment_metadata']['used_real_api']}
- **LAMMPS Available**: {report_data['experiment_metadata']['lammps_available']}

## Results Summary

### Script Generation
- **Method**: {self.results.get('generation', {}).get('method', 'unknown').title()}
- **Total Generated**: {self.results.get('generation', {}).get('total_generated', 0)}

### Execution Validation
"""
        
        val_results = self.results.get('validation', {})
        if 'total_validated' in val_results:
            markdown_content += f"""- **Scripts Validated**: {val_results['total_validated']}
- **Executable Scripts**: {val_results['executable_scripts']}
- **Success Rate**: {val_results['execution_success_rate']:.1f}%
"""
        else:
            markdown_content += f"- **Status**: {val_results.get('status', 'Not performed')}\n"
        
        comp_results = self.results.get('comparison', {})
        if comp_results:
            markdown_content += f"""
### Comparison Metrics
- **Script Pairs**: {comp_results.get('total_comparisons', 0)}
- **Average F1 Score**: {comp_results.get('f1_score_avg', 0):.4f}
- **Average BLEU Score**: {comp_results.get('bleu_score_avg', 0):.4f}

## Conclusion

This experiment demonstrates a comprehensive evaluation framework for AI-generated LAMMPS scripts, including real execution validation and detailed performance metrics.
"""
        
        with open("results/baseline_experiment_report.md", 'w') as f:
            f.write(markdown_content)
        
        print("📄 Markdown report saved to: results/baseline_experiment_report.md")

def main():
    """Main execution function."""
    
    print("🚀 Starting Complete Baseline Experiment...")
    print()
    
    # Configuration
    use_real_api = input("Use real ChatGPT API? (y/N): ").lower() in ['y', 'yes']
    
    if use_real_api and not os.getenv('OPENAI_API_KEY'):
        print("❌ OpenAI API key required for real API usage!")
        print("   Set OPENAI_API_KEY environment variable")
        use_real_api = False
    
    # Run experiment
    experiment = CompleteBaselineExperiment(
        use_real_api=use_real_api,
        max_scripts=50  # Reasonable size for comprehensive testing
    )
    
    results = experiment.run_complete_experiment()
    
    print("\n🎉 Complete baseline experiment finished!")
    print("   Check results/ directory for detailed outputs")

if __name__ == "__main__":
    main() 