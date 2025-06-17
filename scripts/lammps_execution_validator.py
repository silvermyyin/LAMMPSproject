#!/usr/bin/env python3
"""
LAMMPS Execution Validator
Tests whether generated scripts are syntactically correct and executable.
"""

import os
import json
import subprocess
import tempfile
import time
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

class LAMMPSExecutionValidator:
    def __init__(self, lammps_executable="lmp"):
        self.lammps_executable = lammps_executable
        self.validation_results = []
        
    def check_lammps_available(self):
        """Check if LAMMPS is available in the system."""
        try:
            result = subprocess.run([self.lammps_executable, "-help"], 
                                  capture_output=True, text=True, timeout=10)
            return True
        except (subprocess.TimeoutExpired, FileNotFoundError, subprocess.CalledProcessError):
            return False
    
    def validate_script_syntax(self, script_content, script_id):
        """Validate LAMMPS script syntax by attempting to parse it."""
        
        # Create temporary file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.in', delete=False) as temp_file:
            temp_file.write(script_content)
            temp_path = temp_file.name
        
        validation_result = {
            'script_id': script_id,
            'syntax_valid': False,
            'executable': False,
            'error_message': '',
            'execution_time': 0,
            'warnings': [],
            'validation_timestamp': time.strftime('%Y-%m-%d %H:%M:%S')
        }
        
        try:
            # Test syntax by running LAMMPS with -echo none and immediate exit
            cmd = [self.lammps_executable, '-echo', 'none', '-log', 'none', '-in', temp_path]
            
            start_time = time.time()
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
            validation_result['execution_time'] = time.time() - start_time
            
            if result.returncode == 0:
                validation_result['syntax_valid'] = True
                validation_result['executable'] = True
            else:
                validation_result['error_message'] = result.stderr
                
                # Check if it's just a warning vs actual error
                if "WARNING" in result.stderr and "ERROR" not in result.stderr:
                    validation_result['syntax_valid'] = True
                    validation_result['warnings'] = [line for line in result.stderr.split('\n') if 'WARNING' in line]
                    
        except subprocess.TimeoutExpired:
            validation_result['error_message'] = "Execution timeout (>30s)"
        except Exception as e:
            validation_result['error_message'] = f"Execution error: {str(e)}"
        
        finally:
            # Clean up temporary file
            try:
                os.unlink(temp_path)
            except:
                pass
        
        return validation_result
    
    def analyze_common_errors(self, validation_results):
        """Analyze common errors in failed validations."""
        
        error_categories = {
            'syntax_errors': [],
            'missing_files': [],
            'undefined_variables': [],
            'physics_errors': [],
            'other_errors': []
        }
        
        for result in validation_results:
            if not result['syntax_valid'] and result['error_message']:
                error = result['error_message'].lower()
                
                if 'syntax error' in error or 'invalid command' in error:
                    error_categories['syntax_errors'].append(result)
                elif 'no such file' in error or 'cannot open' in error:
                    error_categories['missing_files'].append(result)
                elif 'undefined' in error or 'unknown' in error:
                    error_categories['undefined_variables'].append(result)
                elif 'energy' in error or 'force' in error or 'potential' in error:
                    error_categories['physics_errors'].append(result)
                else:
                    error_categories['other_errors'].append(result)
        
        return error_categories
    
    def validate_generated_scripts(self, generated_scripts_file, max_scripts=None):
        """Validate all generated scripts."""
        
        print("🔍 LAMMPS EXECUTION VALIDATION STARTING...")
        print("=" * 60)
        
        # Check if LAMMPS is available
        if not self.check_lammps_available():
            print("⚠️  LAMMPS not found in system PATH!")
            print("   Install LAMMPS or ensure it's accessible as 'lmp'")
            return None
        
        print("✅ LAMMPS executable found")
        
        # Load generated scripts
        with open(generated_scripts_file, 'r') as f:
            data = json.load(f)
        
        scripts = data['generated_scripts']
        if max_scripts:
            scripts = scripts[:max_scripts]
        
        print(f"📊 Validating {len(scripts)} generated scripts...")
        
        # Validate scripts
        validation_results = []
        successful = 0
        failed = 0
        
        for i, script_data in enumerate(scripts):
            if (i + 1) % 25 == 0:
                print(f"   Progress: {i + 1}/{len(scripts)} scripts")
            
            result = self.validate_script_syntax(
                script_data['generated_content'], 
                script_data['script_id']
            )
            
            validation_results.append(result)
            
            if result['syntax_valid']:
                successful += 1
            else:
                failed += 1
        
        self.validation_results = validation_results
        
        # Calculate statistics
        success_rate = (successful / len(scripts)) * 100 if scripts else 0
        
        print("\n" + "=" * 60)
        print("🏆 VALIDATION RESULTS")
        print("=" * 60)
        print(f"✅ Successful: {successful}/{len(scripts)} ({success_rate:.1f}%)")
        print(f"❌ Failed: {failed}/{len(scripts)} ({(100-success_rate):.1f}%)")
        
        # Analyze errors
        if failed > 0:
            print(f"\n🔍 ERROR ANALYSIS:")
            error_categories = self.analyze_common_errors(validation_results)
            
            for category, errors in error_categories.items():
                if errors:
                    print(f"   • {category.replace('_', ' ').title()}: {len(errors)} scripts")
        
        return validation_results
    
    def save_validation_results(self, output_file="results/lammps_validation_results.json"):
        """Save validation results to file."""
        
        output_data = {
            'metadata': {
                'total_validated': len(self.validation_results),
                'successful_validations': sum(1 for r in self.validation_results if r['syntax_valid']),
                'validation_timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
                'lammps_executable': self.lammps_executable
            },
            'validation_results': self.validation_results
        }
        
        with open(output_file, 'w') as f:
            json.dump(output_data, f, indent=2)
        
        print(f"💾 Validation results saved to: {output_file}")
        return output_file

def main():
    """Main execution function."""
    
    print("🚀 LAMMPS EXECUTION VALIDATION")
    print("=" * 60)
    print("Testing whether generated scripts are actually executable!")
    print()
    
    # Initialize validator
    validator = LAMMPSExecutionValidator()
    
    # Validate scripts
    try:
        results = validator.validate_generated_scripts(
            "results/generated_scripts.json", 
            max_scripts=50  # Start with smaller batch
        )
        
        if results:
            # Save results
            validator.save_validation_results()
            
            print(f"\n🎯 EXECUTION VALIDATION COMPLETE!")
            print(f"   This tests the REAL executability of generated scripts!")
            
    except FileNotFoundError:
        print("❌ Generated scripts file not found!")
        print("   Run the generation pipeline first.")
    except Exception as e:
        print(f"❌ Validation error: {e}")

if __name__ == "__main__":
    main() 