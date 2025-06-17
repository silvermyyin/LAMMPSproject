#!/usr/bin/env python3
"""
Comprehensive script to extract prompts from ALL consolidated real-world LAMMPS scripts.
This will process all 1,089 unique scripts and generate corresponding prompts.
"""

import os
import re
import json
from pathlib import Path
from collections import defaultdict
import openai
from concurrent.futures import ThreadPoolExecutor, as_completed
import time
import sys

# Import the improved prompt templates
sys.path.append('.')
from improved_prompt_templates import create_natural_language_prompt

class ComprehensivePromptExtractor:
    def __init__(self, scripts_dir="data/real_world_consolidated"):
        self.scripts_dir = Path(scripts_dir)
        self.extracted_prompts = {}
        self.extraction_stats = defaultdict(int)
        
    def analyze_script_content(self, script_path):
        """Analyze script content to extract key information for prompt generation."""
        try:
            with open(script_path, 'r', encoding='utf-8') as f:
                content = f.read()
        except UnicodeDecodeError:
            try:
                with open(script_path, 'r', encoding='latin1') as f:
                    content = f.read()
            except Exception as e:
                print(f"Error reading {script_path}: {e}")
                return None
        
        analysis = {
            'filename': script_path.name,
            'path': str(script_path),
            'content': content,
            'lines': content.split('\n'),
            'commands': [],
            'simulation_type': 'unknown',
            'physics_type': 'unknown',
            'key_features': [],
            'comments': []
        }
        
        # Extract LAMMPS commands
        for line in analysis['lines']:
            line = line.strip()
            if line and not line.startswith('#'):
                cmd = line.split()[0] if line.split() else ''
                if cmd:
                    analysis['commands'].append(cmd)
        
        # Extract comments for context
        for line in analysis['lines']:
            line = line.strip()
            if line.startswith('#'):
                analysis['comments'].append(line)
        
        # Determine simulation type
        analysis['simulation_type'] = self._determine_simulation_type(content)
        analysis['physics_type'] = self._determine_physics_type(content)
        analysis['key_features'] = self._extract_key_features(content)
        
        return analysis
    
    def _determine_simulation_type(self, content):
        """Determine the type of simulation based on content."""
        content_lower = content.lower()
        
        if 'minimize' in content_lower:
            return 'minimization'
        elif 'fix nvt' in content_lower:
            return 'nvt_dynamics'
        elif 'fix npt' in content_lower:
            return 'npt_dynamics'
        elif 'fix nve' in content_lower:
            return 'nve_dynamics'
        elif 'fix rigid' in content_lower:
            return 'rigid_body'
        elif 'fix spring' in content_lower:
            return 'constrained'
        elif 'fix wall' in content_lower:
            return 'confined'
        elif 'fix langevin' in content_lower:
            return 'langevin_dynamics'
        else:
            return 'molecular_dynamics'
    
    def _determine_physics_type(self, content):
        """Determine the physics/material type."""
        content_lower = content.lower()
        
        if any(word in content_lower for word in ['water', 'spce', 'tip4p', 'spc']):
            return 'water_simulation'
        elif any(word in content_lower for word in ['protein', 'peptide', 'amino']):
            return 'biomolecular'
        elif any(word in content_lower for word in ['metal', 'eam', 'alloy']):
            return 'metallic'
        elif any(word in content_lower for word in ['polymer', 'chain', 'bond']):
            return 'polymer'
        elif any(word in content_lower for word in ['crystal', 'lattice', 'fcc', 'bcc']):
            return 'crystalline'
        elif any(word in content_lower for word in ['gas', 'vapor', 'fluid']):
            return 'fluid'
        elif any(word in content_lower for word in ['graphene', 'carbon', 'cnt']):
            return 'carbon_materials'
        elif any(word in content_lower for word in ['spin', 'magnetic']):
            return 'magnetic'
        else:
            return 'generic'
    
    def _extract_key_features(self, content):
        """Extract key features from the script."""
        features = []
        content_lower = content.lower()
        
        # Temperature control
        if 'temp' in content_lower:
            features.append('temperature_control')
        
        # Pressure control
        if 'press' in content_lower:
            features.append('pressure_control')
        
        # Potential types
        if 'pair_style lj' in content_lower:
            features.append('lennard_jones')
        elif 'pair_style eam' in content_lower:
            features.append('embedded_atom')
        elif 'pair_style reax' in content_lower:
            features.append('reactive')
        
        # Boundary conditions
        if 'boundary' in content_lower:
            features.append('custom_boundaries')
        
        # Output features
        if 'dump' in content_lower:
            features.append('trajectory_output')
        if 'thermo' in content_lower:
            features.append('thermodynamic_output')
        
        return features
    
    def generate_prompt_for_script(self, analysis):
        """Generate a descriptive prompt for a given script analysis."""
        
        # Use the improved prompt template function
        prompt = create_natural_language_prompt(analysis['path'], analysis['content'])
        
        return prompt
    
    def process_all_scripts(self):
        """Process all scripts in the consolidated directory."""
        
        print("Starting comprehensive prompt extraction from all consolidated scripts...")
        print(f"Processing scripts from: {self.scripts_dir}")
        
        # Get all script files
        script_files = []
        for script_file in self.scripts_dir.glob("script_*"):
            if script_file.is_file():
                script_files.append(script_file)
        
        print(f"Found {len(script_files)} scripts to process")
        
        # Process each script
        results = []
        for i, script_path in enumerate(script_files):
            if i % 100 == 0:
                print(f"Processing script {i+1}/{len(script_files)}")
            
            # Analyze script
            analysis = self.analyze_script_content(script_path)
            if not analysis:
                continue
            
            # Generate prompt
            prompt = self.generate_prompt_for_script(analysis)
            
            result = {
                'script_id': i,
                'original_filename': analysis['filename'],
                'script_path': analysis['path'],
                'simulation_type': analysis['simulation_type'],
                'physics_type': analysis['physics_type'],
                'key_features': analysis['key_features'],
                'prompt': prompt,
                'script_content': analysis['content']
            }
            
            results.append(result)
            self.extraction_stats[analysis['simulation_type']] += 1
        
        return results
    
    def save_prompts(self, results, output_file="results/all_extracted_prompts.json"):
        """Save all extracted prompts to a file."""
        
        output_data = {
            'metadata': {
                'total_scripts': len(results),
                'extraction_timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
                'simulation_type_distribution': dict(self.extraction_stats)
            },
            'prompts': results
        }
        
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(output_data, f, indent=2, ensure_ascii=False)
        
        print(f"Saved {len(results)} prompts to {output_file}")
        
        # Also save a summary
        summary_file = output_file.replace('.json', '_summary.json')
        summary = {
            'total_scripts': len(results),
            'simulation_types': dict(self.extraction_stats),
            'sample_prompts': results[:5] if results else []
        }
        
        with open(summary_file, 'w', encoding='utf-8') as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)
        
        return output_file

def main():
    """Main execution function."""
    
    print("=" * 80)
    print("COMPREHENSIVE LAMMPS PROMPT EXTRACTION")
    print("=" * 80)
    print(f"Processing ALL consolidated real-world LAMMPS scripts")
    print()
    
    # Initialize extractor
    extractor = ComprehensivePromptExtractor()
    
    # Process all scripts
    results = extractor.process_all_scripts()
    
    # Save results
    output_file = extractor.save_prompts(results)
    
    print("\n" + "=" * 60)
    print("EXTRACTION COMPLETE!")
    print("=" * 60)
    print(f"✓ Processed {len(results)} scripts")
    print(f"✓ Results saved to: {output_file}")
    print(f"✓ Simulation type distribution:")
    
    for sim_type, count in extractor.extraction_stats.items():
        print(f"  - {sim_type}: {count} scripts")
    
    print(f"\nNext step: Use these prompts to generate new scripts with ChatGPT!")
    
    return results

if __name__ == "__main__":
    results = main() 