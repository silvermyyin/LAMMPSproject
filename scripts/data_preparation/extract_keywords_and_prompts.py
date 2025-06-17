#!/usr/bin/env python3
"""
Extract Keywords and Generate Natural Language Prompts from Reference LAMMPS Scripts

This script analyzes all reference LAMMPS scripts and creates natural language prompts
for true NL→LAMMPS generation testing.
"""

import os
import re
import pandas as pd
from typing import Dict, List, Tuple, Optional
import argparse
from pathlib import Path

def extract_lammps_features(script_content: str, script_path: str) -> Dict:
    """
    Extract comprehensive features from a LAMMPS script
    """
    features = {
        'script_path': script_path,
        'script_name': os.path.basename(script_path),
        'units': None,
        'dimension': None,
        'atom_style': None,
        'boundary': None,
        'pair_style': None,
        'timestep': None,
        'ensemble': None,
        'temperature': None,
        'pressure': None,
        'run_steps': None,
        'box_type': None,
        'atom_types': [],
        'fixes': [],
        'dumps': [],
        'computes': [],
        'groups': [],
        'special_features': [],
        'simulation_type': None,
        'complexity_score': 0
    }
    
    lines = script_content.split('\n')
    
    for line in lines:
        line = line.strip()
        if not line or line.startswith('#'):
            continue
            
        # Extract basic simulation parameters
        if line.startswith('units '):
            features['units'] = line.split()[1]
            
        elif line.startswith('dimension '):
            features['dimension'] = line.split()[1]
            
        elif line.startswith('atom_style '):
            features['atom_style'] = ' '.join(line.split()[1:])
            
        elif line.startswith('boundary '):
            features['boundary'] = ' '.join(line.split()[1:])
            
        elif line.startswith('pair_style '):
            features['pair_style'] = ' '.join(line.split()[1:])
            
        elif line.startswith('timestep '):
            features['timestep'] = line.split()[1]
            
        elif line.startswith('run '):
            try:
                features['run_steps'] = int(line.split()[1])
            except (ValueError, IndexError):
                features['run_steps'] = line.split()[1]
        
        # Extract ensemble and thermodynamic settings
        elif line.startswith('fix ') and ('nvt' in line or 'npt' in line or 'nve' in line):
            fix_parts = line.split()
            if len(fix_parts) >= 4:
                ensemble_type = fix_parts[3]
                features['ensemble'] = ensemble_type
                features['fixes'].append(line)
                
                # Extract temperature
                if 'temp' in line:
                    temp_match = re.search(r'temp\s+([\d\.]+)', line)
                    if temp_match:
                        features['temperature'] = float(temp_match.group(1))
                
                # Extract pressure  
                if 'iso' in line or 'aniso' in line:
                    press_match = re.search(r'(?:iso|aniso)\s+([\d\.]+)', line)
                    if press_match:
                        features['pressure'] = float(press_match.group(1))
        
        elif line.startswith('fix '):
            features['fixes'].append(line)
            
        # Extract output settings
        elif line.startswith('dump '):
            features['dumps'].append(line)
            
        elif line.startswith('compute '):
            features['computes'].append(line)
            
        elif line.startswith('group '):
            features['groups'].append(line)
            
        # Extract box/geometry information
        elif 'create_box' in line or 'read_data' in line or 'read_restart' in line:
            features['box_type'] = 'created' if 'create_box' in line else 'read_from_file'
            
        # Extract masses (atom types)
        elif line.startswith('mass '):
            try:
                atom_type = int(line.split()[1])
                features['atom_types'].append(atom_type)
            except (ValueError, IndexError):
                pass
    
    # Determine simulation type based on features
    features['simulation_type'] = classify_simulation_type(features, script_content)
    
    # Identify special features
    features['special_features'] = identify_special_features(script_content)
    
    # Calculate complexity score
    features['complexity_score'] = calculate_complexity_score(features)
    
    return features

def classify_simulation_type(features: Dict, script_content: str) -> str:
    """
    Classify the type of simulation based on extracted features
    """
    content_lower = script_content.lower()
    
    # Check for specific simulation types
    if 'reaxff' in content_lower:
        return 'reactive_md'
    elif 'rigid' in content_lower:
        return 'rigid_body'
    elif 'fep' in content_lower or 'lambda' in content_lower:
        return 'free_energy_perturbation'
    elif 'deposit' in content_lower:
        return 'deposition'
    elif 'water' in content_lower:
        return 'water_simulation'
    elif 'equilibrat' in content_lower:
        return 'equilibration'
    elif features['pair_style'] and 'lj' in features['pair_style']:
        return 'lennard_jones'
    elif 'melt' in content_lower:
        return 'melting'
    elif 'crystal' in content_lower:
        return 'crystal_simulation'
    elif features['ensemble'] == 'nvt':
        return 'nvt_ensemble'
    elif features['ensemble'] == 'npt':
        return 'npt_ensemble'
    elif features['ensemble'] == 'nve':
        return 'nve_ensemble'
    else:
        return 'general_md'

def identify_special_features(script_content: str) -> List[str]:
    """
    Identify special features in the simulation
    """
    special_features = []
    content_lower = script_content.lower()
    
    feature_keywords = {
        'ewald': 'long_range_electrostatics',
        'pppm': 'particle_mesh_ewald',
        'shake': 'constraint_dynamics', 
        'rattle': 'constraint_dynamics',
        'qeq': 'charge_equilibration',
        'minimiz': 'energy_minimization',
        'velocity': 'velocity_initialization',
        'restart': 'restart_capability',
        'variable': 'dynamic_variables',
        'if': 'conditional_logic',
        'loop': 'loop_structures',
        'bond': 'bonded_interactions',
        'angle': 'angular_interactions',
        'dihedral': 'dihedral_interactions',
        'improper': 'improper_interactions'
    }
    
    for keyword, feature in feature_keywords.items():
        if keyword in content_lower:
            special_features.append(feature)
    
    return special_features

def calculate_complexity_score(features: Dict) -> int:
    """
    Calculate a complexity score for the simulation
    """
    score = 0
    
    # Base complexity
    score += 1
    
    # Add points for different features
    if features['atom_types']:
        score += len(features['atom_types'])
    
    score += len(features['fixes'])
    score += len(features['dumps']) 
    score += len(features['computes'])
    score += len(features['groups'])
    score += len(features['special_features'])
    
    # Bonus for complex simulation types
    complex_types = ['reactive_md', 'free_energy_perturbation', 'rigid_body']
    if features['simulation_type'] in complex_types:
        score += 3
        
    return score

def generate_natural_language_prompt(features: Dict) -> str:
    """
    Generate natural language prompt from extracted features
    """
    
    # Base prompt templates for different simulation types
    type_templates = {
        'reactive_md': "Create a reactive molecular dynamics simulation using ReaxFF potential",
        'rigid_body': "Set up a rigid body molecular dynamics simulation", 
        'free_energy_perturbation': "Create a free energy perturbation calculation",
        'deposition': "Set up a molecular deposition simulation",
        'water_simulation': "Create a water molecular dynamics simulation",
        'equilibration': "Set up an equilibration molecular dynamics run",
        'lennard_jones': "Create a Lennard-Jones molecular dynamics simulation",
        'melting': "Set up a melting point calculation simulation",
        'crystal_simulation': "Create a crystal structure simulation",
        'nvt_ensemble': "Set up an NVT (canonical) ensemble simulation",
        'npt_ensemble': "Create an NPT (isothermal-isobaric) ensemble simulation", 
        'nve_ensemble': "Set up an NVE (microcanonical) ensemble simulation",
        'general_md': "Create a molecular dynamics simulation"
    }
    
    # Start with base description
    base_desc = type_templates.get(features['simulation_type'], type_templates['general_md'])
    
    requirements = []
    
    # Add system setup requirements
    if features['units']:
        requirements.append(f"Use {features['units']} units")
        
    if features['atom_style']:
        requirements.append(f"Set atom_style to {features['atom_style']}")
        
    if features['dimension']:
        requirements.append(f"Use {features['dimension']}D simulation")
        
    if features['boundary']:
        requirements.append(f"Apply boundary conditions: {features['boundary']}")
    
    # Add force field requirements
    if features['pair_style']:
        requirements.append(f"Use {features['pair_style']} pair potential")
        
    # Add ensemble requirements
    if features['ensemble'] and features['temperature']:
        if features['ensemble'] == 'nvt':
            requirements.append(f"Apply NVT ensemble at {features['temperature']}K")
        elif features['ensemble'] == 'npt':
            temp_press = f"at {features['temperature']}K"
            if features['pressure']:
                temp_press += f" and {features['pressure']} pressure"
            requirements.append(f"Apply NPT ensemble {temp_press}")
        elif features['ensemble'] == 'nve':
            requirements.append(f"Use NVE ensemble with initial temperature {features['temperature']}K")
    
    # Add simulation parameters
    if features['timestep']:
        requirements.append(f"Set timestep to {features['timestep']}")
        
    if features['run_steps']:
        if isinstance(features['run_steps'], int):
            requirements.append(f"Run for {features['run_steps']:,} timesteps")
        else:
            requirements.append(f"Run for {features['run_steps']} timesteps")
    
    # Add atom type requirements
    if features['atom_types']:
        num_types = len(set(features['atom_types']))
        requirements.append(f"Include {num_types} atom type(s)")
    
    # Add output requirements
    if features['dumps']:
        requirements.append("Include trajectory output")
    requirements.append("Include thermodynamic output")
    
    # Add special features
    if 'charge_equilibration' in features['special_features']:
        requirements.append("Include charge equilibration (QEQ)")
    if 'energy_minimization' in features['special_features']:
        requirements.append("Include energy minimization")
    if 'long_range_electrostatics' in features['special_features']:
        requirements.append("Include long-range electrostatic interactions")
        
    # Construct final prompt
    prompt = f"{base_desc}.\n\nRequirements:\n"
    for req in requirements:
        prompt += f"- {req}\n"
        
    prompt += "\nGenerate a complete and syntactically correct LAMMPS input script that fulfills these requirements."
    
    return prompt.strip()

def process_all_scripts(reference_dir: str, output_file: str) -> pd.DataFrame:
    """
    Process all reference scripts and create prompt table
    """
    
    print(f"🔍 Scanning reference scripts in: {reference_dir}")
    
    # Find all LAMMPS script files
    script_files = []
    for root, _, files in os.walk(reference_dir):
        for file in files:
            if file.endswith(('.in', '.lammps', '.lmp', '.input', '.nvt', '.nve', '.min')):
                script_files.append(os.path.join(root, file))
    
    print(f"📁 Found {len(script_files)} reference scripts")
    
    # Process each script
    results = []
    
    for i, script_path in enumerate(script_files):
        print(f"⚙️  Processing {i+1}/{len(script_files)}: {os.path.basename(script_path)}")
        
        try:
            # Read script content
            with open(script_path, 'r', encoding='utf-8', errors='ignore') as f:
                script_content = f.read()
            
            # Extract features
            features = extract_lammps_features(script_content, script_path)
            
            # Generate natural language prompt
            nl_prompt = generate_natural_language_prompt(features)
            
            # Create result record
            result = {
                'script_name': features['script_name'],
                'script_path': features['script_path'],
                'natural_language_prompt': nl_prompt,
                'simulation_type': features['simulation_type'],
                'units': features['units'],
                'atom_style': features['atom_style'],
                'pair_style': features['pair_style'],
                'ensemble': features['ensemble'],
                'temperature': features['temperature'],
                'pressure': features['pressure'],
                'timestep': features['timestep'],
                'run_steps': features['run_steps'],
                'num_atom_types': len(set(features['atom_types'])) if features['atom_types'] else 0,
                'num_fixes': len(features['fixes']),
                'num_dumps': len(features['dumps']),
                'special_features': '; '.join(features['special_features']),
                'complexity_score': features['complexity_score'],
                'script_length_lines': len(script_content.split('\n'))
            }
            
            results.append(result)
            
        except Exception as e:
            print(f"❌ Error processing {script_path}: {e}")
            continue
    
    # Create DataFrame and save
    df = pd.DataFrame(results)
    
    # Sort by complexity score (simpler first)
    df = df.sort_values('complexity_score')
    
    # Save to CSV
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    df.to_csv(output_file, index=False)
    
    print(f"✅ Saved {len(df)} prompt-script pairs to: {output_file}")
    
    # Print summary statistics
    print(f"\n📊 SUMMARY STATISTICS:")
    print(f"Total scripts processed: {len(df)}")
    print(f"Simulation types: {df['simulation_type'].value_counts().to_dict()}")
    print(f"Average complexity score: {df['complexity_score'].mean():.2f}")
    print(f"Units distribution: {df['units'].value_counts().to_dict()}")
    
    return df

def main():
    """
    Main execution function
    """
    parser = argparse.ArgumentParser(description='Extract keywords and generate prompts from LAMMPS scripts')
    parser.add_argument('--reference_dir', default='data/real_world/research_scripts',
                        help='Directory containing reference LAMMPS scripts')
    parser.add_argument('--output_file', default='data/prompts/user_prompts/script_prompt_mapping.csv',
                        help='Output CSV file for prompt-script mapping')
    
    args = parser.parse_args()
    
    # Process all scripts
    df = process_all_scripts(args.reference_dir, args.output_file)
    
    print(f"\n🎯 Next steps:")
    print(f"1. Review the generated prompts in: {args.output_file}")
    print(f"2. Run baseline experiment with natural language prompts")
    print(f"3. Compare NL→LAMMPS generation vs reference scripts")

if __name__ == "__main__":
    main() 