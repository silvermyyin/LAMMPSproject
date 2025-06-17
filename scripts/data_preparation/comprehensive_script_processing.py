#!/usr/bin/env python3
"""
Comprehensive Real-World LAMMPS Script Processing

This script:
1. Finds and deduplicates ALL LAMMPS scripts from multiple sources
2. Extracts natural language prompts from EVERY script
3. Runs comprehensive baseline experiments
4. Provides detailed one-on-one comparisons
"""

import os
import re
import pandas as pd
import hashlib
import shutil
from typing import Dict, List, Tuple, Optional, Set
import argparse
from pathlib import Path
import json
from datetime import datetime
import sys

# Add the project root to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

def find_all_lammps_scripts(search_dirs: List[str]) -> List[Tuple[str, str]]:
    """
    Find all LAMMPS script files from multiple directories
    
    Returns:
        List of (file_path, source_directory) tuples
    """
    print("🔍 SCANNING FOR ALL LAMMPS SCRIPTS...")
    
    # LAMMPS file extensions
    extensions = ['.in', '.lammps', '.lmp', '.input', '.nvt', '.nve', '.min']
    
    all_scripts = []
    
    for search_dir in search_dirs:
        if not os.path.exists(search_dir):
            print(f"⚠️  Directory not found: {search_dir}")
            continue
            
        print(f"📁 Scanning: {search_dir}")
        
        # Walk through directory tree
        for root, dirs, files in os.walk(search_dir):
            for file in files:
                # Check if file has LAMMPS extension
                if any(file.endswith(ext) for ext in extensions):
                    file_path = os.path.join(root, file)
                    all_scripts.append((file_path, search_dir))
                    
                # Also check files without extension that might be LAMMPS scripts
                elif '.' not in file and file.startswith('in.'):
                    file_path = os.path.join(root, file)
                    all_scripts.append((file_path, search_dir))
    
    print(f"✅ Found {len(all_scripts)} potential LAMMPS scripts")
    return all_scripts

def calculate_file_hash(file_path: str) -> str:
    """Calculate MD5 hash of file content for deduplication"""
    try:
        with open(file_path, 'rb') as f:
            content = f.read()
            return hashlib.md5(content).hexdigest()
    except Exception as e:
        print(f"❌ Error hashing {file_path}: {e}")
        return ""

def is_valid_lammps_script(file_path: str) -> bool:
    """
    Check if file is actually a LAMMPS script by content analysis
    """
    try:
        with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
            content = f.read().lower()
            
        # Check for LAMMPS keywords
        lammps_keywords = [
            'units', 'atom_style', 'boundary', 'pair_style', 'fix', 'run',
            'timestep', 'thermo', 'dump', 'minimize', 'create_atoms',
            'read_data', 'pair_coeff', 'mass', 'velocity'
        ]
        
        # Must have at least 3 LAMMPS keywords to be considered valid
        keyword_count = sum(1 for keyword in lammps_keywords if keyword in content)
        
        # Additional checks
        has_run_or_minimize = 'run ' in content or 'minimize' in content
        not_too_short = len(content.strip()) > 50
        not_binary = all(ord(c) < 128 for c in content[:100])  # Check first 100 chars for ASCII
        
        return keyword_count >= 3 and has_run_or_minimize and not_too_short and not_binary
        
    except Exception as e:
        print(f"❌ Error validating {file_path}: {e}")
        return False

def deduplicate_scripts(script_list: List[Tuple[str, str]]) -> List[Tuple[str, str, str]]:
    """
    Deduplicate scripts based on content hash
    
    Returns:
        List of (file_path, source_dir, hash) tuples for unique scripts
    """
    print("🔄 DEDUPLICATING SCRIPTS...")
    
    seen_hashes = set()
    unique_scripts = []
    duplicates_removed = 0
    invalid_scripts = 0
    
    for file_path, source_dir in script_list:
        # First check if it's a valid LAMMPS script
        if not is_valid_lammps_script(file_path):
            invalid_scripts += 1
            continue
            
        # Calculate hash for deduplication
        file_hash = calculate_file_hash(file_path)
        if not file_hash:
            continue
            
        if file_hash in seen_hashes:
            duplicates_removed += 1
            print(f"   🗑️  Duplicate: {os.path.basename(file_path)}")
        else:
            seen_hashes.add(file_hash)
            unique_scripts.append((file_path, source_dir, file_hash))
    
    print(f"✅ Deduplication complete:")
    print(f"   - Unique scripts: {len(unique_scripts)}")
    print(f"   - Duplicates removed: {duplicates_removed}")
    print(f"   - Invalid scripts filtered: {invalid_scripts}")
    
    return unique_scripts

def create_consolidated_directory(unique_scripts: List[Tuple[str, str, str]], 
                                output_dir: str = "data/real_world_consolidated") -> str:
    """
    Create a consolidated directory with all unique scripts
    """
    print(f"📁 CREATING CONSOLIDATED DIRECTORY: {output_dir}")
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Copy unique scripts with descriptive names
    consolidated_scripts = []
    
    for i, (file_path, source_dir, file_hash) in enumerate(unique_scripts):
        # Create descriptive filename
        original_name = os.path.basename(file_path)
        source_name = os.path.basename(source_dir.rstrip('/'))
        
        # Clean filename
        clean_name = re.sub(r'[^\w\-_\.]', '_', original_name)
        new_filename = f"{i+1:04d}_{source_name}_{clean_name}"
        
        # Ensure .in extension
        if not new_filename.endswith('.in'):
            new_filename += '.in'
            
        new_path = os.path.join(output_dir, new_filename)
        
        # Copy file
        try:
            shutil.copy2(file_path, new_path)
            consolidated_scripts.append({
                'consolidated_path': new_path,
                'original_path': file_path,
                'source_directory': source_dir,
                'file_hash': file_hash,
                'script_id': i + 1
            })
        except Exception as e:
            print(f"❌ Error copying {file_path}: {e}")
    
    print(f"✅ Consolidated {len(consolidated_scripts)} scripts")
    
    # Save mapping file
    mapping_df = pd.DataFrame(consolidated_scripts)
    mapping_file = os.path.join(output_dir, 'script_mapping.csv')
    mapping_df.to_csv(mapping_file, index=False)
    print(f"📊 Script mapping saved to: {mapping_file}")
    
    return output_dir

def extract_comprehensive_features(script_content: str, script_path: str) -> Dict:
    """
    Extract comprehensive features from a LAMMPS script (enhanced version)
    """
    features = {
        'script_path': script_path,
        'script_name': os.path.basename(script_path),
        'units': None,
        'dimension': None,
        'atom_style': None,
        'boundary': None,
        'pair_style': None,
        'bond_style': None,
        'angle_style': None,
        'dihedral_style': None,
        'improper_style': None,
        'kspace_style': None,
        'timestep': None,
        'ensemble': None,
        'temperature': None,
        'pressure': None,
        'run_steps': None,
        'minimize_steps': None,
        'box_type': None,
        'atom_types': [],
        'fixes': [],
        'dumps': [],
        'computes': [],
        'groups': [],
        'variables': [],
        'special_features': [],
        'simulation_type': None,
        'complexity_score': 0,
        'has_data_file': False,
        'has_restart': False,
        'has_minimization': False,
        'has_equilibration': False,
        'has_production': False
    }
    
    lines = script_content.split('\n')
    
    for line in lines:
        line = line.strip()
        if not line or line.startswith('#'):
            continue
            
        # Basic simulation parameters
        if line.startswith('units '):
            features['units'] = line.split()[1] if len(line.split()) > 1 else None
            
        elif line.startswith('dimension '):
            features['dimension'] = line.split()[1] if len(line.split()) > 1 else None
            
        elif line.startswith('atom_style '):
            features['atom_style'] = ' '.join(line.split()[1:])
            
        elif line.startswith('boundary '):
            features['boundary'] = ' '.join(line.split()[1:])
            
        elif line.startswith('pair_style '):
            features['pair_style'] = ' '.join(line.split()[1:])
            
        elif line.startswith('bond_style '):
            features['bond_style'] = ' '.join(line.split()[1:])
            
        elif line.startswith('angle_style '):
            features['angle_style'] = ' '.join(line.split()[1:])
            
        elif line.startswith('dihedral_style '):
            features['dihedral_style'] = ' '.join(line.split()[1:])
            
        elif line.startswith('improper_style '):
            features['improper_style'] = ' '.join(line.split()[1:])
            
        elif line.startswith('kspace_style '):
            features['kspace_style'] = ' '.join(line.split()[1:])
            
        elif line.startswith('timestep '):
            features['timestep'] = line.split()[1] if len(line.split()) > 1 else None
            
        elif line.startswith('run '):
            try:
                features['run_steps'] = int(line.split()[1])
            except (ValueError, IndexError):
                features['run_steps'] = line.split()[1] if len(line.split()) > 1 else None
                
        elif line.startswith('minimize '):
            features['has_minimization'] = True
            try:
                features['minimize_steps'] = int(line.split()[3]) if len(line.split()) > 3 else None
            except (ValueError, IndexError):
                pass
        
        # Ensemble and thermodynamic settings
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
            
        # Output settings
        elif line.startswith('dump '):
            features['dumps'].append(line)
            
        elif line.startswith('compute '):
            features['computes'].append(line)
            
        elif line.startswith('group '):
            features['groups'].append(line)
            
        elif line.startswith('variable '):
            features['variables'].append(line)
            
        # Box/geometry information
        elif 'create_box' in line:
            features['box_type'] = 'created'
        elif 'read_data' in line:
            features['box_type'] = 'read_from_file'
            features['has_data_file'] = True
        elif 'read_restart' in line:
            features['has_restart'] = True
            
        # Extract masses (atom types)
        elif line.startswith('mass '):
            try:
                atom_type = int(line.split()[1])
                features['atom_types'].append(atom_type)
            except (ValueError, IndexError):
                pass
    
    # Determine simulation phases
    content_lower = script_content.lower()
    if 'equilibrat' in content_lower:
        features['has_equilibration'] = True
    if 'production' in content_lower or features['run_steps']:
        features['has_production'] = True
    
    # Determine simulation type based on features
    features['simulation_type'] = classify_simulation_type_enhanced(features, script_content)
    
    # Identify special features
    features['special_features'] = identify_special_features_enhanced(script_content)
    
    # Calculate complexity score
    features['complexity_score'] = calculate_complexity_score_enhanced(features)
    
    return features

def classify_simulation_type_enhanced(features: Dict, script_content: str) -> str:
    """
    Enhanced simulation type classification
    """
    content_lower = script_content.lower()
    
    # Check for specific simulation types (more comprehensive)
    if 'reaxff' in content_lower:
        return 'reactive_md'
    elif 'rigid' in content_lower:
        return 'rigid_body'
    elif 'fep' in content_lower or 'lambda' in content_lower or 'free energy' in content_lower:
        return 'free_energy_perturbation'
    elif 'deposit' in content_lower:
        return 'deposition'
    elif 'water' in content_lower or 'h2o' in content_lower:
        return 'water_simulation'
    elif 'equilibrat' in content_lower:
        return 'equilibration'
    elif features['pair_style'] and 'lj' in features['pair_style']:
        return 'lennard_jones'
    elif 'melt' in content_lower:
        return 'melting'
    elif 'crystal' in content_lower:
        return 'crystal_simulation'
    elif 'polymer' in content_lower:
        return 'polymer_simulation'
    elif 'protein' in content_lower:
        return 'protein_simulation'
    elif 'metal' in content_lower and features['units'] == 'metal':
        return 'metal_simulation'
    elif 'eam' in content_lower:
        return 'embedded_atom_method'
    elif 'tersoff' in content_lower:
        return 'tersoff_potential'
    elif features['ensemble'] == 'nvt':
        return 'nvt_ensemble'
    elif features['ensemble'] == 'npt':
        return 'npt_ensemble'
    elif features['ensemble'] == 'nve':
        return 'nve_ensemble'
    elif features['has_minimization']:
        return 'energy_minimization'
    else:
        return 'general_md'

def identify_special_features_enhanced(script_content: str) -> List[str]:
    """
    Enhanced special feature identification
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
        'improper': 'improper_interactions',
        'kspace': 'long_range_interactions',
        'neigh_modify': 'neighbor_list_optimization',
        'comm_modify': 'communication_optimization',
        'newton': 'newton_optimization',
        'processors': 'parallel_decomposition',
        'balance': 'load_balancing',
        'temper': 'replica_exchange',
        'neb': 'nudged_elastic_band',
        'smd': 'steered_molecular_dynamics',
        'umbrella': 'umbrella_sampling',
        'metadynamics': 'metadynamics',
        'bias': 'biased_sampling'
    }
    
    for keyword, feature in feature_keywords.items():
        if keyword in content_lower:
            special_features.append(feature)
    
    return list(set(special_features))  # Remove duplicates

def calculate_complexity_score_enhanced(features: Dict) -> int:
    """
    Enhanced complexity score calculation
    """
    score = 0
    
    # Base complexity
    score += 1
    
    # Add points for different features
    if features['atom_types']:
        score += len(set(features['atom_types']))
    
    score += len(features['fixes'])
    score += len(features['dumps']) 
    score += len(features['computes'])
    score += len(features['groups'])
    score += len(features['variables'])
    score += len(features['special_features'])
    
    # Bonus for complex simulation types
    complex_types = ['reactive_md', 'free_energy_perturbation', 'rigid_body', 
                    'protein_simulation', 'polymer_simulation']
    if features['simulation_type'] in complex_types:
        score += 5
        
    # Bonus for advanced features
    if features['bond_style']:
        score += 2
    if features['angle_style']:
        score += 2
    if features['dihedral_style']:
        score += 3
    if features['kspace_style']:
        score += 3
    if features['has_minimization']:
        score += 2
    if features['has_equilibration']:
        score += 2
    if features['has_data_file']:
        score += 1
        
    return score

def generate_enhanced_natural_language_prompt(features: Dict) -> str:
    """
    Generate enhanced natural language prompt from extracted features
    """
    
    # Enhanced base prompt templates
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
        'polymer_simulation': "Set up a polymer molecular dynamics simulation",
        'protein_simulation': "Create a protein molecular dynamics simulation",
        'metal_simulation': "Set up a metal system molecular dynamics simulation",
        'embedded_atom_method': "Create a simulation using Embedded Atom Method (EAM) potential",
        'tersoff_potential': "Set up a simulation with Tersoff potential",
        'nvt_ensemble': "Set up an NVT (canonical) ensemble simulation",
        'npt_ensemble': "Create an NPT (isothermal-isobaric) ensemble simulation", 
        'nve_ensemble': "Set up an NVE (microcanonical) ensemble simulation",
        'energy_minimization': "Create an energy minimization calculation",
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
        
    if features['bond_style']:
        requirements.append(f"Include {features['bond_style']} bond interactions")
        
    if features['angle_style']:
        requirements.append(f"Include {features['angle_style']} angle interactions")
        
    if features['dihedral_style']:
        requirements.append(f"Include {features['dihedral_style']} dihedral interactions")
        
    if features['kspace_style']:
        requirements.append(f"Use {features['kspace_style']} for long-range electrostatics")
        
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
            
    if features['has_minimization']:
        if features['minimize_steps']:
            requirements.append(f"Include energy minimization for {features['minimize_steps']} steps")
        else:
            requirements.append("Include energy minimization")
    
    # Add atom type requirements
    if features['atom_types']:
        num_types = len(set(features['atom_types']))
        requirements.append(f"Include {num_types} atom type(s)")
    
    # Add output requirements
    if features['dumps']:
        requirements.append("Include trajectory output")
    requirements.append("Include thermodynamic output")
    
    # Add special features
    special_feature_descriptions = {
        'charge_equilibration': "Include charge equilibration (QEQ)",
        'energy_minimization': "Include energy minimization",
        'long_range_electrostatics': "Include long-range electrostatic interactions",
        'constraint_dynamics': "Apply constraint dynamics (SHAKE/RATTLE)",
        'velocity_initialization': "Initialize particle velocities",
        'restart_capability': "Include restart file capability",
        'bonded_interactions': "Include bonded interactions",
        'angular_interactions': "Include angular interactions",
        'dihedral_interactions': "Include dihedral interactions"
    }
    
    for feature in features['special_features']:
        if feature in special_feature_descriptions:
            requirements.append(special_feature_descriptions[feature])
        
    # Construct final prompt
    prompt = f"{base_desc}.\n\nRequirements:\n"
    for req in requirements:
        prompt += f"- {req}\n"
        
    prompt += "\nGenerate a complete and syntactically correct LAMMPS input script that fulfills these requirements."
    
    return prompt.strip()

def process_all_consolidated_scripts(consolidated_dir: str, 
                                   output_file: str = "data/prompts/user_prompts/comprehensive_script_prompt_mapping.csv") -> pd.DataFrame:
    """
    Process all consolidated scripts and create comprehensive prompt table
    """
    
    print(f"🔍 PROCESSING ALL CONSOLIDATED SCRIPTS...")
    print(f"📁 Source directory: {consolidated_dir}")
    
    # Find all script files
    script_files = []
    for file in os.listdir(consolidated_dir):
        if file.endswith('.in') and file != 'script_mapping.csv':
            script_files.append(os.path.join(consolidated_dir, file))
    
    print(f"📊 Found {len(script_files)} consolidated scripts to process")
    
    # Process each script
    results = []
    
    for i, script_path in enumerate(script_files):
        print(f"⚙️  Processing {i+1}/{len(script_files)}: {os.path.basename(script_path)}")
        
        try:
            # Read script content
            with open(script_path, 'r', encoding='utf-8', errors='ignore') as f:
                script_content = f.read()
            
            # Skip if script is too short or empty
            if len(script_content.strip()) < 50:
                print(f"   ⚠️  Skipping short script")
                continue
            
            # Extract features
            features = extract_comprehensive_features(script_content, script_path)
            
            # Generate natural language prompt
            nl_prompt = generate_enhanced_natural_language_prompt(features)
            
            # Create result record
            result = {
                'script_id': i + 1,
                'script_name': features['script_name'],
                'script_path': features['script_path'],
                'natural_language_prompt': nl_prompt,
                'simulation_type': features['simulation_type'],
                'units': features['units'],
                'dimension': features['dimension'],
                'atom_style': features['atom_style'],
                'boundary': features['boundary'],
                'pair_style': features['pair_style'],
                'bond_style': features['bond_style'],
                'angle_style': features['angle_style'],
                'dihedral_style': features['dihedral_style'],
                'kspace_style': features['kspace_style'],
                'ensemble': features['ensemble'],
                'temperature': features['temperature'],
                'pressure': features['pressure'],
                'timestep': features['timestep'],
                'run_steps': features['run_steps'],
                'minimize_steps': features['minimize_steps'],
                'num_atom_types': len(set(features['atom_types'])) if features['atom_types'] else 0,
                'num_fixes': len(features['fixes']),
                'num_dumps': len(features['dumps']),
                'num_computes': len(features['computes']),
                'num_groups': len(features['groups']),
                'num_variables': len(features['variables']),
                'special_features': '; '.join(features['special_features']),
                'complexity_score': features['complexity_score'],
                'script_length_lines': len(script_content.split('\n')),
                'script_length_chars': len(script_content),
                'has_data_file': features['has_data_file'],
                'has_restart': features['has_restart'],
                'has_minimization': features['has_minimization'],
                'has_equilibration': features['has_equilibration'],
                'has_production': features['has_production']
            }
            
            results.append(result)
            
        except Exception as e:
            print(f"❌ Error processing {script_path}: {e}")
            continue
    
    # Create DataFrame and save
    df = pd.DataFrame(results)
    
    # Sort by complexity score (simpler first for easier testing)
    df = df.sort_values('complexity_score')
    
    # Save to CSV
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    df.to_csv(output_file, index=False)
    
    print(f"✅ COMPREHENSIVE PROCESSING COMPLETE!")
    print(f"📊 Processed {len(df)} scripts successfully")
    print(f"💾 Saved comprehensive mapping to: {output_file}")
    
    # Print detailed summary statistics
    print(f"\n📈 COMPREHENSIVE SUMMARY STATISTICS:")
    print(f"Total scripts processed: {len(df)}")
    print(f"Simulation types: {dict(df['simulation_type'].value_counts())}")
    print(f"Units distribution: {dict(df['units'].value_counts())}")
    print(f"Complexity score range: {df['complexity_score'].min():.1f} - {df['complexity_score'].max():.1f}")
    print(f"Average complexity: {df['complexity_score'].mean():.2f}")
    print(f"Scripts with minimization: {df['has_minimization'].sum()}")
    print(f"Scripts with data files: {df['has_data_file'].sum()}")
    print(f"Scripts with bonded interactions: {df['bond_style'].notna().sum()}")
    
    return df

def main():
    """
    Main execution function for comprehensive script processing
    """
    parser = argparse.ArgumentParser(description='Comprehensive LAMMPS script processing and deduplication')
    parser.add_argument('--search_dirs', nargs='+', 
                        default=[
                            'data/real_world',
                            'data_backup_20250612_003400/reference_scripts_flat'
                        ],
                        help='Directories to search for LAMMPS scripts')
    parser.add_argument('--output_dir', default='data/real_world_consolidated',
                        help='Output directory for consolidated scripts')
    parser.add_argument('--mapping_file', default='data/prompts/user_prompts/comprehensive_script_prompt_mapping.csv',
                        help='Output file for comprehensive prompt-script mapping')
    
    args = parser.parse_args()
    
    print("🚀 COMPREHENSIVE LAMMPS SCRIPT PROCESSING")
    print("="*60)
    
    # Step 1: Find all LAMMPS scripts
    all_scripts = find_all_lammps_scripts(args.search_dirs)
    
    if not all_scripts:
        print("❌ No LAMMPS scripts found!")
        return
    
    # Step 2: Deduplicate scripts
    unique_scripts = deduplicate_scripts(all_scripts)
    
    if not unique_scripts:
        print("❌ No valid unique scripts found!")
        return
    
    # Step 3: Create consolidated directory
    consolidated_dir = create_consolidated_directory(unique_scripts, args.output_dir)
    
    # Step 4: Process all scripts and extract prompts
    df = process_all_consolidated_scripts(consolidated_dir, args.mapping_file)
    
    print(f"\n🎉 COMPREHENSIVE PROCESSING COMPLETE!")
    print(f"📁 Consolidated scripts: {consolidated_dir}")
    print(f"📊 Comprehensive mapping: {args.mapping_file}")
    print(f"🎯 Ready for comprehensive baseline experiments!")

if __name__ == "__main__":
    main() 