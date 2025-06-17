#!/usr/bin/env python3
"""
Clean Unified LAMMPS Script Collection (FIXED VERSION)

This script filters the unified collection to remove:
- LAMMPS data files
- LAMMPS output/log files  
- Other non-input script files

Keeps only actual LAMMPS input scripts with commands.
FIXED: More accurate recognition of valid input scripts.
"""

import os
import shutil
import csv
import re
from pathlib import Path
from typing import List, Dict, Tuple

def is_lammps_input_script(filepath: str) -> Tuple[bool, str]:
    """
    Determine if a file is a genuine LAMMPS input script.
    FIXED: More lenient and accurate criteria.
    Returns: (is_input_script, file_type)
    """
    try:
        with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
            content = f.read(4000)  # Read first 4000 chars for better analysis
            lines = content.split('\n')[:100]  # Check first 100 lines
        
        # Remove empty lines and comments
        non_empty_lines = [line.strip() for line in lines if line.strip() and not line.strip().startswith('#')]
        content_lower = content.lower()
        
        # DEFINITIVE DATA FILE indicators (must reject)
        data_file_indicators = [
            r'^\s*\d+\s+atoms\s*$',
            r'^\s*\d+\s+bonds\s*$', 
            r'^\s*\d+\s+angles\s*$',
            r'^\s*\d+\s+dihedrals\s*$',
            r'^\s*\d+\s+atom\s+types\s*$',
            r'^\s*\d+\s+bond\s+types\s*$',
            'lammps data file',
            'lammps description',
            r'^\s*atoms\s*$',
            r'^\s*masses\s*$',
            r'^\s*pair\s+coeffs\s*$',
            r'^\s*bond\s+coeffs\s*$',
            r'^\s*angle\s+coeffs\s*$'
        ]
        
        # Check for definitive data file patterns
        for indicator in data_file_indicators:
            if re.search(indicator, content, re.IGNORECASE | re.MULTILINE):
                return False, "data_file"
        
        # Check for output/log file patterns
        output_patterns = [
            r'lammps\s*\(',
            'timestep',
            'step\s+temp\s+press',
            'step\s+temp\s+pe\s+ke',
            'step\s+pe\s+ke\s+etotal',
            'total wall time:',
            'performance:'
        ]
        
        for pattern in output_patterns:
            if re.search(pattern, content, re.IGNORECASE):
                # Check if this looks more like output than input
                if len(re.findall(r'^\s*\d+\s+[\d\.\-e]+\s+[\d\.\-e]+', content, re.MULTILINE)) > 5:
                    return False, "output_file"
        
        # LAMMPS INPUT COMMAND indicators (positive indicators)
        input_commands = [
            'units', 'dimension', 'boundary', 'atom_style', 'lattice', 'region',
            'create_box', 'create_atoms', 'read_data', 'read_restart',
            'pair_style', 'pair_coeff', 'bond_style', 'bond_coeff',
            'angle_style', 'angle_coeff', 'dihedral_style', 'dihedral_coeff',
            'improper_style', 'improper_coeff', 'kspace_style',
            'neighbor', 'neigh_modify', 'group', 'set', 'velocity',
            'fix', 'unfix', 'compute', 'uncompute', 'variable',
            'thermo_style', 'thermo', 'thermo_modify',
            'dump', 'undump', 'restart', 'run', 'minimize',
            'reset_timestep', 'timestep', 'replicate', 'change_box',
            'displace_atoms', 'move_atoms', 'delete_atoms',
            'mass', 'temperature', 'pressure'
        ]
        
        # Count actual LAMMPS commands (not in comments)
        command_count = 0
        for line in non_empty_lines:
            # Skip comment lines
            if line.startswith('#'):
                continue
            
            # Check if line contains LAMMPS commands
            line_words = line.split()
            if len(line_words) > 0:
                first_word = line_words[0].lower()
                if first_word in input_commands:
                    command_count += 1
        
        # Also check for variable definitions and control structures
        control_patterns = [
            r'^\s*if\s+\(',
            r'^\s*loop\s+',
            r'^\s*jump\s+',
            r'^\s*next\s+',
            r'^\s*print\s+',
            r'^\s*echo\s+',
            r'^\s*log\s+',
            r'^\s*shell\s+',
            r'variable\s+\w+\s+(equal|atom|string|format)',
        ]
        
        for pattern in control_patterns:
            matches = re.findall(pattern, content, re.IGNORECASE | re.MULTILINE)
            command_count += len(matches)
        
        # Very short files with minimal content
        if len(content.strip()) < 50:
            return False, "unknown_file"
        
        # Files that are mostly numbers (likely data files)
        numeric_lines = 0
        for line in non_empty_lines[:20]:  # Check first 20 non-empty lines
            # Count lines that are mostly numbers
            if re.match(r'^\s*[\d\.\-e\s]+$', line) and len(line.split()) >= 3:
                numeric_lines += 1
        
        if numeric_lines > len(non_empty_lines) * 0.7 and numeric_lines > 5:
            return False, "data_file"
        
        # Decision logic (RELAXED criteria)
        if command_count >= 2:  # Reduced from 3 to 2
            return True, "input_script"
        elif command_count >= 1 and len(non_empty_lines) <= 30:  # Short files with at least 1 command
            return True, "input_script"
        elif command_count >= 1 and any(keyword in content_lower for keyword in ['run', 'minimize', 'fix', 'compute']):
            return True, "input_script"  # Has at least one command and execution keywords
        elif command_count == 0:
            return False, "unknown_file"
        else:
            return False, "script_fragment"
            
    except Exception as e:
        print(f"Error analyzing {filepath}: {e}")
        return False, "unknown_file"

def clean_script_collection():
    """Clean the unified script collection with improved accuracy."""
    
    input_dir = Path("data/all_real_world_scripts")
    output_dir = Path("data/cleaned_real_world_scripts")
    
    # Remove existing output directory and recreate
    if output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.mkdir(exist_ok=True)
    
    # Read the original metadata
    metadata_file = input_dir / "scripts_index.csv"
    if not metadata_file.exists():
        print("❌ Original metadata file not found!")
        return
    
    # Process files
    script_files = list(input_dir.glob("script_*.in"))
    print(f"🔍 Found {len(script_files)} files to analyze...")
    
    kept_files = []
    removed_files = []
    
    stats = {
        "input_script": 0,
        "data_file": 0,
        "output_file": 0,
        "parameter_file": 0,
        "script_fragment": 0,
        "unknown_file": 0
    }
    
    # Analyze each file
    for i, script_file in enumerate(script_files, 1):
        if i % 100 == 0:
            print(f"📊 Processed {i}/{len(script_files)} files...")
        
        is_script, file_type = is_lammps_input_script(str(script_file))
        stats[file_type] += 1
        
        if is_script:
            # Copy to cleaned directory with new sequential naming
            new_index = len(kept_files) + 1
            new_filename = f"script_{new_index:05d}.in"
            new_path = output_dir / new_filename
            shutil.copy2(script_file, new_path)
            kept_files.append((script_file.name, new_filename, file_type))
        else:
            removed_files.append((script_file.name, file_type))
    
    # Read original metadata
    original_metadata = {}
    with open(metadata_file, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            original_metadata[row['unified_filename']] = row
    
    # Create new metadata for kept files
    kept_metadata_file = output_dir / "cleaned_scripts_index.csv"
    with open(kept_metadata_file, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['script_id', 'cleaned_filename', 'cleaned_path', 'original_filename', 
                        'original_path', 'source_directory', 'content_hash', 'file_size', 'determined_type'])
        
        for old_filename, new_filename, file_type in kept_files:
            if old_filename in original_metadata:
                orig = original_metadata[old_filename]
                script_id = new_filename.replace('script_', '').replace('.in', '')
                writer.writerow([
                    script_id, new_filename, f"data/cleaned_real_world_scripts/{new_filename}",
                    orig['original_filename'], orig['original_path'], orig['source_directory'],
                    orig['content_hash'], orig['file_size'], file_type
                ])
    
    # Create removed files report
    removed_report_file = output_dir / "removed_files_report.csv"
    with open(removed_report_file, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['original_unified_filename', 'original_filename', 'original_path', 
                        'source_directory', 'content_hash', 'file_size', 'removal_reason'])
        
        for old_filename, file_type in removed_files:
            if old_filename in original_metadata:
                orig = original_metadata[old_filename]
                writer.writerow([
                    old_filename, orig['original_filename'], orig['original_path'],
                    orig['source_directory'], orig['content_hash'], orig['file_size'], file_type
                ])
    
    # Create summary
    summary_file = output_dir / "cleaning_summary.txt"
    with open(summary_file, 'w', encoding='utf-8') as f:
        f.write("LAMMPS Script Collection Cleaning Summary (FIXED VERSION)\n")
        f.write("=" * 60 + "\n\n")
        f.write(f"📊 PROCESSING RESULTS:\n")
        f.write(f"Total files processed: {len(script_files)}\n")
        f.write(f"Files kept as input scripts: {len(kept_files)}\n")
        f.write(f"Files removed: {len(removed_files)}\n")
        f.write(f"Retention rate: {len(kept_files)/len(script_files)*100:.1f}%\n\n")
        
        f.write(f"📈 FILE TYPE ANALYSIS:\n")
        for file_type, count in stats.items():
            percentage = count/len(script_files)*100
            status = "✅ KEPT" if file_type == "input_script" else "❌ REMOVED"
            f.write(f"{file_type}: {count} files ({percentage:.1f}%) - {status}\n")
        
        f.write(f"\n🎯 IMPROVEMENTS IN THIS VERSION:\n")
        f.write(f"- Reduced minimum command threshold from 3 to 2\n")
        f.write(f"- Added special handling for short files with valid commands\n")
        f.write(f"- Improved recognition of control structures and variables\n")
        f.write(f"- More accurate data file detection\n")
        f.write(f"- Better handling of edge cases\n")
    
    print(f"\n✅ CLEANING COMPLETED!")
    print(f"📁 Cleaned scripts: {len(kept_files)}")
    print(f"🗑️  Removed files: {len(removed_files)}")
    print(f"📈 Retention rate: {len(kept_files)/len(script_files)*100:.1f}%")
    print(f"📊 File type breakdown:")
    for file_type, count in stats.items():
        status = "✅" if file_type == "input_script" else "❌"
        print(f"   {status} {file_type}: {count}")

if __name__ == "__main__":
    clean_script_collection() 