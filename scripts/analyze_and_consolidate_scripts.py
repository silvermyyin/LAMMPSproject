#!/usr/bin/env python3
"""
Comprehensive script to analyze, deduplicate, and consolidate all real-world LAMMPS scripts.
"""

import os
import hashlib
import shutil
from pathlib import Path
from collections import defaultdict
import json

def get_file_hash(filepath):
    """Get MD5 hash of file content."""
    try:
        with open(filepath, 'rb') as f:
            return hashlib.md5(f.read()).hexdigest()
    except Exception as e:
        print(f"Error reading {filepath}: {e}")
        return None

def analyze_scripts_directory(base_dir):
    """Analyze all LAMMPS scripts in the directory structure."""
    base_path = Path(base_dir)
    
    # Find all LAMMPS input files
    script_patterns = ['*.in', 'in.*', '*.lmp', '*.lammps']
    all_scripts = []
    
    for pattern in script_patterns:
        all_scripts.extend(base_path.rglob(pattern))
    
    print(f"Found {len(all_scripts)} total script files")
    
    # Group by content hash to find duplicates
    hash_to_files = defaultdict(list)
    file_info = {}
    
    for script_path in all_scripts:
        if script_path.is_file():
            file_hash = get_file_hash(script_path)
            if file_hash:
                hash_to_files[file_hash].append(script_path)
                file_info[str(script_path)] = {
                    'hash': file_hash,
                    'size': script_path.stat().st_size,
                    'name': script_path.name,
                    'relative_path': str(script_path.relative_to(base_path))
                }
    
    # Find duplicates
    duplicates = {hash_val: files for hash_val, files in hash_to_files.items() if len(files) > 1}
    unique_scripts = {hash_val: files[0] for hash_val, files in hash_to_files.items()}
    
    print(f"Found {len(duplicates)} groups of duplicate files")
    print(f"Total unique scripts: {len(unique_scripts)}")
    
    # Calculate stats
    total_duplicate_files = sum(len(files) - 1 for files in duplicates.values())
    print(f"Total duplicate files to remove: {total_duplicate_files}")
    
    return {
        'all_scripts': [str(p) for p in all_scripts],
        'duplicates': {hash_val: [str(p) for p in files] for hash_val, files in duplicates.items()},
        'unique_scripts': {hash_val: str(file) for hash_val, file in unique_scripts.items()},
        'file_info': file_info,
        'stats': {
            'total_files': len(all_scripts),
            'unique_files': len(unique_scripts),
            'duplicate_groups': len(duplicates),
            'duplicate_files_to_remove': total_duplicate_files
        }
    }

def consolidate_scripts(analysis_result, output_dir):
    """Consolidate all unique scripts into a single directory."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    consolidated_scripts = []
    
    for i, (hash_val, script_path) in enumerate(analysis_result['unique_scripts'].items()):
        source_path = Path(script_path)
        
        # Create a unique filename combining index and original name
        original_name = source_path.name
        if not original_name.startswith('in.') and not original_name.endswith('.in'):
            # Ensure it follows LAMMPS naming convention
            if '.' in original_name:
                new_name = f"script_{i:04d}_{original_name}"
            else:
                new_name = f"script_{i:04d}_{original_name}.in"
        else:
            new_name = f"script_{i:04d}_{original_name}"
        
        dest_path = output_path / new_name
        
        try:
            shutil.copy2(source_path, dest_path)
            consolidated_scripts.append({
                'index': i,
                'original_path': str(source_path),
                'consolidated_path': str(dest_path),
                'name': new_name,
                'hash': hash_val
            })
            print(f"Copied: {source_path} -> {dest_path}")
        except Exception as e:
            print(f"Error copying {source_path}: {e}")
    
    return consolidated_scripts

def remove_duplicates(analysis_result, dry_run=True):
    """Remove duplicate files, keeping only one copy of each unique script."""
    removed_files = []
    
    for hash_val, duplicate_files in analysis_result['duplicates'].items():
        # Keep the first file, remove the rest
        files_to_remove = duplicate_files[1:]
        
        for file_path in files_to_remove:
            print(f"{'[DRY RUN] ' if dry_run else ''}Removing duplicate: {file_path}")
            
            if not dry_run:
                try:
                    os.remove(file_path)
                    removed_files.append(file_path)
                except Exception as e:
                    print(f"Error removing {file_path}: {e}")
            else:
                removed_files.append(file_path)
    
    return removed_files

def main():
    base_dir = "data/real_world"
    output_dir = "data/real_world_consolidated"
    
    print("=" * 60)
    print("LAMMPS Scripts Analysis and Consolidation")
    print("=" * 60)
    
    # Analyze current structure
    print("\n1. Analyzing current script structure...")
    analysis = analyze_scripts_directory(base_dir)
    
    # Save analysis results
    with open("results/script_analysis.json", "w") as f:
        json.dump(analysis, f, indent=2)
    
    print("\n2. Analysis Summary:")
    print(f"   Total files found: {analysis['stats']['total_files']}")
    print(f"   Unique files: {analysis['stats']['unique_files']}")
    print(f"   Duplicate groups: {analysis['stats']['duplicate_groups']}")
    print(f"   Files to remove: {analysis['stats']['duplicate_files_to_remove']}")
    
    # Show some duplicate examples
    if analysis['duplicates']:
        print("\n3. Example duplicate groups:")
        for i, (hash_val, files) in enumerate(list(analysis['duplicates'].items())[:5]):
            print(f"   Group {i+1} ({len(files)} files):")
            for file in files[:3]:  # Show first 3
                print(f"     - {file}")
            if len(files) > 3:
                print(f"     ... and {len(files)-3} more")
    
    # Consolidate unique scripts
    print("\n4. Consolidating unique scripts...")
    consolidated = consolidate_scripts(analysis, output_dir)
    
    print(f"\n5. Consolidation complete!")
    print(f"   Consolidated {len(consolidated)} unique scripts to: {output_dir}")
    
    # Save consolidation mapping
    with open("results/consolidation_mapping.json", "w") as f:
        json.dump(consolidated, f, indent=2)
    
    print(f"\n6. Next steps:")
    print(f"   - Review the consolidated scripts in: {output_dir}")
    print(f"   - Analysis saved to: results/script_analysis.json")
    print(f"   - Mapping saved to: results/consolidation_mapping.json")
    print(f"   - Run with remove_duplicates=False for dry run of duplicate removal")

if __name__ == "__main__":
    # Ensure results directory exists
    os.makedirs("results", exist_ok=True)
    main() 