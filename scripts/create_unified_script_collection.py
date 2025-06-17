#!/usr/bin/env python3
"""
Unified Real-World LAMMPS Script Collection Creator

This script collects all LAMMPS input scripts from multiple sources:
- data/real_world/ (all subdirectories)
- data/real_world_consolidated/

Identifies LAMMPS scripts by:
- Files ending with: .in, .lammps, .lmp
- Files starting with: in.
- Content analysis for verification

Creates:
- Unified collection with consistent naming: script_XXXXX.in
- Comprehensive metadata with deduplication tracking
- Content-based hash deduplication
"""

import os
import hashlib
import csv
import shutil
from pathlib import Path
import re
from typing import Set, Dict, List, Tuple

def calculate_file_hash(filepath: str) -> str:
    """Calculate SHA-256 hash of file content"""
    with open(filepath, 'rb') as f:
        return hashlib.sha256(f.read()).hexdigest()

def is_lammps_script(filepath: str) -> bool:
    """
    Determine if a file is a LAMMPS input script based on:
    1. File extension (.in, .lammps, .lmp)
    2. Filename starting with 'in.'
    3. Content analysis (looking for LAMMPS commands)
    """
    filename = os.path.basename(filepath).lower()
    
    # Check file extensions
    if filename.endswith(('.in', '.lammps', '.lmp')):
        return True
    
    # Check if filename starts with 'in.'
    if filename.startswith('in.'):
        return True
    
    # Content-based detection for other files
    try:
        with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
            content = f.read(1000).lower()  # Read first 1000 chars
            # Look for common LAMMPS commands
            lammps_keywords = [
                'units', 'dimension', 'boundary', 'atom_style', 'pair_style',
                'read_data', 'create_box', 'create_atoms', 'minimize',
                'run', 'dump', 'thermo', 'timestep', 'fix', 'compute'
            ]
            
            # If file contains multiple LAMMPS keywords, likely a script
            keyword_count = sum(1 for keyword in lammps_keywords if keyword in content)
            return keyword_count >= 3
    except:
        return False

def find_all_scripts(source_dirs: List[str]) -> List[Tuple[str, str]]:
    """
    Find all LAMMPS scripts in source directories
    Returns: List of (filepath, source_directory) tuples
    """
    scripts = []
    
    for source_dir in source_dirs:
        if not os.path.exists(source_dir):
            print(f"Warning: {source_dir} does not exist, skipping...")
            continue
            
        print(f"Scanning {source_dir}...")
        
        for root, dirs, files in os.walk(source_dir):
            for file in files:
                filepath = os.path.join(root, file)
                
                # Skip binary files and very large files
                try:
                    file_size = os.path.getsize(filepath)
                    if file_size > 10_000_000:  # Skip files > 10MB
                        continue
                except:
                    continue
                
                if is_lammps_script(filepath):
                    scripts.append((filepath, source_dir))
                    
        print(f"Found {len([s for s in scripts if s[1] == source_dir])} scripts in {source_dir}")
    
    return scripts

def deduplicate_scripts(scripts: List[Tuple[str, str]]) -> Tuple[List[Dict], Dict[str, List[str]]]:
    """
    Deduplicate scripts based on content hash
    Returns: (unique_scripts_info, duplicates_map)
    """
    hash_to_script = {}
    duplicates = {}
    unique_scripts = []
    
    for i, (filepath, source_dir) in enumerate(scripts):
        try:
            file_hash = calculate_file_hash(filepath)
            file_size = os.path.getsize(filepath)
            
            script_info = {
                'original_path': filepath,
                'source_directory': source_dir,
                'content_hash': file_hash,
                'file_size': file_size,
                'original_filename': os.path.basename(filepath)
            }
            
            if file_hash in hash_to_script:
                # Duplicate found
                if file_hash not in duplicates:
                    duplicates[file_hash] = [hash_to_script[file_hash]['original_path']]
                duplicates[file_hash].append(filepath)
            else:
                # Unique script
                hash_to_script[file_hash] = script_info
                script_info['script_id'] = len(unique_scripts) + 1
                unique_scripts.append(script_info)
                
        except Exception as e:
            print(f"Error processing {filepath}: {e}")
    
    return unique_scripts, duplicates

def create_unified_collection(unique_scripts: List[Dict], output_dir: str):
    """Create unified collection with consistent naming"""
    
    os.makedirs(output_dir, exist_ok=True)
    
    metadata = []
    
    for script_info in unique_scripts:
        script_id = script_info['script_id']
        new_filename = f"script_{script_id:05d}.in"
        new_filepath = os.path.join(output_dir, new_filename)
        
        # Copy script with new name
        try:
            shutil.copy2(script_info['original_path'], new_filepath)
            
            # Add to metadata
            metadata.append({
                'script_id': script_id,
                'unified_filename': new_filename,
                'unified_path': new_filepath,
                'original_filename': script_info['original_filename'],
                'original_path': script_info['original_path'],
                'source_directory': script_info['source_directory'],
                'content_hash': script_info['content_hash'],
                'file_size': script_info['file_size']
            })
            
        except Exception as e:
            print(f"Error copying {script_info['original_path']}: {e}")
    
    return metadata

def save_metadata(metadata: List[Dict], duplicates: Dict, output_dir: str):
    """Save comprehensive metadata"""
    
    # Save main metadata
    metadata_file = os.path.join(output_dir, 'scripts_index.csv')
    with open(metadata_file, 'w', newline='', encoding='utf-8') as f:
        if metadata:
            writer = csv.DictWriter(f, fieldnames=metadata[0].keys())
            writer.writeheader()
            writer.writerows(metadata)
    
    # Save duplicates information
    duplicates_file = os.path.join(output_dir, 'duplicates_report.csv')
    with open(duplicates_file, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['content_hash', 'duplicate_count', 'file_paths'])
        
        for file_hash, paths in duplicates.items():
            writer.writerow([file_hash, len(paths), '; '.join(paths)])
    
    # Save summary report
    summary_file = os.path.join(output_dir, 'collection_summary.txt')
    with open(summary_file, 'w') as f:
        f.write("Unified Real-World LAMMPS Script Collection Summary\n")
        f.write("=" * 55 + "\n\n")
        f.write(f"Total unique scripts: {len(metadata)}\n")
        f.write(f"Total duplicates found: {sum(len(paths) for paths in duplicates.values())}\n")
        f.write(f"Duplicate groups: {len(duplicates)}\n\n")
        
        # Source breakdown
        source_counts = {}
        for item in metadata:
            source = item['source_directory']
            source_counts[source] = source_counts.get(source, 0) + 1
        
        f.write("Scripts by source:\n")
        for source, count in sorted(source_counts.items()):
            f.write(f"  {source}: {count} scripts\n")

def main():
    """Main execution function"""
    
    # Define source directories
    source_directories = [
        'data/real_world',
        'data/real_world_consolidated'
    ]
    
    output_directory = 'data/all_real_world_scripts'
    
    print("🚀 Creating Unified Real-World LAMMPS Script Collection")
    print("=" * 55)
    
    # Step 1: Find all scripts
    print("\n📂 Step 1: Finding all LAMMPS scripts...")
    all_scripts = find_all_scripts(source_directories)
    print(f"Found {len(all_scripts)} total script files")
    
    # Step 2: Deduplicate
    print("\n🔍 Step 2: Deduplicating scripts...")
    unique_scripts, duplicates = deduplicate_scripts(all_scripts)
    print(f"Unique scripts: {len(unique_scripts)}")
    print(f"Duplicate groups: {len(duplicates)}")
    
    # Step 3: Create unified collection
    print("\n📋 Step 3: Creating unified collection...")
    metadata = create_unified_collection(unique_scripts, output_directory)
    print(f"Created {len(metadata)} unified script files")
    
    # Step 4: Save metadata
    print("\n💾 Step 4: Saving metadata...")
    save_metadata(metadata, duplicates, output_directory)
    
    print(f"\n✅ Unified collection created in: {output_directory}")
    print(f"📊 Check {output_directory}/collection_summary.txt for details")

if __name__ == "__main__":
    main() 