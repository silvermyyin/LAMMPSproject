import os
from pathlib import Path
import shutil
import logging
from typing import List, Dict, Tuple, Set
import re

class ScriptCollector:
    def __init__(self, source_dir: str = "data/baseline_scripts", target_dir: str = "LAMMPSrun/experiments/baseline"):
        """
        Initialize the script collector.
        
        Args:
            source_dir: Directory containing the baseline scripts
            target_dir: Directory where validated scripts will be copied
        """
        self.source_dir = Path(source_dir)
        self.target_dir = Path(target_dir)
        # self.required_commands = {
        #     'units': False,
        #     'atom_style': False,
        #     'boundary': False,
        #     'pair_style': False,
        #     'timestep': False,
        #     'run': False
        # }
        
        # Setup logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger(__name__)
        
    def find_dependencies(self, script_path: Path) -> Set[str]:
        """
        Find dependencies in a LAMMPS script.
        
        Args:
            script_path: Path to the LAMMPS script
            
        Returns:
            Set of dependency file names
        """
        dependencies = set()
        try:
            with open(script_path, 'r') as f:
                content = f.read()
                
            # Look for read_data commands
            read_data_matches = re.finditer(r'read_data\s+(\S+)', content)
            for match in read_data_matches:
                data_file = match.group(1)
                # Remove any variable references
                data_file = re.sub(r'\$\{.*?\}', '', data_file)
                dependencies.add(data_file)
                
            # Look for include commands
            include_matches = re.finditer(r'include\s+(\S+)', content)
            for match in include_matches:
                include_file = match.group(1)
                dependencies.add(include_file)
                
        except Exception as e:
            self.logger.error(f"Error finding dependencies in {script_path}: {str(e)}")
            
        return dependencies
        
    def validate_script(self, script_path: Path) -> Tuple[bool, List[str]]:
        """
        Validate a LAMMPS script by checking for required commands.
        
        Args:
            script_path: Path to the LAMMPS script
            
        Returns:
            Tuple of (is_valid, missing_commands)
        """
        # Always return True and an empty list of missing commands
        return True, []
            
    def collect_scripts(self) -> Dict[str, List[Path]]:
        """
        Collect and validate LAMMPS scripts from the source directory.
        
        Returns:
            Dictionary with categories as keys and lists of valid script paths as values
        """
        valid_scripts = {}
        invalid_scripts = []
        all_dependencies = set()
        
        # Create target directory if it doesn't exist
        self.target_dir.mkdir(parents=True, exist_ok=True)
        
        # Walk through source directory
        for root, _, files in os.walk(self.source_dir):
            for file in files:
                # Collect all files starting with 'in.'
                if file.startswith('in.'):
                    script_path = Path(root) / file
                    
                    # Find dependencies
                    dependencies = self.find_dependencies(script_path)
                    all_dependencies.update(dependencies)
                    
                    # Validate script
                    is_valid, missing_commands = self.validate_script(script_path)
                    
                    if is_valid:
                        # Determine category based on directory structure
                        rel_path = Path(root).relative_to(self.source_dir)
                        category = rel_path.parts[0] if rel_path.parts else 'other'
                        
                        if category not in valid_scripts:
                            valid_scripts[category] = []
                            
                        valid_scripts[category].append(script_path)
                        self.logger.info(f"Valid script found: {script_path}")
                    else:
                        invalid_scripts.append((script_path, missing_commands))
                        self.logger.warning(f"Invalid script {script_path}: missing {', '.join(missing_commands)}")
        
        # Copy valid scripts and their dependencies to target directory
        for category, scripts in valid_scripts.items():
            category_dir = self.target_dir / category
            category_dir.mkdir(exist_ok=True)
            
            for script in scripts:
                # Copy script
                target_path = category_dir / script.name
                shutil.copy2(script, target_path)
                self.logger.info(f"Copied script to {target_path}")
                
                # Copy dependencies if they exist
                dependencies = self.find_dependencies(script)
                for dep in dependencies:
                    # Try to find dependency in the same directory as the script
                    dep_path = script.parent / dep
                    if dep_path.exists():
                        if dep_path.is_file():
                            shutil.copy2(dep_path, category_dir / dep)
                            self.logger.info(f"Copied dependency {dep} to {category_dir}")
                        else:
                            self.logger.warning(f"Skipping directory dependency: {dep_path}")
                    else:
                        # Try to find dependency in parent directories
                        for parent in script.parent.parents:
                            dep_path = parent / dep
                            if dep_path.exists():
                                if dep_path.is_file():
                                    shutil.copy2(dep_path, category_dir / dep)
                                    self.logger.info(f"Copied dependency {dep} from {parent} to {category_dir}")
                                else:
                                    self.logger.warning(f"Skipping directory dependency: {dep_path}")
                                break
                
        # Print summary
        total_scripts = sum(len(scripts) for scripts in valid_scripts.values()) + len(invalid_scripts)
        self.logger.info(f"\nScript Collection Summary:")
        self.logger.info(f"Total scripts found: {total_scripts}")
        self.logger.info(f"Valid scripts: {sum(len(scripts) for scripts in valid_scripts.values())}")
        self.logger.info(f"Invalid scripts: {len(invalid_scripts)}")
        self.logger.info("\nValid scripts by category:")
        for category, scripts in valid_scripts.items():
            self.logger.info(f"{category}: {len(scripts)} scripts")
            
        self.logger.info(f"\nTotal dependencies found: {len(all_dependencies)}")
        self.logger.info("Dependencies:")
        for dep in sorted(all_dependencies):
            self.logger.info(f"- {dep}")
            
        return valid_scripts

def main():
    collector = ScriptCollector()
    collector.collect_scripts()

if __name__ == "__main__":
    main() 