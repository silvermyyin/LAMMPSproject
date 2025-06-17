import os
import shutil
from datetime import datetime
from pathlib import Path
import subprocess
import logging

class ScriptManager:
    def __init__(self, base_dir: str = "LAMMPSrun"):
        """
        Initialize the script manager.
        
        Args:
            base_dir: Base directory for LAMMPS runs
        """
        self.base_dir = Path(base_dir)
        self.experiments_dir = self.base_dir / "experiments"
        self.current_run_dir = None
        
        # Setup logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger(__name__)
        
    def create_experiment_run(self, experiment_type: str = "baseline") -> Path:
        """
        Create a new experiment run directory.
        
        Args:
            experiment_type: Type of experiment (e.g., 'baseline', 'fine_tuning')
            
        Returns:
            Path to the created run directory
        """
        # Create timestamp for the run
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Create experiment directory structure
        experiment_dir = self.experiments_dir / experiment_type
        run_dir = experiment_dir / timestamp
        
        # Create directories
        run_dir.mkdir(parents=True, exist_ok=True)
        
        self.current_run_dir = run_dir
        self.logger.info(f"Created experiment run directory: {run_dir}")
        
        return run_dir
    
    def prepare_script(self, script_content: str, script_name: str = None) -> Path:
        """
        Prepare a LAMMPS script for execution.
        
        Args:
            script_content: Content of the LAMMPS script
            script_name: Name of the script file (optional)
            
        Returns:
            Path to the prepared script
        """
        if not self.current_run_dir:
            raise RuntimeError("No experiment run directory created. Call create_experiment_run first.")
            
        # Generate script name if not provided
        if script_name is None:
            script_name = f"script_{datetime.now().strftime('%Y%m%d_%H%M%S')}.in"
            
        # Write script to file
        script_path = self.current_run_dir / script_name
        with open(script_path, 'w') as f:
            f.write(script_content)
            
        self.logger.info(f"Prepared script: {script_path}")
        return script_path
    
    def run_script(self, script_path: Path, output_name: str = None) -> tuple[bool, str]:
        """
        Run a LAMMPS script.
        
        Args:
            script_path: Path to the LAMMPS script
            output_name: Name for the output file (optional)
            
        Returns:
            Tuple of (success, output_message)
        """
        if not script_path.exists():
            raise FileNotFoundError(f"Script not found: {script_path}")
            
        # Generate output name if not provided
        if output_name is None:
            output_name = f"output_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
            
        output_path = self.current_run_dir / output_name
        
        try:
            # Run LAMMPS
            result = subprocess.run(
                ['lmp', '-in', str(script_path)],
                capture_output=True,
                text=True,
                cwd=str(self.current_run_dir)
            )
            
            # Write output to file
            with open(output_path, 'w') as f:
                f.write(result.stdout)
                if result.stderr:
                    f.write("\nErrors:\n")
                    f.write(result.stderr)
            
            success = result.returncode == 0
            message = "Script executed successfully" if success else "Script execution failed"
            
            self.logger.info(f"{message}: {script_path}")
            return success, message
            
        except Exception as e:
            self.logger.error(f"Error running script {script_path}: {str(e)}")
            return False, str(e)
    
    def cleanup(self):
        """Clean up temporary files and directories."""
        if self.current_run_dir:
            # Keep only essential files
            for item in self.current_run_dir.glob("*"):
                if item.is_file() and not (item.name.endswith('.in') or item.name.endswith('.log')):
                    item.unlink()
            self.logger.info(f"Cleaned up run directory: {self.current_run_dir}") 