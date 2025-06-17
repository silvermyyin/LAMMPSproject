import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

from core.script_manager import ScriptManager

def run_baseline_experiment():
    # Initialize script manager
    manager = ScriptManager()
    
    # Create a new experiment run
    run_dir = manager.create_experiment_run(experiment_type="baseline")
    print(f"Created experiment run directory: {run_dir}")
    
    # Example LAMMPS script
    script_content = """
# LAMMPS input script for simple MD simulation
units           lj
atom_style      atomic
boundary        p p p

# Create simple cubic lattice
lattice         sc 1.0
region          box block 0 10 0 10 0 10
create_box      1 box
create_atoms    1 box

# Define pair potential
pair_style      lj/cut 2.5
pair_coeff      1 1 1.0 1.0 2.5

# Setup MD
velocity        all create 1.0 12345
fix             1 all nve
timestep        0.005

# Run simulation
thermo          100
run             1000
"""
    
    # Prepare and run the script
    script_path = manager.prepare_script(script_content, "simple_md.in")
    success, message = manager.run_script(script_path)
    
    print(f"Script execution: {message}")
    
    # Cleanup
    manager.cleanup()

if __name__ == "__main__":
    run_baseline_experiment() 