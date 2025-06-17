import os
import subprocess
import glob
from tqdm import tqdm

def run_all_simulations():
    """
    Runs all generated LAMMPS scripts and saves their log files.
    """
    # --- Configuration ---
    # Path to the new, powerful LAMMPS executable we just compiled.
    lammps_executable = "/Users/yinjunsheng/lammps/src/lmp_mpi"
    
    # Directory containing the generated scripts to be run.
    scripts_base_dir = "/Users/yinjunsheng/Desktop/FinalProject/results/experiments/baseline_dated_runs"
    
    # Directory where all the output log files will be saved.
    log_output_dir = "/Users/yinjunsheng/Desktop/FinalProject/results/be_description_generated_logs"
    
    # --- Script Execution ---
    
    print(f"LAMMPS executable: {lammps_executable}")
    print(f"Scripts directory: {scripts_base_dir}")
    print(f"Log output directory: {log_output_dir}")

    # Ensure the LAMMPS executable exists and is executable
    if not (os.path.exists(lammps_executable) and os.access(lammps_executable, os.X_OK)):
        print(f"Error: LAMMPS executable not found or not executable at {lammps_executable}")
        print("Please double-check the path and file permissions.")
        return

    # Find all script files, searching recursively in all subdirectories
    script_files = glob.glob(os.path.join(scripts_base_dir, '**', '*.in'), recursive=True)

    if not script_files:
        print(f"Error: No '.in' script files found in {scripts_base_dir} or its subdirectories.")
        return

    print(f"Found {len(script_files)} scripts to run. Starting simulations...")

    # Loop through all found scripts and run them
    for script_path in tqdm(script_files, desc="Running LAMMPS Scripts"):
        # Create a clean and informative log file name based on the script name
        script_filename = os.path.basename(script_path)
        log_filename = os.path.splitext(script_filename)[0] + '.log'
        log_path = os.path.join(log_output_dir, log_filename)

        # Construct the full command to run the simulation
        # The -log flag tells LAMMPS where to save the log file.
        command = [lammps_executable, "-in", script_path, "-log", log_path]
        
        # Execute the command
        # We capture stdout and stderr to prevent them from flooding the console,
        # but they can be printed for debugging if a run fails.
        subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

    print("\nAll simulations have been completed.")
    print(f"All log files have been saved to: {log_output_dir}")

if __name__ == "__main__":
    run_all_simulations() 