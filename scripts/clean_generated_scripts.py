import os
import glob
from tqdm import tqdm

def clean_all_scripts():
    """
    Removes markdown code block fences (```) from generated scripts.
    """
    # Directory containing the generated scripts to be cleaned.
    scripts_base_dir = "/Users/yinjunsheng/Desktop/FinalProject/results/experiments/dated_runs"

    print(f"Starting script cleaning in: {scripts_base_dir}")

    # Find all script files, searching recursively.
    script_files = glob.glob(os.path.join(scripts_base_dir, '**', '*.in'), recursive=True)

    if not script_files:
        print(f"Error: No '.in' script files found in {scripts_base_dir} to clean.")
        return

    print(f"Found {len(script_files)} scripts to process...")

    cleaned_count = 0
    # Loop through all found scripts and clean them.
    for script_path in tqdm(script_files, desc="Cleaning Scripts"):
        try:
            with open(script_path, 'r', encoding='utf-8') as f:
                lines = f.readlines()

            # Remove every line that contains triple backticks
            cleaned_lines = [ln for ln in lines if "```" not in ln]

            # If any line was removed, overwrite the file
            if len(cleaned_lines) != len(lines):
                with open(script_path, 'w', encoding='utf-8') as f:
                    f.writelines(cleaned_lines)
                cleaned_count += 1

        except Exception as e:
            print(f"Could not process file {script_path}: {e}")

    print(f"\nCleaning complete. {cleaned_count} scripts were modified.")

if __name__ == "__main__":
    clean_all_scripts() 