import os
import sys
import shutil
from typing import List

INIT_KEYWORDS = [
    "units", "atom_style", "dimension", "boundary", "lattice", "region",
    "create_box", "read_data", "read_restart", "reset_timestep"
]

RUN_KEYWORDS = [
    "run", "minimize", "fix", "timestep", "velocity", "thermo", "dump",
    "kspace_style"
]


def is_runnable_script(lines: List[str]) -> bool:
    """Return True if file looks like a top-level runnable LAMMPS input."""
    text = "\n".join(lines).lower()
    has_init = any(kw in text for kw in INIT_KEYWORDS)
    has_run = any(kw in text for kw in RUN_KEYWORDS)
    return has_init and has_run


def process_directory(target_dir: str, fragment_dir: str):
    if not os.path.exists(fragment_dir):
        os.makedirs(fragment_dir)

    moved = []
    for fname in os.listdir(target_dir):
        if not fname.endswith(".in"):
            continue
        fpath = os.path.join(target_dir, fname)
        with open(fpath, "r", errors="ignore") as f:
            lines = f.readlines()
        if not is_runnable_script(lines):
            # Move to fragment directory
            dest_path = os.path.join(fragment_dir, fname)
            shutil.move(fpath, dest_path)
            moved.append(fname)
    return moved


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python filter_runnable_inputs.py <cleaned_dir>")
        sys.exit(1)

    cleaned_dir = sys.argv[1]
    fragments_dir = os.path.join(cleaned_dir, "parameter_fragments")

    moved_files = process_directory(cleaned_dir, fragments_dir)
    if moved_files:
        print("Moved the following non-runnable fragment files to 'parameter_fragments':")
        for f in moved_files:
            print(f"  {f}")
    else:
        print("All scripts appear runnable; no files moved.") 