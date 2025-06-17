import os
import random
import json

RULE_BASED_DIR = "data/train/rule_based"
METADATA_FILE = os.path.join(RULE_BASED_DIR, "metadata.json")

# Predefined LAMMPS templates with placeholders for dynamic values
LAMMPS_TEMPLATES = [
    """# Metal system simulation
units metal
atom_style atomic
boundary {boundary}
lattice {lattice} {lattice_param}
region box block 0 {xmax} 0 {ymax} 0 {zmax}
create_box 1 box
create_atoms 1 box
pair_style eam
pair_coeff * * Cu_u3.eam
fix 1 all nvt temp {temp_start} {temp_end} {damp}
thermo {thermo_freq}
timestep {timestep}
run {num_steps}
""",
    """# Polymer simulation
units real
atom_style full
boundary {boundary}
read_data polymer.data
pair_style lj/cut {pair_cutoff}
special_bonds lj 0.0 0.0 0.5
fix 1 all npt temp {temp_start} {temp_end} {damp} iso 1.0 1.0 1000
thermo {thermo_freq}
timestep {timestep}
run {num_steps}
""",
    """# Water simulation using SPC/E model
units real
atom_style full
boundary {boundary}
read_data water.data
pair_style lj/cut {pair_cutoff}
pair_coeff 1 1 0.1553 3.166
pair_coeff 2 2 0.046 3.02
fix 1 all nvt temp {temp_start} {temp_end} {damp}
thermo {thermo_freq}
timestep {timestep}
run {num_steps}
""",
    """# Gas phase simulation
units real
atom_style atomic
boundary {boundary}
region box block 0 {xmax} 0 {ymax} 0 {zmax}
create_box 1 box
create_atoms 1 box
pair_style lj/cut {pair_cutoff}
velocity all create {temp_start} 12345
fix 1 all nve
thermo {thermo_freq}
timestep {timestep}
run {num_steps}
"""
]

# Define possible values for dynamic placeholders
PARAMETER_RANGES = {
    "boundary": ["p p p", "f f f", "p p f"],
    "lattice": ["fcc", "bcc", "sc"],
    "lattice_param": [3.52, 2.85, 4.0],
    "xmax": [10, 20, 30],
    "ymax": [10, 20, 30],
    "zmax": [10, 20, 30],
    "temp_start": [300, 500, 700],
    "temp_end": [300, 600, 800],
    "damp": [0.1, 0.5, 1.0],
    "thermo_freq": [100, 500, 1000],
    "timestep": [0.5, 1.0, 2.0],
    "num_steps": [5000, 10000, 20000],
    "pair_cutoff": [10.0, 12.0, 15.0]
}

def generate_random_params():
    """ Generate random parameters for LAMMPS script """
    return {key: random.choice(values) for key, values in PARAMETER_RANGES.items()}

def generate_rule_based_lammps(num_samples=10):
    """ Generate rule-based LAMMPS input scripts with dynamic parameters """
    os.makedirs(RULE_BASED_DIR, exist_ok=True)
    metadata = []

    for i in range(num_samples):
        template = random.choice(LAMMPS_TEMPLATES)
        params = generate_random_params()
        lammps_script = template.format(**params)

        filename = os.path.join(RULE_BASED_DIR, f"sample_{i+1}.in")
        with open(filename, "w") as f:
            f.write(lammps_script)

        # Save metadata
        metadata.append({"file": f"sample_{i+1}.in", "params": params})

        print(f"Generated {filename}")

    # Save metadata as JSON for tracking
    with open(METADATA_FILE, "w") as f:
        json.dump(metadata, f, indent=4)
    print(f"Metadata saved at {METADATA_FILE}")

if __name__ == "__main__":
    num_samples = int(input("Enter the number of rule-based samples to generate: "))
    generate_rule_based_lammps(num_samples)
