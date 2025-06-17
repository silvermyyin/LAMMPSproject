"""
Improved Prompt Templates for Natural Language → LAMMPS Generation Testing
"""

def extract_simulation_description(reference_script_path: str, reference_content: str) -> str:
    """
    Extract natural language description from reference script metadata and content
    """
    
    # Mapping of script patterns to natural language descriptions
    script_descriptions = {
        
        # ReaxFF Carbon Systems
        "reaxff.*carbon": """
        Create a molecular dynamics simulation for studying amorphous carbon nanoparticle formation.
        Requirements:
        - Use ReaxFF reactive force field for carbon chemistry
        - Set up a 70x70x70 cubic simulation box with periodic boundaries
        - Place approximately 1372 carbon atoms randomly in the box
        - Include Silicon, Oxygen, Carbon, and Fluorine atom types
        - Run at high temperature (6000K initial, cooling to 2500K) using NVT ensemble
        - Use appropriate timestep for reactive MD (0.1 fs)
        - Include charge equilibration for ReaxFF
        - Run for 25 million timesteps
        - Output thermodynamic properties and atomic trajectories
        """,
        
        # Rigid Body Systems  
        "rigid.*nve": """
        Set up a molecular dynamics simulation of rigid body dynamics.
        Requirements:
        - Use Lennard-Jones units and atomic style
        - Read initial configuration from data file
        - Create 9 rigid body groups (clumps) with 9 atoms each
        - Apply NVE ensemble with rigid body constraints
        - Use LJ/cut pair potential with 2.5 cutoff
        - Initialize velocities at temperature 100.0
        - Exclude intra-clump interactions
        - Run for 10,000 timesteps with 0.0001 timestep
        - Output thermodynamic data every 50 steps
        """,
        
        # NVT Equilibration
        "equilibrat": """
        Create an equilibration molecular dynamics simulation.
        Requirements:
        - Set up NVT ensemble for temperature equilibration
        - Use appropriate thermostat for the system
        - Run sufficient steps for equilibration
        - Monitor temperature and energy convergence
        - Output trajectory and thermodynamic data
        """,
        
        # Deposition Simulations
        "deposit": """
        Set up a molecular deposition simulation.
        Requirements:
        - Create substrate and depositing species
        - Define deposition region and parameters
        - Use appropriate ensemble (NVT or NVE)
        - Include surface interactions
        - Monitor deposition process through trajectory output
        """,
        
        # FEP (Free Energy Perturbation)
        "fep": """
        Create a free energy perturbation molecular dynamics simulation.
        Requirements:
        - Set up dual topology for FEP calculation
        - Define lambda coupling parameters
        - Use appropriate ensemble for free energy calculation
        - Include proper sampling and output for FEP analysis
        """,
        
        # General LJ systems
        "lj": """
        Set up a Lennard-Jones molecular dynamics simulation.
        Requirements:
        - Use LJ pair potential with appropriate cutoff
        - Define system box and particle configuration
        - Choose appropriate ensemble (NVE, NVT, or NPT)
        - Set reasonable timestep and run length
        - Include thermodynamic and trajectory output
        """
    }
    
    # Extract filename for pattern matching
    filename = reference_script_path.lower()
    
    # Find matching description
    for pattern, description in script_descriptions.items():
        if pattern in filename:
            return description.strip()
    
    # Default generic description
    return """
    Create a molecular dynamics simulation.
    Requirements:
    - Set up appropriate simulation box and boundary conditions
    - Define atom types and masses
    - Choose suitable force field and parameters
    - Select appropriate ensemble (NVE, NVT, or NPT)
    - Set reasonable timestep and run parameters
    - Include thermodynamic output and data saving
    """

def create_natural_language_prompt(reference_script_path: str, reference_content: str) -> str:
    """
    Create natural language prompt that tests true NL→LAMMPS generation
    """
    
    description = extract_simulation_description(reference_script_path, reference_content)
    
    prompt_template = f"""You are a LAMMPS expert. Generate a complete and valid LAMMPS input script based on the following natural language description:

SIMULATION REQUIREMENTS:
{description}

Please write a complete LAMMPS input script that fulfills these requirements. Include all necessary commands for:
- System setup (units, atom_style, boundary conditions)
- Force field definition (pair_style, pair_coeff)
- Simulation parameters (timestep, ensemble)
- Output settings (thermo, dump)
- Execution (run command)

Make sure the script is syntactically correct and physically reasonable."""

    return prompt_template

# Example usage for different script types
def demonstrate_improved_prompts():
    """
    Show examples of improved prompts vs old prompts
    """
    
    examples = {
        "ReaxFF Carbon": create_natural_language_prompt(
            "in.amorphous.carbon.lammps", ""
        ),
        
        "Rigid Body": create_natural_language_prompt(
            "in.rigid.nve", ""
        ),
        
        "LJ System": create_natural_language_prompt(
            "in.lj.melt", ""
        )
    }
    
    return examples

if __name__ == "__main__":
    examples = demonstrate_improved_prompts()
    for sim_type, prompt in examples.items():
        print(f"\n{'='*50}")
        print(f"SIMULATION TYPE: {sim_type}")
        print(f"{'='*50}")
        print(prompt)
        print("\n") 