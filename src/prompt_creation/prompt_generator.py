from cot_templates import generate_prompt

def basic_prompt(task):
    return f"Generate a LAMMPS input script for the following task: {task}"

def cot_prompt(task, style='step_by_step'):
    return generate_prompt(style, task)

# Example usage:
if __name__ == '__main__':
    task = "Simulate an NVT ensemble of argon atoms at 300K using Lennard-Jones potential."
    print(basic_prompt(task))
    print(cot_prompt(task, 'step_by_step')) 