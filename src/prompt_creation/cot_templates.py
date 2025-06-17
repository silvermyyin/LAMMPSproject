# Chain-of-Thought (CoT) Prompt Templates for LAMMPS

COT_TEMPLATES = {
    'step_by_step': (
        "You are a LAMMPS expert assistant. Given the following task, reason step by step and generate a valid LAMMPS input script.\n"
        "Task: {task}\n"
        "Reasoning: Let's think step by step.\n"
    ),
    'reasoning_first': (
        "Analyze the requirements and explain your reasoning before generating the LAMMPS input script.\n"
        "Task: {task}\n"
        "Reasoning:"
    ),
    'action_first': (
        "Given the following task, directly generate the LAMMPS input script, then explain your reasoning.\n"
        "Task: {task}\n"
        "Script:"
    ),
    'expert_consultation': (
        "You are a LAMMPS domain expert. For the following task, first list all relevant considerations, then provide the best input script.\n"
        "Task: {task}\n"
        "Considerations:"
    ),
    'error_analysis': (
        "Given the following LAMMPS task, generate a script and explain possible sources of error or failure.\n"
        "Task: {task}\n"
        "Script:"
        "\nError Analysis:"
    ),
    'minimal_script': (
        "Generate the minimal valid LAMMPS input script for the following task, with no extra comments.\n"
        "Task: {task}\n"
        "Script:"
    ),
    'explanation_only': (
        "Explain in detail how you would approach the following LAMMPS task, but do not generate the script.\n"
        "Task: {task}\n"
        "Explanation:"
    ),
    'compare_and_choose': (
        "For the following LAMMPS task, list at least two possible approaches, compare them, and recommend the best one.\n"
        "Task: {task}\n"
        "Approach 1: \nApproach 2: \nComparison: \nRecommendation:"
    ),
    'multi_turn': (
        "This is a multi-turn LAMMPS assistant. First, ask clarifying questions if needed, then generate the script.\n"
        "Task: {task}\n"
        "Questions: \nScript:"
    ),
}

def generate_prompt(template_name, task):
    """Generate a prompt using the specified template and task description."""
    template = COT_TEMPLATES.get(template_name, COT_TEMPLATES['step_by_step'])
    return template.format(task=task) 