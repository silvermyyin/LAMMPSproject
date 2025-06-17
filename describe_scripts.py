import os
import json
import time
import argparse
from typing import Dict

from openai import OpenAI

client = OpenAI()  # uses OPENAI_API_KEY env var

MODEL = "gpt-4o"
SYS_PROMPT = "You are an expert in molecular dynamics simulations using LAMMPS."
USER_PREFIX = """Analyze the following LAMMPS input script and generate a concise, single-paragraph description of the simulation it performs.

The description must:
1.  Be a single, well-formed paragraph.
2.  Start with the exact phrase: "Generate a LAMMPS input script about"
3.  Summarize the key physical models, simulation setup, and goals.

**Excellent Example:**
"Generate a LAMMPS input script about a 2D granular flow simulation where spherical particles are poured into a rectangular container under gravity. The system uses Hertzian contact mechanics (`pair_style gran/hertz/history`) to model inter-particle and particle-wall interactions with simplified stiffness settings for quicker runtime. The domain is defined as a 2D box, and particles are inserted dynamically using the `fix pour` command within a specified slab region, with random diameters ranging from 0.5 to 1.0. Gravity acts downward at an angle of -180 degrees, while boundaries in the x and y directions are enforced using granular wall fixes. The simulation runs with a small timestep of 0.001, appropriate for resolving contact forces, and is integrated using `fix nve/sphere`. Thermodynamic output includes kinetic energy, rotational energy, and system volume, providing insight into the dynamics of granular material deposition."

**LAMMPS Script to Describe:**
"""
MAX_PROMPT_CHARS = 8000  # safeguard against very large inputs
SLEEP_BETWEEN_CALLS = 1.2  # seconds


def load_existing(path: str) -> Dict[str, str]:
    if os.path.exists(path):
        with open(path, "r") as f:
            return json.load(f)
    return {}


def save_json(path: str, data: Dict[str, str]):
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(data, f, indent=2)
    os.replace(tmp, path)


def save_csv(path: str, data: Dict[str, str]):
    with open(path, "w") as f:
        f.write("filename,description\n")
        for k, v in data.items():
            line = (
                k.replace("\"", "'")
                + ",\""
                + v.replace("\"", "'").replace("\n", " ")
                + "\"\n"
            )
            f.write(line)


def describe_script(content: str) -> str:
    trimmed = content[:MAX_PROMPT_CHARS]
    messages = [
        {"role": "system", "content": SYS_PROMPT},
        {"role": "user", "content": f"{USER_PREFIX}\n{trimmed}"},
    ]
    response = client.chat.completions.create(
        model=MODEL,
        messages=messages,
        temperature=0.2,
        max_tokens=300,
    )
    return response.choices[0].message.content.strip()


def main(directory: str, limit: int):
    descriptions_json = os.path.join(directory, "descriptions.json")
    descriptions_csv = os.path.join(directory, "descriptions.csv")

    done: Dict[str, str] = load_existing(descriptions_json)
    
    files = sorted(f for f in os.listdir(directory) if f.endswith(".in"))
    
    newly_done = 0
    for idx, fname in enumerate(files, 1):
        if fname in done:
            continue
        
        if limit is not None and newly_done >= limit:
            print(f"Reached limit of {limit} new descriptions.")
            break
            
        path = os.path.join(directory, fname)
        with open(path, "r", errors="ignore") as f:
            content = f.read()

        try:
            desc = describe_script(content)
        except Exception as e:
            print(f"Error generating description for {fname}: {e}. Retrying in 5s…")
            time.sleep(5)
            continue  # skip this iteration, will retry next run

        done[fname] = desc
        newly_done += 1
        print(f"[{idx}/{len(files)}] {fname} -> {desc[:60]}…")

        # periodic checkpoint
        if newly_done % 10 == 0:
            save_json(descriptions_json, done)
            save_csv(descriptions_csv, done)
            print("Checkpoint saved.")
        time.sleep(SLEEP_BETWEEN_CALLS)

    # final save
    save_json(descriptions_json, done)
    save_csv(descriptions_csv, done)
    print("All done. Descriptions written to JSON and CSV files.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate short descriptions for LAMMPS scripts via OpenAI")
    parser.add_argument("directory", help="Path to cleaned_real_world_scripts")
    parser.add_argument("--limit", type=int, default=None, help="Maximum number of new descriptions to generate")
    args = parser.parse_args()

    if not os.getenv("OPENAI_API_KEY"):
        print("OPENAI_API_KEY environment variable not set. Export it first.")
        exit(1)

    main(args.directory, args.limit) 