#!/usr/bin/env python3
"""
Real ChatGPT Integration for LAMMPS Script Generation
Uses actual OpenAI API to generate scripts from natural language prompts.
"""

import os
import json
import time
from pathlib import Path
from openai import OpenAI
import asyncio
import aiohttp
import logging

class RealChatGPTGenerator:
    def __init__(self, api_key=None, model="gpt-4o-mini"):
        """Initialize with OpenAI API key."""
        self.api_key = api_key or os.getenv('OPENAI_API_KEY')
        self.model = model
        self.generated_scripts = []
        
        if not self.api_key:
            print("⚠️  No OpenAI API key provided!")
            print("   Set OPENAI_API_KEY environment variable or pass api_key parameter")
            print("   Example: export OPENAI_API_KEY='your-key-here'")
            self.client = None
        else:
            self.client = OpenAI(api_key=self.api_key)
            print(f"✅ OpenAI API configured with model: {self.model}")
    
    def generate_script_with_chatgpt(self, prompt, script_id, max_retries=3):
        """Generate a LAMMPS script using real ChatGPT API."""
        
        if not self.client:
            return self._fallback_generation(prompt, script_id)
        
        system_prompt = """You are a LAMMPS molecular dynamics expert. Generate complete, syntactically correct LAMMPS input scripts based on natural language descriptions. 

Requirements:
- Include ALL necessary commands for a complete simulation
- Use proper LAMMPS syntax and command ordering
- Include realistic and stable parameters
- Add brief comments explaining key sections
- Ensure the script is executable and physically stable
- Use appropriate simulation box sizes and particle densities
- Choose conservative timesteps to avoid instabilities

Return ONLY the LAMMPS script content, no additional explanation."""

        for attempt in range(max_retries):
            try:
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": prompt}
                    ],
                    max_tokens=2000,
                    temperature=0.3  # Lower temperature for more consistent code generation
                )
                
                generated_content = response.choices[0].message.content.strip()
                
                # Clean up the response (remove markdown formatting if present)
                if generated_content.startswith("```"):
                    lines = generated_content.split('\n')
                    # Remove first line (```lammps or ```) and last line (```)
                    start_idx = 1
                    end_idx = -1
                    if lines[-1].strip() == "```":
                        end_idx = -1
                    else:
                        end_idx = len(lines)
                    generated_content = '\n'.join(lines[start_idx:end_idx])
                
                return {
                    'script_id': script_id,
                    'prompt': prompt,
                    'generated_content': generated_content,
                    'generation_timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
                    'success': True,
                    'model_used': self.model,
                    'api_response': {
                        'usage': response.usage.model_dump() if response.usage else {},
                        'attempt': attempt + 1
                    }
                }
                
            except Exception as e:
                error_msg = str(e)
                print(f"   API error on attempt {attempt + 1}: {error_msg}")
                
                if "rate_limit" in error_msg.lower():
                    wait_time = (attempt + 1) * 10
                    print(f"   Rate limit hit, waiting {wait_time}s...")
                    time.sleep(wait_time)
                elif attempt == max_retries - 1:
                    return self._fallback_generation(prompt, script_id, error=error_msg)
                else:
                    time.sleep(5)
        
        return self._fallback_generation(prompt, script_id, error="Max retries exceeded")
    
    def _fallback_generation(self, prompt, script_id, error=None):
        """Fallback generation when API is not available."""
        
        # Generate a reasonable LAMMPS script based on prompt analysis
        fallback_script = self._analyze_prompt_and_generate(prompt)
        
        return {
            'script_id': script_id,
            'prompt': prompt,
            'generated_content': fallback_script,
            'generation_timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
            'success': False,
            'model_used': 'fallback',
            'error': error or "No API key provided",
            'note': "Generated using fallback method"
        }
    
    def _analyze_prompt_and_generate(self, prompt):
        """Analyze prompt and generate appropriate LAMMPS script."""
        
        prompt_lower = prompt.lower()
        
        # Determine simulation type from prompt
        if 'water' in prompt_lower or 'spce' in prompt_lower:
            return self._generate_water_script()
        elif 'metal' in prompt_lower or 'eam' in prompt_lower:
            return self._generate_metal_script()
        elif 'polymer' in prompt_lower or 'chain' in prompt_lower:
            return self._generate_polymer_script()
        elif 'minimize' in prompt_lower or 'minimization' in prompt_lower:
            return self._generate_minimization_script()
        else:
            return self._generate_generic_lj_script()
    
    def _generate_water_script(self):
        """Generate a water simulation script."""
        return """# Water simulation using SPC/E model
units real
dimension 3
boundary p p p
atom_style full

# Read water configuration
region box block -20 20 -20 20 -20 20
create_box 2 box

# Create water molecules
molecule water water.txt
create_atoms 0 random 500 12345 box mol water 25634

# Set masses
mass 1 15.999  # Oxygen
mass 2 1.008   # Hydrogen

# SPC/E water potential
pair_style lj/cut/coul/long 10.0
pair_coeff 1 1 0.1553 3.166
pair_coeff 2 2 0.0000 0.0000
pair_coeff 1 2 0.0000 0.0000

# Bonds and angles
bond_style harmonic
bond_coeff 1 553.0 1.0

angle_style harmonic
angle_coeff 1 100.0 109.47

# Long-range electrostatics
kspace_style pppm 1e-4

# Set charges
set type 1 charge -0.8476
set type 2 charge 0.4238

# Thermodynamics
thermo 100
thermo_style custom step temp pe ke etotal press vol

# Initialize velocities
velocity all create 300.0 12345

# NVT dynamics
fix 1 all nvt temp 300.0 300.0 100.0

# Run simulation
timestep 1.0
run 50000

write_data final_water.data"""

    def _generate_metal_script(self):
        """Generate a metal simulation script."""
        return """# Metal simulation using EAM potential
units metal
dimension 3
boundary p p p
atom_style atomic

# Create FCC lattice
lattice fcc 3.615
region box block 0 10 0 10 0 10
create_box 1 box
create_atoms 1 box

# Set mass
mass 1 63.546  # Copper

# EAM potential
pair_style eam/alloy
pair_coeff * * Cu_u3.eam Cu

# Thermodynamics
thermo 50
thermo_style custom step temp pe ke etotal press vol

# Initialize velocities
velocity all create 300.0 12345

# NPT dynamics
fix 1 all npt temp 300.0 300.0 0.1 iso 0.0 0.0 1.0

# Run simulation
timestep 0.001
run 10000

write_data final_metal.data"""

    def _generate_polymer_script(self):
        """Generate a polymer simulation script."""
        return """# Polymer chain simulation
units lj
dimension 3
boundary p p p
atom_style bond

# Create simulation box
region box block -20 20 -20 20 -20 20
create_box 1 box

# Create polymer chain
create_atoms 1 random 1000 12345 box

# Set mass
mass 1 1.0

# LJ potential
pair_style lj/cut 2.5
pair_coeff 1 1 1.0 1.0 2.5

# Bond potential
bond_style harmonic
bond_coeff 1 30.0 1.5

# Create bonds randomly
create_bonds many all all 1 1.0 1.8

# Thermodynamics
thermo 100
thermo_style custom step temp pe ke etotal press

# Initialize velocities
velocity all create 1.0 12345

# NVT dynamics
fix 1 all nvt temp 1.0 1.0 0.1

# Run simulation
timestep 0.005
run 20000

write_data final_polymer.data"""

    def _generate_minimization_script(self):
        """Generate an energy minimization script."""
        return """# Energy minimization
units lj
dimension 3
boundary p p p
atom_style atomic

# Read initial configuration
read_data initial.data

# Set mass
mass * 1.0

# LJ potential
pair_style lj/cut 2.5
pair_coeff * * 1.0 1.0 2.5

# Thermodynamics
thermo 10
thermo_style custom step pe fnorm

# Energy minimization
minimize 1e-6 1e-8 1000 10000

# Write minimized structure
write_data minimized.data"""

    def _generate_generic_lj_script(self):
        """Generate a generic Lennard-Jones script."""
        return """# Generic Lennard-Jones simulation
units lj
dimension 3
boundary p p p
atom_style atomic

# Create simulation box
region box block -10 10 -10 10 -10 10
create_box 1 box
create_atoms 1 random 1000 12345 box

# Set mass
mass 1 1.0

# LJ potential
pair_style lj/cut 2.5
pair_coeff 1 1 1.0 1.0 2.5

# Initialize velocities
velocity all create 1.0 12345

# Thermodynamics
thermo 100
thermo_style custom step temp pe ke etotal press

# NVE dynamics
fix 1 all nve

# Run simulation
timestep 0.005
run 10000

write_data final.data"""

    def generate_batch_scripts(self, prompts_file, max_scripts=None, batch_size=5):
        """Generate scripts for multiple prompts."""
        
        print("🚀 REAL CHATGPT SCRIPT GENERATION")
        print("=" * 60)
        
        # Load prompts
        with open(prompts_file, 'r') as f:
            data = json.load(f)
        
        prompts = data['prompts']
        if max_scripts:
            prompts = prompts[:max_scripts]
        
        print(f"📊 Generating {len(prompts)} scripts using ChatGPT...")
        
        if self.client:
            print("✅ Using real ChatGPT API")
            print(f"⚡ Model: {self.model}")
            print(f"🔄 Batch size: {batch_size} (for rate limit management)")
        else:
            print("⚠️  Using fallback generation (no API key)")
        
        # Generate scripts
        generated_scripts = []
        total_cost = 0
        successful_api_calls = 0
        
        start_time = time.time()
        
        for i, prompt_data in enumerate(prompts):
            # Progress reporting
            if (i + 1) % 50 == 0:
                elapsed = time.time() - start_time
                rate = (i + 1) / elapsed * 60  # scripts per minute
                remaining = len(prompts) - (i + 1)
                eta_minutes = remaining / rate if rate > 0 else 0
                
                print(f"   Progress: {i + 1}/{len(prompts)} scripts ({(i+1)/len(prompts)*100:.1f}%)")
                print(f"   Rate: {rate:.1f} scripts/minute, ETA: {eta_minutes:.0f} minutes")
                print(f"   API Success: {successful_api_calls}/{i+1} ({successful_api_calls/(i+1)*100:.1f}%)")
            
            result = self.generate_script_with_chatgpt(
                prompt_data['prompt'],
                prompt_data['script_id']
            )
            
            generated_scripts.append(result)
            
            if result['success']:
                successful_api_calls += 1
            
            # Estimate cost (rough)
            if result.get('api_response') and 'usage' in result['api_response']:
                usage = result['api_response']['usage']
                if usage and 'total_tokens' in usage:
                    tokens = usage['total_tokens']
                    # GPT-4o-mini pricing: $0.00015 per 1K input tokens, $0.0006 per 1K output tokens
                    cost = tokens * 0.0003  # Average estimate
                    total_cost += cost
            
            # Rate limiting - be conservative
            if self.client and (i + 1) % batch_size == 0:
                print(f"   Completed batch {(i+1)//batch_size}, pausing 2s...")
                time.sleep(2)  # Conservative pause between batches
        
        self.generated_scripts = generated_scripts
        
        final_elapsed = time.time() - start_time
        
        print(f"\n✅ Generation complete!")
        print(f"⏱️  Total time: {final_elapsed/60:.1f} minutes")
        print(f"🎯 Success rate: {successful_api_calls}/{len(prompts)} ({successful_api_calls/len(prompts)*100:.1f}%)")
        if total_cost > 0:
            print(f"💰 Estimated cost: ${total_cost:.2f}")
        
        return generated_scripts
    
    def save_results(self, output_file="results/real_chatgpt_generated_scripts.json"):
        """Save generated scripts to file."""
        
        output_data = {
            'metadata': {
                'total_generated': len(self.generated_scripts),
                'successful_api_calls': sum(1 for s in self.generated_scripts if s['success']),
                'model_used': self.model,
                'generation_timestamp': time.strftime('%Y-%m-%d %H:%M:%S')
            },
            'generated_scripts': self.generated_scripts
        }
        
        with open(output_file, 'w') as f:
            json.dump(output_data, f, indent=2)
        
        print(f"💾 Results saved to: {output_file}")
        return output_file

def main():
    """Main execution function."""
    
    # Check for API key
    api_key = os.getenv('OPENAI_API_KEY')
    if not api_key:
        print("🔑 OpenAI API Key Setup Required!")
        print("=" * 40)
        print("To use real ChatGPT generation:")
        print("1. Get an API key from https://platform.openai.com/")
        print("2. Set environment variable: export OPENAI_API_KEY='your-key'")
        print("3. Re-run this script")
        print()
        print("Running with fallback generation for now...")
        print()
    
    # Initialize generator
    generator = RealChatGPTGenerator(api_key=api_key)
    
    # Generate scripts - process ALL 1090 scripts!
    try:
        print("🚀 STARTING REAL CHATGPT GENERATION FOR ALL 1090 SCRIPTS!")
        print("=" * 70)
        print("This is the REAL baseline experiment!")
        print("Estimated time: 1-2 hours depending on API rate limits")
        print("Cost estimate: $50-100 depending on script complexity")
        print()
        
        scripts = generator.generate_batch_scripts(
            "results/all_extracted_prompts.json",
            max_scripts=None,  # Process ALL scripts!
            batch_size=5      # Smaller batches to respect rate limits
        )
        
        # Save results
        generator.save_results()
        
        print(f"\n🎯 REAL GENERATION COMPLETE!")
        print(f"   Generated {len(scripts)} scripts using real ChatGPT API")
        print(f"   Success rate: {sum(1 for s in scripts if s['success']) / len(scripts) * 100:.1f}%")
        
    except FileNotFoundError:
        print("❌ Prompts file not found!")
        print("   Run prompt extraction first.")
    except Exception as e:
        print(f"❌ Generation error: {e}")

if __name__ == "__main__":
    main() 