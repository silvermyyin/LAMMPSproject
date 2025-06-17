import os
import json
from typing import List, Dict, Any
import openai
from openai import OpenAI

class LLMInterface:
    def __init__(self, config_path: str = "configs/model_configs.json"):
        """Initialize the LLM interface with configuration."""
        with open(config_path, 'r') as f:
            self.config = json.load(f)
        
        # Read API key from environment variable, fall back to default placeholder if provided
        api_key = os.getenv("OPENAI_API_KEY")
        if api_key is None or api_key.strip() == "":
            raise ValueError(
                "OPENAI_API_KEY environment variable is not set. Export it before running baseline:\n\n   export OPENAI_API_KEY=\"your_api_key_here\"\n"
            )

        # Initialize OpenAI client
        self.client = OpenAI(api_key=api_key)
        
    def call_llm(self, prompt: str, model: str = "gpt-4", system_prompt: str = None) -> str:
        """
        Call the LLM to generate LAMMPS code.
        
        Args:
            prompt (str): The user prompt
            model (str): The model to use
            system_prompt (str): Optional custom system prompt
            
        Returns:
            str: Generated LAMMPS script
        """
        try:
            # Use default system prompt if none provided
            if system_prompt is None:
                system_prompt = "You are a LAMMPS expert. Generate valid LAMMPS input scripts based on the given requirements."
            
            response = self.client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.7,
                max_tokens=1000
            )
            return response.choices[0].message.content
        except Exception as e:
            print(f"Error calling LLM: {str(e)}")
            return ""
    
    def batch_call_llm(self, prompts: List[str], model: str = "gpt-4", system_prompt: str = None) -> List[str]:
        """
        Call the LLM for a batch of prompts.
        
        Args:
            prompts (List[str]): List of user prompts
            model (str): The model to use
            system_prompt (str): Optional custom system prompt
            
        Returns:
            List[str]: List of generated LAMMPS scripts
        """
        return [self.call_llm(prompt, model, system_prompt) for prompt in prompts] 