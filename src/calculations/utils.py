import os
import json
import openai
from openai import OpenAI

class LLMInterface:
    def __init__(self, config_path: str = "configs/model_configs.json"):
        """Initialize the LLM interface with configuration."""
        with open(config_path, 'r') as f:
            self.config = json.load(f)
        
        api_key = os.getenv("OPENAI_API_KEY")
        if api_key is None or api_key.strip() == "":
            raise ValueError("OPENAI_API_KEY environment variable not set.")

        self.client = OpenAI(api_key=api_key)
        
    def call_llm(self, prompt: str, model: str = "gpt-4", system_prompt: str = None) -> str:
        """Call the LLM to generate LAMMPS code."""
        response = self.client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system_prompt or "You are a helpful assistant."},
                {"role": "user", "content": prompt}
            ]
        )
        return response.choices[0].message.content 