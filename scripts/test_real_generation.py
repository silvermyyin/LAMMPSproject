#!/usr/bin/env python3
"""
Test script for real ChatGPT generation - small batch first
"""

import os
import sys
sys.path.append('.')
from scripts.real_chatgpt_generation import RealChatGPTGenerator

def main():
    """Test with just 10 scripts first."""
    
    print("🧪 TESTING REAL CHATGPT GENERATION")
    print("=" * 50)
    print("Testing with 10 scripts first to verify everything works")
    print()
    
    # Initialize generator
    generator = RealChatGPTGenerator()
    
    if not generator.client:
        print("❌ No API client available!")
        return
    
    # Test with just 10 scripts
    try:
        scripts = generator.generate_batch_scripts(
            "results/all_extracted_prompts.json",
            max_scripts=10,  # Just 10 for testing
            batch_size=3     # Small batches for testing
        )
        
        # Save test results
        generator.save_results("results/test_chatgpt_scripts.json")
        
        print(f"\n🎯 TEST COMPLETE!")
        print(f"   Generated {len(scripts)} test scripts")
        
        # Show sample content
        if scripts:
            print(f"\n📝 Sample generated script:")
            print("-" * 40)
            sample = scripts[0]['generated_content'][:500]
            print(sample + "..." if len(sample) == 500 else sample)
            print("-" * 40)
        
        success_rate = sum(1 for s in scripts if s['success']) / len(scripts) * 100
        print(f"✅ Success rate: {success_rate:.1f}%")
        
        if success_rate > 80:
            print("🚀 API working well! Ready for full 1090 script generation.")
        else:
            print("⚠️  Low success rate. Check API key and network connection.")
            
    except Exception as e:
        print(f"❌ Test failed: {e}")

if __name__ == "__main__":
    main() 