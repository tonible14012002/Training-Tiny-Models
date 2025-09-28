#!/usr/bin/env python3
"""
Test script for the updated PromptBuilder service with consolidated prompts.
Demonstrates LLM-based prompt generation and balanced generation planning.
"""

import json
import sys
import os
from pathlib import Path

# Add the app directory to Python path
sys.path.insert(0, str(Path(__file__).parent / "app"))

from core.services.prompt_builder import PromptBuilderService, ConsolidatedPrompt


def main():
    # Set required environment variable for LLM client
    if not os.getenv("OPENAI_API_KEY"):
        print("⚠️  Warning: OPENAI_API_KEY not set. Using fallback prompts.")
        print("   Set OPENAI_API_KEY to test LLM-generated prompts.\n")

    # Load the analyzer output from json.json
    with open("json.json", "r") as f:
        analysis_result = json.load(f)

    # Create the prompt builder service
    prompt_builder = PromptBuilderService()

    print("🔍 Step 1: Building Consolidated Prompts")
    print("=" * 60)

    # Build consolidated prompts from the analysis
    consolidated_prompts = prompt_builder.build_consolidated_prompts(analysis_result)

    summary = prompt_builder.summarize_prompts(consolidated_prompts)
    print(f"Generated {summary['total_prompts']} consolidated prompts")
    print(f"Error patterns: {summary['error_patterns']}")
    print(f"Labels involved: {summary['labels_involved']}")
    print(f"Total issues addressed: {summary['total_issues']}")

    print("\n📋 Step 2: Consolidated Prompts Overview")
    print("=" * 60)

    for i, prompt in enumerate(consolidated_prompts, 1):
        print(f"\n{i}. Error Pattern: {prompt.error_pattern_key}")
        print(f"   Issues: {len(prompt.identified_issues)} identified")
        print(f"   Actions: {len(prompt.consolidated_actions)} action types")

    print("\n📝 Step 3: Sample Generated Prompt")
    print("=" * 60)

    if consolidated_prompts:
        sample_prompt = consolidated_prompts[0]
        print(f"Pattern: {sample_prompt.error_pattern_key}")
        print(f"Placeholder: {sample_prompt.generation_placeholder}")
        print("\n--- Generated Prompt Text ---")
        print(sample_prompt.prompt_text)
        print("--- End Prompt ---")

        # Show how to use the placeholder
        print(f"\n🔧 Step 4: Using the Quantity Placeholder")
        print("=" * 60)
        formatted_prompt = prompt_builder.format_prompt_with_quantity(sample_prompt, 50)
        print("Example with quantity=50:")
        print(formatted_prompt[:500] + "..." if len(formatted_prompt) > 500 else formatted_prompt)

    print("\n⚖️ Step 5: Balanced Generation Plan")
    print("=" * 60)

    # Create a balanced generation plan
    generation_plan = prompt_builder.get_balanced_generation_plan(
        consolidated_prompts, total_examples=200
    )

    print("Generation plan for 200 total examples:")
    total_planned = 0
    for pattern, label_counts in generation_plan.items():
        pattern_total = sum(label_counts.values())
        total_planned += pattern_total
        print(f"\n{pattern}:")
        for label, count in label_counts.items():
            print(f"  {label}: {count} examples")
        print(f"  Subtotal: {pattern_total} examples")

    print(f"\nTotal planned examples: {total_planned}")

    print("\n✅ Step 6: Integration Example")
    print("=" * 60)
    print("1. For each consolidated prompt:")
    print("   - Use get_balanced_generation_plan() for quantities")
    print("   - Format prompt with format_prompt_with_quantity()")
    print("   - Pass to DataGeneratorV2 for generation")
    print("2. Result: Balanced training data addressing model weaknesses")


if __name__ == "__main__":
    main()