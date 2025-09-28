#!/usr/bin/env python3
"""
Test script for PromptBuilder service.
Demonstrates how to use the service with analyzer output.
"""

import json
import sys
from pathlib import Path

# Add the app directory to Python path
sys.path.insert(0, str(Path(__file__).parent / "app"))

from core.services.prompt_builder import PromptBuilderService


def main():
    # Load the analyzer output from json.json
    with open("json.json", "r") as f:
        analysis_result = json.load(f)

    # Create the prompt builder service
    prompt_builder = PromptBuilderService()

    # Build prompts from the analysis
    print("Building prompts from analysis result...")
    prompts = prompt_builder.build_prompts_from_analysis(analysis_result)

    print(f"\nGenerated {len(prompts)} prompts:")
    print("=" * 60)

    # Show high priority prompts
    high_priority = prompt_builder.get_high_priority_prompts(prompts, limit=3)
    print(f"\nTop {len(high_priority)} High Priority Prompts:")
    print("-" * 40)

    for i, prompt in enumerate(high_priority, 1):
        print(f"\n{i}. Target Label: {prompt.target_label}")
        print(f"   Action: {prompt.action_type}")
        print(f"   Expected Count: {prompt.expected_count}")
        print(f"   Priority: {prompt.priority}")
        print(f"\n   Prompt Text:")
        print("   " + "\n   ".join(prompt.prompt_text.split("\n")))
        print("\n   Generation Constraints:")
        for key, value in prompt.generation_constraints.items():
            if isinstance(value, list) and len(value) > 3:
                print(f"     {key}: {value[:3]}... ({len(value)} total)")
            else:
                print(f"     {key}: {value}")
        print("-" * 40)

    # Show summary
    summary = prompt_builder.summarize_prompts(prompts)
    print(f"\nPrompt Summary:")
    print(f"  Total prompts: {summary['total_prompts']}")
    print(f"  High priority: {summary['high_priority_count']}")
    print(f"  Total expected examples: {summary['total_expected_examples']}")
    print(f"  By action type: {summary['by_action_type']}")
    print(f"  By target label: {summary['by_target_label']}")


if __name__ == "__main__":
    main()