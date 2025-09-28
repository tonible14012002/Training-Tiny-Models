#!/usr/bin/env python3
"""
Example showing how PromptBuilder integrates with the existing workflow.
Demonstrates the flow: Analyze → Build Prompts → Generate Data
"""

import json
import sys
from pathlib import Path

# Add the app directory to Python path
sys.path.insert(0, str(Path(__file__).parent / "app"))

from core.services.prompt_builder import PromptBuilderService


def example_integration_workflow():
    """
    Example workflow showing how PromptBuilder fits into the data generation pipeline
    """
    print("🔍 Step 1: Load Model Analysis Results")
    print("-" * 50)

    # In real workflow, this would come from ModelAnalyzer
    with open("json.json", "r") as f:
        analysis_result = json.load(f)

    print(f"Loaded analysis with {len(analysis_result['error_analyses'])} error patterns")
    print(f"Overall accuracy: {analysis_result['evaluation_summary']['overall_accuracy']:.3f}")
    print(f"Total errors: {analysis_result['evaluation_summary']['total_errors']}")

    print("\n🏗️ Step 2: Build Targeted Generation Prompts")
    print("-" * 50)

    prompt_builder = PromptBuilderService()
    prompts = prompt_builder.build_prompts_from_analysis(analysis_result)
    summary = prompt_builder.summarize_prompts(prompts)

    print(f"Generated {summary['total_prompts']} targeted prompts")
    print(f"High priority prompts: {summary['high_priority_count']}")
    print(f"Expected new examples: {summary['total_expected_examples']}")

    print("\n📝 Step 3: Select Priority Prompts for Generation")
    print("-" * 50)

    # Get the most important prompts to address first
    priority_prompts = prompt_builder.get_high_priority_prompts(prompts, limit=5)

    for i, prompt in enumerate(priority_prompts, 1):
        print(f"\n{i}. {prompt.target_label} ({prompt.action_type})")
        print(f"   Expected: {prompt.expected_count} examples")

        # Show key requirements from the prompt
        constraints = prompt.generation_constraints
        if constraints.get("keywords_to_include"):
            keywords = constraints["keywords_to_include"][:3]
            print(f"   Key words: {', '.join(keywords)}")

        if constraints.get("sentence_patterns"):
            patterns = constraints["sentence_patterns"][:2]
            print(f"   Patterns: {patterns[0]}")

    print("\n🎯 Step 4: Example Prompt for Data Generator")
    print("-" * 50)

    # Show how a prompt would be used by DataGenerator
    example_prompt = priority_prompts[0]
    print(f"Prompt for DataGenerator:")
    print(f"Target Label: {example_prompt.target_label}")
    print(f"Count: {example_prompt.expected_count}")
    print("\nPrompt Text:")
    print(example_prompt.prompt_text)

    print("\n🔄 Step 5: Next Steps in Workflow")
    print("-" * 50)
    print("1. Pass each prompt to DataGeneratorV2")
    print("2. Generate targeted examples based on constraints")
    print("3. Apply deduplication and quality filtering")
    print("4. Add new examples to training dataset")
    print("5. Retrain model and evaluate improvements")

    return prompts


if __name__ == "__main__":
    prompts = example_integration_workflow()

    print(f"\n✅ Generated {len(prompts)} prompts ready for data generation")