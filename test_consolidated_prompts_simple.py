#!/usr/bin/env python3
"""
Simple test script for the updated PromptBuilder service with sample data.
"""

import sys
import os
from pathlib import Path

# Add the app directory to Python path
sys.path.insert(0, str(Path(__file__).parent / "app"))

from core.services.prompt_builder import PromptBuilderService, ConsolidatedPrompt


def create_sample_analysis_result():
    """Create sample analysis data for testing"""
    return {
        "message": "Error pattern analysis completed. Analyzed 2 error patterns.",
        "status": "completed",
        "error_analyses": [
            {
                "predicted_label": "payment_request",
                "expected_label": "open_intent",
                "identified_issues": [
                    "Over-reliance on payment-related keywords like 'pay', 'paid', 'transfer'",
                    "Lack of examples in payment_request with explicit requests (e.g., 'Can you...', 'Please...')",
                    "Missing question patterns in open_intent that reference payment"
                ],
                "data_actions": [
                    {
                        "action_type": "generate_more",
                        "target_label": "open_intent",
                        "expected_count": 40,
                        "keywords_to_include": ["pay", "paid", "transfer", "fee", "payment"],
                        "sentence_patterns": [
                            "Did you [action] the [fee/bill/payment] yet?",
                            "I already [action] the [fee/bill/payment].",
                            "Was the [payment/fee] included in...?"
                        ],
                        "diversity_constraints": [
                            "Vary tense (past, present, future)",
                            "Include questions and statements"
                        ]
                    },
                    {
                        "action_type": "generate_more",
                        "target_label": "payment_request",
                        "expected_count": 30,
                        "keywords_to_include": ["can you", "please", "I need you to"],
                        "sentence_patterns": [
                            "Can you pay the [fee/bill] for me?",
                            "Please send me [amount] for..."
                        ],
                        "diversity_constraints": ["Use different request phrasings"]
                    }
                ]
            },
            {
                "predicted_label": "open_intent",
                "expected_label": "payment_intent",
                "identified_issues": [
                    "Insufficient examples of informal payment intents",
                    "Missing intent expressions where payment is implied by context"
                ],
                "data_actions": [
                    {
                        "action_type": "generate_more",
                        "target_label": "payment_intent",
                        "expected_count": 30,
                        "keywords_to_include": ["cover", "settle", "Venmo-ing", "processing"],
                        "sentence_patterns": [
                            "I'm going to [verb] the [item] this time.",
                            "I'll [verb] the [funds] by [timeframe]."
                        ],
                        "diversity_constraints": [
                            "vary payment types (cash, crypto, Venmo)",
                            "different contexts (meals, bills, purchases)"
                        ]
                    }
                ]
            }
        ],
        "evaluation_summary": {
            "overall_accuracy": 0.621,
            "macro_f1": 0.616,
            "total_errors": 100
        }
    }


def main():
    print("🚀 Testing Consolidated PromptBuilder Service")
    print("=" * 60)

    if not os.getenv("OPENAI_API_KEY"):
        print("⚠️  Warning: OPENAI_API_KEY not set. Using fallback prompts.")
        print("   Set OPENAI_API_KEY to test LLM-generated prompts.\n")

    # Create sample analysis data
    analysis_result = create_sample_analysis_result()

    # Create the prompt builder service
    prompt_builder = PromptBuilderService()

    print("🔍 Step 1: Building Consolidated Prompts")
    print("-" * 50)

    # Build consolidated prompts from the analysis
    consolidated_prompts = prompt_builder.build_consolidated_prompts(analysis_result)

    summary = prompt_builder.summarize_prompts(consolidated_prompts)
    print(f"Generated {summary['total_prompts']} consolidated prompts")
    print(f"Error patterns: {summary['error_patterns']}")
    print(f"Labels involved: {summary['labels_involved']}")
    print(f"Total issues addressed: {summary['total_issues']}")

    print("\n📋 Step 2: Consolidated Prompts Overview")
    print("-" * 50)

    for i, prompt in enumerate(consolidated_prompts, 1):
        print(f"\n{i}. Error Pattern: {prompt.error_pattern_key}")
        print(f"   Predicted: {prompt.predicted_label} → Expected: {prompt.expected_label}")
        print(f"   Issues: {len(prompt.identified_issues)} identified")
        print(f"   Actions: {len(prompt.consolidated_actions)} action types")

    if consolidated_prompts:
        print("\n📝 Step 3: Sample Generated Prompt")
        print("-" * 50)

        sample_prompt = consolidated_prompts[0]
        print(f"Pattern: {sample_prompt.error_pattern_key}")
        print(f"Placeholder: {sample_prompt.generation_placeholder}")
        print("\n--- Generated Prompt Text ---")
        print(sample_prompt.prompt_text)
        print("--- End Prompt ---")

        print(f"\n🔧 Step 4: Using the Quantity Placeholder")
        print("-" * 50)
        formatted_prompt = prompt_builder.format_prompt_with_quantity(sample_prompt, 50)
        print("Example with quantity=50:")
        print(formatted_prompt[:400] + "..." if len(formatted_prompt) > 400 else formatted_prompt)

    print("\n⚖️ Step 5: Balanced Generation Plan")
    print("-" * 50)

    # Create a balanced generation plan
    generation_plan = prompt_builder.get_balanced_generation_plan(
        consolidated_prompts, total_examples=100
    )

    print("Generation plan for 100 total examples:")
    total_planned = 0
    for pattern, label_counts in generation_plan.items():
        pattern_total = sum(label_counts.values())
        total_planned += pattern_total
        print(f"\n{pattern}:")
        for label, count in label_counts.items():
            print(f"  {label}: {count} examples")
        print(f"  Subtotal: {pattern_total} examples")

    print(f"\nTotal planned examples: {total_planned}")

    print("\n✅ Integration Ready!")
    print("-" * 50)
    print("The PromptBuilder service now provides:")
    print("• Consolidated prompts per error pattern")
    print("• LLM-generated fine-tuned prompts (with fallback)")
    print("• Quantity placeholders for balanced generation")
    print("• Balanced generation planning across labels")


if __name__ == "__main__":
    main()