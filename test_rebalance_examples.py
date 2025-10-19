#!/usr/bin/env python3
"""
Test script to demonstrate rebalance instruction outputs for different scenarios
"""

from app.core.services.data_generator.data_generator_v2 import DataGeneratorV2


def print_example(title, current_counts, expected_counts):
    """Print an example case with the generated instruction"""
    print(f"\n{'='*80}")
    print(f"📊 {title}")
    print(f"{'='*80}")
    print(f"Current counts:  {current_counts}")
    print(f"Expected counts: {expected_counts}")
    print(f"\n🔹 Generated Instruction:")

    # Create a mock instance just to use the method
    generator = DataGeneratorV2(llm=None, prompt_mgr=None, data_manager=None)
    instruction = generator._build_rebalance_instruction(current_counts, expected_counts)

    if instruction:
        print(f"{instruction}")
    else:
        print("  (No instruction - all labels have sufficient samples)")

    # Show what the full user prompt would look like
    print(f"\n📝 Full User Prompt Example:")
    seed_examples = "- Example message 1\n- Example message 2"
    count = 30
    full_prompt = f"## Examples: {seed_examples} \nContinue generate {count} diverse examples{instruction}"
    print(f"  {full_prompt}")


if __name__ == "__main__":
    print("\n" + "🎯 REBALANCE INSTRUCTION EXAMPLES".center(80, "="))

    # Case 1: Beginning of generation - significant imbalance
    print_example(
        "CASE 1: Early Stage - Major Imbalance",
        current_counts={"payment_intent": 10, "payment_request": 45, "open_intent": 30},
        expected_counts={"payment_intent": 60, "payment_request": 60, "open_intent": 60}
    )

    # Case 2: One label complete, others need more
    print_example(
        "CASE 2: One Label Complete, Others Behind",
        current_counts={"payment_intent": 60, "payment_request": 35, "open_intent": 40},
        expected_counts={"payment_intent": 60, "payment_request": 60, "open_intent": 60}
    )

    # Case 3: Only one label needs more samples
    print_example(
        "CASE 3: Only One Label Remaining",
        current_counts={"payment_intent": 60, "payment_request": 60, "open_intent": 45},
        expected_counts={"payment_intent": 60, "payment_request": 60, "open_intent": 60}
    )

    # Case 4: All labels complete
    print_example(
        "CASE 4: All Labels Complete",
        current_counts={"payment_intent": 60, "payment_request": 60, "open_intent": 60},
        expected_counts={"payment_intent": 60, "payment_request": 60, "open_intent": 60}
    )

    # Case 5: Very unbalanced - one label way behind
    print_example(
        "CASE 5: Severe Imbalance - One Label Far Behind",
        current_counts={"payment_intent": 5, "payment_request": 55, "open_intent": 58},
        expected_counts={"payment_intent": 60, "payment_request": 60, "open_intent": 60}
    )

    # Case 6: Near completion with slight imbalance
    print_example(
        "CASE 6: Near Completion - Slight Imbalance",
        current_counts={"payment_intent": 58, "payment_request": 60, "open_intent": 57},
        expected_counts={"payment_intent": 60, "payment_request": 60, "open_intent": 60}
    )

    # Case 7: Multiple labels with different priorities
    print_example(
        "CASE 7: Five Labels - Different Priorities",
        current_counts={
            "label_a": 10,
            "label_b": 45,
            "label_c": 30,
            "label_d": 5,
            "label_e": 50
        },
        expected_counts={
            "label_a": 60,
            "label_b": 60,
            "label_c": 60,
            "label_d": 60,
            "label_e": 60
        }
    )

    # Case 8: Some labels exceeded (edge case)
    print_example(
        "CASE 8: Some Labels Exceeded Target (Edge Case)",
        current_counts={"payment_intent": 65, "payment_request": 40, "open_intent": 30},
        expected_counts={"payment_intent": 60, "payment_request": 60, "open_intent": 60}
    )

    print("\n" + "="*80)
    print("✅ All examples generated successfully!")
    print("="*80 + "\n")
