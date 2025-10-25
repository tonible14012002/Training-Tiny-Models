"""
Interactive test for DataValidator - test your own samples!

Usage:
    python test_validator_custom.py

Then follow the prompts to add samples and see validation results.
"""
import asyncio
import json
from app.core.services.data_validator import DataValidator
from app.core.schemas.workflow import Sample
from app.core.models.models import LabelConfig
from src.payment_classifier.llm.litellm import LiteLLMProvider
from src.payment_classifier.prompts import InmemoryPromptManager
from src.payment_classifier.llm.settings import LLMSettings
from app.core.settings import settings


def print_label_guide():
    """Print label definitions to help user."""
    print("\n" + "=" * 70)
    print("LABEL GUIDE:")
    print("=" * 70)
    print("payment_intent:")
    print("  - Sender will send/pay money to someone")
    print("  - Examples: 'I'll send $10', 'Pay Alice 20 BTC', 'Sending now'")
    print()
    print("payment_request:")
    print("  - Sender requests/demands money from someone")
    print("  - Examples: 'Send me $10', 'You owe me', 'Pay me back'")
    print()
    print("open_intent:")
    print("  - Past payments, acknowledgments, or unrelated messages")
    print("  - Examples: 'I sent it already', 'Thanks!', 'Weather is nice'")
    print("=" * 70 + "\n")


async def test_custom_samples():
    """Interactive test with user-provided samples."""

    # Setup
    label_config = LabelConfig(
        id2label=json.dumps({
            "0": "payment_intent",
            "1": "payment_request",
            "2": "open_intent"
        }),
        label2id=json.dumps({
            "payment_intent": 0,
            "payment_request": 1,
            "open_intent": 2
        })
    )

    llm = LiteLLMProvider(
        settings=LLMSettings(
            llm_model_name="gpt-4o-mini",
            api_key=settings.OPENAI_API_KEY,
            temperature=0.0,
            num_retries=2,
        )
    )

    prompt_mgr = InmemoryPromptManager()

    validator = DataValidator(
        llm=llm,
        prompt_mgr=prompt_mgr,
        label_config=label_config
    )

    print("\n" + "=" * 70)
    print("CUSTOM SAMPLE VALIDATOR TEST")
    print("=" * 70)
    print_label_guide()

    # Collect samples from user
    samples = []
    print("Enter your samples (or 'done' to finish):\n")

    while True:
        print(f"\nSample {len(samples) + 1}:")
        msg = input("  Message: ").strip()

        if msg.lower() == 'done':
            break

        if not msg:
            print("  ⚠ Empty message, skipping...")
            continue

        print("  Choose label:")
        print("    1. payment_intent")
        print("    2. payment_request")
        print("    3. open_intent")

        label_choice = input("  Enter 1, 2, or 3: ").strip()

        label_map = {
            "1": "payment_intent",
            "2": "payment_request",
            "3": "open_intent"
        }

        label = label_map.get(label_choice)
        if not label:
            print("  ⚠ Invalid choice, skipping sample...")
            continue

        samples.append(Sample(msg=msg, label=label))
        print(f"  ✓ Added: [{label}] {msg}")

    if not samples:
        print("\n⚠ No samples provided. Exiting.")
        return

    # Show collected samples
    print("\n" + "=" * 70)
    print(f"COLLECTED {len(samples)} SAMPLES:")
    print("=" * 70)
    for i, sample in enumerate(samples):
        print(f"  [{i}] {sample.label:20} | {sample.msg}")

    # Run validation
    print("\n" + "=" * 70)
    print("RUNNING VALIDATION...")
    print("=" * 70)

    corrected_samples = await validator.validate_and_fix(samples)

    # Show results
    print("\n" + "=" * 70)
    print("VALIDATION RESULTS:")
    print("=" * 70)

    changes_made = 0
    for i, (original, corrected) in enumerate(zip(samples, corrected_samples)):
        if original.label != corrected.label:
            changes_made += 1
            print(f"\n  [{i}] ⚠ CORRECTED:")
            print(f"      Message:  '{original.msg}'")
            print(f"      Original: {original.label}")
            print(f"      Fixed to: {corrected.label}")
        else:
            print(f"  [{i}] ✓ UNCHANGED: {original.label}")

    print("\n" + "=" * 70)
    print(f"SUMMARY: {changes_made} corrections made out of {len(samples)} samples")
    print("=" * 70)

    # Ask if user wants to save results
    save = input("\nSave corrected samples to file? (y/n): ").strip().lower()
    if save == 'y':
        filename = "validator_custom_output.jsonl"
        with open(filename, 'w', encoding='utf-8') as f:
            for sample in corrected_samples:
                label_id = label_config.get_label2id()[sample.label]
                json.dump({"msg": sample.msg, "label": label_id}, f, ensure_ascii=False)
                f.write('\n')
        print(f"✓ Saved to {filename}")


if __name__ == "__main__":
    asyncio.run(test_custom_samples())
