"""
Test file for DataValidator with intentionally mislabeled samples.

This script creates a dataset with known labeling errors and tests
if the DataValidator can identify and correct them.
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


# Test samples with intentional labeling errors
MISLABELED_SAMPLES = [
    # Correct labels (for baseline)
    Sample(msg="I'll send you $50 later", label="payment_intent"),
    Sample(msg="Send me the money you owe", label="payment_request"),
    Sample(msg="Thanks for the payment yesterday", label="open_intent"),

    # MISLABELED: payment_request labeled as payment_intent
    Sample(msg="Send me 10 BTC please", label="payment_intent"),  # WRONG! Should be payment_request
    Sample(msg="You owe me $200", label="payment_intent"),  # WRONG! Should be payment_request
    Sample(msg="Pay me back the 50 bucks", label="payment_intent"),  # WRONG! Should be payment_request

    # MISLABELED: payment_intent labeled as payment_request
    Sample(msg="I'll transfer $100 to Alice", label="payment_request"),  # WRONG! Should be payment_intent
    Sample(msg="Pay Bob 20 sats", label="payment_request"),  # WRONG! Should be payment_intent
    Sample(msg="Sending 5 ETH now", label="payment_request"),  # WRONG! Should be payment_intent

    # MISLABELED: open_intent labeled as payment_intent
    Sample(msg="I already sent the payment", label="payment_intent"),  # WRONG! Should be open_intent
    Sample(msg="The transaction was completed", label="payment_intent"),  # WRONG! Should be open_intent
    Sample(msg="Thanks for paying me", label="payment_intent"),  # WRONG! Should be open_intent

    # MISLABELED: open_intent labeled as payment_request
    Sample(msg="I received your payment", label="payment_request"),  # WRONG! Should be open_intent
    Sample(msg="The money arrived safely", label="payment_request"),  # WRONG! Should be open_intent

    # MISLABELED: payment_intent labeled as open_intent
    Sample(msg="I'll give you $30 tomorrow", label="open_intent"),  # WRONG! Should be payment_intent
    Sample(msg="Send Alice 100 sats", label="open_intent"),  # WRONG! Should be payment_intent

    # MISLABELED: payment_request labeled as open_intent
    Sample(msg="Can you send me the $50?", label="open_intent"),  # WRONG! Should be payment_request
    Sample(msg="I need you to pay me back", label="open_intent"),  # WRONG! Should be payment_request

    # Correct labels (mixed in)
    Sample(msg="I'll pay the bill now", label="payment_intent"),
    Sample(msg="Send me what you owe", label="payment_request"),
    Sample(msg="How's the weather today?", label="open_intent"),
]


EXPECTED_CORRECTIONS = {
    3: "payment_request",   # "Send me 10 BTC please"
    4: "payment_request",   # "You owe me $200"
    5: "payment_request",   # "Pay me back the 50 bucks"
    6: "payment_intent",    # "I'll transfer $100 to Alice"
    7: "payment_intent",    # "Pay Bob 20 sats"
    8: "payment_intent",    # "Sending 5 ETH now"
    9: "open_intent",       # "I already sent the payment"
    10: "open_intent",      # "The transaction was completed"
    11: "open_intent",      # "Thanks for paying me"
    12: "open_intent",      # "I received your payment"
    13: "open_intent",      # "The money arrived safely"
    14: "payment_intent",   # "I'll give you $30 tomorrow"
    15: "payment_intent",   # "Send Alice 100 sats"
    16: "payment_request",  # "Can you send me the $50?"
    17: "payment_request",  # "I need you to pay me back"
}


async def test_validator():
    """Test the DataValidator with mislabeled samples."""

    # Setup label config
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

    # Setup LLM
    llm = LiteLLMProvider(
        settings=LLMSettings(
            llm_model_name="gpt-4o-mini",
            api_key=settings.OPENAI_API_KEY,
            temperature=0.0,  # Use 0 for deterministic validation
            num_retries=2,
        )
    )

    # Setup prompt manager
    prompt_mgr = InmemoryPromptManager()

    # Create validator
    validator = DataValidator(
        llm=llm,
        prompt_mgr=prompt_mgr,
        label_config=label_config
    )

    print("=" * 80)
    print("TESTING DATA VALIDATOR WITH MISLABELED SAMPLES")
    print("=" * 80)
    print(f"\nTotal samples: {len(MISLABELED_SAMPLES)}")
    print(f"Expected errors: {len(EXPECTED_CORRECTIONS)}")
    print(f"Correctly labeled: {len(MISLABELED_SAMPLES) - len(EXPECTED_CORRECTIONS)}")

    # Show original mislabeled samples
    print("\n" + "=" * 80)
    print("ORIGINAL MISLABELED SAMPLES:")
    print("=" * 80)
    for idx, sample in enumerate(MISLABELED_SAMPLES):
        expected = EXPECTED_CORRECTIONS.get(idx)
        marker = "❌ WRONG" if expected else "✓ Correct"
        expected_str = f" → should be: {expected}" if expected else ""
        print(f"[{idx}] {marker} | '{sample.msg[:50]}...' | label={sample.label}{expected_str}")

    # Validate and fix
    print("\n" + "=" * 80)
    print("RUNNING VALIDATION...")
    print("=" * 80)

    corrected_samples = await validator.validate_and_fix(MISLABELED_SAMPLES)

    # Analyze results
    print("\n" + "=" * 80)
    print("VALIDATION RESULTS:")
    print("=" * 80)

    corrections_found = 0
    corrections_missed = 0
    false_positives = 0
    correct_fixes = 0
    incorrect_fixes = 0

    for idx, (original, corrected) in enumerate(zip(MISLABELED_SAMPLES, corrected_samples)):
        expected_label = EXPECTED_CORRECTIONS.get(idx)

        if original.label != corrected.label:
            # Validator made a correction
            corrections_found += 1

            if expected_label:
                # This was indeed mislabeled
                if corrected.label == expected_label:
                    correct_fixes += 1
                    print(f"[{idx}] ✓ CORRECT FIX: '{original.msg[:40]}...'")
                    print(f"      {original.label} → {corrected.label} (expected: {expected_label})")
                else:
                    incorrect_fixes += 1
                    print(f"[{idx}] ⚠ WRONG FIX: '{original.msg[:40]}...'")
                    print(f"      {original.label} → {corrected.label} (expected: {expected_label})")
            else:
                # This was correctly labeled, but validator changed it (false positive)
                false_positives += 1
                print(f"[{idx}] ❌ FALSE POSITIVE: '{original.msg[:40]}...'")
                print(f"      {original.label} → {corrected.label} (should stay: {original.label})")
        else:
            # Validator kept the label
            if expected_label:
                # This should have been corrected but wasn't (missed)
                corrections_missed += 1
                print(f"[{idx}] ❌ MISSED: '{original.msg[:40]}...'")
                print(f"      Stayed as: {original.label} (should be: {expected_label})")

    # Summary
    print("\n" + "=" * 80)
    print("SUMMARY:")
    print("=" * 80)
    print(f"Expected corrections:     {len(EXPECTED_CORRECTIONS)}")
    print(f"Corrections found:        {corrections_found}")
    print(f"  ✓ Correct fixes:        {correct_fixes}")
    print(f"  ⚠ Incorrect fixes:      {incorrect_fixes}")
    print(f"  ❌ False positives:     {false_positives}")
    print(f"❌ Missed corrections:    {corrections_missed}")

    accuracy = (correct_fixes / len(EXPECTED_CORRECTIONS) * 100) if len(EXPECTED_CORRECTIONS) > 0 else 0
    precision = (correct_fixes / corrections_found * 100) if corrections_found > 0 else 0

    print(f"\nAccuracy (correct/expected): {accuracy:.1f}%")
    print(f"Precision (correct/found):   {precision:.1f}%")

    # Detailed comparison
    print("\n" + "=" * 80)
    print("DETAILED COMPARISON:")
    print("=" * 80)
    print(f"{'Idx':<5} {'Original':<20} {'Corrected':<20} {'Expected':<20} {'Status':<15}")
    print("-" * 80)
    for idx, (original, corrected) in enumerate(zip(MISLABELED_SAMPLES, corrected_samples)):
        expected = EXPECTED_CORRECTIONS.get(idx, original.label)
        status = "✓" if corrected.label == expected else "❌"
        print(f"{idx:<5} {original.label:<20} {corrected.label:<20} {expected:<20} {status:<15}")

    return corrected_samples


if __name__ == "__main__":
    asyncio.run(test_validator())
