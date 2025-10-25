"""
Test validator on actual incorrect cases found in production.
These are real samples that slipped through the validator.
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


async def test_actual_errors():
    """Test on actual errors found in production."""

    # These are ACTUAL cases that were incorrectly labeled
    samples = [
        # "Send me" patterns - should be payment_request
        Sample(msg="Send me the invoice for the 100$ payment!", label="payment_intent"),  # WRONG
        Sample(msg="Send me 30$ before the end of the week.", label="payment_intent"),  # WRONG
        Sample(msg="Send me 0.001 BTC when you can!", label="payment_intent"),  # WRONG
        Sample(msg="Remember to send me that 5$ for lunch.", label="payment_intent"),  # WRONG
        Sample(msg="Send me 1000 sats next week.", label="open_intent"),  # WRONG

        # Non-payment "Send me" - should be open_intent
        Sample(msg="Can you send me that invoice? I need to check it.", label="open_intent"),  # CORRECT

        # "Pay me" patterns - should be payment_request
        Sample(msg="Pay me 1000 msats for the artwork!", label="payment_intent"),  # WRONG
        Sample(msg="Pay me back the 10 bucks when you can!", label="payment_intent"),  # WRONG
    ]

    # Expected corrections
    expected = {
        0: "payment_request",  # Send me the invoice for the 100$ payment!
        1: "payment_request",  # Send me 30$ before the end of the week.
        2: "payment_request",  # Send me 0.001 BTC when you can!
        3: "payment_request",  # Remember to send me that 5$ for lunch.
        4: "payment_request",  # Send me 1000 sats next week.
        5: "open_intent",      # Can you send me that invoice? (CORRECT - no change)
        6: "payment_request",  # Pay me 1000 msats for the artwork!
        7: "payment_request",  # Pay me back the 10 bucks when you can!
    }

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

    validator = DataValidator(
        llm=llm,
        prompt_mgr=InmemoryPromptManager(),
        label_config=label_config
    )

    print("=" * 80)
    print("TESTING VALIDATOR ON ACTUAL PRODUCTION ERRORS")
    print("=" * 80)
    print("\nThese are real samples that were incorrectly labeled in production.\n")

    print("Original labels:")
    for i, sample in enumerate(samples):
        expected_label = expected[i]
        marker = "❌" if sample.label != expected_label else "✓"
        print(f"  [{i}] {marker} {sample.label:20} (should be: {expected_label})")
        print(f"       '{sample.msg}'")

    print("\n" + "=" * 80)
    print("RUNNING VALIDATION...")
    print("=" * 80)

    corrected_samples = await validator.validate_and_fix(samples)

    print("\nCorrected labels:")
    for i, (original, corrected) in enumerate(zip(samples, corrected_samples)):
        expected_label = expected[i]
        if corrected.label == expected_label:
            marker = "✅"
            status = "CORRECT"
        else:
            marker = "❌"
            status = "STILL WRONG"

        changed = "→" if original.label != corrected.label else " "
        print(f"  [{i}] {marker} {status:12} {original.label:20} {changed} {corrected.label:20}")
        print(f"       Expected: {expected_label}")
        print(f"       '{corrected.msg}'")

    # Count results
    correct = sum(1 for i, (_, c) in enumerate(zip(samples, corrected_samples)) if c.label == expected[i])
    errors_fixed = sum(1 for i, (o, c) in enumerate(zip(samples, corrected_samples))
                      if o.label != expected[i] and c.label == expected[i])

    print("\n" + "=" * 80)
    print("RESULTS:")
    print("=" * 80)
    print(f"Total samples:        {len(samples)}")
    print(f"Originally incorrect: 7")
    print(f"Errors fixed:         {errors_fixed}/7")
    print(f"Final correct:        {correct}/{len(samples)}")

    if correct == len(samples):
        print("\n✅ PERFECT: All samples now have correct labels!")
    elif errors_fixed >= 6:
        print(f"\n✅ GOOD: Fixed {errors_fixed}/7 errors")
    else:
        print(f"\n⚠️  NEEDS WORK: Only fixed {errors_fixed}/7 errors")

    print("=" * 80)

    return corrected_samples


if __name__ == "__main__":
    asyncio.run(test_actual_errors())
