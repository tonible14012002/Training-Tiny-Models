"""
Test validator with "Pay me" samples that are mislabeled.

These samples all start with "Pay me" and should be payment_request,
but are incorrectly labeled as payment_intent.
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


async def test_pay_me_samples():
    """Test specific 'Pay me' samples that are mislabeled."""

    # These are all WRONG - labeled as payment_intent but should be payment_request
    samples = [
        Sample(
            msg="Pay me in bitcoin, about 0.01 BTC will do.",
            label="payment_intent"  # WRONG! Should be payment_request
        ),
        Sample(
            msg="Pay me 10$ for the graphic design work, okay?",
            label="payment_intent"  # WRONG! Should be payment_request
        ),
        Sample(
            msg="Pay me 25 bucks tomorrow, please!",
            label="payment_intent"  # WRONG! Should be payment_request
        ),
        Sample(
            msg="Pay me back those 1000 msats at your convenience.",
            label="payment_intent"  # WRONG! Should be payment_request
        ),
    ]

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

    print("=" * 80)
    print("TESTING VALIDATOR WITH 'PAY ME' SAMPLES")
    print("=" * 80)
    print("\nThese samples all use 'Pay me' which means the sender is requesting")
    print("payment, so they should be labeled as 'payment_request', not 'payment_intent'.")
    print()
    print("Key distinction:")
    print("  - 'Pay Alice $10' = payment_intent (sender pays someone else)")
    print("  - 'Pay me $10' = payment_request (sender receives payment)")
    print("=" * 80)

    print("\nOriginal mislabeled samples:")
    for i, sample in enumerate(samples):
        print(f"  [{i}] ❌ WRONG: payment_intent")
        print(f"      Message: '{sample.msg}'")
        print(f"      Should be: payment_request")
        print()

    print("=" * 80)
    print("RUNNING VALIDATION...")
    print("=" * 80)

    corrected_samples = await validator.validate_and_fix(samples)

    print("\n" + "=" * 80)
    print("VALIDATION RESULTS:")
    print("=" * 80)

    all_correct = True
    for i, (original, corrected) in enumerate(zip(samples, corrected_samples)):
        if corrected.label == "payment_request":
            print(f"\n  [{i}] ✅ CORRECTLY FIXED:")
            print(f"      Message: '{original.msg}'")
            print(f"      {original.label} → {corrected.label}")
        else:
            all_correct = False
            print(f"\n  [{i}] ❌ NOT FIXED:")
            print(f"      Message: '{original.msg}'")
            print(f"      Stayed as: {original.label}")
            print(f"      Expected: payment_request")

    print("\n" + "=" * 80)
    print("SUMMARY:")
    print("=" * 80)
    corrections_made = sum(1 for o, c in zip(samples, corrected_samples) if o.label != c.label)
    correct_fixes = sum(1 for o, c in zip(samples, corrected_samples) if c.label == "payment_request")

    print(f"Total samples:        {len(samples)}")
    print(f"Expected corrections: {len(samples)}")
    print(f"Corrections made:     {corrections_made}")
    print(f"Correct fixes:        {correct_fixes}")

    if all_correct:
        print("\n✅ ALL SAMPLES CORRECTLY FIXED!")
        print("The validator successfully identified that 'Pay me' requests are")
        print("payment_request, not payment_intent.")
    else:
        print(f"\n⚠️  SOME SAMPLES NOT FIXED: {len(samples) - correct_fixes}/{len(samples)}")
        print("The validator may need prompt adjustments to handle 'Pay me' patterns.")

    print("=" * 80)

    return corrected_samples


if __name__ == "__main__":
    asyncio.run(test_pay_me_samples())
