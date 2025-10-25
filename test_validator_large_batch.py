"""
Test validator with a large batch of samples (>30) to verify batching works.
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


async def test_large_batch():
    """Test validator with 50 samples (should split into 2 batches of 30 and 20)."""

    # Create 50 samples: mix of correct and intentionally mislabeled
    samples = []

    # Add 25 correctly labeled samples
    for i in range(25):
        if i % 3 == 0:
            samples.append(Sample(msg=f"I'll send you ${10+i} later", label="payment_intent"))
        elif i % 3 == 1:
            samples.append(Sample(msg=f"Send me ${10+i} please", label="payment_request"))
        else:
            samples.append(Sample(msg=f"I sent ${10+i} yesterday", label="open_intent"))

    # Add 25 MISLABELED samples (all "pay me" as payment_intent - WRONG!)
    for i in range(25):
        samples.append(Sample(
            msg=f"Pay me back the ${20+i} you owe",
            label="payment_intent"  # WRONG! Should be payment_request
        ))

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
    print("TESTING VALIDATOR WITH LARGE BATCH (50 samples)")
    print("=" * 80)
    print(f"\nTotal samples: {len(samples)}")
    print(f"Expected to split into 2 batches (30 + 20)")
    print(f"\nComposition:")
    print(f"  - 25 correctly labeled samples")
    print(f"  - 25 mislabeled 'pay me' samples (payment_intent → should be payment_request)")

    print("\n" + "=" * 80)
    print("RUNNING VALIDATION...")
    print("=" * 80)

    corrected_samples = await validator.validate_and_fix(samples, batch_size=30)

    print("\n" + "=" * 80)
    print("ANALYZING RESULTS...")
    print("=" * 80)

    # Count corrections
    corrections_made = 0
    correct_corrections = 0

    for i, (original, corrected) in enumerate(zip(samples, corrected_samples)):
        if original.label != corrected.label:
            corrections_made += 1
            # Check if "pay me" was corrected to payment_request
            if "pay me" in corrected.msg.lower() and corrected.label == "payment_request":
                correct_corrections += 1

    print(f"\nCorrections made:     {corrections_made}")
    print(f"Correct corrections:  {correct_corrections}/25 'pay me' samples")

    # Show some examples
    print("\nSample corrections:")
    shown = 0
    for i, (original, corrected) in enumerate(zip(samples, corrected_samples)):
        if original.label != corrected.label and shown < 5:
            shown += 1
            print(f"  [{i}] {original.label:20} → {corrected.label:20}")
            print(f"       '{corrected.msg}'")

    print("\n" + "=" * 80)
    print("SUMMARY:")
    print("=" * 80)

    success_rate = (correct_corrections / 25 * 100) if corrections_made > 0 else 0
    print(f"Success rate: {success_rate:.1f}% ({correct_corrections}/25 'pay me' samples fixed)")

    if correct_corrections >= 23:  # Allow 2 edge cases
        print("✅ PASS: Validator successfully handles large batches!")
    else:
        print(f"⚠️  WARNING: Only {correct_corrections}/25 corrections made")

    print("=" * 80)

    return corrected_samples


if __name__ == "__main__":
    asyncio.run(test_large_batch())
