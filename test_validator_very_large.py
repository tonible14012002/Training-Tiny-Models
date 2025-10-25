"""
Test validator with a very large batch (~100 samples) to simulate first-gen.
This tests the batching mechanism with multiple batches.
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


async def test_very_large_batch():
    """Test validator with 100 samples (should split into 4 batches of 30+30+30+10)."""

    samples = []

    # Add 50 correctly labeled samples
    for i in range(50):
        if i % 3 == 0:
            samples.append(Sample(msg=f"I'll send you ${10+i} later", label="payment_intent"))
        elif i % 3 == 1:
            samples.append(Sample(msg=f"Send me ${10+i} please", label="payment_request"))
        else:
            samples.append(Sample(msg=f"I sent ${10+i} yesterday", label="open_intent"))

    # Add 50 MISLABELED samples
    for i in range(50):
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

    validator = DataValidator(
        llm=llm,
        prompt_mgr=InmemoryPromptManager(),
        label_config=label_config
    )

    print("=" * 80)
    print("TESTING VALIDATOR WITH VERY LARGE BATCH (100 samples)")
    print("=" * 80)
    print(f"\nTotal samples: {len(samples)}")
    print(f"Expected to split into ~4 batches (30+30+30+10)")
    print(f"\nComposition:")
    print(f"  - 50 correctly labeled samples")
    print(f"  - 50 mislabeled 'pay me' samples")

    print("\n" + "=" * 80)
    print("RUNNING VALIDATION (this may take 30-60 seconds)...")
    print("=" * 80)

    import time
    start_time = time.time()
    corrected_samples = await validator.validate_and_fix(samples, batch_size=30)
    elapsed = time.time() - start_time

    print("\n" + "=" * 80)
    print("RESULTS:")
    print("=" * 80)

    # Count corrections
    corrections_made = 0
    correct_corrections = 0

    for original, corrected in zip(samples, corrected_samples):
        if original.label != corrected.label:
            corrections_made += 1
            if "pay me" in corrected.msg.lower() and corrected.label == "payment_request":
                correct_corrections += 1

    print(f"\nTime taken:           {elapsed:.1f} seconds")
    print(f"Corrections made:     {corrections_made}")
    print(f"Correct corrections:  {correct_corrections}/50 'pay me' samples")

    success_rate = (correct_corrections / 50 * 100) if corrections_made > 0 else 0
    print(f"Success rate:         {success_rate:.1f}%")

    if correct_corrections >= 47:  # Allow 3 edge cases
        print("\n✅ PASS: Validator successfully handles very large batches!")
    else:
        print(f"\n⚠️  WARNING: Only {correct_corrections}/50 corrections made")

    print("=" * 80)

    return corrected_samples


if __name__ == "__main__":
    asyncio.run(test_very_large_batch())
