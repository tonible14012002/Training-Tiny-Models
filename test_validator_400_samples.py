"""
Test validator with 400 samples - simulating real first-gen output.
This is the realistic scale for first-gen batches.
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


async def test_400_samples():
    """Test validator with 400 samples - real first-gen scale."""

    samples = []

    # Add 300 correctly labeled samples
    for i in range(300):
        if i % 3 == 0:
            samples.append(Sample(msg=f"I'll send you ${10+i} later", label="payment_intent"))
        elif i % 3 == 1:
            samples.append(Sample(msg=f"Send me ${10+i} please", label="payment_request"))
        else:
            samples.append(Sample(msg=f"I sent ${10+i} yesterday", label="open_intent"))

    # Add 100 MISLABELED samples (simulating typical error rate)
    mislabeled_patterns = [
        "Pay me back the ${} you owe",
        "Don't forget to pay me ${}",
        "Please pay me ${} for the work",
        "I need you to pay me back ${}",
        "Pay me ${} when you can",
    ]

    for i in range(100):
        pattern = mislabeled_patterns[i % len(mislabeled_patterns)]
        samples.append(Sample(
            msg=pattern.format(20+i),
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
    print("TESTING VALIDATOR WITH 400 SAMPLES (REAL FIRST-GEN SCALE)")
    print("=" * 80)
    print(f"\nTotal samples: {len(samples)}")
    print(f"Expected to split into ~14 batches (30×13 + 10)")
    print(f"\nComposition:")
    print(f"  - 300 correctly labeled samples (75%)")
    print(f"  - 100 mislabeled 'pay me' samples (25%)")
    print(f"\nThis simulates a realistic first-gen output with typical error rates.")

    print("\n" + "=" * 80)
    print("RUNNING VALIDATION...")
    print("This will take ~60-90 seconds for 14 batches...")
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
    false_positives = 0

    for i, (original, corrected) in enumerate(zip(samples, corrected_samples)):
        if original.label != corrected.label:
            corrections_made += 1
            # Check if correction is valid
            if i >= 300:  # These were the mislabeled ones
                if "pay me" in corrected.msg.lower() and corrected.label == "payment_request":
                    correct_corrections += 1
            else:  # These were correct originally
                false_positives += 1

    print(f"\nTime taken:           {elapsed:.1f} seconds")
    print(f"Avg per batch:        {elapsed/14:.1f} seconds")
    print(f"\nCorrections made:     {corrections_made}")
    print(f"  - Correct fixes:    {correct_corrections}/100 mislabeled samples")
    print(f"  - False positives:  {false_positives}/300 correct samples")

    success_rate = (correct_corrections / 100 * 100) if corrections_made > 0 else 0
    precision = (correct_corrections / corrections_made * 100) if corrections_made > 0 else 0

    print(f"\nSuccess rate (recall): {success_rate:.1f}% (caught how many errors)")
    print(f"Precision:             {precision:.1f}% (how many fixes were correct)")

    # Show some examples
    print("\n" + "=" * 80)
    print("SAMPLE CORRECTIONS:")
    print("=" * 80)
    shown = 0
    for i, (original, corrected) in enumerate(zip(samples, corrected_samples)):
        if original.label != corrected.label and shown < 10:
            shown += 1
            status = "✅" if i >= 300 else "⚠️ "
            print(f"{status} [{i}] {original.label:20} → {corrected.label:20}")
            print(f"   '{corrected.msg}'")

    print("\n" + "=" * 80)
    print("FINAL ASSESSMENT:")
    print("=" * 80)

    if correct_corrections >= 95:  # 95%+ of errors caught
        print("✅ EXCELLENT: Validator successfully handles 400-sample batches!")
        print(f"   Caught {correct_corrections}/100 errors ({success_rate:.1f}%)")
    elif correct_corrections >= 90:
        print("✅ GOOD: Validator performs well on large batches")
        print(f"   Caught {correct_corrections}/100 errors ({success_rate:.1f}%)")
    else:
        print(f"⚠️  NEEDS IMPROVEMENT: Only caught {correct_corrections}/100 errors")

    if false_positives <= 5:  # Less than 2% false positive rate
        print(f"✅ LOW FALSE POSITIVES: Only {false_positives}/300 ({false_positives/3:.1f}%)")
    else:
        print(f"⚠️  HIGH FALSE POSITIVES: {false_positives}/300 ({false_positives/3:.1f}%)")

    print("=" * 80)

    return corrected_samples


if __name__ == "__main__":
    asyncio.run(test_400_samples())
