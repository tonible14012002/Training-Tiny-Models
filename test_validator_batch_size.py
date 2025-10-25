"""
Test validator with different batch sizes on the same samples.
Critical: Even small batches should have high accuracy.
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


async def test_batch_sizes():
    """Test the same samples with different batch sizes."""

    # These are ACTUAL production errors - should ALL be fixed
    samples = [
        Sample(msg="Send me the invoice for the 100$ payment!", label="payment_intent"),
        Sample(msg="Send me 30$ before the end of the week.", label="payment_intent"),
        Sample(msg="Send me 0.001 BTC when you can!", label="payment_intent"),
        Sample(msg="Remember to send me that 5$ for lunch.", label="payment_intent"),
        Sample(msg="Send me 1000 sats next week.", label="open_intent"),
        Sample(msg="Pay me 1000 msats for the artwork!", label="payment_intent"),
        Sample(msg="Pay me back the 10 bucks when you can!", label="payment_intent"),
        Sample(msg="Pay me 25 bucks tomorrow, please!", label="payment_intent"),
    ]

    expected_corrections = 8  # All should be payment_request

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
    print("TESTING DIFFERENT BATCH SIZES ON SAME SAMPLES")
    print("=" * 80)
    print(f"\nTotal samples: {len(samples)}")
    print(f"All samples are mislabeled and should be corrected to payment_request")

    # Test with different batch sizes
    batch_sizes = [5, 8, 10, 30]  # 5=2 batches, 8=1 batch, 10=1 batch, 30=1 batch

    results = {}

    for batch_size in batch_sizes:
        print(f"\n" + "=" * 80)
        print(f"TESTING WITH BATCH_SIZE = {batch_size}")
        print("=" * 80)

        corrected_samples = await validator.validate_and_fix(samples, batch_size=batch_size)

        # Count corrections
        correct_fixes = sum(1 for c in corrected_samples if c.label == "payment_request")

        results[batch_size] = {
            'correct': correct_fixes,
            'total': len(samples),
            'accuracy': correct_fixes / len(samples) * 100
        }

        print(f"\nResults for batch_size={batch_size}:")
        print(f"  Correct fixes: {correct_fixes}/{len(samples)}")
        print(f"  Accuracy:      {correct_fixes/len(samples)*100:.1f}%")

        # Show which ones failed
        if correct_fixes < len(samples):
            print(f"\n  ❌ Failed samples:")
            for i, (orig, corr) in enumerate(zip(samples, corrected_samples)):
                if corr.label != "payment_request":
                    print(f"    [{i}] {corr.label:20} | '{corr.msg}'")

    # Summary
    print("\n" + "=" * 80)
    print("SUMMARY - BATCH SIZE COMPARISON")
    print("=" * 80)
    print(f"{'Batch Size':<12} | {'Batches':<10} | {'Correct':<10} | {'Accuracy':<10}")
    print("-" * 80)

    for batch_size in batch_sizes:
        num_batches = (len(samples) + batch_size - 1) // batch_size
        r = results[batch_size]
        status = "✅" if r['accuracy'] == 100 else "❌"
        print(f"{status} {batch_size:<10} | {num_batches:<10} | {r['correct']}/{r['total']:<8} | {r['accuracy']:.1f}%")

    print("\n" + "=" * 80)
    print("ANALYSIS:")
    print("=" * 80)

    best_accuracy = max(r['accuracy'] for r in results.values())
    worst_accuracy = min(r['accuracy'] for r in results.values())

    if best_accuracy == worst_accuracy == 100:
        print("✅ EXCELLENT: Consistent 100% accuracy across all batch sizes")
    elif worst_accuracy >= 90:
        print(f"✅ GOOD: All batch sizes achieve ≥90% accuracy")
        print(f"   Range: {worst_accuracy:.1f}% - {best_accuracy:.1f}%")
    else:
        print(f"⚠️  CONCERNING: Accuracy varies significantly")
        print(f"   Range: {worst_accuracy:.1f}% - {best_accuracy:.1f}%")
        print(f"   Even simple 8-sample batches should be 100% accurate")

    print("=" * 80)

    return results


if __name__ == "__main__":
    asyncio.run(test_batch_sizes())
