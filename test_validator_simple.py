"""
Simple test for DataValidator with a few mislabeled samples.
Quick test to verify the validator is working correctly.
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


async def test_simple_validation():
    """Simple test with a few samples."""

    # Test samples - mix of correct and incorrect labels
    samples = [
        # Correctly labeled
        Sample(msg="I'll send you $50 later", label="payment_intent"),

        # WRONG: This is a payment_request, not payment_intent
        Sample(msg="Send me 10 BTC please", label="payment_intent"),

        # Correctly labeled
        Sample(msg="Thanks for the payment", label="open_intent"),

        # WRONG: This is payment_intent, not open_intent
        Sample(msg="I'll transfer $100 to Alice", label="open_intent"),

        # WRONG: This is open_intent (past tense), not payment_intent
        Sample(msg="I already sent the money", label="payment_intent"),

        # Correctly labeled
        Sample(msg="You owe me $20", label="payment_request"),
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

    print("=" * 70)
    print("SIMPLE VALIDATOR TEST")
    print("=" * 70)

    print("\nOriginal samples:")
    for i, sample in enumerate(samples):
        print(f"  [{i}] {sample.label:20} | {sample.msg}")

    print("\nRunning validation...")
    corrected_samples = await validator.validate_and_fix(samples)

    print("\nCorrected samples:")
    for i, (original, corrected) in enumerate(zip(samples, corrected_samples)):
        changed = "→" if original.label != corrected.label else " "
        new_label = corrected.label if original.label != corrected.label else ""
        print(f"  [{i}] {original.label:20} {changed} {new_label:20} | {corrected.msg}")

    print("\n" + "=" * 70)
    print("Expected corrections:")
    print("  [1] payment_intent → payment_request")
    print("  [3] open_intent → payment_intent")
    print("  [4] payment_intent → open_intent")
    print("=" * 70)


if __name__ == "__main__":
    asyncio.run(test_simple_validation())
