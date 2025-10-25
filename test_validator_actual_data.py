"""Test validator on actual mislabeled data from the generated file"""
import asyncio
import json
from app.core.services.data_validator import DataValidator
from app.core.schemas.workflow import Sample
from app.core.models.models import LabelConfig
from src.payment_classifier.llm.litellm import LiteLLMProvider
from src.payment_classifier.prompts import InmemoryPromptManager
from src.payment_classifier.llm.settings import LLMSettings
from app.core.settings import settings

async def test():
    # These are ACTUAL samples from the generated file that are WRONG
    samples = [
        Sample(msg="Don't forget to pay me back the 60$ for last month!", label="payment_intent"),
        Sample(msg="Please pay me back the 5$ you owe!", label="payment_intent"),
        Sample(msg="Make sure to pay me back the 50$ soon!", label="payment_intent"),
        Sample(msg="Please remember to pay me 40$ next week!", label="payment_intent"),
        Sample(msg="I need you to pay me back 60$ ASAP.", label="payment_intent"),
        Sample(msg="I need you to pay me back those 50$ you borrowed.", label="payment_intent"),
        Sample(msg="Pay me back that 50$ when you can, no rush!", label="payment_intent"),
        Sample(msg="Pay me 5$ for the coffee I bought you.", label="payment_intent"),
        Sample(msg="Pay me back, I covered your dinner, it was $25!", label="payment_intent"),
        Sample(msg="Make sure to pay me the $40 for that gift!", label="payment_intent"),
    ]

    label_config = LabelConfig(
        id2label=json.dumps({"0": "payment_intent", "1": "payment_request", "2": "open_intent"}),
        label2id=json.dumps({"payment_intent": 0, "payment_request": 1, "open_intent": 2})
    )

    llm = LiteLLMProvider(
        settings=LLMSettings(llm_model_name="gpt-4o-mini", api_key=settings.OPENAI_API_KEY, temperature=0.0, num_retries=2)
    )

    validator = DataValidator(llm=llm, prompt_mgr=InmemoryPromptManager(), label_config=label_config)

    print("=" * 80)
    print("TESTING VALIDATOR ON ACTUAL MISLABELED DATA FROM GENERATED FILE")
    print("=" * 80)
    print(f"\nOriginal labels (all WRONG - should be payment_request):")
    for i, s in enumerate(samples):
        print(f"  [{i}] ❌ {s.label:20} | {s.msg}")

    print("\n" + "=" * 80)
    print("RUNNING VALIDATION...")
    print("=" * 80)

    corrected = await validator.validate_and_fix(samples)

    print("\nCorrected labels:")
    for i, (orig, corr) in enumerate(zip(samples, corrected)):
        if corr.label == "payment_request":
            marker = "✅"
            status = "FIXED"
        else:
            marker = "❌"
            status = "NOT FIXED"
        print(f"  [{i}] {marker} {status:12} {orig.label:20} → {corr.label:20}")
        print(f"       '{corr.msg}'")

    fixes = sum(1 for o, c in zip(samples, corrected) if c.label == "payment_request")
    print("\n" + "=" * 80)
    print(f"RESULT: {fixes}/{len(samples)} correctly fixed to payment_request")
    if fixes == len(samples):
        print("✅ ALL SAMPLES FIXED! Validator is working perfectly.")
    else:
        print(f"⚠️  {len(samples) - fixes} samples were NOT fixed.")
    print("=" * 80)

asyncio.run(test())
