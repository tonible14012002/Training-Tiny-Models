from src.payment_classifier.llm import LiteLLMProvider, LLMSettings
from src.payment_classifier.prompts.inmemory_prompt_manager import InmemoryPromptManager
from training.config.settings import settings
# from training.constant.seed import PAYMENT_TEXT_SEED_VARIATIONS

from pydantic import BaseModel
import asyncio
from typing import List
from datetime import datetime
import json
import os

class AIResponse(BaseModel):
    examples: List[str]

async def batch_generate_examples(prompt: str, llm_provider: LiteLLMProvider, count: int = 100) -> List[AIResponse]:
    messages = [
        {
            "role": "system",
            "content": prompt
        },
        {
            "role": "user", 
            "content": f"Generate \n {count} messages"
        }
    ]

    response = await llm_provider.generate_structured_output(messages, AIResponse)
    return response.examples

async def main():
    # Setup
    now = datetime.now().timestamp()
    num_of_calls = 100
    examples_per_call = 20

    output_dir = f"training_data/payment_requests/{int(now)}"
    os.makedirs(output_dir, exist_ok=True)


    prompt_mgr = InmemoryPromptManager()
    llm_model = LiteLLMProvider(
        settings=LLMSettings(
            api_key=settings.OPENAI_API_KEY,
            llm_model_name="gpt-4o-mini",
            temperature=0.8,
            num_retries=3,
            max_tokens=16000,
        )
    )

    # Get the payment intent prompt
    transform_text_to_payment = prompt_mgr.get_prompt("payment_related_no_seed")

    # Generate examples asynchronously
    tasks = []
    for i in range(num_of_calls):
        # subset seed messages for each call
        task = batch_generate_examples(transform_text_to_payment, llm_model, examples_per_call)
        tasks.append(task)
    
    # Execute all tasks concurrently
    results: List[AIResponse] = await asyncio.gather(*tasks)
    
    # Save each batch to separate files and collect all examples
    all_examples = []
    for i, examples in enumerate(results):
        example_strs = [{
            "text": example,
            "label": 1
        } for example in examples]
        all_examples.extend(example_strs)
    
    # Save consolidated file with all examples
    final_filename = f"payment_request_{int(now)}.json"
    final_filepath = os.path.join(output_dir, final_filename)
    
    with open(final_filepath, 'w', encoding='utf-8') as f:
        json.dump(all_examples, f, indent=2, ensure_ascii=False)
    
    total_examples = len(all_examples)
    print(f"Saved consolidated file with {total_examples} examples to {final_filepath}")
    print(f"Generated {total_examples} total examples across {num_of_calls} batches")
    

if __name__ == "__main__":
    asyncio.run(main())