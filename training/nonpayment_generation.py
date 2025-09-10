from src.payment_classifier.llm import LiteLLMProvider, LLMSettings
from src.payment_classifier.prompts.inmemory_prompt_manager import InmemoryPromptManager
from training.config.settings import settings
from training.constant.seed import PAYMENT_TEXT_SEED_VARIATIONS

from pydantic import BaseModel
import asyncio
from typing import List
from datetime import datetime
import json
import os
import time

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
            "content": f"Generate {count} of examples."
        }
    ]

    response = await llm_provider.generate_structured_output(messages, AIResponse)
    return response.examples

def make_seed_texts(seed_variation):
    """Generate seed texts for payment-related messages."""
    def map_seed_text(item) -> str:
        result = {}
        result['relationships'] = ", ".join(item['relationships'])
        result['topics'] = ", ".join(item['topics'])

        result['name'] = item['name']
        return result

    return list(map(map_seed_text, seed_variation))

def next_seed_index(index: int, step: int, total_seed: int) -> int:
    return (index + step) % total_seed

async def main():
    # Setup
    now = datetime.now().timestamp()
    num_of_calls = 100
    now = datetime.now().timestamp()
    examples_per_call = 100
    # seeds = make_seed_texts(PAYMENT_TEXT_SEED_VARIATIONS)
    # rotate_step = 2

    output_dir = f"training_data/nonpayment_generation/{int(now)}"
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
    non_payment_related = prompt_mgr.get_prompt("non_payment_related_message")

    # Generate examples asynchronously
    tasks = []
    for i in range(num_of_calls):
        formatted_prompt = non_payment_related.format(
            count=examples_per_call,
        )

        task = batch_generate_examples(formatted_prompt, llm_model, examples_per_call)
        tasks.append(task)
    
    # Execute all tasks concurrently
    results: List[AIResponse] = await asyncio.gather(*tasks)
    
    # Save each batch to separate files and collect all examples
    all_examples = []
    for i, examples in enumerate(results):
        filename = f"batch_{i+1}_{int(now)}.json"
        filepath = os.path.join(output_dir, filename)

        # Convert examples to dict for JSON serialization
        example_strs = [example for example in examples]

        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(example_strs, f, indent=2, ensure_ascii=False)
        
        print(f"Saved {len(examples)} examples to {filepath}")
        
        # Add to consolidated list
        all_examples.extend(example_strs)
    
    # Save consolidated file with all examples
    final_filename = f"all_batches_consolidated_{int(now)}.json"
    final_filepath = os.path.join(output_dir, final_filename)
    
    with open(final_filepath, 'w', encoding='utf-8') as f:
        json.dump(all_examples, f, indent=2, ensure_ascii=False)
    
    total_examples = len(all_examples)
    print(f"Saved consolidated file with {total_examples} examples to {final_filepath}")
    print(f"Generated {total_examples} total examples across {num_of_calls} batches")
    

if __name__ == "__main__":
    asyncio.run(main())