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

async def batch_generate_examples(prompt: str, llm_provider: LiteLLMProvider, count: int = 100, to_excludes = []) -> List[AIResponse]:

    formatted = "\n-".join(to_excludes)
    
    messages = [
        {
            "role": "system",
            "content": prompt
        },
        {
            "role": "user", 
            "content": f"Input Messages:\n {formatted}"
        }
    ]

    response = await llm_provider.generate_structured_output(messages, AIResponse)
    return response.examples

async def main():
    # Setup
    now = datetime.now().timestamp()
    num_of_calls = 100
    now = datetime.now().timestamp()
    examples_per_call = 40
    
    output_dir = f"training_data/non_payment/{int(now)}"
    os.makedirs(output_dir, exist_ok=True)

    # Read payment msgs from file
    with open('training_data/human_sms.json', 'r', encoding='utf-8') as f:
        payment_msgs = json.load(f)
        import random
        random.shuffle(payment_msgs)

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
    transform_to_non_payment = prompt_mgr.get_prompt("transform_text_to_non_payment_relate")

    # Generate examples asynchronously
    tasks = []
    for i in range(num_of_calls):
        # seed_index = next_seed_index(i, rotate_step, len(seeds))

        seed_msgs = payment_msgs[i*examples_per_call:(i+1)*examples_per_call]

        task = batch_generate_examples(transform_to_non_payment, llm_model, examples_per_call, seed_msgs)
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