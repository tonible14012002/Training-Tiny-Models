


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

async def batch_label_messages(prompt: str, llm_provider: LiteLLMProvider, messages_to_label: List[str]):
    formatted_messages = "\n".join([f"{i+1}. {msg}" for i, msg in enumerate(messages_to_label)])
    
    messages = [
        {
            "role": "system",
            "content": prompt
        },
        {
            "role": "user", 
            "content": f"Label each of these messages as PAYMENT or NON-PAYMENT. Return JSON array with format: [{{'message': '<input_message>', 'label': '<PAYMENT or NON-PAYMENT>'}}]\n\nMessages to label:\n{formatted_messages}"
        }
    ]

    response = await llm_provider.generate_output(messages)
    
    try:
        # Parse JSON response
        import json
        labels = json.loads(response)
        return labels
    except:
        # Fallback if JSON parsing fails
        return [{"message": msg, "label": "NON-PAYMENT"} for msg in messages_to_label]

async def main():
    # Setup
    now = datetime.now().timestamp()
    batch_size = 1500  # Messages per batch group
    num_llm_calls_per_batch = 100  # LLM calls per batch group
    messages_per_llm_call = 50  # Messages per individual LLM call
    
    output_dir = f"training_data/labeled/{int(now)}"
    os.makedirs(output_dir, exist_ok=True)

    # Read messages from file
    with open('final_1757647158.json', 'r', encoding='utf-8') as f:
        data = json.load(f)
        import random
        random.shuffle(data)
    
    # Extract text messages
    all_messages = []
    for item in data:
        if isinstance(item, dict) and "text" in item:
            all_messages.append(item["text"])
        elif isinstance(item, str):
            all_messages.append(item)

    print(f"Total messages to process: {len(all_messages)}")

    # Setup LLM
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
    label_payment_intent = prompt_mgr.get_prompt("label_payment_intent")

    # Split messages into batch groups of 1500
    batch_groups = []
    for i in range(0, len(all_messages), batch_size):
        batch_groups.append(all_messages[i:i + batch_size])

    all_labeled_data = []
    
    for batch_group_idx, batch_group in enumerate(batch_groups):
        print(f"Processing batch group {batch_group_idx + 1}/{len(batch_groups)} ({len(batch_group)} messages)")
        
        # Create 100 LLM tasks, each processing 50 messages (or remaining messages)
        tasks = []
        for llm_call_idx in range(num_llm_calls_per_batch):
            start_idx = llm_call_idx * messages_per_llm_call
            end_idx = min(start_idx + messages_per_llm_call, len(batch_group))
            
            if start_idx >= len(batch_group):
                break
                
            messages_chunk = batch_group[start_idx:end_idx]
            task = batch_label_messages(label_payment_intent, llm_model, messages_chunk)
            tasks.append(task)
        
        # Execute all tasks in this batch group concurrently
        print(f"Executing {len(tasks)} LLM calls in parallel...")
        batch_results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Process results
        batch_labeled_data = []
        for result in batch_results:
            if isinstance(result, Exception):
                print(f"Error in LLM call: {result}")
                continue
            if isinstance(result, list):
                batch_labeled_data.extend(result)
        
        # Save batch group results
        batch_filename = f"batch_group_{batch_group_idx + 1}_{int(now)}.json"
        batch_filepath = os.path.join(output_dir, batch_filename)
        
        with open(batch_filepath, 'w', encoding='utf-8') as f:
            json.dump(batch_labeled_data, f, indent=2, ensure_ascii=False)
        
        print(f"Saved {len(batch_labeled_data)} labeled messages to {batch_filepath}")
        all_labeled_data.extend(batch_labeled_data)
        
        # Wait before processing next batch group (except for the last one)
        if batch_group_idx < len(batch_groups) - 1:
            print("Waiting 5 seconds before next batch group...")
            await asyncio.sleep(7)
    
    # Save consolidated file with all labeled data
    final_filename = f"all_labeled_messages_{int(now)}.json"
    final_filepath = os.path.join(output_dir, final_filename)
    
    with open(final_filepath, 'w', encoding='utf-8') as f:
        json.dump(all_labeled_data, f, indent=2, ensure_ascii=False)
    
    print(f"Processing complete!")
    print(f"Total labeled messages: {len(all_labeled_data)}")
    print(f"Consolidated file saved to: {final_filepath}")
    

if __name__ == "__main__":
    asyncio.run(main())