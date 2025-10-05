import random
import logging
import asyncio
from typing import List
from src.payment_classifier.personas.personas import PersonasSeeder
from pydantic import BaseModel
from app.core.schemas.workflow import Sample, Result
from app.core.services.data_manager.data_manager import DataManager
from app.core.services.eval_data_manager.eval_data_manager import EvalDataManager
from src.payment_classifier.llm.base import BaseLLM
from src.payment_classifier.prompts.base import BasePromptManager

logger = logging.getLogger(__name__)

class EvalGenerator:
    EVAL_PROMPT_KEY = "v2/eval/seed"

    def __init__(self, llm: BaseLLM, prompt_mgr: BasePromptManager, eval_data_manager: EvalDataManager = None, generate_open_intent: bool = True):
        self.llm = llm
        self.prompt_mgr = prompt_mgr
        self.eval_data_manager = eval_data_manager
        self.generate_open_intent = generate_open_intent

    def _calculate_eval_generation_params(self, expect_total_message: int) -> tuple[int, int, int]:
        """
        Calculate generation parameters for evaluation data.

        Args:
            expect_total_message: Expected total number of messages to generate
            is_open_intent: Whether this is for open intent generation

        Returns:
            Tuple of (parallel_generations, messages_per_batch, batches_needed)
        """
        messages_per_api_call = 30

        # Different limits for intent vs open intent
        max_parallel = 10
        min_parallel = 3

        # Calculate optimal parallel generations
        parallel_gens = min(max_parallel, max(min_parallel, expect_total_message // messages_per_api_call))

        # Calculate batches needed
        messages_per_batch_cycle = parallel_gens * messages_per_api_call
        batches_needed = max(1, (expect_total_message + messages_per_batch_cycle - 1) // messages_per_batch_cycle)

        return parallel_gens, messages_per_api_call, batches_needed

    async def fresh_gen(self, human_seeds: List[Sample] = None, human_seed_messages: List[str] = None, expect_total_intent_message: int = None) -> dict:
        """
        Generate intent test cases and optionally open intent messages in the same iteration folder.

        Args:
            human_seeds: Human seed samples for intent generation
            human_seed_messages: Human seed messages for open intent generation

        Returns:
            Dictionary containing generated data and iteration info
        """
        if human_seeds is None:
            human_seeds = []
        if human_seed_messages is None:
            human_seed_messages = []


        if self.generate_open_intent:
            logger.info("Starting fresh generation of both intent and open intent evaluation data...")
        else:
            logger.info("Starting fresh generation of intent evaluation data only...")

        # Generate intent test cases first (this creates the iteration number)
        intent_results = await self.intent_gen(human_seeds, expect_total_intent_message)
        iteration_number = self.eval_data_manager.get_latest_item_number()

        # Conditionally generate open intent messages using the same iteration number
        return {
            "iteration_number": iteration_number,
            "intent_samples": intent_results,
            "intent_count": len(intent_results),
        }

    async def intent_gen(self, human_seeds: List[Sample] = None, target_messages: int = None) -> List[Sample]:
        """
        Generate intent test cases using parallel generation.

        Args:
            human_seeds: Human seed samples for intent generation
            expect_total_message: Expected total number of messages to generate

        Returns:
            Generated intent samples
        """
        if human_seeds is None:
            human_seeds = []

        # Calculate generation parameters
        parallel_intent_generations, batch_size, total_batch_per_gen = self._calculate_eval_generation_params(target_messages, False)
        logger.info(f"Intent generation: targeting {target_messages} messages with {parallel_intent_generations} parallel calls, {batch_size} per call, {total_batch_per_gen} batches")

        # Load the prompt at the caller level
        prompt = self.prompt_mgr.get_prompt(self.EVAL_PROMPT_KEY)

        # First iteration with parallel generation
        seed_examples = random.sample(human_seeds, k=min(len(human_seeds), 2))

        # Generate parallel batches
        generation_tasks = []
        for _ in range(parallel_intent_generations):
            generation_tasks.append(self._gen_eval(seed_examples, prompt, batch_size))

        # Run all generations in parallel
        parallel_results = await asyncio.gather(*generation_tasks)

        # Combine all parallel results
        all_results = []
        for results in parallel_results:
            all_results.extend(results)

        # Internal deduplication within the batch
        internal_deduped = await self.eval_data_manager._dedup_helper.deduplicate(all_results)

        # Save initial deduplicated results
        iteration_number = self.eval_data_manager.save(internal_deduped)

        # Subsequent batches with parallel generation
        for _ in range(total_batch_per_gen - 1):
            # Create seeds for each parallel generation
            generation_tasks = []
            for _ in range(parallel_intent_generations):
                # Make new seed for each parallel task
                new_seed = random.sample(all_results, k=min(len(all_results), 2))
                new_human_seeds = random.sample(human_seeds, k=min(len(human_seeds), 6))
                seed = new_seed + new_human_seeds
                generation_tasks.append(self._gen_eval(seed, prompt, total_message_gen_per_batch))

            # Run parallel generations
            parallel_results = await asyncio.gather(*generation_tasks)
            logger.debug(f"Generated {sum(len(results) for results in parallel_results)} new evaluation examples across {parallel_intent_generations} parallel tasks")

            # Combine all parallel results
            batch_results = []
            for results in parallel_results:
                batch_results.extend(results)

            # Internal deduplication within the batch
            internal_deduped = await self.eval_data_manager._dedup_helper.deduplicate(batch_results)

            # Filter against existing iteration data and append
            new_data = await self.eval_data_manager.filter(internal_deduped, iteration_number)
            self.eval_data_manager.append(new_data, iteration_number)

            # Update all_results for next iteration
            all_results.extend(batch_results)

        logger.info(f"Generated evaluation samples with parallel processing in iteration {iteration_number}")
        return all_results

    async def _gen_eval(self, seed: List[Sample], prompt: str, batch_size: int = None) -> List[Sample]:
        """
        Generate evaluation data from given seed samples.

        Args:
            seed: Seed samples to base generation on
            prompt: The prompt template to use for generation

        Returns:
            Generated samples
        """
        # Use provided batch size or default
        personas_txt = "\n- ".join(PersonasSeeder.random(min(20, batch_size)))  # Limit personas to reasonable number
        seed_examples_txt = "\n".join([f"- {s.msg}" for s in seed])

        formatted_prompt = prompt.format(personas=personas_txt)

        data = (await self.llm.generate_structured_output([
            {"role": "system", "content": formatted_prompt},
            {"role": "user", "content": f"Examples: {seed_examples_txt}\n Continue generate {batch_size} diverse examples"}
        ], Result)).messages

        return data
