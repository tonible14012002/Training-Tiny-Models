from src.payment_classifier.personas.personas import PersonasSeeder
from .data_generator import DataGenerator
from app.core.schemas import Sample
from typing import List
import asyncio
import random
import logging

logger = logging.getLogger(__name__)

class DataGeneratorV2(DataGenerator):
    SEED_PROMPT_KEY = "v2/train/seed"
    DEFAULT_TOTAL_MESSAGES = 600  # Default target for V2 (higher due to parallel generation)

    def _calculate_v2_generation_params(self, expect_total_message: int) -> tuple[int, int, int]:
        """
        Calculate PARALLEL_GENERATIONS, TOTAL_MESSAGE_PER_BATCH, and batches for V2 generator.

        Args:
            expect_total_message: Expected total number of messages to generate

        Returns:
            Tuple of (PARALLEL_GENERATIONS, TOTAL_MESSAGE_PER_BATCH, batches_needed)
        """
        # Each API call generates ~30 messages
        messages_per_api_call = self.ESTIMATED_MESSAGES_PER_API_CALL

        # For V2, we run parallel generations, so total per batch = parallel_gens * messages_per_api_call
        # But we also run multiple batches, so: total = parallel_gens * messages_per_api_call * batches

        # Optimize for reasonable parallel calls (not too many to avoid rate limits)
        max_parallel = 20
        min_parallel = 5

        # Start with a reasonable parallel count
        parallel_gens = min(max_parallel, max(min_parallel, expect_total_message // messages_per_api_call))

        # Calculate how many batches we need
        messages_per_batch_cycle = parallel_gens * messages_per_api_call
        batches_needed = max(1, (expect_total_message + messages_per_batch_cycle - 1) // messages_per_batch_cycle)

        return parallel_gens, messages_per_api_call, batches_needed

    async def fresh_gen(self, human_seeds: List[Sample], expect_total_message: int = None) -> List[Sample]:
        """Generate fresh data using the default prompt"""
        # Use provided target or default
        target_messages = expect_total_message or self.DEFAULT_TOTAL_MESSAGES

        # Calculate generation parameters
        parallel_generations, total_message_per_batch, batches = self._calculate_v2_generation_params(target_messages)
        logger.info(f"V2 Dynamic calculation: targeting {target_messages} messages with {parallel_generations} parallel calls, {total_message_per_batch} per call, {batches} batches")

        # Load the default prompt
        prompt = self.prompt_mgr.get_prompt(self.SEED_PROMPT_KEY)

        # Use the common generation logic with calculated parameters
        return await self.iterative_gen(human_seeds, prompt, parallel_generations, total_message_per_batch, batches)
        
    async def iterative_gen(self, human_seeds: List[Sample], prompt: str, parallel_generations: int = None, total_message_per_batch: int = None, total_batches: int = None) -> List[Sample]:
        """Core generation logic that can be reused with different prompts"""
        # Use provided parameters or calculate defaults
        if parallel_generations is None or total_message_per_batch is None or total_batches is None:
            parallel_generations, total_message_per_batch, total_batches = self._calculate_v2_generation_params(self.DEFAULT_TOTAL_MESSAGES)

        # First iterate with parallel generation
        seed_examples = random.sample(human_seeds, k=min(len(human_seeds), 2))

        # Generate parallel batches
        generation_tasks = []
        for _ in range(parallel_generations):
            generation_tasks.append(self._gen(seed_examples, prompt))

        # Run all generations in parallel
        parallel_results = await asyncio.gather(*generation_tasks)

        # Combine all parallel results
        all_results = []
        for results in parallel_results:
            all_results.extend(results)

        # Deduplicate within the combined results
        internal_deduped = await self._internal_deduplicate(all_results)

        # Filter and deduplicate against existing data
        filtered_results = await self.data_manager.deduplicate(internal_deduped)
        self.data_manager.save(filtered_results)

        # Subsequent batches with parallel generation
        for _ in range(total_batches - 1):
            # Create seeds for each parallel generation
            generation_tasks = []
            for _ in range(parallel_generations):
                # Make new seed for each parallel task
                new_seed = random.sample(all_results, k=min(len(all_results), 2))
                new_human_seeds = random.sample(human_seeds, k=min(len(human_seeds), 6))
                seed = new_seed + new_human_seeds
                generation_tasks.append(self._gen(seed, prompt))

            # Run parallel generations
            parallel_results = await asyncio.gather(*generation_tasks)
            logger.debug(f"Generated {sum(len(results) for results in parallel_results)} new examples across {parallel_generations} parallel tasks")

            # Combine all parallel results
            batch_results = []
            for results in parallel_results:
                batch_results.extend(results)

            # Deduplicate within the batch
            internal_deduped = await self._internal_deduplicate(batch_results)

            # Filter against existing data and save
            new_data = await self.data_manager.filter(internal_deduped)
            self.data_manager.save(new_data)

            # Update all_results for next iteration
            all_results.extend(batch_results)

        return all_results

    async def fix_gen(self, prompt: str, amount: int = None) -> List[Sample]:
        """Generate data using a custom prompt from PromptBuilder for fixing errors"""
        # Load human seeds for generation
        human_seeds = self.data_manager.get_human_seeds()

        # Use provided amount or default
        target_messages = amount or self.DEFAULT_TOTAL_MESSAGES

        # Calculate generation parameters
        parallel_generations, total_message_per_batch, total_batches = self._calculate_v2_generation_params(target_messages)
        logger.info(f"Fix generation: targeting {target_messages} messages with {parallel_generations} parallel calls, {total_message_per_batch} per call, {total_batches} batches")

        return await self.iterative_gen(human_seeds, prompt, parallel_generations, total_message_per_batch, total_batches)

    async def _internal_deduplicate(self, samples: List[Sample]) -> List[Sample]:
        """Deduplicate samples within the batch itself before filtering against existing data"""
        if not samples:
            return samples

        # Use the same deduplication logic as data_manager but only within this batch
        from app.utils.scorer import EvaluationUtils

        deduped_samples = []

        for sample in samples:
            is_duplicate = False
            for existing_sample in deduped_samples:
                rouge_score = await EvaluationUtils.score_rouge(
                    ref=existing_sample.msg,
                    pred=sample.msg,
                    rouge_type="rougeL",
                    mode="precision"
                )

                if rouge_score >= 0.6:  # Same threshold as in data_manager
                    is_duplicate = True
                    break

            if not is_duplicate:
                deduped_samples.append(sample)

        logger.debug(f"Internal deduplication: {len(samples)} -> {len(deduped_samples)} samples")
        return deduped_samples