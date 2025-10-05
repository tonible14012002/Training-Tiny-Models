from src.payment_classifier.personas.personas import PersonasSeeder
from src.payment_classifier.llm.base import BaseLLM
from src.payment_classifier.prompts.base import BasePromptManager
from app.core.services.data_manager import DataManager
from app.core.schemas.workflow import Result
from app.core.schemas import Sample
from typing import List, Dict, Union
import asyncio
import random
import logging

logger = logging.getLogger(__name__)

class DataGeneratorV2:
    SEED_PROMPT_KEY = "v2/train/seed"
    DEFAULT_TOTAL_MESSAGES = 600  # Default target for V2 (higher due to parallel generation)

    def __init__(self, llm: BaseLLM, prompt_mgr: BasePromptManager, data_manager: DataManager):
        self.llm = llm
        self.prompt_mgr = prompt_mgr
        self.data_manager = data_manager

    async def fresh_gen(self, human_seeds: List[Sample], expect_total_message: int = None) -> List[Sample]:
        """Generate fresh data using the default prompt"""
        # Use provided target or default
        target_messages = expect_total_message or self.DEFAULT_TOTAL_MESSAGES
        parallel_generations, total_message_per_batch, batches = self._calculate_v2_generation_params(target_messages)
        prompt = self.prompt_mgr.get_prompt(self.SEED_PROMPT_KEY)

        return await self.iterative_gen(human_seeds, prompt, parallel_generations, total_message_per_batch, batches)

    async def fresh_gen_v2(self, human_seeds: List[Sample], expect_total_each_label: Dict[Union[str, int], int]) -> List[Sample]:
        """Generate fresh data with label-based quantity tracking

        Args:
            human_seeds: Initial seed samples for generation
            expect_total_each_label: Dictionary mapping label (str or int) to expected quantity
                                    e.g., {"payment_intent": 200, "payment_request": 200, "open_intent": 200}
                                    or {0: 200, 1: 200, 2: 200}

        Returns:
            List of all generated samples
        """
        prompt = self.prompt_mgr.get_prompt(self.SEED_PROMPT_KEY)

        # Initialize quantity tracker for each label
        quantity_tracker: Dict[Union[str, int], List[Sample]] = {
            label: [] for label in expect_total_each_label.keys()
        }

        # Calculate parallel generation parameters based on total expected
        total_expected = sum(expect_total_each_label.values())
        parallel_generations, messages_per_call, _ = self._calculate_v2_generation_params(total_expected)

        all_results = []
        previous_batch_results = []
        iteration = 0

        # Continue generating until all labels have enough samples
        while not self._is_quantity_sufficient(quantity_tracker, expect_total_each_label):
            iteration += 1
            logger.info(f"Generation iteration {iteration}: Current counts = {self._get_current_counts(quantity_tracker)}, Target = {expect_total_each_label}")

            # Prepare seed examples
            if iteration == 1:
                # First iteration: use human seeds
                seed_examples = random.sample(human_seeds, k=min(len(human_seeds), 2))
            else:
                # Subsequent iterations: mix previous batch results with human seeds
                new_seed = random.sample(previous_batch_results, k=min(len(previous_batch_results), 2))
                new_human_seeds = random.sample(human_seeds, k=min(len(human_seeds), 6))
                seed_examples = new_seed + new_human_seeds

            # Generate parallel batches
            generation_tasks = []
            for _ in range(parallel_generations):
                generation_tasks.append(self._gen(seed_examples, prompt, messages_per_call))

            # Run all generations in parallel
            parallel_results = await asyncio.gather(*generation_tasks)

            # Combine all parallel results
            batch_results = []
            for results in parallel_results:
                batch_results.extend(results)

            # Deduplicate within the batch
            internal_deduped = await self.data_manager._dedup_helper.deduplicate(batch_results)

            # Filter against existing data
            filtered_results = await self.data_manager.filter(internal_deduped)

            # Append filtered results to the correct label tracker
            for sample in filtered_results:
                if sample.label in quantity_tracker:
                    quantity_tracker[sample.label].append(sample)

            # Save the filtered results
            self.data_manager.save(filtered_results)

            # Update all_results and previous_batch_results for next iteration
            all_results.extend(batch_results)
            previous_batch_results = batch_results

            logger.debug(f"Iteration {iteration}: Generated {len(batch_results)} samples, {len(filtered_results)} after filtering")

        logger.info(f"Generation complete after {iteration} iterations. Final counts: {self._get_current_counts(quantity_tracker)}")

        return all_results

    def _is_quantity_sufficient(self, quantity_tracker: Dict[Union[str, int], List[Sample]],
                                expect_total_each_label: Dict[Union[str, int], int]) -> bool:
        """Check if all labels have enough samples"""
        for label, expected_count in expect_total_each_label.items():
            if len(quantity_tracker[label]) < expected_count:
                return False
        return True

    def _get_current_counts(self, quantity_tracker: Dict[Union[str, int], List[Sample]]) -> Dict[Union[str, int], int]:
        """Get current sample counts for each label"""
        return {label: len(samples) for label, samples in quantity_tracker.items()}
        
    async def iterative_gen(self, human_seeds: List[Sample], prompt: str, parallel_generations: int = None, total_message_per_batch: int = None, total_batches: int = None, track_saved: bool = False) -> tuple[List[Sample], List[Sample]] | List[Sample]:
        """Core generation logic that can be reused with different prompts

        Args:
            human_seeds: Initial seed samples for generation
            prompt: Prompt to use for generation
            parallel_generations: Number of parallel generations to run
            total_message_per_batch: Number of messages per batch
            total_batches: Total number of batches to run
            track_saved: If True, returns (all_results, saved_samples). If False, returns all_results only

        Returns:
            If track_saved=False: List of all generated samples
            If track_saved=True: Tuple of (all_results, saved_samples) where saved_samples are the deduplicated ones
        """
        saved_samples = [] if track_saved else None

        # First iterate with parallel generation
        seed_examples = random.sample(human_seeds, k=min(len(human_seeds), 2))

        # Generate parallel batches
        generation_tasks = []
        for _ in range(parallel_generations):
            generation_tasks.append(self._gen(seed_examples, prompt, total_message_per_batch))

        # Run all generations in parallel
        parallel_results = await asyncio.gather(*generation_tasks)

        # Combine all parallel results
        all_results = []
        for results in parallel_results:
            all_results.extend(results)

        # Deduplicate within the combined results
        internal_deduped = await self.data_manager._dedup_helper.deduplicate(all_results)
        
        # Filter against existing data
        filtered_results = await self.data_manager.filter(internal_deduped)
        self.data_manager.save(filtered_results)

        if track_saved:
            saved_samples.extend(filtered_results)

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
            internal_deduped = await self.data_manager._dedup_helper.deduplicate(batch_results)

            # Filter against existing data and save
            new_data = await self.data_manager.filter(internal_deduped)
            self.data_manager.save(new_data)

            if track_saved:
                saved_samples.extend(new_data)

            # Update all_results for next iteration
            all_results.extend(batch_results)

        if track_saved:
            return all_results, saved_samples
        return all_results

    async def fix_gen(self, human_seeds: List[Sample], prompt: str, amount: int = None) -> tuple[List[Sample], str, int]:
        """Generate data using a custom prompt from PromptBuilder for fixing errors

        This method:
        1. Deduplicates and filters against existing data (standard behavior)
        2. Appends deduplicated data to the main data file
        3. Also saves deduplicated data to a separate versioned file for tracking

        Returns:
            Tuple of (all_results, versioned_file_path, saved_count)
            - all_results: List of all generated samples (before deduplication)
            - versioned_file_path: Path to the versioned file created
            - saved_count: Number of samples saved after deduplication
        """
        # Use provided amount or default
        target_messages = amount or self.DEFAULT_TOTAL_MESSAGES

        # Calculate generation parameters
        parallel_generations, total_message_per_batch, total_batches = self._calculate_v2_generation_params(target_messages)
        logger.info(f"Fix generation: targeting {target_messages} messages with {parallel_generations} parallel calls, {total_message_per_batch} per call, {total_batches} batches")

        # Track saved samples to save them to a separate file
        all_results, saved_samples = await self.iterative_gen(
            human_seeds, prompt, parallel_generations, total_message_per_batch, total_batches, track_saved=True
        )

        # Generate a unique suffix with timestamp
        from datetime import datetime
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        file_suffix = f"fix_gen_{timestamp}"

        # Save deduplicated samples to a separate versioned file
        versioned_file_path = None
        saved_count = len(saved_samples) if saved_samples else 0

        if saved_samples:
            versioned_file_path = self.data_manager.save_to_versioned_file(saved_samples, file_suffix)
            logger.info(f"Saved {saved_count} deduplicated samples to {versioned_file_path}")
        else:
            logger.info("No new samples were saved after deduplication")

        return all_results, versioned_file_path, saved_count

    def _calculate_v2_generation_params(self, expect_total_message: int) -> tuple[int, int, int]:
        """
        Calculate PARALLEL_GENERATIONS, TOTAL_MESSAGE_PER_BATCH, and batches for V2 generator.

        Args:
            expect_total_message: Expected total number of messages to generate

        Returns:
            Tuple of (PARALLEL_GENERATIONS, TOTAL_MESSAGE_PER_BATCH, batches_needed)
        """
        # Each API call generates ~30 messages
        messages_per_api_call = 30

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

    async def _gen(self, seed: List[Sample], prompt: str, count: int) -> List[Sample]:
        '''
        Generate data from given seed and prompt
        '''
        personas_txt = "n- ".join(PersonasSeeder.random(10))  # Use a reasonable default for personas

        seed_examples_txt = "\n".join([f"- {s.msg}" for s in seed])
        formatted_prompt = prompt.format(personas=personas_txt)

        user_input = f"## Examples: {seed_examples_txt} \nContinue generate {count} diverse examples"

        data = (await self.llm.generate_structured_output([
            {"role": "system", "content": formatted_prompt},
            {"role": "user", "content": user_input}
        ], Result)).messages

        return data
