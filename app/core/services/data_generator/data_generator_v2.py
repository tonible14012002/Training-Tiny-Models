from src.payment_classifier.personas.personas import PersonasSeeder
from src.payment_classifier.llm.base import BaseLLM
from src.payment_classifier.prompts.base import BasePromptManager
from app.core.services.data_manager import DataManager
from app.core.schemas.workflow import Result
from app.core.schemas import Sample
from typing import List, Dict, Union, Optional, Callable, Awaitable
import asyncio
import random
import logging

logger = logging.getLogger(__name__)

class DataGeneratorV2:
    SEED_PROMPT_KEY = "v2/train/seed"

    def __init__(self, llm: BaseLLM, prompt_mgr: BasePromptManager, data_manager: DataManager):
        self.llm = llm
        self.prompt_mgr = prompt_mgr
        self.data_manager = data_manager

    async def fresh_gen_v2(
        self,
        human_seeds: List[Sample],
        expect_total_each_label: Dict[Union[str, int], int],
        on_batch_generated: Optional[Callable[[int, List[Sample], str], Awaitable[None]]] = None,
        composal_ds_mgr: Optional[DataManager] = None
    ) -> tuple[List[Sample], str]:
        """Generate fresh data with label-based quantity tracking

        Args:
            human_seeds: Initial seed samples for generation
            expect_total_each_label: Dictionary mapping label (str or int) to expected quantity
                                    e.g., {"payment_intent": 200, "payment_request": 200, "open_intent": 200}
                                    or {0: 200, 1: 200, 2: 200}
            on_batch_generated: Optional async callback called after each batch generation
                               Signature: async def callback(batch_number: int, samples: List[Sample], temp_file_path: str)

        Returns:
            Tuple of (final_samples, base_file_path)
            - final_samples: List of final samples (exact count per label as specified)
            - base_file_path: Path to the base data file where samples are saved
        """
        prompt = self.prompt_mgr.get_prompt(self.SEED_PROMPT_KEY)

        # Initialize quantity tracker for each label
        quantity_tracker: Dict[Union[str, int], List[Sample]] = {
            label: [] for label in expect_total_each_label.keys()
        }

        # Create temp file path for incremental saves
        from datetime import datetime
        from pathlib import Path
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        temp_file_path = Path(self.data_manager.LOCAL_FILE).parent / f"temp_gen_{timestamp}.jsonl"
        logger.info(f"Incremental saves will be written to: {temp_file_path}")

        # Calculate parallel generation parameters based on total expected
        total_expected = sum(expect_total_each_label.values())
        parallel_generations, messages_per_call, _ = self._calculate_v2_generation_params(total_expected)

        all_results = []
        previous_batch_results = []
        iteration = 0

        # Continue generating until all labels have enough samples
        while not self._is_quantity_sufficient(quantity_tracker, expect_total_each_label):
            iteration += 1
            current_counts = self._get_current_counts(quantity_tracker)
            logger.info(f"Generation iteration {iteration}: Current counts = {current_counts}, Target = {expect_total_each_label}")

            # Build rebalance instruction to guide LLM toward underrepresented labels
            rebalance_instruction = ""
            if iteration > 1:  # Start rebalancing from iteration 2 onwards
                rebalance_instruction = self._build_rebalance_instruction(current_counts, expect_total_each_label)
                if rebalance_instruction:
                    logger.info(f"Rebalance instruction: {rebalance_instruction.strip()}")

            # Prepare seed examples
            if iteration == 1:
                # First iteration: use human seeds
                seed_examples = random.sample(human_seeds, k=min(len(human_seeds), 2))
            else:
                # Subsequent iterations: mix previous batch results with human seeds
                new_seed = random.sample(previous_batch_results, k=min(len(previous_batch_results), 6))
                new_human_seeds = random.sample(human_seeds, k=min(len(human_seeds), 2))
                seed_examples = new_seed + new_human_seeds

            # Generate parallel batches with rebalance instruction
            generation_tasks = []
            for _ in range(parallel_generations):
                generation_tasks.append(self._gen(seed_examples, prompt, messages_per_call, rebalance_instruction))

            # Run all generations in parallel
            parallel_results = await asyncio.gather(*generation_tasks)

            # Combine all parallel results
            batch_results = []
            for results in parallel_results:
                batch_results.extend(results)

            # Step 1: Deduplicate within the batch and against all_results so far (internal dedup)
            internal_deduped = await self.data_manager._dedup_helper.deduplicate(batch_results)
            internal_deduped = await self.data_manager._dedup_helper.filter_against_existing(internal_deduped, all_results)

            # If provided composal file => should also filter against existing composal data
            if composal_ds_mgr:
                internal_deduped = await composal_ds_mgr.filter(internal_deduped)

            new_unique = [s for s in internal_deduped if s not in all_results]

            logger.debug(f"Internal dedup: {len(batch_results)} -> {len(new_unique)} samples")

            # Step 3: Add externally filtered samples to quantity tracker (only if not exceeding target)
            added_samples = []
            added_count = 0
            for sample in new_unique:
                if sample.label in quantity_tracker:
                    # Only add if this label hasn't reached its target yet
                    if len(quantity_tracker[sample.label]) < expect_total_each_label[sample.label]:
                        quantity_tracker[sample.label].append(sample)
                        added_samples.append(sample)
                        added_count += 1

            # Save newly added samples to temp file incrementally
            if added_samples:
                self._append_to_file(temp_file_path, added_samples)
                logger.info(f"Saved {len(added_samples)} new samples to temp file")

                # Call the batch callback if provided
                if on_batch_generated:
                    await on_batch_generated(iteration, added_samples, str(temp_file_path))

            # Update all_results and previous_batch_results for next iteration
            all_results.extend(batch_results)
            previous_batch_results = batch_results

            logger.debug(f"Iteration {iteration}: Generated {len(batch_results)} samples, added {added_count} unique samples to tracker")

            # Wait 20 seconds before next iteration to avoid rate limits
            if not self._is_quantity_sufficient(quantity_tracker, expect_total_each_label):
                logger.info("Waiting 20 seconds before next batch to avoid rate limits...")
                await asyncio.sleep(20)

        # Flatten quantity_tracker to get final samples
        final_samples = []
        for samples in quantity_tracker.values():
            final_samples.extend(samples)

        logger.info(f"Generation complete after {iteration} iterations. Final counts: {self._get_current_counts(quantity_tracker)}, Total: {len(final_samples)}")

        # Use hard_save to overwrite the file with exact final samples
        self.data_manager.hard_save(final_samples)
        if composal_ds_mgr:
            # Append new examples to composal dataset manager as well
            composal_ds_mgr.save(final_samples)

        logger.info(f"Final dataset saved to: {self.data_manager.LOCAL_FILE}, temp backup at: {temp_file_path}")

        return final_samples, self.data_manager.LOCAL_FILE

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

    def _build_rebalance_instruction(
        self,
        current_counts: Dict[Union[str, int], int],
        expected_counts: Dict[Union[str, int], int]
    ) -> str:
        """
        Build a prompt instruction to rebalance label generation based on current progress.

        Args:
            current_counts: Current sample count for each label {label: count}
            expected_counts: Expected/target sample count for each label {label: count}

        Returns:
            A string instruction to append to the prompt to guide the LLM to generate
            more samples for labels that are behind schedule.

        Example:
            current: {"label_a": 10, "label_b": 45, "label_c": 30}
            expected: {"label_a": 60, "label_b": 60, "label_c": 60}
            Returns: "Focus on generating more samples for: label_a (need 50 more),
                     label_c (need 30 more), label_b (need 15 more)."
        """
        # Calculate remaining needed for each label
        remaining = {}
        for label, expected in expected_counts.items():
            current = current_counts.get(label, 0)
            remaining[label] = max(0, expected - current)

        # Sort labels by remaining count (descending) to prioritize labels that need more samples
        sorted_labels = sorted(remaining.items(), key=lambda x: x[1], reverse=True)

        # Filter out labels that have reached their target
        labels_needing_more = [(label, count) for label, count in sorted_labels if count > 0]

        if not labels_needing_more:
            return ""  # All labels have sufficient samples

        # Build the instruction
        if len(labels_needing_more) == 1:
            label, count = labels_needing_more[0]
            return f"\n\n**IMPORTANT**: Generate ONLY '{label}' examples (need {count} more samples)."
        else:
            # Build priority list
            priority_parts = []
            for label, count in labels_needing_more:
                priority_parts.append(f"'{label}' (need {count} more)")

            priority_str = ", ".join(priority_parts)
            return f"\n\n**IMPORTANT**: Focus on generating more samples with this priority: {priority_str}. Generate more examples for labels that need the most samples."

    def _append_to_file(self, file_path, samples: List[Sample]):
        """Append samples to a file in JSONL format"""
        import json
        with open(file_path, 'a', encoding='utf-8') as f:
            for sample in samples:
                json.dump({"msg": sample.msg, "label": sample.label}, f, ensure_ascii=False)
                f.write('\n')

    def _calculate_v2_generation_params(self, expect_total_message: int) -> tuple[int, int, int]:
        """
        Calculate PARALLEL_GENERATIONS, TOTAL_MESSAGE_PER_BATCH, and batches for V2 generator.

        Args:
            expect_total_message: Expected total number of messages to generate

        Returns:
            Tuple of (PARALLEL_GENERATIONS, TOTAL_MESSAGE_PER_BATCH, batches_needed)
        """
        # Each API call generates ~30 messages
        messages_per_api_call = 40

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

    async def _gen(
        self,
        seed: List[Sample],
        prompt: str,
        count: int,
        rebalance_instruction: str = ""
    ) -> List[Sample]:
        '''
        Generate data from given seed and prompt

        Args:
            seed: Seed samples for generation
            prompt: System prompt template
            count: Number of examples to generate
            rebalance_instruction: Optional instruction to guide label balancing

        Returns:
            List of generated samples
        '''
        personas_txt = "n- ".join(PersonasSeeder.random(6))  # Use a reasonable default for personas

        seed_examples_txt = "\n".join([f"- {s.msg}" for s in seed])
        formatted_prompt = prompt.format(personas=personas_txt)

        user_input = f"## Examples: {seed_examples_txt} \nContinue generate {count} diverse examples{rebalance_instruction}"

        data = (await self.llm.generate_structured_output([
            {"role": "system", "content": formatted_prompt},
            {"role": "user", "content": user_input}
        ], Result)).messages

        return data
