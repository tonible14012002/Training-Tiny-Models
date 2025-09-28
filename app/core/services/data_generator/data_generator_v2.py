from .data_generator import DataGenerator
from app.core.schemas import Sample
from typing import List
import asyncio
import random
import logging

logger = logging.getLogger(__name__)

class DataGeneratorV2(DataGenerator):
    PARALLEL_GENERATIONS = 20
    SEED_PROMPT_KEY = "v2/train/seed"

    async def fresh_gen(self, human_seeds: List[Sample]) -> List[Sample]:
        # First iterate with parallel generation
        seed_examples = random.sample(human_seeds, k=min(len(human_seeds), 2))

        # Generate 10 parallel batches
        generation_tasks = []
        for _ in range(self.PARALLEL_GENERATIONS):
            generation_tasks.append(self._gen(seed_examples))

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
        for _ in range(self.TOTAL_MESSAGE_PER_BATCH - 1):
            # Create seeds for each parallel generation
            generation_tasks = []
            for _ in range(self.PARALLEL_GENERATIONS):
                # Make new seed for each parallel task
                new_seed = random.sample(all_results, k=min(len(all_results), 2))
                new_human_seeds = random.sample(human_seeds, k=min(len(human_seeds), 6))
                seed = new_seed + new_human_seeds
                generation_tasks.append(self._gen(seed))

            # Run parallel generations
            parallel_results = await asyncio.gather(*generation_tasks)
            logger.debug(f"Generated {sum(len(results) for results in parallel_results)} new examples across {self.PARALLEL_GENERATIONS} parallel tasks")

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