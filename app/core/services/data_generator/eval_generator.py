import random
import logging
import asyncio
from typing import List
from datasets import load_dataset
from pydantic import BaseModel
from app.core.schemas.workflow import Sample, Result
from app.core.services.data_manager.data_manager import DataManager
from app.core.services.eval_data_manager.eval_data_manager import EvalDataManager
from src.payment_classifier.llm.base import BaseLLM
from src.payment_classifier.prompts.base import BasePromptManager

logger = logging.getLogger(__name__)

class EvalGenerator:

    TOTAL_MESSAGE_GEN_PER_BATCH = 30
    TOTAL_BATCH_PER_GEN = 5

    TOTAL_OPENINTENT_MESSAGE_GEN_PER_BATCH = 30
    TOTAL_OPENINTENT_BATCH_PER_GEN = 5

    # Parallel generation settings
    PARALLEL_INTENT_GENERATIONS = 5  # Parallel LLM calls for intent generation
    PARALLEL_OPEN_INTENT_GENERATIONS = 3  # Parallel LLM calls for open intent generation

    EVAL_PROMPT_KEY = "v2/eval/seed"
    OPEN_INTENT_PROMPT_KEY = "v2/eval/open_intent"


    def __init__(self, llm: BaseLLM, prompt_mgr: BasePromptManager, data_manager: DataManager, eval_data_manager: EvalDataManager = None, generate_open_intent: bool = True):
        self.llm = llm
        self.prompt_mgr = prompt_mgr
        self.eval_data_manager = eval_data_manager or EvalDataManager()
        self.personas_ds = load_dataset("proj-persona/PersonaHub", "persona")
        self.generate_open_intent = generate_open_intent

    async def fresh_gen(self, human_seeds: List[Sample] = None, human_seed_messages: List[str] = None) -> dict:
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
        intent_results = await self.intent_gen(human_seeds)
        iteration_number = self.eval_data_manager.get_latest_item_number()

        # Conditionally generate open intent messages using the same iteration number
        open_intent_results = []
        if self.generate_open_intent:
            open_intent_results = await self.gen_open_intent_with_iteration(human_seed_messages, iteration_number)

        return {
            "iteration_number": iteration_number,
            "intent_samples": intent_results,
            "open_intent_messages": open_intent_results,
            "intent_count": len(intent_results),
            "open_intent_count": len(open_intent_results),
            "total_generated": len(intent_results) + len(open_intent_results)
        }

    async def intent_gen(self, human_seeds: List[Sample] = None) -> List[Sample]:
        """
        Generate intent test cases using parallel generation.

        Args:
            human_seeds: Human seed samples for intent generation

        Returns:
            Generated intent samples
        """
        if human_seeds is None:
            human_seeds = []

        # First iteration with parallel generation
        seed_examples = random.sample(human_seeds, k=min(len(human_seeds), 2))

        # Generate parallel batches
        generation_tasks = []
        for _ in range(self.PARALLEL_INTENT_GENERATIONS):
            generation_tasks.append(self._gen_eval(seed_examples))

        # Run all generations in parallel
        parallel_results = await asyncio.gather(*generation_tasks)

        # Combine all parallel results
        all_results = []
        for results in parallel_results:
            all_results.extend(results)

        # Internal deduplication within the batch
        internal_deduped = await self._internal_deduplicate_samples(all_results)

        # Deduplicate against any existing data and save
        filtered_result = await self.eval_data_manager.deduplicate(internal_deduped)
        iteration_number = self.eval_data_manager.save(filtered_result)

        # Subsequent batches with parallel generation
        for _ in range(self.TOTAL_BATCH_PER_GEN - 1):
            # Create seeds for each parallel generation
            generation_tasks = []
            for _ in range(self.PARALLEL_INTENT_GENERATIONS):
                # Make new seed for each parallel task
                new_seed = random.sample(all_results, k=min(len(all_results), 2))
                new_human_seeds = random.sample(human_seeds, k=min(len(human_seeds), 6))
                seed = new_seed + new_human_seeds
                generation_tasks.append(self._gen_eval(seed))

            # Run parallel generations
            parallel_results = await asyncio.gather(*generation_tasks)
            logger.debug(f"Generated {sum(len(results) for results in parallel_results)} new evaluation examples across {self.PARALLEL_INTENT_GENERATIONS} parallel tasks")

            # Combine all parallel results
            batch_results = []
            for results in parallel_results:
                batch_results.extend(results)

            # Internal deduplication within the batch
            internal_deduped = await self._internal_deduplicate_samples(batch_results)

            # Filter against existing iteration data and append
            new_data = await self.eval_data_manager.filter(internal_deduped, iteration_number)
            self.eval_data_manager.append(new_data, iteration_number)

            # Update all_results for next iteration
            all_results.extend(batch_results)

        logger.info(f"Generated evaluation samples with parallel processing in iteration {iteration_number}")
        return all_results

    async def gen_open_intent_with_iteration(self, human_seeds: List[str] = None, iteration_number: int = None) -> List[str]:
        """
        Generate open intent messages using parallel generation and save to a specific iteration number.

        Args:
            human_seeds: Optional human seed messages to base generation on
            iteration_number: Specific iteration number to save to

        Returns:
            Generated open intent messages as simple strings
        """
        if human_seeds is None:
            human_seeds = []

        logger.info(f"Starting parallel open intent generation for iteration {iteration_number}...")

        # First iteration with parallel generation
        seed_examples = random.sample(human_seeds, k=min(len(human_seeds), 2))

        # Generate parallel batches
        generation_tasks = []
        for _ in range(self.PARALLEL_OPEN_INTENT_GENERATIONS):
            generation_tasks.append(self._gen_open_intent(seed_examples))

        # Run all generations in parallel
        parallel_results = await asyncio.gather(*generation_tasks)

        # Combine all parallel results
        all_results = []
        for results in parallel_results:
            all_results.extend(results)

        # Internal deduplication within the batch
        internal_deduped = await self._internal_deduplicate_strings(all_results)

        # Deduplicate against any existing data and save
        filtered_result = await self.eval_data_manager.deduplicate_open_intent(internal_deduped)
        self.eval_data_manager._save_open_intent(filtered_result, iteration_number)

        # Subsequent batches with parallel generation
        for _ in range(self.TOTAL_OPENINTENT_BATCH_PER_GEN - 1):
            # Create seeds for each parallel generation
            generation_tasks = []
            for _ in range(self.PARALLEL_OPEN_INTENT_GENERATIONS):
                # Make new seed for each parallel task
                new_seed = random.sample(all_results, k=min(len(all_results), 2))
                new_human_seeds = random.sample(human_seeds, k=min(len(human_seeds), 6))
                seed = new_seed + new_human_seeds
                generation_tasks.append(self._gen_open_intent(seed))

            # Run parallel generations
            parallel_results = await asyncio.gather(*generation_tasks)
            logger.debug(f"Generated {sum(len(results) for results in parallel_results)} new open intent examples across {self.PARALLEL_OPEN_INTENT_GENERATIONS} parallel tasks")

            # Combine all parallel results
            batch_results = []
            for results in parallel_results:
                batch_results.extend(results)

            # Internal deduplication within the batch
            internal_deduped = await self._internal_deduplicate_strings(batch_results)

            # Filter against existing iteration data and append
            new_data = await self.eval_data_manager.filter_open_intent(internal_deduped, iteration_number)
            self.eval_data_manager.append_open_intent(new_data, iteration_number)

            # Update all_results for next iteration
            all_results.extend(batch_results)

        logger.info(f"Generated open intent samples with parallel processing in iteration {iteration_number}")
        return all_results

    async def _gen_eval(self, seed: List[Sample]) -> List[Sample]:
        """
        Generate evaluation data from given seed samples.

        Args:
            seed: Seed samples to base generation on

        Returns:
            Generated samples
        """
        seed_rd = random.randint(0, 1000)
        personas = self.personas_ds["train"].shuffle(seed=seed_rd).select(range(self.TOTAL_MESSAGE_GEN_PER_BATCH))
        personas_txt = "\n- ".join([p['persona'] for p in personas])

        prompt = self.prompt_mgr.get_prompt(self.EVAL_PROMPT_KEY).format(personas=personas_txt)

        data = (await self.llm.generate_structured_output([
            {"role": "system", "content": prompt},
            {"role": "user", "content": f"generate {self.TOTAL_MESSAGE_GEN_PER_BATCH} diverse evaluation examples"}
        ], Result)).messages

        return data

    async def _gen_open_intent(self, seed: List[str]) -> List[str]:
        """
        Generate open intent evaluation data from given seed messages.

        Args:
            seed: Seed messages to base generation on

        Returns:
            Generated open intent messages as strings
        """
        seed_rd = random.randint(0, 1000)
        personas = self.personas_ds["train"].shuffle(seed=seed_rd).select(range(self.TOTAL_OPENINTENT_MESSAGE_GEN_PER_BATCH))
        personas_txt = "\n- ".join([p['persona'] for p in personas])

        # Add seed context if available
        seed_context = ""
        if seed:
            seed_context = f"\n\nExample messages to inspire generation (but create different ones):\n" + "\n".join([f"- {msg}" for msg in seed[:5]])

        prompt = self.prompt_mgr.get_prompt(self.OPEN_INTENT_PROMPT_KEY).format(personas=personas_txt) + seed_context

        # Generate raw text output instead of structured
        class OpenIntentResults(BaseModel):
            messages: List[str]

        messages = (await self.llm.generate_structured_output([
            {"role": "system", "content": prompt},
            {"role": "user", "content": f"Generate {self.TOTAL_OPENINTENT_MESSAGE_GEN_PER_BATCH} diverse open intent messages. Return only the messages, one per line, no labels or extra formatting."}
        ], OpenIntentResults)).messages

        return messages

    async def _internal_deduplicate_samples(self, samples: List[Sample]) -> List[Sample]:
        """
        Internal deduplication for Sample objects within a batch.
        Uses label-aware filtering similar to DeduplicationMixin but optimized for batch processing.
        """
        if not samples:
            return samples

        from app.utils.scorer import EvaluationUtils

        # Group samples by label for efficient comparison
        samples_by_label = {}
        for sample in samples:
            label = sample.label
            if label not in samples_by_label:
                samples_by_label[label] = []
            samples_by_label[label].append(sample)

        # Deduplicate within each label group
        deduped_samples = []
        for label, label_samples in samples_by_label.items():
            deduped_for_label = []

            for sample in label_samples:
                is_duplicate = False
                for existing_sample in deduped_for_label:
                    rouge_score = await EvaluationUtils.score_rouge(
                        ref=existing_sample.msg,
                        pred=sample.msg,
                        rouge_type="rougeL",
                        mode="precision"
                    )

                    if rouge_score >= self.eval_data_manager.rouge_threshold:
                        is_duplicate = True
                        break

                if not is_duplicate:
                    deduped_for_label.append(sample)

            deduped_samples.extend(deduped_for_label)

        logger.debug(f"Internal sample deduplication: {len(samples)} -> {len(deduped_samples)} samples")
        return deduped_samples

    async def _internal_deduplicate_strings(self, messages: List[str]) -> List[str]:
        """
        Internal deduplication for string messages within a batch.
        """
        if not messages:
            return messages

        from app.utils.scorer import EvaluationUtils

        deduped_messages = []

        for message in messages:
            is_duplicate = False
            for existing_message in deduped_messages:
                rouge_score = await EvaluationUtils.score_rouge(
                    ref=existing_message,
                    pred=message,
                    rouge_type="rougeL",
                    mode="precision"
                )

                if rouge_score >= self.eval_data_manager.rouge_threshold:
                    is_duplicate = True
                    break

            if not is_duplicate:
                deduped_messages.append(message)

        logger.debug(f"Internal string deduplication: {len(messages)} -> {len(deduped_messages)} messages")
        return deduped_messages