from src.payment_classifier.personas.personas import PersonasSeeder
from src.payment_classifier.llm.base import BaseLLM
from src.payment_classifier.prompts.base import BasePromptManager
from app.core.services.data_manager import DataManager
from app.core.schemas.workflow import Result
from app.core.schemas import Sample
from app.core.models.models import LabelConfig
from typing import List, Dict, Union, Optional, Callable, Awaitable
from datetime import datetime
from pathlib import Path
import asyncio
import random
import logging
import json

logger = logging.getLogger(__name__)

class DataGeneratorV2:
    SEED_PROMPT_KEY = "v2/train/seed"

    def __init__(self, llm: BaseLLM, prompt_mgr: BasePromptManager, data_manager: DataManager, high_llm: Optional[BaseLLM] = None):
        self.llm = llm
        self.high_llm = high_llm  # Higher capability model for when generation gets stuck
        self.prompt_mgr = prompt_mgr
        self.data_manager = data_manager
        self.current_llm = llm  # Track which LLM is currently being used

    async def fresh_gen_v2(
        self,
        human_seeds: List[Sample],
        expect_total_each_label: Dict[Union[str, int], int],
        label_config: LabelConfig,
        on_batch_generated: Optional[Callable[[int, List[Sample], str, str], Awaitable[None]]] = None,
        composal_ds_mgr: Optional[DataManager] = None
    ) -> tuple[List[Sample], str]:
        """Generate new data
        Args:
            human_seeds: Initial seed samples for generation
            expect_total_each_label: Dictionary mapping label (str or int) to expected quantity
                                    e.g., {"payment_intent": 200, "payment_request": 200, "open_intent": 200}
                                    or {0: 200, 1: 200, 2: 200}
            label_config: LabelConfig object for label to ID conversion
            on_batch_generated: Optional async callback called after each batch generation
                               Signature: async def callback(batch_number: int, samples: List[Sample],
                                                            batch_file_path: str, metadata_path: str)

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

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        temp_file_path = Path(self.data_manager.LOCAL_FILE).parent / f"temp_gen_{timestamp}.jsonl"
        batch_dir = Path(self.data_manager.LOCAL_FILE).parent / f"batches_{timestamp}"
        batch_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"Batch files will be saved to: {batch_dir}")

        # Calculate parallel generation parameters based on total expected
        total_expected = sum(expect_total_each_label.values())
        parallel_generations, messages_per_call, _ = self._calculate_v2_generation_params(total_expected)

        all_results = []
        previous_batch_results = []
        iteration = 0

        # Track last K iterations for model switching logic
        iteration_history = []  # List of tuples: (iteration_num, added_count)
        model_switched = False  # Track if we've already switched models

        # Reset to base LLM at start
        self.current_llm = self.llm

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

            # Prepare seed examples - use new balanced selection method
            needed_labels = self._get_needed_labels(current_counts, expect_total_each_label)
            seed_examples = self._select_balanced_seeds(
                human_seeds=human_seeds,
                needed_labels=needed_labels,
                total_seeds=6,
                iteration=iteration
            )

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

            # Step 1: Deduplicate within the batch (internal dedup)
            internal_dedup_result = await self.data_manager._dedup_helper.deduplicate(batch_results)
            internal_duplicates = internal_dedup_result.rejected

            # Step 2: Filter against all_results accumulated so far
            external_dedup_result = await self.data_manager._dedup_helper.filter_against_existing(
                internal_dedup_result.accepted, all_results
            )
            external_duplicates_vs_all = external_dedup_result.rejected

            # Step 3: If provided composal file => filter against existing composal data
            composal_duplicates = []
            if composal_ds_mgr:
                composal_filter_result = await composal_ds_mgr.filter(external_dedup_result.accepted)
                final_unique = composal_filter_result.accepted
                composal_duplicates = composal_filter_result.rejected
            else:
                final_unique = external_dedup_result.accepted

            # Combine all external duplicates
            all_external_duplicates = external_duplicates_vs_all + composal_duplicates

            logger.debug(f"Dedup: {len(batch_results)} -> {len(final_unique)} samples "
                        f"(internal_dup: {len(internal_duplicates)}, external_dup: {len(all_external_duplicates)})")

            # Step 4: Add filtered samples to quantity tracker (only if not exceeding target)
            added_samples = []
            quota_exceeded = []
            for sample in final_unique:
                if sample.label in quantity_tracker:
                    # Only add if this label hasn't reached its target yet
                    if len(quantity_tracker[sample.label]) < expect_total_each_label[sample.label]:
                        quantity_tracker[sample.label].append(sample)
                        added_samples.append(sample)
                    else:
                        quota_exceeded.append(sample)

            # Step 5: Save batch file and metadata
            label2id = label_config.get_label2id()

            # Save accepted samples to batch file
            batch_file_path = None
            metadata_path = None
            if added_samples:
                batch_file_path, _ = self._save_batch_file(
                    batch_number=iteration,
                    samples=added_samples,
                    label2id=label2id,
                    output_dir=batch_dir
                )

            # Save metadata with all filtering information and generation context
            # Generate a representative personas text for logging
            personas_txt = "\n- ".join(PersonasSeeder.random(6))

            metadata_path = self._save_batch_metadata(
                batch_number=iteration,
                all_generated=batch_results,
                accepted=added_samples,
                internal_duplicates=internal_duplicates,
                external_duplicates=all_external_duplicates,
                quota_exceeded=quota_exceeded,
                output_dir=batch_dir,
                prompt=prompt,
                seed_examples=seed_examples,
                personas_txt=personas_txt,
                rebalance_instruction=rebalance_instruction
            )

            logger.info(f"Batch {iteration}: {len(added_samples)} accepted, "
                       f"{len(internal_duplicates) + len(all_external_duplicates) + len(quota_exceeded)} filtered")

            # Track iteration history for model switching
            iteration_history.append((iteration, len(added_samples)))

            # Check if we should switch to higher model
            model_switched = self._check_and_switch_model(iteration_history, model_switched)

            # Call the batch callback if provided
            if on_batch_generated and batch_file_path:
                await on_batch_generated(iteration, added_samples, batch_file_path, metadata_path)

            # Update all_results and previous_batch_results for next iteration
            all_results.extend(batch_results)
            previous_batch_results = batch_results

            logger.debug(f"Iteration {iteration}: Generated {len(batch_results)} samples, added {len(added_samples)} unique samples to tracker")

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
            Returns: "\n\n**IMPORTANT**: Prioritize generating: 'label_a' (50), 'label_c' (30), 'label_b' (15)"
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
            # Build concise priority list
            priority_parts = [f"'{label}' ({count})" for label, count in labels_needing_more]
            priority_str = ", ".join(priority_parts)
            return f"\n\n**IMPORTANT**: Prioritize generating: {priority_str}"

    def _save_batch_file(
        self,
        batch_number: int,
        samples: List[Sample],
        label2id: Dict[str, int],
        output_dir: Path
    ) -> tuple[str, int]:
        """
        Save batch samples to a new file with proper label2id format.

        Args:
            batch_number: Current batch number
            samples: Samples to save
            label2id: Label to ID mapping
            output_dir: Directory to save batch files

        Returns:
            tuple: (batch_file_path, sample_count)
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        batch_file = output_dir / f"batch_{batch_number}_{timestamp}.jsonl"

        with open(batch_file, 'w', encoding='utf-8') as f:
            for sample in samples:
                # Convert label to numeric ID (same format as DataManager._save)
                data_dict = {"msg": sample.msg, "label": label2id[sample.label]}
                json.dump(data_dict, f, ensure_ascii=False)
                f.write('\n')

        logger.debug(f"Saved batch {batch_number} to {batch_file}")
        return str(batch_file), len(samples)

    def _save_batch_metadata(
        self,
        batch_number: int,
        all_generated: List[Sample],
        accepted: List[Sample],
        internal_duplicates: List[tuple[Sample, str]],
        external_duplicates: List[tuple[Sample, str]],
        quota_exceeded: List[Sample],
        output_dir: Path,
        prompt: str = "",
        seed_examples: List[Sample] = None,
        personas_txt: str = "",
        rebalance_instruction: str = ""
    ) -> str:
        """
        Save comprehensive batch metadata including filtered items and generation context.

        Args:
            batch_number: Current batch number
            all_generated: All samples generated in this batch
            accepted: Samples that were accepted
            internal_duplicates: Samples rejected due to internal duplication
            external_duplicates: Samples rejected due to external duplication
            quota_exceeded: Samples rejected due to quota limits
            output_dir: Directory to save metadata
            prompt: The system prompt used for generation
            seed_examples: The seed samples used for this batch
            personas_txt: The personas text used for generation
            rebalance_instruction: The rebalance instruction (if any) for this batch

        Returns:
            metadata_file_path: Path to metadata JSON file
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        metadata_file = output_dir / f"batch_{batch_number}_{timestamp}_metadata.json"

        # Build filter breakdown
        filter_breakdown = {
            "internal_duplicates": len(internal_duplicates),
            "external_duplicates": len(external_duplicates),
            "quota_exceeded": len(quota_exceeded)
        }

        # Prepare filtered samples with reasons
        filtered_samples = []

        # Add internal duplicates
        for sample, reason in internal_duplicates:
            filtered_samples.append({
                "msg": sample.msg,
                "label": sample.label,
                "filter_reason": reason
            })

        # Add external duplicates
        for sample, reason in external_duplicates:
            filtered_samples.append({
                "msg": sample.msg,
                "label": sample.label,
                "filter_reason": reason
            })

        # Add quota exceeded samples
        for sample in quota_exceeded:
            filtered_samples.append({
                "msg": sample.msg,
                "label": sample.label,
                "filter_reason": "quota_exceeded"
            })

        # Prepare seed examples for logging
        seed_examples_data = []
        if seed_examples:
            seed_examples_data = [
                {"msg": s.msg, "label": s.label}
                for s in seed_examples
            ]

        metadata = {
            "batch_number": batch_number,
            "timestamp": timestamp,
            "total_generated": len(all_generated),
            "accepted_count": len(accepted),
            "filtered_count": len(filtered_samples),
            "filter_breakdown": filter_breakdown,
            "filtered_samples": filtered_samples,
            "generation_context": {
                "prompt": prompt,
                "seed_examples": seed_examples_data,
                "personas": personas_txt,
                "rebalance_instruction": rebalance_instruction if rebalance_instruction else None
            }
        }

        with open(metadata_file, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, ensure_ascii=False, indent=2)

        logger.debug(f"Saved batch {batch_number} metadata to {metadata_file}")
        return str(metadata_file)

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

        seed_examples_txt = json.dumps([
            { "label": s.label, "msg": s.msg } for s in seed
        ], ensure_ascii=False, indent=2)

        formatted_prompt = prompt.format(personas=personas_txt)

        user_input = f"## Examples: {seed_examples_txt} \nContinue generate {count} diverse examples {rebalance_instruction}"

        data = (await self.current_llm.generate_structured_output([
            {"role": "system", "content": formatted_prompt},
            {"role": "user", "content": user_input}
        ], Result)).messages

        return data

    def _get_needed_labels(
        self,
        current_counts: Dict[Union[str, int], int],
        expected_counts: Dict[Union[str, int], int]
    ) -> List[Union[str, int]]:
        """
        Get list of labels that still need more samples.

        Args:
            current_counts: Current sample count for each label
            expected_counts: Expected/target sample count for each label

        Returns:
            List of labels that need more samples (have not reached target)
        """
        needed_labels = []
        for label, expected in expected_counts.items():
            current = current_counts.get(label, 0)
            if current < expected:
                needed_labels.append(label)
        return needed_labels

    def _select_balanced_seeds(
        self,
        human_seeds: List[Sample],
        needed_labels: List[Union[str, int]],
        total_seeds: int = 6,
        iteration: int = 1
    ) -> List[Sample]:
        """
        Select seed samples ensuring each needed label is represented at least once.
        Each batch gets different seed samples for variety.

        Args:
            human_seeds: Pool of human seed samples to select from
            needed_labels: Labels that still need more samples
            total_seeds: Total number of seed samples to select (default: 6)
            iteration: Current iteration number (for logging)

        Returns:
            List of selected seed samples with balanced label coverage

        Example:
            - If needed_labels = ["payment_intent", "open_intent"] and total_seeds = 6
            - Will select at least 1 example of each needed label
            - Remaining slots filled randomly from human_seeds pool
        """
        if not human_seeds:
            logger.warning("No human seeds available for selection")
            return []

        if not needed_labels:
            # All labels are satisfied, return random sample
            return random.sample(human_seeds, k=min(len(human_seeds), total_seeds))

        # Group human seeds by label
        seeds_by_label: Dict[Union[str, int], List[Sample]] = {}
        for seed in human_seeds:
            if seed.label not in seeds_by_label:
                seeds_by_label[seed.label] = []
            seeds_by_label[seed.label].append(seed)

        selected_seeds = []

        # Step 1: Ensure at least one example from each needed label
        for label in needed_labels:
            if label in seeds_by_label and seeds_by_label[label]:
                # Randomly select one sample from this label
                selected = random.choice(seeds_by_label[label])
                selected_seeds.append(selected)
            else:
                logger.warning(f"No human seed samples available for needed label: {label}")

        # Step 2: Fill remaining slots randomly from all human seeds
        remaining_slots = total_seeds - len(selected_seeds)
        if remaining_slots > 0:
            # Get samples that weren't already selected
            available_seeds = [s for s in human_seeds if s not in selected_seeds]
            if available_seeds:
                additional = random.sample(available_seeds, k=min(len(available_seeds), remaining_slots))
                selected_seeds.extend(additional)

        # Shuffle to avoid consistent ordering
        random.shuffle(selected_seeds)

        logger.debug(f"Iteration {iteration}: Selected {len(selected_seeds)} seeds with labels: "
                    f"{[s.label for s in selected_seeds]}")

        return selected_seeds

    def _select_seeds_legacy(
        self,
        human_seeds: List[Sample],
        previous_batch_results: List[Sample],
        iteration: int
    ) -> List[Sample]:
        """
        LEGACY: Original seed selection logic (no longer used).
        Kept for reference only.

        This method mixed human seeds with previous batch results but did not
        guarantee label balance coverage.

        Args:
            human_seeds: Human seed samples
            previous_batch_results: Results from previous batch
            iteration: Current iteration number

        Returns:
            List of selected seed samples
        """
        if iteration == 1:
            # First iteration: use human seeds
            seed_examples = random.sample(human_seeds, k=min(len(human_seeds), 2))
        else:
            # Subsequent iterations: mix previous batch results with human seeds
            new_seed = random.sample(previous_batch_results, k=min(len(previous_batch_results), 3))
            new_human_seeds = random.sample(human_seeds, k=min(len(human_seeds), 4))
            seed_examples = new_seed + new_human_seeds
        return seed_examples

    def _check_and_switch_model(
        self,
        iteration_history: List[tuple[int, int]],
        model_switched: bool
    ) -> bool:
        """
        Check if we should switch to higher capability model based on iteration history.

        Switches to high_llm if:
        - high_llm is available
        - Model hasn't been switched yet
        - Last 2 consecutive iterations produced < 7 samples each

        Args:
            iteration_history: List of (iteration_num, added_count) tuples
            model_switched: Whether model has already been switched

        Returns:
            Updated model_switched status
        """
        # Only switch if high_llm is available and we haven't switched yet
        if not self.high_llm or model_switched:
            return model_switched

        # Need at least 2 iterations to check
        if len(iteration_history) < 2:
            return model_switched

        # Check last 2 consecutive iterations
        last_two = iteration_history[-2:]
        if all(count < 7 for _, count in last_two):
            logger.warning(f"Low generation detected: last 2 iterations produced {[c for _, c in last_two]} samples")
            logger.info(f"Switching to higher capability model (from {self.current_llm.__class__.__name__} to high_llm)")
            self.current_llm = self.high_llm
            logger.info("Model switch completed - using higher capability model for subsequent iterations")
            return True

        return model_switched
