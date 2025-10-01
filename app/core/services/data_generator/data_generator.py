from src.payment_classifier.llm.base import BaseLLM
from src.payment_classifier.prompts.base import BasePromptManager
from app.core.services.data_manager.data_manager import DataManager
from app.core.services.eval_data_manager.eval_data_manager import EvalDataManager
from src.payment_classifier.personas.personas import PersonasSeeder

import random
from app.core.schemas import Sample, Result
from typing import *
import logging

logger = logging.getLogger(__name__)

class DataGenerator:
    SEED_PROMPT_KEY = "train/seed"

    # Estimation: each API call generates ~30 messages
    ESTIMATED_MESSAGES_PER_API_CALL = 30
    DEFAULT_TOTAL_MESSAGES = 300  # Default target when not specified

    def __init__(self, llm: BaseLLM, prompt_mgr: BasePromptManager, data_manager: DataManager, eval_data_manager: EvalDataManager):
        self.llm = llm
        self.prompt_mgr = prompt_mgr
        self.data_manager = data_manager
        self.eval_data_manager = eval_data_manager

    def _calculate_generation_params(self, expect_total_message: int) -> tuple[int, int]:
        """
        Calculate TOTAL_MESSAGE_PER_BATCH and TOTAL_BATCH_PER_GEN based on expected total messages.

        Args:
            expect_total_message: Expected total number of messages to generate

        Returns:
            Tuple of (TOTAL_MESSAGE_PER_BATCH, TOTAL_BATCH_PER_GEN)
        """
        # Each API call generates ~30 messages
        # TOTAL_MESSAGE_PER_BATCH is how many messages per API call
        # TOTAL_BATCH_PER_GEN is how many API calls (batches) to make

        # Use the estimated messages per API call
        messages_per_batch = self.ESTIMATED_MESSAGES_PER_API_CALL

        # Calculate how many batches we need
        batches_needed = max(1, (expect_total_message + messages_per_batch - 1) // messages_per_batch)

        return messages_per_batch, batches_needed

    async def fresh_gen(self, human_seeds: List[Sample], expect_total_message: int = None) -> List[Sample]:
        # Use provided target or default
        target_messages = expect_total_message or self.DEFAULT_TOTAL_MESSAGES

        # Calculate generation parameters
        total_message_per_batch, total_batch_per_gen = self._calculate_generation_params(target_messages)
        logger.info(f"Dynamic calculation: targeting {target_messages} messages with {total_message_per_batch} per batch and {total_batch_per_gen} batches")

        # Load the prompt at the caller level
        prompt = self.prompt_mgr.get_prompt(self.SEED_PROMPT_KEY)

        # First iterate
        seed_examples = random.sample(human_seeds, k=min(len(human_seeds), 2))
        results = await self._gen(seed_examples, prompt)

        # Deduplicate initial results
        filted_result = await self.data_manager.deduplicate(results)
        self.data_manager.save(filted_result)

        # Subsequent batches
        for _ in range(total_batch_per_gen - 1):
            # Make new seed
            new_seed = random.sample(results, k=min(len(results), 2))
            new_human_seeds = random.sample(human_seeds, k=min(len(human_seeds), 6))
            seed = new_seed + new_human_seeds

            results = await self._gen(seed, prompt)
            logger.debug(f"Generated {len(results)} new examples")

            # Deduplicate and save
            new_data = await self.data_manager.filter(results)
            self.data_manager.save(new_data)

    async def _gen(self, seed: List[Sample], prompt: str) -> List[Sample]:
        '''
        Generate data from given seed and prompt
        '''
        personas_txt = "\n- ".join(PersonasSeeder.random(10))  # Use a reasonable default for personas

        seed_examples_txt = "\n".join([f"- {s.msg}" for s in seed])
        formatted_prompt = prompt.format(personas=personas_txt)

        user_input = f"## Examples: {seed_examples_txt} \nContinue generate {self.ESTIMATED_MESSAGES_PER_API_CALL} diverse examples"

        data = (await self.llm.generate_structured_output([
            {"role": "system", "content": formatted_prompt},
            {"role": "user", "content": user_input}
        ], Result)).messages

        return data

    async def _gen_open_intent(self, seed: List[Sample], prompt: str) -> List[Sample]:
        '''
        Generate data from given seed and prompt
        '''
        personas_txt = "\n- ".join(PersonasSeeder.random(10))  # Use a reasonable default for personas

        seed_examples_txt = "\n".join([f"- {s.msg}" for s in seed])
        formatted_prompt = prompt.format(personas=personas_txt)

        user_input = f"## Examples: {seed_examples_txt} \nContinue generate {self.ESTIMATED_MESSAGES_PER_API_CALL} diverse examples"

        data = (await self.llm.generate_structured_output([
            {"role": "system", "content": formatted_prompt},
            {"role": "user", "content": user_input}
        ], Result)).messages

        return data