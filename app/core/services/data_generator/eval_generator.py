import random
import logging
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

    TOTAL_MESSAGE_GEN_PER_BATCH = 20
    TOTAL_BATCH_PER_GEN = 20
    # ~ 180

    TOTAL_OPENINTENT_MESSAGE_GEN_PER_BATCH = 15
    TOTAL_OPENINTENT_BATCH_PER_GEN = 4
    # ~ 60

    EVAL_PROMPT_KEY = "eval/seed"
    OPEN_INTENT_PROMPT_KEY = "eval/open_intent"


    def __init__(self, llm: BaseLLM, prompt_mgr: BasePromptManager, data_manager: DataManager, eval_data_manager: EvalDataManager = None):
        self.llm = llm
        self.prompt_mgr = prompt_mgr
        self.eval_data_manager = eval_data_manager or EvalDataManager()
        self.personas_ds = load_dataset("proj-persona/PersonaHub", "persona")

    async def fresh_gen(self, human_seeds: List[Sample] = None, human_seed_messages: List[str] = None) -> dict:
        """
        Generate both intent test cases and open intent messages in the same iteration folder.

        Args:
            human_seeds: Human seed samples for intent generation
            human_seed_messages: Human seed messages for open intent generation

        Returns:
            Dictionary containing both types of generated data and iteration info
        """
        if human_seeds is None:
            human_seeds = []
        if human_seed_messages is None:
            human_seed_messages = []

        logger.info("Starting fresh generation of both intent and open intent evaluation data...")

        # Generate intent test cases first (this creates the iteration number)
        intent_results = await self.intent_gen(human_seeds)
        iteration_number = self.eval_data_manager.get_latest_item_number()

        # Generate open intent messages using the same iteration number
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
        Generate intent test cases (original fresh_gen logic).

        Args:
            human_seeds: Human seed samples for intent generation

        Returns:
            Generated intent samples
        """
        if human_seeds is None:
            human_seeds = []

        # First iteration
        results = await self._gen_eval(human_seeds)

        # Deduplicate initial results
        filtered_result = await self.eval_data_manager.deduplicate(results)

        # Save initial batch to new iteration folder
        iteration_number = self.eval_data_manager.save(filtered_result)

        for _ in range(self.TOTAL_BATCH_PER_GEN - 1):
            # Make new seed from previous results
            new_seed = random.sample(results, k=min(len(results), 2))
            new_human_seeds = random.sample(human_seeds, k=min(len(human_seeds), 6))
            seed = new_seed + new_human_seeds

            # Generate new batch
            new_results = await self._gen_eval(seed)
            logger.debug(f"Generated {len(new_results)} new evaluation examples")

            # Deduplicate and filter against current iteration data
            new_data = await self.eval_data_manager.filter(new_results, iteration_number)

            # Append filtered data to the same iteration folder
            self.eval_data_manager.append(new_data, iteration_number)

            # Update results for next iteration
            results.extend(new_results)

        logger.info(f"Generated {len(results)} total evaluation samples in iteration {iteration_number}")
        return results

    async def gen_open_intent_with_iteration(self, human_seeds: List[str] = None, iteration_number: int = None) -> List[str]:
        """
        Generate open intent messages and save to a specific iteration number.

        Args:
            human_seeds: Optional human seed messages to base generation on
            iteration_number: Specific iteration number to save to

        Returns:
            Generated open intent messages as simple strings
        """
        if human_seeds is None:
            human_seeds = []

        logger.info(f"Starting open intent generation for iteration {iteration_number}...")

        # First iteration
        results = await self._gen_open_intent(human_seeds)

        # Deduplicate initial results
        filtered_result = await self.eval_data_manager.deduplicate_open_intent(results)

        # Save to the specified iteration folder
        self.eval_data_manager._save_open_intent(filtered_result, iteration_number)

        for _ in range(self.TOTAL_OPENINTENT_BATCH_PER_GEN - 1):
            # Make new seed from previous results
            new_seed = random.sample(results, k=min(len(results), 2))
            new_human_seeds = random.sample(human_seeds, k=min(len(human_seeds), 6))
            seed = new_seed + new_human_seeds

            # Generate new batch
            new_results = await self._gen_open_intent(seed)
            logger.debug(f"Generated {len(new_results)} new open intent evaluation examples")

            # Deduplicate and filter against current iteration data
            new_data = await self.eval_data_manager.filter_open_intent(new_results, iteration_number)

            # Append filtered data to the same iteration folder
            self.eval_data_manager.append_open_intent(new_data, iteration_number)

            # Update results for next iteration
            results.extend(new_results)

        logger.info(f"Generated {len(results)} total open intent evaluation samples in iteration {iteration_number}")
        return results

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