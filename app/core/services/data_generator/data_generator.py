from src.payment_classifier.llm.base import BaseLLM
from src.payment_classifier.prompts.base import BasePromptManager
from app.core.services.data_manager.data_manager import DataManager
from datasets import load_dataset

import random
from app.core.schemas import Sample, Result
from typing import *
import logging

logger = logging.getLogger(__name__)

class DataGenerator:
    NUMBER_PER_ITER = 15
    MAX_GEN_PER_ITER = 20

    SEED_PROMPT_KEY = "seed"

    def __init__(self, llm: BaseLLM, prompt_mgr: BasePromptManager, data_manager: DataManager):
        self.llm = llm
        self.prompt_mgr = prompt_mgr
        self.data_manager = data_manager
        self.personas_ds = load_dataset("proj-persona/PersonaHub", "persona")

    async def fresh_gen(self, human_seeds: List[Sample]) -> List[Sample]:
        # First iterate
        results = await self._gen(human_seeds)
        
        print(results[0])

        filted_result = await self.data_manager.deduplicate(results)
        self.data_manager.save(filted_result)

        for _ in range(self.NUMBER_PER_ITER):
            # Make new seed
            new_seed = random.sample(results, k=min(len(results), 2))
            new_human_seeds = random.sample(human_seeds, k=min(len(human_seeds), 6))
            seed = new_seed + new_human_seeds

            results = await self._gen(seed)
            logger.debug(f"Generated {len(results)} new examples")

            # Deduplicate and save
            new_data = await self.data_manager.filter(results)
            self.data_manager.save(new_data)

    async def _gen(self, seed: List[Sample]) -> List[Sample]:
        '''
        Generate data from given seed and prompt
        '''
        seed_rd = random.randint(0, 1000)
        personas = self.personas_ds["train"].shuffle(seed=seed_rd).select(range(self.MAX_GEN_PER_ITER))
        personas_txt = "\n- ".join([p['persona'] for p in personas])
        
        prompt = self.prompt_mgr.get_prompt(self.SEED_PROMPT_KEY).format(personas=personas_txt)

        data = (await self.llm.generate_structured_output([
            {"role": "system", "content": prompt},
            {"role": "user", "content": f"generate {self.NUMBER_PER_ITER} diverse examples"}
        ], Result)).messages

        return data