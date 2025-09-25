from typing import List
from app.core import schemas

class EvalGenerator:
    def __init__(self):
        pass

    async def fresh_gen(self, human_seeds: List[Sample] = None) -> List[Sample]:
        """
        Generate fresh evaluation data and save using eval_data_manager.
        Iterates multiple times like fresh_gen with deduplication and filtering.

        Args:
            human_seeds: Optional human seed examples

        Returns:
            Generated evaluation samples
        """
        if human_seeds is None:
            human_seeds = []

        # First iteration
        results = await self._gen_eval(human_seeds)

        # Deduplicate initial results
        filtered_result = await self.eval_data_manager.deduplicate(results)

        # Save initial batch to new iteration folder
        iteration_number = self.eval_data_manager.save(filtered_result)

        for _ in range(self.NUMBER_EVAL_PER_ITER - 1):
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

    async def _gen(self, seed: List[Sample]) -> List[Sample]:
        """
        Generate evaluation data from given seed samples.

        Args:
            seed: Seed samples to base generation on

        Returns:
            Generated samples
        """
        seed_rd = random.randint(0, 1000)
        personas = self.personas_ds["train"].shuffle(seed=seed_rd).select(range(self.MAX_GEN_EVAL_PER_ITER))
        personas_txt = "\n- ".join([p['persona'] for p in personas])

        prompt = self.prompt_mgr.get_prompt(self.EVAL_PROMPT_KEY).format(personas=personas_txt)

        data = (await self.llm.generate_structured_output([
            {"role": "system", "content": prompt},
            {"role": "user", "content": f"generate {self.NUMBER_EVAL_PER_ITER} diverse evaluation examples"}
        ], Result)).messages

        return data