from datasets import load_dataset
import random

class PersonasSeeder:
    personas = load_dataset("proj-persona/PersonaHub", "persona")

    @classmethod
    def random(cls, k: int = 1) -> list[str]:
        seed = random.randint(0, 1_000_000)
        personas = cls.personas["train"].shuffle(seed=seed).select(range(k))
        return [p['persona'] for p in personas]