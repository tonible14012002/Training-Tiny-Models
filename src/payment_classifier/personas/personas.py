from datasets import load_dataset

class PersonasSeeder:
    personas = load_dataset("proj-persona/PersonaHub", "persona")

    @classmethod
    def random(cls, k: int = 1) -> list[str]:
        personas = cls.personas["train"].shuffle(seed=42).select(range(k))
        return [p['persona'] for p in personas]