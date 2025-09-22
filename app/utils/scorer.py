from ragas.metrics import RougeScore
from ragas.dataset_schema import SingleTurnSample
import typing as t

class EvaluationUtils:
    @staticmethod
    async def score_rouge(
        ref: str,
        pred: str,
        mode: t.Literal["fmeasure", "precision", "recall"] = "fmeasure",
        rouge_type: t.Literal["rouge1", "rougeL"] = "rougeL"
    ) -> float:
        sample = SingleTurnSample(
            response=pred,
            reference=ref
        )

        scorer = RougeScore(
            mode=mode,
            rouge_type=rouge_type
        )
        return await scorer.single_turn_ascore(sample)