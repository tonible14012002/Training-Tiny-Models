from app.utils.scorer import EvaluationUtils
import asyncio

if __name__ == "__main__":
    score = asyncio.run(EvaluationUtils.score_rouge(
        ref="Hey, can you cover me for lunch today?",
        pred="Hey, can you cover the invoice for the construction permit application today?",
        rouge_type="rougeL",
        mode="precision"
    ))
    print(score)