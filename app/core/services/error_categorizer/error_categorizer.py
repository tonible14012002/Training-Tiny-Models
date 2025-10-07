from app.core import schemas
from src.payment_classifier.llm.base import BaseLLM
from src.payment_classifier.prompts.base import BasePromptManager
import typing as t
from sentence_transformers import SentenceTransformer
from pydantic import BaseModel, Field

def get_init_buckets() -> t.Dict[str, schemas.ErrorBucket]:
    return schemas.INIT_ERROR_BUCKETS.copy()

class CategorizeResult(BaseModel):
    bucket: str = Field(description="The name of the error bucket the test case is categorized into")
    reason: str = Field(description="A brief explanation of why the test case was categorized into this bucket")

class ErrorCategorizer:
    def __init__(
            self,
            label_config: schemas.BaseLabelConfig,
            llm: BaseLLM,
            prompt_mgr: BasePromptManager
        ):
        self.label_config = label_config
        self.buckets = get_init_buckets()
        self.llm = llm
        self.prompt_mgr = prompt_mgr
    
    async def categorize_error(self, testcase: schemas.AnalyzeTestCase) -> CategorizeResult:
        """Categorize a single test case into error buckets"""
        prompt = self.prompt_mgr.get_prompt("v2/analyze/categorize_error")

        bucket_details = ""
        for bucket in self.buckets.values():
            bucket_details += f"- {bucket.name}: {bucket.description}\n"

        prompt = prompt.format(
            message=testcase.sample.msg,
            true_label=testcase.sample.label,
            pred_label=testcase.predicted if testcase.predicted else "N/A",
            bucket_details=bucket_details
        )

        result: CategorizeResult = await self.llm.generate_structured_output([
            {"role": "system", "content": prompt},
        ], CategorizeResult)

        return result
