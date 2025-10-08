from app.core import schemas
from app.core.models.models import LabelConfig, ErrorBucket
from src.payment_classifier.llm.base import BaseLLM
from src.payment_classifier.prompts.base import BasePromptManager
import typing as t
import asyncio
import logging

logger = logging.getLogger(__name__)


class ErrorCategorizer:
    def __init__(
            self,
            label_config: LabelConfig,
            llm: BaseLLM,
            prompt_mgr: BasePromptManager,
            error_buckets: t.List[ErrorBucket]
        ):
        self.label_config = label_config
        self.error_buckets = error_buckets
        self.bucket_map = {bucket.name: bucket for bucket in error_buckets}
        self.llm = llm
        self.prompt_mgr = prompt_mgr
    
    async def batch_categorize_errors(self, testcases: t.List[schemas.AnalyzeTestCase]):
        batch = 30
        results: t.List[t.Tuple[schemas.AnalyzeTestCase, schemas.CategorizeResult]] = []

        for i in range(0, len(testcases), batch):
            logger.info(f"Classifying batch {i // batch + 1} over {((len(testcases) - 1) // batch) + 1}")

            batch_testcases = testcases[i:i+batch]
            async def do_analyze(testcase: schemas.AnalyzeTestCase):
                cat_result = await self.categorize_error(testcase)
                return testcase, cat_result

            tasks = [do_analyze(testcase) for testcase in batch_testcases]

            batch_results = await asyncio.gather(*tasks)
            results.extend(batch_results)
        
        return results

    async def categorize_error(self, testcase: schemas.AnalyzeTestCase) -> schemas.CategorizeResult:
        """Categorize a single test case into error buckets"""
        prompt = self.prompt_mgr.get_prompt("v2/analyze/categorize_error")

        bucket_details = ""
        for bucket in self.error_buckets:
            bucket_details += f"- {bucket.name}: {bucket.description}\n"

        prompt = prompt.format(
            message=testcase.sample.msg,
            true_label=testcase.sample.label,
            pred_label=testcase.predicted if testcase.predicted else "N/A",
            bucket_details=bucket_details
        )

        retries = 50
        result = None
        while retries > 0:
            try:
                result: schemas.CategorizeResult = await self.llm.generate_structured_output([
                    {"role": "system", "content": prompt},
                ], schemas.CategorizeResult)
                break
            except Exception as e:
                retries -= 1
                await asyncio.sleep(51 - retries)
                print(f"Error during categorization: {e}, Retrying...")

        return result
