import typing as t
from app.core import services
from app.core import schemas
from src.payment_classifier.llm.litellm import LiteLLMProvider
from src.payment_classifier.prompts import InmemoryPromptManager
from src.payment_classifier.llm.settings import LLMSettings
from src.payment_classifier.inference.prob_inference import ProbModelInference
from app.core.settings import settings
from app import utils as app_utils
from datasets import Dataset

class AutoPipelineV2:
    def __init__(self):
        base_dir = "./cache/pipeline"
        label_config = schemas.PAYMENT_LABEL_V2
        prompt_mgr = InmemoryPromptManager()
        data_manager = services.DataManager(
            label_config=label_config,
            rouge_threshold=0.65,
            base_dir=f'{base_dir}/data'
        )

        llm = LiteLLMProvider(
            settings=LLMSettings(
            llm_model_name="gpt-4.1",
            api_key=settings.OPENAI_API_KEY,
            temperature=0.7,
            num_retries=2,
            )
        )

        self.data_generator = services.DataGeneratorV2(
            llm=llm,
            prompt_mgr=prompt_mgr,
            data_manager=data_manager,
        )
        self.trainer_service = services.TrainerService(
            base_model="prajjwal1/bert-tiny",
            label_config=label_config,
            base_dir=f'{base_dir}/trainer'
        )
        # TODO: Update to use database ErrorBucket instances instead of schema-based buckets
        # This orchestrator needs to be refactored to work with the new ErrorCategorizer API
        # For now, this will fail - use api/v2 endpoints instead
        self.error_categorizer = services.ErrorCategorizer(
            label_config=label_config,
            llm=llm,
            prompt_mgr=prompt_mgr,
            error_buckets=[],  # FIXME: Pass actual ErrorBucket model instances from database
        )

        self.label_config = label_config

    async def run(
        self,
        human_seed_path: t.Optional[str] = None,
    ) -> t.Dict:

        human_seed = self.load_human_seeds() if human_seed_path else []

        # First Iteration
        generated, base_file_path = await self.data_generator.fresh_gen_v2(
            human_seeds=human_seed,
            expect_total_each_label={
                schemas.PAYMENT_LABEL_V2.PAYMENT_INTENT: 200,
                schemas.PAYMENT_LABEL_V2.PAYMENT_REQUEST: 200,
                schemas.PAYMENT_LABEL_V2.OPEN_INTENT: 200
            }
        )

        ds = self.data_generator.data_manager.to_datasets()
        base_checkpoint_id, _ = await self.trainer_service.train(ds, "prob")
        human_eval_ds = self.load_frozen_human_eval()

        inferencer = ProbModelInference(
            peft_path=base_checkpoint_id,
            label_config=schemas.PAYMENT_LABEL_V2
        )

        eval_result = inferencer.evaluate(human_eval_ds)
        samples = app_utils.DatasetHelper.ds_to_samples(human_eval_ds, self.label_config)
        
        self.error_categorizer.categorize_errors(samples)

    def load_frozen_human_eval(self):
        import json

        human_eval_file = ".cache/frozen_human_eval.json"
        msgs = []
        labels = []
        try:
            with open(human_eval_file, 'r') as f:
                for line in f:
                    item = json.loads(line)
                    msgs.append(item['msg'])
                    labels.append(item['label'])
        except Exception as e:
            self.logger.warning(f"Failed to load frozen human eval: {e}")
            raise e
        
        return Dataset.from_dict({
            "msg": msgs,
            "label": labels
        })

    def load_human_seeds(self):
        """Load human seeds from the cache"""
        import json
        import os
        from app.core.schemas.workflow import Sample

        seed_file = ".cache/human_seed.json"
        if not os.path.exists(seed_file):
            return []

        try:
            with open(seed_file, 'r') as f:
                data = json.load(f)
                return [Sample(**item) for item in data]
        except Exception as e:
            self.logger.warning(f"Failed to load human seeds: {e}")
            return []