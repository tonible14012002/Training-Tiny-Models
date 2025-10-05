import typing as t
from app.core.services import *
from app.core.schemas import BaseLabelConfig, PAYMENT_LABEL_V2
from src.payment_classifier.llm.litellm import LiteLLMProvider
from src.payment_classifier.prompts import InmemoryPromptManager
from src.payment_classifier.llm.settings import LLMSettings
from src.payment_classifier.inference.prob_inference import ProbModelInference
from app.core.settings import settings
from datasets import Dataset

class AutoPipelineV2:
    def __init__(self):
        base_dir = "./cache/pipeline"
        label_config = PAYMENT_LABEL_V2
        prompt_mgr = InmemoryPromptManager()
        data_manager = DataManager(
            label_config=label_config,
            rouge_threshold=0.6,
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

        self.data_generator = DataGeneratorV2(
            llm=llm,
            prompt_mgr=prompt_mgr,
            data_manager=data_manager,
        )
        self.trainer_service = TrainerService(
            base_model="prajjwal1/bert-tiny",
            label_config=label_config,
            base_dir=f'{base_dir}/trainer'
        )
        self.label_config = label_config

        # eval_data_manager = EvalDataManager(
        #     label_config=label_config,
        #     rouge_threshold=0.6,
        #     base_dir=f'{base_dir}/eval'
        # )
        #
        # self.eval_generator = EvalGenerator(
        #     llm=llm,
        #     prompt_mgr=prompt_mgr,
        #     eval_data_manager=eval_data_manager,
        #     data_manager=data_manager,
        # )


    async def run(
        self,
        base_dir: str = "./cache/pipeline",
        max_iterations: int = 20,
        target_f1_per_label: float = 0.7,
        samples_per_action: int = 500,
        iteration_number: t.Optional[int] = None,
        human_seed_path: t.Optional[str] = None,
    ) -> t.Dict:

        human_seed = self.load_human_seeds() if human_seed_path else []

        # First Iteration
        await self.data_generator.fresh_gen_v2(
            human_seeds=human_seed,
            expect_total_each_label={
                PAYMENT_LABEL_V2.PAYMENT_INTENT: 200,
                PAYMENT_LABEL_V2.PAYMENT_REQUEST: 200,
                PAYMENT_LABEL_V2.OPEN_INTENT: 200
            }
        )

        ds = self.data_generator.data_manager.to_datasets()
        base_checkpoint_id = await self.trainer_service.train(ds, "prob")
        human_eval_ds = self.load_frozen_human_eval()

        inferencer = ProbModelInference(
            peft_path=base_checkpoint_id,
            label_config=PAYMENT_LABEL_V2
        )

        eval_result = inferencer.evaluate(human_eval_ds)

    def load_frozen_human_eval(self):
        import json
        import os

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