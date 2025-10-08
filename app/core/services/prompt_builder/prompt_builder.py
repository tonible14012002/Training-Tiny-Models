from app.core import schemas
from app.core.models.models import LabelConfig
from src.payment_classifier.llm.base import BaseLLM
from src.payment_classifier.prompts.base import BasePromptManager

import typing as t

class PromptBuilder:

    TEMPLATE_PROMPT_KEY = "v2/fix/seed"

    def __init__(
            self,
            label_config: LabelConfig,
            llm: BaseLLM,
            prompt_mgr: BasePromptManager
        ):
        self.label_config = label_config
        self.llm = llm
        self.prompt_mgr = prompt_mgr

    def build_generate_data_prompt(self, data_gen_action: t.List[schemas.DataGenerationAction]) -> str | None:
        
        label_instrucs = [] # {"label": str, "instruction": str}]
        for action in data_gen_action:
            label_instrucs.append(self._make_label_instruct(action))
        
        prompt = self.prompt_mgr.get_prompt(self.TEMPLATE_PROMPT_KEY)
        requirement_txt = self._make_instruction(label_instrucs)
        prompt = prompt.replace(
            "{requirement_txt}",
            requirement_txt,
        )

        return prompt
    
    def _make_instruction(self, instructs: t.List[t.Dict[str, str]]) -> str:
        final_txt = ""
        for label_instruct in instructs:
            key, val = label_instruct["label"], label_instruct["instruction"]
            if val:
                final_txt += f"## {key}:\n{val}\n"
            else:
                final_txt += f"## {key}:\nNo specific requirements. Be diverse in examples and ensure equal distribution.\n"

        return final_txt.strip()

    def _make_label_instruct(self, action: schemas.DataGenerationAction):
        if action.ignore:
            return {
                "label": action.label,
                "instruction": ""
            }

        keyword_req = "- Keywords to include: " + ", ".join(action.keywords_to_include)
        keyword_avoid = "- Keywords to avoid: " + ", ".join(action.keywords_to_avoid)
        patterns = "- Patterns: \n" + "\n- ".join(action.sentence_patterns)
        diversity = "- Diversity: \n" + "\n- ".join(action.diversity_constraints)
        avoid = "- Avoid words: " + ", ".join(action.keywords_to_avoid)

        requirement_instruction = f"""
            {keyword_req if action.keywords_to_include else ""}
            {keyword_avoid if action.keywords_to_avoid else ""}
            {patterns if action.sentence_patterns else ""}
            {avoid if action.keywords_to_avoid else ""}
            {diversity if action.diversity_constraints else ""}
        """

        return  {
            "label": action.label,
            "instruction": requirement_instruction
        }