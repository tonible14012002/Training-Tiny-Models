"""
Prompt Builder Service

Converts model analyzer output into consolidated prompts for data generation.
Uses LLM to generate fine-tuned prompts with placeholders for balanced generation.
"""

import logging
from typing import Dict, List, Any, Optional
from dataclasses import dataclass
import asyncio
from collections import defaultdict

from src.payment_classifier.llm.litellm import LiteLLMProvider
from src.payment_classifier.llm.settings import LLMSettings

logger = logging.getLogger(__name__)


@dataclass
class ConsolidatedPrompt:
    """Represents a consolidated prompt for balanced data generation"""
    error_pattern_key: str  # e.g., "open_intent->payment_intent"
    predicted_label: str
    expected_label: str
    prompt_text: str
    generation_placeholder: str  # e.g., "{quantity}"
    identified_issues: List[str]
    consolidated_actions: Dict[str, Any]


class PromptBuilderService:
    """
    Service that converts model analysis results into consolidated prompts for data generation.

    Groups error patterns by predicted->expected label pairs and uses LLM to generate
    fine-tuned prompts with quantity placeholders for balanced generation.
    """

    def __init__(self, llm_client: Optional[LiteLLMProvider] = None):
        self.llm_client = llm_client or self._create_default_llm_client()

    def _create_default_llm_client(self) -> Optional[LiteLLMProvider]:
        """Create a default LLM client if none provided"""
        import os
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            logger.warning("No OPENAI_API_KEY found, LLM prompt generation will use fallback")
            return None

        return LiteLLMProvider(LLMSettings(
            llm_model_name="gpt-4",
            api_key=api_key,
            temperature=0.3,
            num_retries=2,
        ))

    def build_consolidated_prompts(self, analysis_result: Dict[str, Any]) -> List[ConsolidatedPrompt]:
        """
        Convert analyzer output into consolidated prompts grouped by error patterns.

        Args:
            analysis_result: Output from ModelAnalyzer containing error analyses

        Returns:
            List of ConsolidatedPrompt objects for balanced data generation
        """
        if "error_analyses" not in analysis_result:
            logger.warning("No error_analyses found in analysis result")
            return []

        # Group error analyses by predicted->expected label pairs
        grouped_errors = self._group_errors_by_pattern(analysis_result["error_analyses"])

        # Generate consolidated prompts for each group
        consolidated_prompts = []
        for pattern_key, error_group in grouped_errors.items():
            prompt = self._generate_consolidated_prompt(pattern_key, error_group)
            if prompt:
                consolidated_prompts.append(prompt)

        logger.info(f"Built {len(consolidated_prompts)} consolidated generation prompts")
        return consolidated_prompts

    def _group_errors_by_pattern(self, error_analyses: List[Dict[str, Any]]) -> Dict[str, List[Dict[str, Any]]]:
        """Group error analyses by predicted->expected label pattern"""
        grouped = defaultdict(list)

        for error_analysis in error_analyses:
            predicted = error_analysis.get("predicted_label", "unknown")
            expected = error_analysis.get("expected_label", "unknown")
            pattern_key = f"{predicted}->{expected}"
            grouped[pattern_key].append(error_analysis)

        return dict(grouped)

    def _generate_consolidated_prompt(self, pattern_key: str, error_group: List[Dict[str, Any]]) -> Optional[ConsolidatedPrompt]:
        """Generate a consolidated prompt for a group of related errors"""
        if not error_group:
            return None

        # Extract common information
        first_error = error_group[0]
        predicted_label = first_error.get("predicted_label", "unknown")
        expected_label = first_error.get("expected_label", "unknown")

        # Consolidate all issues and actions
        all_issues = []
        all_actions = defaultdict(list)

        for error in error_group:
            all_issues.extend(error.get("identified_issues", []))
            for action in error.get("data_actions", []):
                target_label = action.get("target_label")
                action_type = action.get("action_type")
                if target_label and action_type:
                    all_actions[f"{target_label}_{action_type}"].append(action)

        # Generate LLM-based prompt
        prompt_text = self._generate_llm_prompt(
            predicted_label=predicted_label,
            expected_label=expected_label,
            issues=all_issues,
            actions=all_actions
        )

        return ConsolidatedPrompt(
            error_pattern_key=pattern_key,
            predicted_label=predicted_label,
            expected_label=expected_label,
            prompt_text=prompt_text,
            generation_placeholder="{quantity}",
            identified_issues=all_issues,
            consolidated_actions=dict(all_actions)
        )

    def _generate_llm_prompt(self, predicted_label: str, expected_label: str,
                           issues: List[str], actions: Dict[str, List[Dict[str, Any]]]) -> str:
        """Use LLM to generate a fine-tuned prompt based on error analysis"""

        # Build the meta-prompt for the LLM
        meta_prompt = f"""You are an expert prompt engineer creating training data generation prompts for a payment intention classification model.

The model incorrectly predicted '{predicted_label}' when it should have predicted '{expected_label}'.

## Identified Issues:
{self._format_issues(issues)}

## Required Data Actions:
{self._format_actions(actions)}

## Task:
Create a comprehensive data generation prompt that addresses these issues. The prompt should:

1. Start with a clear task definition for payment intent classification
2. Define the label categories (payment_intent, payment_request, open_intent)
3. Include specific generation rules for each relevant label
4. Use the keywords, sentence patterns, and constraints from the data actions
5. Include a placeholder {{quantity}} for the number of examples to generate
6. Ensure the prompt promotes balanced generation across all labels
7. Include persona simulation instructions

## Example Format:
You are generating training data for a payment intention detection model.
The goal is to create chat messages that simulate how real humans express payment intentions.

## Task Definition
Generate {{quantity}} examples that show different types of payment intentions in chat messages...

## Important Rules
### [label_name]
Generate examples using these keywords: [keywords]
Follow these sentence patterns: [patterns]
[Additional constraints]

Please generate a complete, production-ready prompt:"""

        if not self.llm_client:
            logger.info("No LLM client available, using fallback prompt")
            return self._generate_fallback_prompt(predicted_label, expected_label, issues, actions)

        try:
            # Use the async interface
            import asyncio
            if asyncio.iscoroutinefunction(self.llm_client.generate_output):
                loop = asyncio.get_event_loop()
                response = loop.run_until_complete(
                    self.llm_client.generate_output([{"role": "user", "content": meta_prompt}])
                )
            else:
                # Fallback if synchronous method exists
                response = self.llm_client.generate_output([{"role": "user", "content": meta_prompt}])

            return response.strip()
        except Exception as e:
            logger.error(f"Failed to generate LLM prompt: {e}")
            return self._generate_fallback_prompt(predicted_label, expected_label, issues, actions)

    def _format_issues(self, issues: List[str]) -> str:
        """Format identified issues for the meta-prompt"""
        formatted = []
        for i, issue in enumerate(issues[:5], 1):  # Limit to top 5 issues
            formatted.append(f"{i}. {issue}")
        return "\n".join(formatted)

    def _format_actions(self, actions: Dict[str, List[Dict[str, Any]]]) -> str:
        """Format data actions for the meta-prompt"""
        formatted = []

        for action_key, action_list in actions.items():
            if not action_list:
                continue

            # Take the first action as representative
            action = action_list[0]
            target_label = action.get("target_label", "")
            action_type = action.get("action_type", "")

            formatted.append(f"\n### {target_label} ({action_type}):")

            if action.get("keywords_to_include"):
                keywords = ", ".join(action["keywords_to_include"][:8])  # Limit keywords
                formatted.append(f"- Keywords to include: {keywords}")

            if action.get("sentence_patterns"):
                patterns = action["sentence_patterns"][:5]  # Limit patterns
                formatted.append("- Sentence patterns:")
                for pattern in patterns:
                    formatted.append(f"  * {pattern}")

            if action.get("diversity_constraints"):
                constraints = action["diversity_constraints"][:3]  # Limit constraints
                formatted.append("- Diversity requirements:")
                for constraint in constraints:
                    formatted.append(f"  * {constraint}")

        return "\n".join(formatted)

    def _generate_fallback_prompt(self, predicted_label: str, expected_label: str,
                                issues: List[str], actions: Dict[str, List[Dict[str, Any]]]) -> str:
        """Generate a fallback prompt if LLM generation fails"""
        return f"""You are generating training data for a payment intention detection model.
The goal is to create chat messages that simulate how real humans express payment intentions.

## Task Definition
Generate {{quantity}} examples that address the confusion between '{predicted_label}' and '{expected_label}'.

## Important Rules
The model incorrectly predicted '{predicted_label}' instead of '{expected_label}'.

### Key Issues to Address:
{chr(10).join(f"- {issue}" for issue in issues[:3])}

### Generation Requirements:
- Create diverse, realistic chat messages
- Vary sentence structures and formality levels
- Include different payment methods and contexts
- Ensure clear intent distinctions

Generate examples that would help the model correctly distinguish between these labels.
"""

    def get_balanced_generation_plan(self, prompts: List[ConsolidatedPrompt],
                                   total_examples: int = 100) -> Dict[str, Dict[str, int]]:
        """
        Create a balanced generation plan across all labels.

        Args:
            prompts: List of consolidated prompts
            total_examples: Total number of examples to generate

        Returns:
            Dictionary mapping pattern_key to label quantities
        """
        if not prompts:
            return {}

        # Calculate examples per pattern
        examples_per_pattern = total_examples // len(prompts)
        remainder = total_examples % len(prompts)

        generation_plan = {}
        for i, prompt in enumerate(prompts):
            # Distribute examples between predicted and expected labels
            pattern_examples = examples_per_pattern + (1 if i < remainder else 0)

            # Split between the two labels (expected gets more weight)
            expected_count = int(pattern_examples * 0.6)
            predicted_count = pattern_examples - expected_count

            generation_plan[prompt.error_pattern_key] = {
                prompt.expected_label: expected_count,
                prompt.predicted_label: predicted_count
            }

        return generation_plan

    def format_prompt_with_quantity(self, prompt: ConsolidatedPrompt, quantity: int) -> str:
        """Format the prompt with a specific quantity"""
        return prompt.prompt_text.replace("{quantity}", str(quantity))

    def summarize_prompts(self, prompts: List[ConsolidatedPrompt]) -> Dict[str, Any]:
        """Provide a summary of consolidated prompts"""
        summary = {
            "total_prompts": len(prompts),
            "error_patterns": [p.error_pattern_key for p in prompts],
            "labels_involved": set(),
            "total_issues": 0
        }

        for prompt in prompts:
            summary["labels_involved"].add(prompt.predicted_label)
            summary["labels_involved"].add(prompt.expected_label)
            summary["total_issues"] += len(prompt.identified_issues)

        summary["labels_involved"] = list(summary["labels_involved"])
        return summary