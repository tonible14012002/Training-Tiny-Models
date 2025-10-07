from pydantic import BaseModel, Field
from typing import Dict, List, Optional, Any
from .workflow import Sample

class DataGenerationAction(BaseModel):
    """Specific data generation action that can be programmatically executed"""
    label: str = Field(description="The label for which to generate/modify examples")
    # expected_count: int = Field(description="Expected number of samples to generate/modify")

    # Specific instructions
    keywords_to_include: List[str] = Field(default=[], description="Specific keywords/phrases that should appear in generated examples")
    keywords_to_avoid: List[str] = Field(default=[], description="Keywords/phrases to avoid or filter out")
    # word_replacements: Dict[str, List[str]] = Field(default={}, description="Word replacement mappings: original -> [alternatives]")

    # Pattern specifications
    sentence_patterns: List[str] = Field(default=[], description="Specific sentence structures/patterns to follow")
    context_requirements: List[str] = Field(default=[], description="Required contextual elements")
    diversity_constraints: List[str] = Field(default=[], description="Diversity requirements (e.g., 'vary sentence length', 'use different personas')")
    ignore: bool = Field(default=False, description="True if no action needed for this label")

    # Distribution targets
    # min_examples: Optional[int] = Field(default=None, description="Minimum examples needed for this pattern")
    # target_distribution: Optional[float] = Field(default=None, description="Target percentage of total dataset")

class ErrorPatternAnalysis(BaseModel):
    """LLM-generated analysis of error patterns for misclassified examples"""
    predicted_label: str = Field(description="The incorrect label that the model predicted")
    expected_label: str = Field(description="The correct label that should have been predicted")

    # What the LLM identified
    identified_issues: List[str] = Field(
        description="Specific, concrete issues found in the data (e.g., 'Missing examples with pronoun I as recipient', 'Overuse of word pay in payment_request')"
    )

    # Programmable fixes
    data_actions: List[DataGenerationAction] = Field(
        description="Specific, programmable actions to fix the identified issues"
    )


class BuildPromptRequest(BaseModel):
    actions: List[DataGenerationAction]