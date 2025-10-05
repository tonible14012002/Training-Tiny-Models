# Module Update Plan - Key Specifications Implementation

**Version:** 1.0
**Date:** 2025-10-05
**Status:** Planning Phase

---

## 📋 Executive Summary

This document outlines the implementation plan for four critical module updates to enhance the automated training pipeline. These updates address error taxonomy, curriculum learning, semantic deduplication, and error bucket tracking.

### Priority Modules (Ordered by Implementation)

1. **Error Taxonomy Enhancement** - Entity-based error categorization
2. **Confidence & Curriculum Learning** - Distribution-aware data generation
3. **Semantic Deduplication** - Cosine similarity filtering
4. **Error Bucket Tracking** - Per-bucket metrics and convergence detection

---

## 🎯 Module 1: Error Taxonomy Enhancement

### Current State
- **Location**: `app/core/services/error_pattern_analyzer/error_pattern_analyzer.py`
- Simple LLM prompting: "What should be generated next, given these errors?"
- Groups errors by pattern: `{expected_label}->{predicted_label}`
- Uses static predefined error buckets

### Target State
- **Entity-based categorization**: Extract entities/keywords causing errors
- **Dynamic bucket extension**: LLM extends bucket list when no match found
- **Bucket weighting**: Priority based on error frequency per bucket

### Implementation Steps

#### Step 1.1: Enhance Schema Definitions
**File**: `app/core/schemas/analysis.py`

**Changes**:
```python
class ErrorBucket(BaseModel):
    name: str
    description: Optional[str] = None
    approach: str
    example_issue: Optional[str] = None
    data_generation_strategy: Optional[str] = None

    # NEW FIELDS
    entities: List[str] = Field(
        default=[],
        description="Keywords/entities causing this error"
    )
    error_count: int = Field(
        default=0,
        description="Number of errors in this bucket"
    )
    priority_weight: float = Field(
        default=1.0,
        description="Priority weight based on error frequency"
    )

class ErrorPatternAnalysis(BaseModel):
    predicted_label: str
    expected_label: str
    identified_issues: List[str]
    data_actions: List[DataGenerationAction]

    # NEW FIELDS
    error_bucket: Optional[ErrorBucket] = None
    extracted_entities: List[str] = Field(
        default=[],
        description="Entities extracted from error samples"
    )
```

#### Step 1.2: Create Entity Extraction Service
**New File**: `app/core/services/error_pattern_analyzer/entity_extractor.py`

**Purpose**: Use LLM to extract problematic entities from error patterns

**Key Methods**:
```python
class ErrorEntityExtractor:
    """Extract entities and keywords from error patterns using LLM"""

    def __init__(self, llm: BaseLLM, prompt_mgr: BasePromptManager):
        self.llm = llm
        self.prompt_mgr = prompt_mgr

    async def extract_entities(
        self,
        error_cases: List[ErrorCase]
    ) -> List[str]:
        """
        Extract keywords/entities causing errors from error samples.

        Returns:
            List of entities like ["payment methods", "informal slang",
            "currency formats", "temporal phrases"]
        """
        # Build prompt with error samples
        # Call LLM to identify common problematic entities
        # Return extracted entity list

    async def categorize_to_bucket(
        self,
        error_cases: List[ErrorCase],
        existing_buckets: List[ErrorBucket]
    ) -> ErrorBucket:
        """
        Categorize errors into existing bucket or create new one.

        Process:
        1. Extract entities from error cases
        2. Compare with existing bucket descriptions
        3. Return matching bucket OR create new bucket
        """

    async def extend_bucket_list(
        self,
        error_pattern: str,
        existing_buckets: List[ErrorBucket],
        extracted_entities: List[str]
    ) -> ErrorBucket:
        """
        Create new error bucket when no existing match found.

        Returns:
            New ErrorBucket with:
            - name: Descriptive bucket name
            - description: Error pattern description
            - entities: Extracted problematic entities
            - approach: Recommended fix strategy
        """
```

#### Step 1.3: Update Error Pattern Analyzer
**File**: `app/core/services/error_pattern_analyzer/error_pattern_analyzer.py`

**Modifications to `_analyze_single_error_group()`**:
```python
async def _analyze_single_error_group(
    self,
    pattern_key: str,
    error_cases: List[ErrorCase],
    label_explanations: Dict[str, str]
) -> Optional[ErrorPatternAnalysis]:
    """Enhanced with entity extraction and bucket categorization"""

    try:
        # EXISTING: Parse pattern key
        expected_label, predicted_label = pattern_key.split('->', 1)

        # NEW: Extract entities from error cases
        extracted_entities = await self.entity_extractor.extract_entities(
            error_cases
        )

        # NEW: Categorize into error bucket
        error_bucket = await self.entity_extractor.categorize_to_bucket(
            error_cases,
            EXAMPLE_ERROR_BUCKETS
        )

        # NEW: Update bucket with error count
        error_bucket.error_count = len(error_cases)
        error_bucket.entities = extracted_entities

        # NEW: Calculate priority weight (more errors = higher priority)
        error_bucket.priority_weight = len(error_cases) / 10.0

        # EXISTING: Format test cases and get LLM analysis
        test_cases = self._format_test_cases(error_cases)
        label_explanation = self._get_label_explanation(
            expected_label, predicted_label, label_explanations
        )

        # EXISTING: Get prompt and call LLM
        prompt_template = self.prompt_mgr.get_prompt(
            "analyze/error_pattern_detect"
        )
        formatted_prompt = prompt_template.format(
            predicted_label=predicted_label,
            expected_label=expected_label,
            label_explanation=label_explanation,
            test_cases=test_cases,
            # NEW: Include extracted entities in prompt
            extracted_entities=", ".join(extracted_entities)
        )

        messages = [{"role": "system", "content": formatted_prompt}]
        result = await self.llm.generate_structured_output(
            messages,
            ErrorPatternAnalysis
        )

        # Set metadata
        result.predicted_label = predicted_label
        result.expected_label = expected_label

        # NEW: Attach bucket and entities
        result.error_bucket = error_bucket
        result.extracted_entities = extracted_entities

        return result

    except Exception as e:
        logger.error(f"Error analyzing pattern {pattern_key}: {e}")
        return None
```

#### Step 1.4: Create Required Prompts
**New Prompt Files** (in `src/payment_classifier/prompts/`):

1. **`analyze/extract_entities.txt`**
```
You are analyzing classification errors to identify problematic entities and keywords.

Given these misclassified examples:
{error_examples}

Expected label: {expected_label}
Predicted label: {predicted_label}

Extract 3-5 specific entities, keywords, or linguistic patterns that are causing these errors.

Examples:
- "informal payment slang" (e.g., "spot me", "front me")
- "currency formats" (e.g., "50 bucks", "0.01 BTC")
- "temporal references" (e.g., "tomorrow", "later")
- "payment platform names" (e.g., "Venmo", "PayPal")

Focus on concrete, actionable patterns that can guide data generation.
```

2. **`analyze/categorize_bucket.txt`**
```
Categorize this error pattern into an existing error bucket or suggest a new one.

Error Pattern: {pattern_description}
Extracted Entities: {entities}
Misclassified Examples: {error_examples}

Existing Error Buckets:
{existing_buckets}

Determine if this pattern matches an existing bucket OR if a new bucket is needed.
If creating a new bucket, provide:
- Bucket name
- Description
- Approach to fix
- Data generation strategy
```

**Update**: `src/payment_classifier/prompts/inmemory_prompt_manager.py` to include new prompts

---

## 🎯 Module 2: Confidence & Curriculum Learning

### Current State
- **Location**: `app/core/services/data_generator/data_generator_v2.py`
- Random sampling from human seeds
- No confidence-based filtering
- No distribution enforcement across error types

### Target State
- **Distribution guard**: 40% error-fixing / 40% low-confidence / 20% random
- **Confidence scoring**: Identify uncertain predictions
- **Weighted sampling**: More errors in bucket → more generated samples

### Implementation Steps

#### Step 2.1: Create Confidence Scoring Utility
**New File**: `app/utils/confidence_scorer.py`

```python
from typing import List
from app.core.schemas.analysis import Prediction, TestCase, ErrorCase

class ConfidenceScorer:
    """Score model predictions by confidence level"""

    def __init__(self, confidence_threshold: float = 0.7):
        self.confidence_threshold = confidence_threshold

    def calculate_confidence(self, prediction: Prediction) -> float:
        """
        Calculate confidence score from prediction.

        For ADB: Combine probability and distance
        For Prob: Use probability directly

        Returns:
            Confidence score in [0, 1]
        """
        if prediction.dis is not None:
            # ADB inference: lower distance = higher confidence
            # Normalize: conf = prob * (1 / (1 + distance))
            distance_factor = 1 / (1 + prediction.dis)
            return prediction.prob * distance_factor
        else:
            # Probability-based inference
            return prediction.prob

    def filter_low_confidence(
        self,
        test_cases: List[TestCase],
        threshold: float = None
    ) -> List[ErrorCase]:
        """
        Extract low-confidence predictions (correct but uncertain).

        Args:
            test_cases: All test cases from evaluation
            threshold: Confidence threshold (uses self.confidence_threshold if None)

        Returns:
            List of ErrorCase for low-confidence CORRECT predictions
        """
        threshold = threshold or self.confidence_threshold
        low_confidence_cases = []

        for test_case in test_cases:
            if not test_case.prediction:
                continue

            # Check if prediction is correct
            is_correct = (test_case.true_label == test_case.prediction.label)

            # Calculate confidence
            confidence = self.calculate_confidence(test_case.prediction)

            # Include if correct but low confidence
            if is_correct and confidence < threshold:
                error_case = ErrorCase(
                    input=test_case.input,
                    true_label=test_case.true_label,
                    predicted_label=test_case.prediction.label,
                    confidence=confidence,
                    distance=test_case.prediction.dis
                )
                low_confidence_cases.append(error_case)

        return low_confidence_cases

    def get_confidence_distribution(
        self,
        test_cases: List[TestCase]
    ) -> dict:
        """
        Analyze confidence distribution across predictions.

        Returns:
            Dict with statistics: mean, median, low_count, high_count
        """
        confidences = [
            self.calculate_confidence(tc.prediction)
            for tc in test_cases
            if tc.prediction
        ]

        return {
            "mean": np.mean(confidences) if confidences else 0,
            "median": np.median(confidences) if confidences else 0,
            "std": np.std(confidences) if confidences else 0,
            "low_confidence_count": sum(1 for c in confidences if c < self.confidence_threshold),
            "high_confidence_count": sum(1 for c in confidences if c >= self.confidence_threshold)
        }
```

#### Step 2.2: Create Distribution Guard
**New File**: `app/core/services/data_generator/distribution_guard.py`

```python
import random
from typing import Dict, List, Tuple
from app.core.schemas.workflow import Sample
from app.core.schemas.analysis import ErrorCase, ErrorBucket

class DistributionGuard:
    """Ensure balanced data generation across error buckets and sample types"""

    def __init__(self, target_distribution: Dict[str, float] = None):
        """
        Initialize with target distribution.

        Args:
            target_distribution: Dict with ratios for each sample type
                Default: {"error_fixing": 0.4, "low_confidence": 0.4, "random": 0.2}
        """
        self.target_distribution = target_distribution or {
            "error_fixing": 0.4,
            "low_confidence": 0.4,
            "random": 0.2
        }

        # Validate distribution sums to 1.0
        total = sum(self.target_distribution.values())
        assert abs(total - 1.0) < 0.01, f"Distribution must sum to 1.0, got {total}"

    def calculate_bucket_weights(
        self,
        error_buckets: List[ErrorBucket]
    ) -> Dict[str, float]:
        """
        Weight buckets by error frequency.

        More errors in a bucket → higher weight → more samples generated

        Args:
            error_buckets: List of error buckets with error_count

        Returns:
            Dict mapping bucket_name to weight (sums to 1.0)
        """
        if not error_buckets:
            return {}

        # Calculate total errors
        total_errors = sum(bucket.error_count for bucket in error_buckets)

        if total_errors == 0:
            # Equal weight if no errors
            weight = 1.0 / len(error_buckets)
            return {bucket.name: weight for bucket in error_buckets}

        # Weight proportional to error count
        weights = {}
        for bucket in error_buckets:
            weights[bucket.name] = bucket.error_count / total_errors

        return weights

    def sample_with_distribution(
        self,
        error_cases: List[ErrorCase],
        low_confidence_cases: List[ErrorCase],
        random_pool: List[Sample],
        total_samples: int
    ) -> Dict[str, List]:
        """
        Sample according to 40/40/20 distribution.

        Args:
            error_cases: Misclassified cases to fix
            low_confidence_cases: Correct but uncertain predictions
            random_pool: Random samples for diversity
            total_samples: Total number of samples to generate

        Returns:
            Dict with three lists:
            - "error_fixing": Samples addressing errors (40%)
            - "low_confidence": Samples addressing uncertainty (40%)
            - "random": Random samples for diversity (20%)
        """
        # Calculate target counts
        n_error = int(total_samples * self.target_distribution["error_fixing"])
        n_low_conf = int(total_samples * self.target_distribution["low_confidence"])
        n_random = total_samples - n_error - n_low_conf  # Remainder

        # Sample from each pool
        sampled_errors = random.sample(
            error_cases,
            min(n_error, len(error_cases))
        )

        sampled_low_conf = random.sample(
            low_confidence_cases,
            min(n_low_conf, len(low_confidence_cases))
        )

        sampled_random = random.sample(
            random_pool,
            min(n_random, len(random_pool))
        )

        return {
            "error_fixing": sampled_errors,
            "low_confidence": sampled_low_conf,
            "random": sampled_random
        }

    def enforce_per_label_balance(
        self,
        samples: List[Sample],
        target_distribution: Dict[str, float] = None
    ) -> List[Sample]:
        """
        Ensure balanced distribution across labels.

        Args:
            samples: Generated samples
            target_distribution: Optional target label distribution

        Returns:
            Balanced sample list
        """
        # Group by label
        by_label = {}
        for sample in samples:
            if sample.label not in by_label:
                by_label[sample.label] = []
            by_label[sample.label].append(sample)

        if target_distribution:
            # Use specified distribution
            balanced = []
            total = len(samples)
            for label, ratio in target_distribution.items():
                target_count = int(total * ratio)
                label_samples = by_label.get(label, [])
                sampled = random.sample(label_samples, min(target_count, len(label_samples)))
                balanced.extend(sampled)
            return balanced
        else:
            # Equal distribution
            min_count = min(len(cases) for cases in by_label.values())
            balanced = []
            for label_samples in by_label.values():
                balanced.extend(random.sample(label_samples, min_count))
            return balanced
```

#### Step 2.3: Update DataGeneratorV2
**File**: `app/core/services/data_generator/data_generator_v2.py`

**Add new method**:
```python
from app.utils.confidence_scorer import ConfidenceScorer
from app.core.services.data_generator.distribution_guard import DistributionGuard

class DataGeneratorV2(DataGenerator):
    # EXISTING code...

    def __init__(self, ...):
        # EXISTING initialization
        super().__init__(...)

        # NEW: Initialize curriculum learning components
        self.confidence_scorer = ConfidenceScorer()
        self.distribution_guard = DistributionGuard()

    async def curriculum_gen(
        self,
        error_analyses: List[ErrorPatternAnalysis],
        evaluation_result: EvaluationResult,
        human_seeds: List[Sample],
        amount: int
    ) -> tuple[List[Sample], str, int]:
        """
        Generate with curriculum learning using 40/40/20 distribution.

        Process:
        1. Extract error cases from error analyses
        2. Extract low-confidence cases from evaluation
        3. Calculate bucket weights based on error frequency
        4. Sample with 40/40/20 distribution
        5. Generate targeted data for each sample set

        Args:
            error_analyses: Analyzed error patterns with buckets
            evaluation_result: Model evaluation with test cases
            human_seeds: Human seed samples for random pool
            amount: Total samples to generate

        Returns:
            Tuple of (all_results, versioned_file_path, saved_count)
        """
        logger.info(f"Starting curriculum generation for {amount} samples")

        # Step 1: Extract error cases from analyses
        error_cases = []
        for analysis in error_analyses:
            # Get error cases from the pattern
            pattern_key = f"{analysis.expected_label}->{analysis.predicted_label}"

            # Extract from errors_by_label
            if evaluation_result.errors_by_label:
                label_errors = evaluation_result.errors_by_label.get(
                    analysis.expected_label, None
                )
                if label_errors:
                    error_cases.extend(label_errors.false_negatives)

        logger.info(f"Extracted {len(error_cases)} error cases")

        # Step 2: Extract low-confidence cases
        low_conf_cases = []
        if evaluation_result.test_cases:
            low_conf_cases = self.confidence_scorer.filter_low_confidence(
                evaluation_result.test_cases
            )

        logger.info(f"Extracted {len(low_conf_cases)} low-confidence cases")

        # Step 3: Calculate bucket weights
        error_buckets = [a.error_bucket for a in error_analyses if a.error_bucket]
        bucket_weights = self.distribution_guard.calculate_bucket_weights(
            error_buckets
        )

        logger.info(f"Bucket weights: {bucket_weights}")

        # Step 4: Sample with distribution
        sampled_sets = self.distribution_guard.sample_with_distribution(
            error_cases=error_cases,
            low_confidence_cases=low_conf_cases,
            random_pool=human_seeds,
            total_samples=amount
        )

        logger.info(
            f"Sampled distribution: "
            f"{len(sampled_sets['error_fixing'])} error-fixing, "
            f"{len(sampled_sets['low_confidence'])} low-confidence, "
            f"{len(sampled_sets['random'])} random"
        )

        # Step 5: Build prompts for each sample set
        all_results = []
        saved_samples = []

        # Generate for error-fixing samples
        if sampled_sets['error_fixing']:
            error_prompt = self._build_error_fixing_prompt(
                sampled_sets['error_fixing'],
                error_analyses
            )
            error_results, error_file, error_count = await self.fix_gen(
                human_seeds=human_seeds,
                prompt=error_prompt,
                amount=len(sampled_sets['error_fixing']) * 2  # Oversample
            )
            all_results.extend(error_results)
            saved_samples.extend(error_results[:error_count])

        # Generate for low-confidence samples
        if sampled_sets['low_confidence']:
            conf_prompt = self._build_confidence_boost_prompt(
                sampled_sets['low_confidence']
            )
            conf_results, conf_file, conf_count = await self.fix_gen(
                human_seeds=human_seeds,
                prompt=conf_prompt,
                amount=len(sampled_sets['low_confidence']) * 2
            )
            all_results.extend(conf_results)
            saved_samples.extend(conf_results[:conf_count])

        # Generate random diversity samples
        if sampled_sets['random']:
            random_results = await self.fresh_gen(
                human_seeds=sampled_sets['random'],
                expect_total_message=len(sampled_sets['random']) * 2
            )
            all_results.extend(random_results)

        # Save combined results
        from datetime import datetime
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        file_suffix = f"curriculum_{timestamp}"

        versioned_file_path = self.data_manager.save_to_versioned_file(
            saved_samples, file_suffix
        )

        logger.info(
            f"Curriculum generation complete: "
            f"{len(all_results)} total, {len(saved_samples)} saved to {versioned_file_path}"
        )

        return all_results, versioned_file_path, len(saved_samples)

    def _build_error_fixing_prompt(
        self,
        error_cases: List[ErrorCase],
        error_analyses: List[ErrorPatternAnalysis]
    ) -> str:
        """Build prompt for generating error-fixing samples"""
        # Use PromptBuilder or create custom prompt focusing on error patterns
        # Include extracted entities and error bucket information
        pass

    def _build_confidence_boost_prompt(
        self,
        low_conf_cases: List[ErrorCase]
    ) -> str:
        """Build prompt for generating clearer boundary examples"""
        # Create prompt to generate more decisive examples
        # Focus on making intent signals clearer
        pass
```

#### Step 2.4: Update TrainingOrchestrator
**File**: `app/core/services/orchestrator/training_orchestrator.py`

**Modify Step 4 in `run()` method** (around line 196-230):
```python
# REPLACE existing data generation with curriculum-aware generation

# Step 4: Generate data using curriculum learning
self.logger.info(
    f"Step 3: Generating {samples_per_action} samples "
    f"with curriculum learning (40/40/20 distribution)..."
)

# Use curriculum generation instead of simple fix_gen
all_results, versioned_file_path, saved_count = await self.data_generator.curriculum_gen(
    error_analyses=error_analyses,
    evaluation_result=evaluation_result,
    human_seeds=human_seeds,
    amount=samples_per_action
)

if saved_count == 0:
    self.logger.warning("No new samples generated. Stopping pipeline.")
    break

self.logger.info(
    f"Curriculum generation complete: "
    f"{len(all_results)} total samples, {saved_count} saved after deduplication"
)

# Prepare dataset for training
combined_dataset = self._combine_versioned_files_to_dataset([versioned_file_path])
```

---

## 🎯 Module 3: Semantic Deduplication

### Current State
- **Location**: `app/core/mixins/deduplication.py`
- Uses ROUGE-L precision with threshold 0.6
- Sequential O(n²) comparison for each sample

### Target State
- **Cosine similarity**: Semantic embedding-based filtering
- **Efficient comparison**: Batch similarity computation
- **Configurable threshold**: 0.85-0.95 for semantic similarity

### Implementation Steps

#### Step 3.1: Add Embedding Service
**New File**: `app/utils/embeddings.py`

```python
from sentence_transformers import SentenceTransformer
import numpy as np
from typing import List
import logging

logger = logging.getLogger(__name__)

class EmbeddingService:
    """Generate and compare semantic embeddings for text similarity"""

    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        """
        Initialize embedding model.

        Args:
            model_name: SentenceTransformer model name
                - "all-MiniLM-L6-v2": Fast, good quality (default)
                - "all-mpnet-base-v2": Better quality, slower
        """
        self.model = SentenceTransformer(model_name)
        self.model_name = model_name
        logger.info(f"Loaded embedding model: {model_name}")

    def encode(self, texts: List[str]) -> np.ndarray:
        """
        Generate embeddings for texts.

        Args:
            texts: List of text strings to encode

        Returns:
            numpy array of shape (len(texts), embedding_dim)
        """
        return self.model.encode(
            texts,
            convert_to_numpy=True,
            show_progress_bar=False
        )

    def cosine_similarity(
        self,
        emb1: np.ndarray,
        emb2: np.ndarray
    ) -> float:
        """
        Calculate cosine similarity between two embeddings.

        Args:
            emb1: First embedding vector
            emb2: Second embedding vector

        Returns:
            Similarity score in [0, 1]
        """
        return float(
            np.dot(emb1, emb2) / (np.linalg.norm(emb1) * np.linalg.norm(emb2))
        )

    def batch_similarity(
        self,
        query_emb: np.ndarray,
        corpus_embs: np.ndarray
    ) -> np.ndarray:
        """
        Calculate similarities between query and corpus efficiently.

        Args:
            query_emb: Single embedding vector (1D array)
            corpus_embs: Multiple embedding vectors (2D array)

        Returns:
            Array of similarity scores
        """
        # Normalize embeddings
        query_norm = query_emb / np.linalg.norm(query_emb)
        corpus_norms = corpus_embs / np.linalg.norm(corpus_embs, axis=1, keepdims=True)

        # Batch dot product
        similarities = np.dot(corpus_norms, query_norm)

        return similarities
```

#### Step 3.2: Update DeduplicationMixin
**File**: `app/core/mixins/deduplication.py`

**Add semantic deduplication methods**:
```python
from app.utils.embeddings import EmbeddingService
import numpy as np

class DeduplicationMixin:
    """
    Mixin class for handling deduplication and filtering logic for data.
    Now supports both ROUGE-L and semantic (cosine similarity) deduplication.
    """

    # EXISTING: rouge_threshold property
    @property
    def rouge_threshold(self) -> float:
        """ROUGE-L threshold for deduplication"""
        raise NotImplementedError(
            "Classes using DeduplicationMixin must define rouge_threshold"
        )

    # NEW: Semantic deduplication properties
    @property
    def cosine_threshold(self) -> float:
        """
        Cosine similarity threshold for semantic deduplication.
        Can be overridden by implementing classes.
        """
        return 0.90  # Default threshold

    @property
    def use_semantic_dedup(self) -> bool:
        """
        Whether to use semantic (cosine) deduplication.
        Can be overridden by implementing classes.
        """
        return True  # Default to semantic dedup

    @property
    def embedding_model(self) -> str:
        """Embedding model name for semantic deduplication"""
        return "all-MiniLM-L6-v2"  # Fast and effective

    # NEW: Semantic deduplication method
    async def deduplicate_semantic(
        self,
        data: List[Sample]
    ) -> List[Sample]:
        """
        Remove duplicates using cosine similarity of embeddings.

        Args:
            data: List of samples to deduplicate

        Returns:
            List of deduplicated samples
        """
        if not data:
            return data

        logger.info(f"Starting semantic deduplication of {len(data)} samples")

        # Initialize embedding service
        embedding_service = EmbeddingService(model_name=self.embedding_model)

        # Encode all messages
        texts = [item.msg for item in data]
        embeddings = embedding_service.encode(texts)

        logger.debug(f"Generated embeddings with shape: {embeddings.shape}")

        # Deduplicate
        deduped = []
        deduped_embeddings = []

        for i, (item, emb) in enumerate(zip(data, embeddings)):
            is_unique = True

            if deduped_embeddings:
                # Batch similarity check against all existing embeddings
                similarities = embedding_service.batch_similarity(
                    emb,
                    np.array(deduped_embeddings)
                )

                # Check if any similarity exceeds threshold
                if np.any(similarities >= self.cosine_threshold):
                    is_unique = False
                    max_sim = np.max(similarities)
                    logger.debug(
                        f"Duplicate found (similarity: {max_sim:.3f}): {item.msg[:50]}..."
                    )

            if is_unique:
                deduped.append(item)
                deduped_embeddings.append(emb)

        logger.info(
            f"Semantic deduplication: {len(data)} → {len(deduped)} samples "
            f"({len(data) - len(deduped)} duplicates removed)"
        )

        return deduped

    # MODIFIED: Main deduplicate method with routing
    async def deduplicate(self, data: List[Sample]) -> List[Sample]:
        """
        Deduplicate samples using configured method.
        Routes to semantic or ROUGE-L based on use_semantic_dedup setting.
        """
        if self.use_semantic_dedup:
            return await self.deduplicate_semantic(data)
        else:
            return await self._deduplicate_rouge(data)

    # RENAMED: Original ROUGE-L method
    async def _deduplicate_rouge(self, data: List[Sample]) -> List[Sample]:
        """
        Remove duplicates using ROUGE-L similarity (original method).

        Args:
            data: List of samples to deduplicate

        Returns:
            List of deduplicated samples
        """
        deduped = {}

        for item in data:
            # First item is always added
            if len(deduped.keys()) == 0:
                deduped[item.msg] = item
                continue

            # Check against existing items
            is_unique = True
            for existing_item in deduped.values():
                rouge = await EvaluationUtils.score_rouge(
                    ref=existing_item.msg,
                    pred=item.msg,
                    rouge_type="rougeL",
                    mode="precision"
                )

                # If ROUGE score is above threshold, consider it a duplicate
                if rouge >= self.rouge_threshold:
                    is_unique = False
                    break

            if is_unique:
                deduped[item.msg] = item

        logger.debug(
            f"ROUGE-L deduplication: {len(data)} → {len(deduped)} samples"
        )
        return list(deduped.values())

    # EXISTING: filter_against_existing (update to support semantic)
    async def filter_against_existing(
        self,
        new_data: List[Sample],
        existing_data: List[Sample],
        window_size: int = 1000
    ) -> List[Sample]:
        """
        Filter new data against existing data to avoid duplicates.
        Now supports semantic similarity filtering.
        """
        if not existing_data:
            return new_data

        # Route based on dedup method
        if self.use_semantic_dedup:
            return await self._filter_semantic(new_data, existing_data, window_size)
        else:
            return await self._filter_rouge(new_data, existing_data, window_size)

    async def _filter_semantic(
        self,
        new_data: List[Sample],
        existing_data: List[Sample],
        window_size: int = 1000
    ) -> List[Sample]:
        """Filter using semantic similarity"""
        embedding_service = EmbeddingService(model_name=self.embedding_model)

        # Group by label
        existing_by_label = {}
        for item in existing_data:
            if item.label not in existing_by_label:
                existing_by_label[item.label] = []
            existing_by_label[item.label].append(item)

        # Apply window
        for label in existing_by_label:
            if len(existing_by_label[label]) > window_size:
                existing_by_label[label] = existing_by_label[label][-window_size:]

        # Encode existing data per label
        existing_embeddings_by_label = {}
        for label, items in existing_by_label.items():
            texts = [item.msg for item in items]
            existing_embeddings_by_label[label] = embedding_service.encode(texts)

        # Filter new data
        filtered = []
        new_texts = [item.msg for item in new_data]
        new_embeddings = embedding_service.encode(new_texts)

        for new_item, new_emb in zip(new_data, new_embeddings):
            is_unique = True

            # Compare against same-label existing items
            existing_embs = existing_embeddings_by_label.get(new_item.label)

            if existing_embs is not None and len(existing_embs) > 0:
                similarities = embedding_service.batch_similarity(
                    new_emb, existing_embs
                )

                if np.any(similarities >= self.cosine_threshold):
                    is_unique = False

            if is_unique:
                filtered.append(new_item)

        logger.debug(
            f"Semantic filtering: {len(new_data)} → {len(filtered)} samples"
        )
        return filtered

    async def _filter_rouge(
        self,
        new_data: List[Sample],
        existing_data: List[Sample],
        window_size: int = 1000
    ) -> List[Sample]:
        """Filter using ROUGE-L (original implementation)"""
        # EXISTING CODE (lines 67-126 from original file)
        # ... keep as is
        pass

    # EXISTING: deduplicate_and_filter method (unchanged)
    async def deduplicate_and_filter(
        self,
        new_data: List[Sample],
        existing_data: List[Sample] = None
    ) -> List[Sample]:
        """Combined deduplication and filtering operation"""
        # First deduplicate within new data
        deduped_data = await self.deduplicate(new_data)

        # Then filter against existing data if provided
        if existing_data:
            filtered_data = await self.filter_against_existing(
                deduped_data, existing_data
            )
            return filtered_data

        return deduped_data
```

#### Step 3.3: Add Dependencies
**File**: `pyproject.toml`

```toml
[tool.poetry.dependencies]
# EXISTING dependencies...

# NEW: For semantic deduplication
sentence-transformers = "^2.2.2"
torch = "^2.0.0"  # Required by sentence-transformers

# OPTIONAL: For faster similarity search (future optimization)
# faiss-cpu = "^1.7.4"
```

**Install**:
```bash
poetry add sentence-transformers
```

---

## 🎯 Module 4: Error Bucket Tracking

### Current State
- No per-bucket metrics tracking
- Only overall model metrics (F1, accuracy)
- No visibility into which error patterns are being fixed

### Target State
- **Per-bucket metrics**: Track error rate for each bucket across iterations
- **Bucket convergence**: Detect when buckets fall below 5% error threshold
- **Historical tracking**: Monitor improvement trends per bucket
- **API endpoints**: Expose bucket metrics for monitoring

### Implementation Steps

#### Step 4.1: Enhance Metrics Schemas
**File**: `app/core/schemas/orchestrator.py`

**Add new schemas**:
```python
from pydantic import BaseModel, Field
from typing import Optional, List

class ErrorBucketMetrics(BaseModel):
    """Metrics for a specific error bucket in one iteration"""

    bucket_name: str = Field(description="Name of the error bucket")
    error_count: int = Field(description="Number of errors in this bucket")
    total_samples: int = Field(
        description="Total samples that could exhibit this error pattern"
    )
    error_rate: float = Field(description="error_count / total_samples")

    improvement_from_last: Optional[float] = Field(
        default=None,
        description="Change in error rate from previous iteration (negative = improvement)"
    )

    is_below_threshold: bool = Field(
        default=False,
        description="Whether bucket is below 5% error threshold"
    )

class IterationMetrics(BaseModel):
    """Metrics for a single training iteration"""

    # EXISTING FIELDS
    iteration: int
    accuracy: float
    macro_f1: float
    unknown_rate: float
    total_samples: int
    checkpoint_path: str
    timestamp: float
    training_time: float
    evaluation_time: float

    # NEW FIELDS for bucket tracking
    bucket_metrics: List[ErrorBucketMetrics] = Field(
        default=[],
        description="Per-bucket error metrics"
    )
    buckets_below_threshold: int = Field(
        default=0,
        description="Number of buckets with error rate < 5%"
    )
    total_buckets: int = Field(
        default=0,
        description="Total number of error buckets tracked"
    )
```

#### Step 4.2: Create Bucket Tracker Service
**New File**: `app/core/services/bucket_tracker/bucket_tracker.py`

```python
import logging
from typing import List, Dict, Optional
from app.core.schemas.analysis import ErrorPatternAnalysis, ErrorBucket
from app.core.schemas.orchestrator import ErrorBucketMetrics
from app.core.schemas.analysis import EvaluationResult

logger = logging.getLogger(__name__)

class ErrorBucketTracker:
    """Track error bucket metrics across training iterations"""

    def __init__(self, error_threshold: float = 0.05):
        """
        Initialize bucket tracker.

        Args:
            error_threshold: Error rate threshold for convergence (default: 5%)
        """
        self.error_threshold = error_threshold
        self.history: List[Dict[str, ErrorBucketMetrics]] = []
        logger.info(f"Initialized ErrorBucketTracker with threshold: {error_threshold}")

    def calculate_bucket_metrics(
        self,
        error_analyses: List[ErrorPatternAnalysis],
        evaluation_result: EvaluationResult
    ) -> List[ErrorBucketMetrics]:
        """
        Calculate current metrics for each error bucket.

        Args:
            error_analyses: List of error pattern analyses with buckets
            evaluation_result: Evaluation result with error counts

        Returns:
            List of ErrorBucketMetrics for this iteration
        """
        metrics = []

        for analysis in error_analyses:
            bucket = analysis.error_bucket
            if not bucket:
                logger.warning(
                    f"Analysis for {analysis.expected_label}->{analysis.predicted_label} "
                    f"has no error bucket"
                )
                continue

            # Get error count from bucket
            error_count = bucket.error_count

            # Estimate total samples that could have this error
            # This is label-dependent: samples with expected_label
            total_samples = self._estimate_bucket_exposure(
                bucket, analysis, evaluation_result
            )

            # Calculate error rate
            error_rate = error_count / total_samples if total_samples > 0 else 0.0

            # Calculate improvement from last iteration
            improvement = None
            if self.history:
                last_metrics = self.history[-1].get(bucket.name)
                if last_metrics:
                    # Negative improvement = error rate increased (bad)
                    # Positive improvement = error rate decreased (good)
                    improvement = last_metrics.error_rate - error_rate

            # Check if below threshold
            is_below_threshold = error_rate < self.error_threshold

            bucket_metric = ErrorBucketMetrics(
                bucket_name=bucket.name,
                error_count=error_count,
                total_samples=total_samples,
                error_rate=error_rate,
                improvement_from_last=improvement,
                is_below_threshold=is_below_threshold
            )

            metrics.append(bucket_metric)

            logger.debug(
                f"Bucket '{bucket.name}': "
                f"{error_rate:.1%} error rate ({error_count}/{total_samples})"
            )

        return metrics

    def _estimate_bucket_exposure(
        self,
        bucket: ErrorBucket,
        analysis: ErrorPatternAnalysis,
        evaluation_result: EvaluationResult
    ) -> int:
        """
        Estimate how many samples could exhibit this error pattern.

        Conservative estimate: samples with the expected label
        """
        expected_label = analysis.expected_label

        # Get label metrics
        if expected_label in evaluation_result.per_label:
            label_metrics = evaluation_result.per_label[expected_label]
            return label_metrics.samples

        # Fallback: use overall sample count
        return evaluation_result.overall.total_samples

    def record_iteration(self, metrics: List[ErrorBucketMetrics]):
        """
        Record metrics for this iteration in history.

        Args:
            metrics: List of bucket metrics for this iteration
        """
        metrics_dict = {m.bucket_name: m for m in metrics}
        self.history.append(metrics_dict)

        logger.info(
            f"Recorded metrics for {len(metrics)} buckets in iteration {len(self.history)}"
        )

    def get_buckets_below_threshold(
        self,
        metrics: List[ErrorBucketMetrics]
    ) -> int:
        """
        Count buckets with error rate below threshold.

        Args:
            metrics: Current bucket metrics

        Returns:
            Number of buckets below error threshold
        """
        return sum(1 for m in metrics if m.is_below_threshold)

    def all_buckets_converged(
        self,
        metrics: List[ErrorBucketMetrics]
    ) -> bool:
        """
        Check if all buckets are below error threshold.

        Args:
            metrics: Current bucket metrics

        Returns:
            True if all buckets have error rate < threshold
        """
        if not metrics:
            return False

        return all(m.is_below_threshold for m in metrics)

    def get_bucket_history(
        self,
        bucket_name: str
    ) -> List[ErrorBucketMetrics]:
        """
        Get historical metrics for a specific bucket.

        Args:
            bucket_name: Name of the bucket

        Returns:
            List of metrics across iterations
        """
        history = []
        for iteration_metrics in self.history:
            if bucket_name in iteration_metrics:
                history.append(iteration_metrics[bucket_name])
        return history

    def get_summary(self) -> Dict:
        """
        Get summary of bucket tracking across all iterations.

        Returns:
            Dict with summary statistics
        """
        if not self.history:
            return {"message": "No tracking history available"}

        latest = self.history[-1]

        return {
            "total_iterations": len(self.history),
            "total_buckets": len(latest),
            "buckets_below_threshold": sum(
                1 for m in latest.values() if m.is_below_threshold
            ),
            "all_converged": all(m.is_below_threshold for m in latest.values()),
            "average_error_rate": sum(m.error_rate for m in latest.values()) / len(latest) if latest else 0,
            "bucket_summary": [
                {
                    "name": name,
                    "error_rate": metric.error_rate,
                    "improvement": metric.improvement_from_last
                }
                for name, metric in latest.items()
            ]
        }
```

**Create module init**:
**New File**: `app/core/services/bucket_tracker/__init__.py`
```python
from .bucket_tracker import ErrorBucketTracker

__all__ = ["ErrorBucketTracker"]
```

#### Step 4.3: Integrate into TrainingOrchestrator
**File**: `app/core/services/orchestrator/training_orchestrator.py`

**Update `__init__` method**:
```python
from app.core.services.bucket_tracker.bucket_tracker import ErrorBucketTracker

class TrainingOrchestrator:
    def __init__(
        self,
        data_generator: DataGeneratorV2,
        trainer_service: TrainerService,
        model_analyzer: ModelAnalyzer,
        data_manager: DataManager,
        eval_data_manager: EvalDataManager,
        error_pattern_analyzer: ErrorPatternAnalysisService,
        prompt_builder: PromptBuilder,
        label_config: Type[BaseLabelConfig]
    ):
        # EXISTING initialization
        self.data_generator = data_generator
        self.trainer_service = trainer_service
        # ... other services

        # NEW: Initialize bucket tracker
        self.bucket_tracker = ErrorBucketTracker(error_threshold=0.05)

        self.status = PipelineStatus()
        self.logger = logging.getLogger(__name__)
```

**Update `run()` method** after error analysis (around line 187-193):
```python
# EXISTING: Error pattern analysis
error_analyses = await self.error_pattern_analyzer.analyze_error_patterns(
    errors_by_label=evaluation_result.errors_by_label,
    label_explanations=label_explanations
)

if not error_analyses:
    self.logger.warning("No error patterns identified. Stopping pipeline.")
    break

self.logger.info(f"Found {len(error_analyses)} error patterns to address")

# NEW: Calculate bucket metrics
self.logger.info("Calculating error bucket metrics...")
bucket_metrics = self.bucket_tracker.calculate_bucket_metrics(
    error_analyses, evaluation_result
)

# NEW: Log bucket status
self.logger.info("\nError Bucket Metrics:")
self.logger.info("-" * 60)
for metric in bucket_metrics:
    status = "✓" if metric.is_below_threshold else "✗"
    improvement = ""
    if metric.improvement_from_last is not None:
        if metric.improvement_from_last > 0:
            improvement = f" (↓ {metric.improvement_from_last:.1%})"
        elif metric.improvement_from_last < 0:
            improvement = f" (↑ {abs(metric.improvement_from_last):.1%})"
        else:
            improvement = " (no change)"

    self.logger.info(
        f"  {status} {metric.bucket_name}: "
        f"{metric.error_rate:.1%} error rate "
        f"({metric.error_count}/{metric.total_samples}){improvement}"
    )
self.logger.info("-" * 60)

# NEW: Check bucket convergence
buckets_below = self.bucket_tracker.get_buckets_below_threshold(bucket_metrics)
self.logger.info(
    f"Buckets below {self.bucket_tracker.error_threshold:.0%} threshold: "
    f"{buckets_below}/{len(bucket_metrics)}"
)

if self.bucket_tracker.all_buckets_converged(bucket_metrics):
    self.logger.info("✓ All error buckets converged (below 5% threshold)")
    self.status.termination_reason = "All error buckets converged below threshold"
    self.status.is_running = False

    return {
        "success": True,
        "status": "completed",
        "termination_reason": self.status.termination_reason,
        "iterations_completed": iteration,
        "final_checkpoint": current_checkpoint_id,
        "bucket_metrics": [
            {
                "name": m.bucket_name,
                "error_rate": m.error_rate,
                "converged": m.is_below_threshold
            }
            for m in bucket_metrics
        ]
    }
```

**Update metrics recording** (around line 266-277):
```python
# Record metrics for this iteration
iteration_metrics = IterationMetrics(
    iteration=iteration,
    accuracy=evaluation_result.overall.accuracy,
    macro_f1=evaluation_result.overall.macro_f1,
    unknown_rate=evaluation_result.overall.unknown_rate,
    total_samples=evaluation_result.overall.total_samples,
    checkpoint_path=str(checkpoint_path),
    timestamp=time.time(),
    training_time=train_time,
    evaluation_time=eval_time,
    # NEW: Add bucket metrics
    bucket_metrics=bucket_metrics,
    buckets_below_threshold=buckets_below,
    total_buckets=len(bucket_metrics)
)

self.status.metrics_history.append(iteration_metrics)

# NEW: Record in bucket tracker
self.bucket_tracker.record_iteration(bucket_metrics)
```

#### Step 4.4: Add API Endpoints
**File**: `app/api/routes/workflow.py`

**Add new endpoint for bucket metrics**:
```python
@router.get("/bucket-metrics")
async def get_bucket_metrics(
    orchestrator: TrainingOrchestrator = Depends(get_orchestrator)
):
    """
    Get error bucket tracking metrics.

    Returns current bucket metrics and historical trends.
    """
    if not orchestrator.bucket_tracker.history:
        return {
            "message": "No bucket tracking data available",
            "status": "no_data"
        }

    # Get latest iteration metrics
    if orchestrator.status.metrics_history:
        latest_iteration = orchestrator.status.metrics_history[-1]
        current_metrics = latest_iteration.bucket_metrics
    else:
        current_metrics = []

    # Get summary
    summary = orchestrator.bucket_tracker.get_summary()

    return {
        "current_iteration": orchestrator.status.current_iteration,
        "bucket_metrics": [
            {
                "bucket_name": m.bucket_name,
                "error_rate": m.error_rate,
                "error_count": m.error_count,
                "total_samples": m.total_samples,
                "improvement": m.improvement_from_last,
                "is_below_threshold": m.is_below_threshold
            }
            for m in current_metrics
        ],
        "buckets_below_threshold": summary.get("buckets_below_threshold", 0),
        "total_buckets": summary.get("total_buckets", 0),
        "convergence_threshold": orchestrator.bucket_tracker.error_threshold,
        "all_converged": summary.get("all_converged", False),
        "average_error_rate": summary.get("average_error_rate", 0)
    }

@router.get("/bucket-metrics/{bucket_name}")
async def get_bucket_history(
    bucket_name: str,
    orchestrator: TrainingOrchestrator = Depends(get_orchestrator)
):
    """
    Get historical metrics for a specific error bucket.

    Args:
        bucket_name: Name of the error bucket

    Returns:
        Historical metrics across iterations
    """
    history = orchestrator.bucket_tracker.get_bucket_history(bucket_name)

    if not history:
        return {
            "message": f"No history found for bucket '{bucket_name}'",
            "bucket_name": bucket_name,
            "status": "not_found"
        }

    return {
        "bucket_name": bucket_name,
        "total_iterations": len(history),
        "history": [
            {
                "error_rate": m.error_rate,
                "error_count": m.error_count,
                "total_samples": m.total_samples,
                "improvement": m.improvement_from_last,
                "is_below_threshold": m.is_below_threshold
            }
            for m in history
        ],
        "current_error_rate": history[-1].error_rate if history else None,
        "converged": history[-1].is_below_threshold if history else False
    }
```

---

## 📊 Implementation Timeline

### Week 1: Foundation
- **Day 1-2**: Error Taxonomy Enhancement (Steps 1.1-1.4)
  - Update schemas
  - Create entity extractor
  - Update error pattern analyzer
  - Create prompts

- **Day 3-5**: Semantic Deduplication (Steps 3.1-3.3)
  - Create embedding service
  - Update deduplication mixin
  - Add dependencies and test

### Week 2: Distribution & Tracking
- **Day 1-3**: Confidence & Curriculum Learning (Steps 2.1-2.4)
  - Create confidence scorer
  - Create distribution guard
  - Update data generator
  - Integrate with orchestrator

- **Day 4-5**: Error Bucket Tracking (Steps 4.1-4.4)
  - Update schemas
  - Create bucket tracker
  - Integrate with orchestrator
  - Add API endpoints

### Week 3: Testing & Refinement
- **Day 1-2**: Unit tests for all modules
- **Day 3-4**: Integration testing
- **Day 5**: Documentation and final validation

---

## 🧪 Testing Strategy

### Unit Tests

#### Test Entity Extraction
```python
# test_entity_extractor.py
async def test_extract_entities():
    """Test entity extraction from error cases"""
    # Given error cases with payment slang
    # When extracting entities
    # Then should identify "informal slang", "payment platforms"

async def test_categorize_to_bucket():
    """Test error categorization to buckets"""
    # Given error cases and existing buckets
    # When categorizing
    # Then should match to appropriate bucket or create new one
```

#### Test Distribution Guard
```python
# test_distribution_guard.py
def test_sample_with_distribution():
    """Test 40/40/20 distribution sampling"""
    # Given pools of error/low-conf/random samples
    # When sampling 100 samples
    # Then should get 40/40/20 distribution

def test_bucket_weights():
    """Test bucket weighting by error frequency"""
    # Given buckets with different error counts
    # When calculating weights
    # Then higher error count should have higher weight
```

#### Test Semantic Deduplication
```python
# test_semantic_dedup.py
async def test_cosine_deduplication():
    """Test semantic similarity deduplication"""
    # Given samples with semantic duplicates
    # When deduplicating
    # Then should remove semantically similar samples

async def test_embedding_efficiency():
    """Test batch embedding performance"""
    # Given 1000 samples
    # When encoding
    # Then should complete in <5 seconds
```

#### Test Bucket Tracker
```python
# test_bucket_tracker.py
def test_calculate_metrics():
    """Test bucket metrics calculation"""
    # Given error analyses and evaluation result
    # When calculating metrics
    # Then should compute correct error rates

def test_convergence_detection():
    """Test convergence detection"""
    # Given buckets with <5% error rates
    # When checking convergence
    # Then should return True
```

### Integration Tests

#### End-to-End Pipeline Test
```python
# test_orchestrator_integration.py
async def test_curriculum_pipeline():
    """Test full pipeline with curriculum learning"""
    # Given initial checkpoint
    # When running orchestrator with curriculum
    # Then should generate with 40/40/20 distribution
    # And track bucket metrics
    # And converge when all buckets <5%
```

### Validation Metrics

- **Error Reduction**: Track per-bucket error rate reduction
  - Target: All buckets <5% within 10 iterations

- **Distribution Adherence**: Verify 40/40/20 distribution
  - Tolerance: ±5% variance acceptable

- **Deduplication Effectiveness**: Compare semantic vs ROUGE-L
  - Measure: Duplicate detection rate, false positive rate

- **Performance Impact**: Monitor overhead of new features
  - Baseline: Current iteration time
  - Target: <20% increase in iteration time

---

## ⚙️ Configuration

### Settings File
**File**: `app/core/settings.py`

```python
class Settings(BaseSettings):
    # EXISTING settings
    OPENAI_API_KEY: str
    LOG_LEVEL: str = "INFO"
    ENVIRONMENT: str = "dev"

    # NEW: Error Taxonomy
    USE_ENTITY_BASED_TAXONOMY: bool = True
    DYNAMIC_BUCKET_EXTENSION: bool = True
    MAX_ERROR_BUCKETS: int = 20

    # NEW: Curriculum Learning
    ERROR_SAMPLE_RATIO: float = 0.4
    LOW_CONFIDENCE_RATIO: float = 0.4
    RANDOM_SAMPLE_RATIO: float = 0.2
    LOW_CONFIDENCE_THRESHOLD: float = 0.7

    # NEW: Deduplication
    USE_SEMANTIC_DEDUP: bool = True
    COSINE_SIMILARITY_THRESHOLD: float = 0.90
    EMBEDDING_MODEL: str = "all-MiniLM-L6-v2"
    FALLBACK_TO_ROUGE: bool = True  # Fallback if embedding fails

    # NEW: Bucket Tracking
    ERROR_BUCKET_THRESHOLD: float = 0.05
    TRACK_BUCKET_HISTORY: bool = True
    ENABLE_BUCKET_CONVERGENCE: bool = True

    class Config:
        env_file = ".env"
```

### Environment Variables
```bash
# .env file
OPENAI_API_KEY=your-key-here

# Error Taxonomy
USE_ENTITY_BASED_TAXONOMY=true
DYNAMIC_BUCKET_EXTENSION=true

# Curriculum Learning
ERROR_SAMPLE_RATIO=0.4
LOW_CONFIDENCE_RATIO=0.4
RANDOM_SAMPLE_RATIO=0.2

# Deduplication
USE_SEMANTIC_DEDUP=true
COSINE_SIMILARITY_THRESHOLD=0.90
EMBEDDING_MODEL=all-MiniLM-L6-v2

# Bucket Tracking
ERROR_BUCKET_THRESHOLD=0.05
TRACK_BUCKET_HISTORY=true
```

---

## 📈 Success Metrics

### Per Module

#### Error Taxonomy
- ✅ 95%+ of errors categorized into buckets
- ✅ <10 total buckets (well-organized taxonomy)
- ✅ Entity extraction accuracy >80%

#### Curriculum Learning
- ✅ Actual distribution within ±5% of 40/40/20 target
- ✅ Low-confidence identification recall >90%
- ✅ Faster convergence vs random sampling (baseline)

#### Semantic Deduplication
- ✅ Duplicate detection rate >95%
- ✅ False positive rate <5%
- ✅ Processing time <10s for 1000 samples

#### Bucket Tracking
- ✅ All buckets tracked across iterations
- ✅ Convergence detected within 10 iterations
- ✅ Metrics API response time <500ms

### Overall System
- ✅ All labels achieve F1 >0.7 within 15 iterations
- ✅ All error buckets <5% error rate at convergence
- ✅ Total training time reduced by 30% vs baseline
- ✅ Generated data diversity score >0.8

---

## 🔍 Monitoring & Debugging

### Logging Strategy

```python
# Add detailed logging for each module

# Error Taxonomy
logger.info(f"Extracted {len(entities)} entities from {len(error_cases)} errors")
logger.debug(f"Entities: {entities}")
logger.info(f"Categorized into bucket: {bucket.name}")

# Curriculum Learning
logger.info(f"Distribution: {len(error_fixing)} error / {len(low_conf)} low-conf / {len(random)} random")
logger.debug(f"Bucket weights: {bucket_weights}")

# Deduplication
logger.info(f"Semantic dedup: {before} → {after} samples ({removed} duplicates)")
logger.debug(f"Cosine threshold: {threshold}, avg similarity: {avg_sim}")

# Bucket Tracking
logger.info(f"Bucket '{name}': {error_rate:.1%} ({count}/{total})")
logger.info(f"Buckets below threshold: {below}/{total_buckets}")
```

### Debug Endpoints

```python
# Add debug endpoints for development

@router.get("/debug/entity-extraction")
async def debug_entity_extraction(...):
    """Debug entity extraction for error cases"""

@router.get("/debug/distribution")
async def debug_distribution(...):
    """Check actual vs target distribution"""

@router.get("/debug/deduplication")
async def debug_deduplication(...):
    """Analyze deduplication effectiveness"""
```

---

## 📝 Documentation Updates

### Update Existing Docs

1. **`docs/AUTO_TRAIN_WORKFLOW.md`**: Add sections for new features
2. **`docs/CLAUDE.md`**: Update with new components and workflows
3. **`docs/API.md`**: Document new endpoints

### New Documentation

1. **`docs/ERROR_TAXONOMY.md`**: Entity-based error categorization guide
2. **`docs/CURRICULUM_LEARNING.md`**: Distribution strategy explanation
3. **`docs/BUCKET_TRACKING.md`**: Metrics and convergence detection

---

## 🎯 Rollout Plan

### Phase 1: Development (Week 1-2)
- Implement all four modules
- Unit test each component
- Integration testing

### Phase 2: Validation (Week 3)
- Run full pipeline with new features
- Compare metrics with baseline
- Tune hyperparameters

### Phase 3: Deployment (Week 4)
- Deploy to production environment
- Monitor performance
- Collect user feedback

### Rollback Strategy
- Feature flags for each module
- Can disable individual features via settings
- Maintain backward compatibility with existing workflows

---

This comprehensive plan provides detailed implementation steps, testing strategies, and success metrics for all four priority modules. Each module is designed to integrate seamlessly with the existing codebase while providing clear value independently.