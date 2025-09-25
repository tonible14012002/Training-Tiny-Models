# Auto-Train Model Workflow Documentation

## Table of Contents
1. [System Overview](#system-overview)
2. [Core Workflow](#core-workflow)
3. [Seven Key Upgrades](#seven-key-upgrades)
4. [Error Buckets](#error-buckets)
5. [Complete Prompt Library](#complete-prompt-library)
6. [System Architecture](#system-architecture)
7. [Implementation Guidelines](#implementation-guidelines)
8. [Best Practices](#best-practices)

---

## System Overview

### Purpose
An iterative training pipeline for payment intention detection that uses persona-driven synthetic data generation to continuously improve a tiny model's performance through targeted error correction.

### Task Definition
Classify chat messages into 4 categories:
- **PAYMENT_REQUEST**: User asking someone to send them money
- **PAYMENT_SEND**: User intends to send/pay money to someone
- **PAYMENT_COMMAND**: User instructing a system to make a payment
- **NO_PAYMENT**: No payment intention present

### Key Innovation
Uses 1 billion diverse personas to generate targeted synthetic training data that fixes specific model weaknesses while maintaining diversity.

---

## Core Workflow

### Initialization Phase
1. **Create Frozen Test Sets**: Human-curated dev/test sets that are never touched by the training loop
2. **Seed Dataset Generation**: Use multi-persona prompting to create initial balanced training data
3. **Initial Model Training**: Train tiny model on seed dataset → baseline checkpoint

### Iterative Loop (Repeat Until Convergence)
1. **Evaluation**: Generate test examples with personas, run model predictions
2. **Error Analysis**: Categorize failures into error buckets, identify low-confidence cases
3. **Targeted Generation**: Create new examples using 40% error fixes + 40% low-confidence + 20% random
4. **Two-Pass Labeling**: LLM A labels, LLM B verifies for quality control
5. **Quality Control**: Deduplication, diversity enforcement, schema validation
6. **Budget Management**: Enforce per-bucket limits, check convergence criteria
7. **Optional Validation**: Real data comparison, human review sampling

### Stopping Criteria
- Dev set F1 improvement < 0.5% for 3+ rounds
- All error buckets < 5% error rate
- Budget constraints reached
- Payment detection F1 > 85% on both classes

---

## Seven Key Upgrades

### 1. Real, Frozen Test Set
- **What**: Human-curated dev/test sets that LLM never sees
- **Why**: Prevents data leakage and overfitting to synthetic patterns
- **Implementation**: Hold out 10-20% of initial human data, never regenerate

### 2. Error Taxonomy → Targeted Data Asks
- **What**: Classify failures into specific error buckets (currency, slang, context, etc.)
- **Why**: Systematic improvement instead of random data generation
- **Implementation**: Analyze wrong predictions, group by failure patterns, generate targeted fixes

### 3. Diversity + Dedup Filters
- **What**: N-gram uniqueness + semantic deduplication using cosine similarity/LSH
- **Why**: Prevents "same pattern with new names" problem
- **Implementation**: Pre-generation penalties + post-generation semantic filtering

### 4. Two-Pass Labeling
- **What**: LLM A generates labels, LLM B verifies for consistency
- **Why**: Catches label drift, inconsistencies, schema violations
- **Implementation**: Auto-reject low-agreement items, repair conflicting labels

### 5. Confidence & Curriculum
- **What**: Mine low-confidence predictions, mix 40% error + 40% low-confidence + 20% random
- **Why**: Addresses both wrong predictions AND model uncertainty
- **Implementation**: Extract entropy/margin scores, generate clearer boundary examples

### 6. Budget-Aware Scheduler
- **What**: Cap examples per bucket, reweight critical intents, early stopping
- **Why**: Prevents resource waste and infinite loops
- **Implementation**: Track per-bucket quotas, monitor dev-set gains, enforce limits

### 7. Periodically Sanity-Check Against Real Data
- **What**: Sample 1-5% production queries for human review
- **Why**: Detect synthetic data drift from real-world usage
- **Implementation**: Regular sampling, drift detection, feedback into next iteration

---

## Error Buckets

### Definition
Categorized groups of model failures organized by underlying cause or pattern.

### Payment Intent Error Buckets

#### Currency Format Errors
- **Pattern**: Model misses non-standard currency representations
- **Examples**: "50 bucks", "0.01 BTC", "twenty dollars", "some money"
- **Fix Strategy**: Generate examples with slang currency, crypto, written numbers

#### Payment Method Confusion
- **Pattern**: Model doesn't recognize payment platforms or confuses direction
- **Examples**: "Venmo me" → NO_PAYMENT, "PayPal me for dinner" → PAYMENT_SEND (wrong direction)
- **Fix Strategy**: Generate platform-specific verbs, directional clarity

#### Context Ambiguity Errors
- **Pattern**: Model struggles with conditional, hypothetical, or temporal contexts
- **Examples**: "I'll pay you back tomorrow", "Did you pay the bill?", "Just kidding about money"
- **Fix Strategy**: Generate conditional statements, questions vs commands, joke contexts

#### Slang and Informal Language
- **Pattern**: Model doesn't understand informal payment terminology
- **Examples**: "Spot me twenty", "Front me cash", "Hit me up with funds"
- **Fix Strategy**: Generate informal payment slang variations

#### Multi-Intent Messages
- **Pattern**: Model confused by multiple intents in one message
- **Examples**: "Thanks for dinner, can you Venmo me for the Uber?"
- **Fix Strategy**: Generate complex sentences with clear primary intent

### Error Bucket Workflow
1. **Identify**: Group similar failures (minimum 5+ examples per bucket)
2. **Analyze**: Understand root cause and failure pattern
3. **Target**: Generate 15-50 specific examples to address the pattern
4. **Measure**: Track error rate reduction per bucket
5. **Iterate**: Continue until bucket error rate < 5%

---

## Prompt Requirements & Expectations

### 1. Seed Dataset Generation
**Purpose**: Bootstrap initial training data with balanced coverage
**Input Requirements**:
- 3-5 diverse personas per call
- Target of 5 examples per persona (25 total)
- Balanced distribution across 4 payment intention categories

**Expected Output**:
- Structured format: Persona | Message | Label | Reasoning | Payment_Method | Context | Formality
- 70% focused examples, 20% contextual, 10% conversational/noisy
- Cross-persona diversity in payment methods, language styles, contexts
- No repeated patterns or vocabulary across personas

**Quality Standards**:
- Realistic chat messages authentic to each persona
- Clear label justification with brief reasoning
- Mix of obvious and edge cases for robustness
- Cultural and linguistic diversity included

### 2. Evaluation Dataset Generation
**Purpose**: Create challenging test cases to identify model weaknesses
**Input Requirements**:
- Single persona focus
- Target of 20 challenging examples
- Focus on edge cases and boundary conditions

**Expected Output**:
- Format: [Message] | [Label] | [Difficulty] | [Focus_Area]
- Examples likely to cause errors (ambiguous contexts, complex scenarios)
- Mix of difficulty levels (easy/medium/hard)
- Test cases for context understanding and nuance

### 3. Error Analysis & Taxonomy
**Purpose**: Categorize model failures into actionable error buckets
**Input Requirements**:
- Set of misclassified examples from model evaluation
- Minimum 5+ examples per potential bucket

**Expected Output**:
- Grouped error buckets with specific names and descriptions
- Failure pattern analysis with root cause identification
- Occurrence counts and example failures (3-5 per bucket)
- Prioritized fix strategies ranked by impact and feasibility

**Error Bucket Categories**:
- Currency format, Payment method confusion, Context ambiguity
- Slang/informal language, Multi-intent messages, Temporal references

### 4. Error Bucket Targeted Generation
**Purpose**: Create specific examples to fix identified error patterns
**Input Requirements**:
- Specific error bucket name and description
- 2-3 relevant personas
- Target of 15-50 examples per bucket

**Expected Output**:
- Examples addressing the specific failure pattern
- Mix: 40% previously misclassified patterns + 40% similar correct examples + 20% edge cases
- Diversity constraints: no bigram reuse, 20% typos/format variants
- Semantic diversity from existing examples

### 5. Two-Pass Labeling System

#### Labeller Prompt (LLM A)
**Purpose**: Initial labeling of generated examples
**Expected Output**:
- Format: [Message] | [Label] | [Confidence: 1-5] | [Key_indicators]
- Consistent labeling criteria across all examples
- Confidence scoring for uncertainty detection

#### Verifier Prompt (LLM B)
**Purpose**: Quality control and label verification
**Expected Output**:
- AGREE/DISAGREE with original label and detailed reasoning
- Confidence score (1-5) for verification quality
- Flag low-agreement items for review
- Schema violation detection

### 6. Deduplication & Diversity Review
**Purpose**: Ensure example diversity and remove duplicates
**Input Requirements**:
- Newly generated examples for filtering
- Similarity thresholds (typically 0.8-0.9)

**Expected Output**:
- Filtered examples with duplicates removed
- Diversity assessment across payment methods, amounts, contexts
- Balance verification across intent categories
- Quality score for overall dataset diversity

### 7. Convergence & Progress Evaluation
**Purpose**: Determine if training should continue or stop
**Input Requirements**:
- Current metrics (Intent F1, Entity F1, per-bucket error rates)
- Dev set improvement history
- Budget and resource constraints

**Expected Output**:
- CONTINUE/STOP recommendation with detailed reasoning
- Performance trend analysis and plateau detection
- Resource utilization assessment
- Next iteration focus areas if continuing

### 8. Real Data Validation
**Purpose**: Detect drift between synthetic and production data
**Input Requirements**:
- Sample of synthetic examples
- Sample of production queries (1-5%)

**Expected Output**:
- Drift analysis across payment methods, language, contexts
- Identification of overrepresented synthetic patterns
- Missing real-world scenarios flagged
- Recommendations for next iteration focus