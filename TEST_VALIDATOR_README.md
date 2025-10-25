# DataValidator Testing Guide

This directory contains test files for validating the `DataValidator` service, which automatically detects and corrects mislabeled samples using LLM-based validation.

## Test Files

### 1. `test_validator_simple.py` - Quick Smoke Test
A lightweight test with 6 samples (3 correct, 3 intentionally mislabeled).

**Run:**
```bash
python test_validator_simple.py
```

**Expected Results:**
- 3 corrections should be made
- All corrections should be accurate
- Output shows before/after comparison

**Sample Output:**
```
Original samples:
  [1] payment_intent       | Send me 10 BTC please

Corrected samples:
  [1] payment_intent       → payment_request      | Send me 10 BTC please
```

---

### 2. `test_validator_mislabeled.py` - Comprehensive Test
A thorough test with 21 samples covering all labeling error patterns.

**Run:**
```bash
python test_validator_mislabeled.py
```

**Test Coverage:**
- ✓ Baseline correct labels (6 samples)
- ✗ payment_request mislabeled as payment_intent (3 samples)
- ✗ payment_intent mislabeled as payment_request (3 samples)
- ✗ open_intent mislabeled as payment_intent (3 samples)
- ✗ open_intent mislabeled as payment_request (2 samples)
- ✗ payment_intent mislabeled as open_intent (2 samples)
- ✗ payment_request mislabeled as open_intent (2 samples)

**Expected Results:**
- 15 total corrections expected
- Metrics reported: Accuracy, Precision, Recall
- Detailed per-sample analysis

**Sample Output:**
```
SUMMARY:
Expected corrections:     15
Corrections found:        15
  ✓ Correct fixes:        15
  ⚠ Incorrect fixes:      0
  ❌ False positives:     0
❌ Missed corrections:    0

Accuracy (correct/expected): 100.0%
Precision (correct/found):   100.0%
```

---

## Label Definitions (from validation prompt)

### payment_intent
The message sender declares they are about to send money or a note to make a payment to someone else.

**Examples:**
- ✓ "I'll send you $5 later."
- ✓ "Pay Alice $20 in BTC."
- ✓ "Send 40$"

**Key characteristics:**
- Future or imminent payment action
- Sender is the one paying/sending money
- NOT if payment already completed

---

### payment_request
The message sender requests, reminds, or demands money from someone else.

**Examples:**
- ✓ "Send me 10 eth"
- ✓ "You owe me $20"
- ✓ "Pay me back"

**Key characteristics:**
- Requesting or demanding money
- Sender is the recipient of payment

---

### open_intent
Any arbitrary messages not related to payment_intent or payment_request, but can include payment-related context.

**Examples:**
- ✓ "I sent $50 already." (past payment)
- ✓ "Thanks for the payment"
- ✓ "What's the weather like?"

**Key characteristics:**
- Past payments (already completed)
- General payment discussion without intent/request
- Non-payment related messages

---

## Common Mislabeling Patterns

### 1. Direction Confusion
**WRONG:** "Send me $10" labeled as `payment_intent`
**RIGHT:** `payment_request` (sender receives money)

**WRONG:** "I'll pay you $20" labeled as `payment_request`
**RIGHT:** `payment_intent` (sender pays money)

### 2. Timing Confusion
**WRONG:** "I already sent the payment" labeled as `payment_intent`
**RIGHT:** `open_intent` (payment already completed - past tense)

**WRONG:** "I'll transfer $50" labeled as `open_intent`
**RIGHT:** `payment_intent` (future payment)

### 3. Context Confusion
**WRONG:** "Thanks for paying me" labeled as `payment_request`
**RIGHT:** `open_intent` (acknowledgment, not request)

**WRONG:** "You owe me $20" labeled as `open_intent`
**RIGHT:** `payment_request` (implicit demand for payment)

---

## How the Validator Works

### Integration with DataGeneratorV2
The validator runs automatically during data generation:

```python
# Step 1-3: Generate & deduplicate samples
batch_results = await generate_samples(...)
deduplicated = await deduplicate(batch_results)

# Step 4: Validate and fix labels (NEW)
validated_samples = await validator.validate_and_fix(deduplicated)

# Step 5: Save to files
save_samples(validated_samples)
```

### Standalone Usage
```python
from app.core.services.data_validator import DataValidator

validator = DataValidator(
    llm=llm,
    prompt_mgr=prompt_mgr,
    label_config=label_config
)

# Returns corrected samples
corrected = await validator.validate_and_fix(samples)
```

---

## Validation Prompt Location

The validation prompt is located at:
```
app/core/prompts/v2/validation/validate_labels.txt
```

This prompt includes:
- Complete label definitions
- Examples for each label
- Important distinctions
- Edge cases and clarifications

---

## Running Tests in CI/CD

To integrate into automated testing:

```bash
# Quick smoke test
python test_validator_simple.py || exit 1

# Comprehensive test
python test_validator_mislabeled.py || exit 1
```

---

## Expected Performance

### Accuracy Targets
- **Simple test**: 100% (3/3 corrections)
- **Comprehensive test**: ≥95% (14+/15 corrections)

### Common Edge Cases
The validator should handle:
- Past vs future tense detection
- Direction of payment (send vs receive)
- Implicit vs explicit requests
- Payment acknowledgments vs requests
- Ambiguous phrasing

---

## Troubleshooting

### Validator making incorrect fixes
1. Check if label definitions in prompt are clear
2. Review specific edge cases in prompt examples
3. Consider adjusting LLM temperature (lower = more deterministic)

### Validator missing errors
1. Increase LLM model capability (e.g., gpt-4o instead of gpt-4o-mini)
2. Add more examples to validation prompt
3. Review specific failure patterns and update prompt

### False positives
1. Review label definitions for ambiguity
2. Add "Guidelines" section to validation prompt
3. Consider adding confidence threshold

---

## Metrics Interpretation

### Accuracy
Percentage of expected corrections that were made correctly:
```
Accuracy = Correct Fixes / Expected Corrections
```

### Precision
Percentage of validator's corrections that were actually needed:
```
Precision = Correct Fixes / Total Corrections Made
```

### Recall
Percentage of mislabeled samples that were detected:
```
Recall = Correct Fixes / (Correct Fixes + Missed Corrections)
```

---

## File Structure

```
finetune/
├── test_validator_simple.py          # Quick 6-sample test
├── test_validator_mislabeled.py      # Comprehensive 21-sample test
├── TEST_VALIDATOR_README.md          # This file
└── app/
    └── core/
        ├── services/
        │   └── data_validator/
        │       └── data_validator.py  # Validator implementation
        └── prompts/
            └── v2/
                └── validation/
                    └── validate_labels.txt  # Validation prompt
```

---

## Next Steps

1. Run simple test: `python test_validator_simple.py`
2. Run comprehensive test: `python test_validator_mislabeled.py`
3. Review results and metrics
4. Adjust validation prompt if needed
5. Integrate into production pipeline

---

## Notes

- Validation uses LLM with temperature=0.0 for deterministic results
- All test samples are intentionally crafted edge cases
- The validator is optional in DataGeneratorV2 (pass `validator=None` to disable)
- Validation adds ~5-10 seconds per batch (depends on batch size)
