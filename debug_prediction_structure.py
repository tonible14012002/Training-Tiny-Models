#!/usr/bin/env python3

# Debug script to check prediction structure

import sys
sys.path.append(".")

from app.core.schemas.analysis import MisclassifiedOpenIntent

# Test what happens if 'dis' field is missing
test_predictions = [
    {
        "label": "PAYMENT_REQUEST",
        "prob": 0.8,
        # Missing 'dis' field - this might be the issue
    },
    {
        "label": "Unknown",
        "prob": 0.0,
        "closest": "PAYMENT_REQUEST",
        "dis": 1.5
    },
    {
        "label": "PAYMENT_SEND",
        "prob": 0.9,
        "dis": 0.5
    }
]

open_intent_samples = ["test1", "test2", "test3"]

print("Testing misclassified extraction logic...")

misclassified = []
for i, pred in enumerate(test_predictions):
    print(f"\nPrediction {i}: {pred}")
    if pred["label"] != "Unknown":
        print(f"  -> Not Unknown, should be misclassified")
        try:
            item = {
                "text": open_intent_samples[i],
                "predicted_as": pred["label"],
                "confidence": pred["prob"],
                "distance": pred.get("dis", 0.0)  # Default to 0.0 if missing
            }
            print(f"  -> Created item: {item}")

            # Test schema creation
            misclassified_item = MisclassifiedOpenIntent(**item)
            misclassified.append(misclassified_item)
            print(f"  -> Schema object: {misclassified_item}")
        except Exception as e:
            print(f"  -> ERROR: {e}")
    else:
        print(f"  -> Unknown, not misclassified")

print(f"\nFinal misclassified list length: {len(misclassified)}")
for item in misclassified:
    print(f"  - {item}")