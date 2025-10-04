#!/usr/bin/env python3

# Debug script to check why misclassified open intents array is empty

import sys
sys.path.append(".")

from app.core.schemas.analysis import MisclassifiedOpenIntent

# Test data that should match what _analyze_open_intents returns
test_misclassified_data = [
    {
        "text": "Test message",
        "predicted_as": "PAYMENT_REQUEST",
        "confidence": 0.8,
        "distance": 1.5
    }
]

print("Testing MisclassifiedOpenIntent creation...")

try:
    # This is what happens in model_analyzer.py line 106
    misclassified = [
        MisclassifiedOpenIntent(**item) for item in test_misclassified_data
    ]
    print(f"SUCCESS: Created {len(misclassified)} misclassified items")
    print(f"First item: {misclassified[0]}")
except Exception as e:
    print(f"ERROR: {e}")
    print(f"Error type: {type(e)}")

print("\nTesting with empty list...")
try:
    misclassified_empty = [
        MisclassifiedOpenIntent(**item) for item in []
    ]
    print(f"SUCCESS: Empty list created {len(misclassified_empty)} items")
except Exception as e:
    print(f"ERROR with empty list: {e}")