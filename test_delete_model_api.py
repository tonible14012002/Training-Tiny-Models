#!/usr/bin/env python3
"""
Test the delete model API endpoint.

Run this after starting the server with: make start
Then in another terminal: python test_delete_model_api.py
"""
import requests
import json
import os

# API configuration
API_BASE_URL = "http://localhost:8000/v2/workflow"
API_KEY = os.getenv("API_KEY", "test")


def list_models():
    """List available trained models"""
    # This is a helper - you'll need to query the database or use another endpoint
    print("To find a model ID, check the database or use:")
    print("  sqlite3 pipeline.db 'SELECT id, model_name, model_save_path, status FROM trained_model LIMIT 5;'")
    print()


def test_delete_model(model_id: str, delete_files: bool = True):
    """Test the delete-model endpoint"""

    endpoint = f"{API_BASE_URL}/delete-model"

    payload = {
        "model_id": model_id,
        "delete_files": delete_files
    }

    headers = {
        "Content-Type": "application/json",
        "X-API-Key": API_KEY
    }

    print(f"Testing delete model endpoint: {endpoint}")
    print(f"Payload: {json.dumps(payload, indent=2)}")
    print()

    try:
        response = requests.post(
            endpoint,
            json=payload,
            headers=headers,
            timeout=30
        )

        print(f"Status Code: {response.status_code}")

        if response.status_code == 200:
            result = response.json()
            print(f"\n✓ Success!")
            print(f"Response: {json.dumps(result, indent=2)}")

            if result.get('database_deleted'):
                print(f"\n✓ Model deleted from database")
            if result.get('files_deleted'):
                print(f"✓ Model files deleted from disk")
                print(f"  Path: {result.get('model_path')}")
            elif not delete_files:
                print(f"✓ Files preserved (delete_files=False)")
            else:
                print(f"⚠ Files not deleted (may not exist)")

        else:
            print(f"\n✗ Error: {response.status_code}")
            try:
                error = response.json()
                print(f"Response: {json.dumps(error, indent=2)}")
            except:
                print(f"Response: {response.text}")

    except requests.exceptions.ConnectionError:
        print("\n✗ Error: Could not connect to the server.")
        print("  Make sure the server is running: make start")
    except Exception as e:
        print(f"\n✗ Error: {e}")
        import traceback
        traceback.print_exc()


def test_delete_without_files(model_id: str):
    """Test deletion keeping files"""
    print("=" * 80)
    print("TEST: Delete model from database only (preserve files)")
    print("=" * 80)
    test_delete_model(model_id, delete_files=False)


def test_delete_with_files(model_id: str):
    """Test deletion including files"""
    print("=" * 80)
    print("TEST: Delete model from database and disk")
    print("=" * 80)
    test_delete_model(model_id, delete_files=True)


def test_delete_nonexistent():
    """Test deleting a non-existent model"""
    print("=" * 80)
    print("TEST: Delete non-existent model")
    print("=" * 80)
    test_delete_model("nonexistent-model-id-12345", delete_files=True)


if __name__ == "__main__":
    import sys

    print("Delete Model API Test")
    print("=" * 80)
    print()

    if len(sys.argv) < 2:
        print("Usage: python test_delete_model_api.py <model_id> [delete_files]")
        print()
        print("Arguments:")
        print("  model_id      - ID of the model to delete")
        print("  delete_files  - 'true' or 'false' (default: true)")
        print()
        list_models()
        print("\nExamples:")
        print("  # Delete model and files:")
        print("  python test_delete_model_api.py abc123-model-id")
        print()
        print("  # Delete model from DB only:")
        print("  python test_delete_model_api.py abc123-model-id false")
        print()
        print("  # Test with non-existent model:")
        print("  python test_delete_model_api.py test")
        sys.exit(1)

    model_id = sys.argv[1]
    delete_files = True if len(sys.argv) < 3 else sys.argv[2].lower() == 'true'

    if model_id == "test":
        # Run test suite
        print("Running test suite...")
        print()
        test_delete_nonexistent()
        print()
    else:
        # Delete specific model
        test_delete_model(model_id, delete_files)
