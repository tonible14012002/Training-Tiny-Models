#!/usr/bin/env python3
"""
Test the ONNX conversion API endpoint.

Run this after starting the server with: make start
Then in another terminal: python test_onnx_api.py
"""
import requests
import json
import os
from pathlib import Path

# API configuration
API_BASE_URL = "http://localhost:8000/v2/workflow"
API_KEY = os.getenv("API_KEY", "test")

def test_convert_to_onnx():
    """Test the convert-to-onnx endpoint"""

    endpoint = f"{API_BASE_URL}/convert-to-onnx"

    # Prepare request payload
    payload = {
        "model_path": ".checkpoints/payment_classification_v2/16",
        "output_name": "payment_model_v16_onnx"
    }

    headers = {
        "Content-Type": "application/json",
        "X-API-Key": API_KEY
    }

    print(f"Testing ONNX conversion endpoint: {endpoint}")
    print(f"Payload: {json.dumps(payload, indent=2)}")

    try:
        response = requests.post(
            endpoint,
            json=payload,
            headers=headers,
            timeout=120  # 2 minutes timeout for conversion
        )

        print(f"\nStatus Code: {response.status_code}")
        print(f"Headers: {dict(response.headers)}")

        if response.status_code == 200:
            # Check if response is a file download
            content_type = response.headers.get('content-type', '')

            if 'application/zip' in content_type:
                # Save the downloaded file
                output_file = Path("test_downloaded_model.zip")
                output_file.write_bytes(response.content)

                print(f"\n✓ Success! Model downloaded to: {output_file}")
                print(f"✓ File size: {output_file.stat().st_size / 1024 / 1024:.2f} MB")

                # Verify it's a valid zip
                import zipfile
                if zipfile.is_zipfile(output_file):
                    with zipfile.ZipFile(output_file, 'r') as zip_ref:
                        print(f"\n✓ Valid zip file with {len(zip_ref.namelist())} files:")
                        for name in zip_ref.namelist():
                            print(f"  - {name}")
                else:
                    print("✗ Downloaded file is not a valid zip file")
            else:
                # JSON response
                print(f"\nResponse: {response.json()}")
        else:
            print(f"\n✗ Error: {response.status_code}")
            try:
                print(f"Response: {response.json()}")
            except:
                print(f"Response: {response.text}")

    except requests.exceptions.ConnectionError:
        print("\n✗ Error: Could not connect to the server.")
        print("  Make sure the server is running: make start")
    except Exception as e:
        print(f"\n✗ Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    test_convert_to_onnx()
