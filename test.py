import argparse
import requests
import base64
import json
import os

BASE_URL = "http://localhost:8000"

def test_text_endpoint(input_text: str, model_alias: str, prompt_name: str = None, dimensions: int = None):
    """Tests the /text endpoint with a given string and model alias."""
    print(f"Testing /text endpoint with model '{model_alias}'...")
    payload = {
        "text": input_text,
        "model": model_alias
    }
    if prompt_name:
        payload["prompt_name"] = prompt_name
    if dimensions:
        payload["dimensions"] = dimensions
    response = requests.post(f"{BASE_URL}/text", json=payload)
    handle_response(response)

def test_image_endpoint(image_path: str, model_alias: str, dimensions: int = None):
    """Tests the /image endpoint with a local image file and model alias."""
    print(f"Testing /image endpoint with model '{model_alias}'...")
    if not os.path.exists(image_path):
        print(f"Error: Image file not found at '{image_path}'")
        return

    try:
        with open(image_path, "rb") as image_file:
            encoded_string = base64.b64encode(image_file.read()).decode('utf-8')
    except IOError:
        print(f"Error: Could not read image file at '{image_path}'")
        return

    payload = {
        "image": encoded_string,
        "model": model_alias
    }
    if dimensions:
        payload["dimensions"] = dimensions
    response = requests.post(f"{BASE_URL}/image", json=payload)
    handle_response(response)

def test_multi_endpoint(input_text: str, image_path: str, model_alias: str, prompt_name: str = None, dimensions: int = None):
    """Tests the /multi endpoint with optional text and image."""
    print(f"Testing /multi endpoint with model '{model_alias}'...")
    payload = {
        "model": model_alias
    }
    if input_text:
        payload["text"] = input_text
    if image_path:
        if not os.path.exists(image_path):
            print(f"Error: Image file not found at '{image_path}'")
            return
        try:
            with open(image_path, "rb") as image_file:
                payload["image"] = base64.b64encode(image_file.read()).decode('utf-8')
        except IOError:
            print(f"Error: Could not read image file at '{image_path}'")
            return

    if prompt_name:
        payload["prompt_name"] = prompt_name
    if dimensions:
        payload["dimensions"] = dimensions

    response = requests.post(f"{BASE_URL}/multi", json=payload)
    handle_response(response)

def test_caption_endpoint(image_path: str, max_length: int):
    """Tests the /caption endpoint with a local image file."""
    print("Testing /caption endpoint...")
    if not os.path.exists(image_path):
        print(f"Error: Image file not found at '{image_path}'")
        return

    try:
        with open(image_path, "rb") as image_file:
            encoded_string = base64.b64encode(image_file.read()).decode('utf-8')
    except IOError:
        print(f"Error: Could not read image file at '{image_path}'")
        return

    # The caption endpoint uses a hardcoded model, so we don't pass an alias.
    payload = {
        "image": encoded_string,
        "max_length": max_length
    }
    try:
        response = requests.post(f"{BASE_URL}/caption", json=payload)
        response.raise_for_status()
        caption = response.json()
        print(f"Caption (max_length={max_length}): {caption}")
    except requests.exceptions.RequestException as e:
        print(f"Error testing caption endpoint: {e}")

def handle_response(response):
    """Prints the response from the API or an error message."""
    print("-" * 50)
    print(f"Status Code: {response.status_code}")
    try:
        if response.status_code == 200:
            data = response.json()
            if isinstance(data, list) and all(isinstance(x, (int, float)) for x in data):
                vector = data
                count = len(vector)
                first_10 = vector[:10]
                last_value = vector[-1]

                print("Response Body:")
                print(f"Vector Count: {count}")
                print(f"First 10 values: {first_10}")
                print(f"Last value: {last_value}")
            else:
                print("Response Body:")
                print(json.dumps(data, indent=2))
        else:
            print("Error Response:")
            print(response.text)
    except json.JSONDecodeError:
        print("Raw Response:")
        print(response.text)
    print("-" * 50)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test FastAPI endpoints for text and image processing.")

    parser.add_argument("endpoint", choices=["text", "image", "caption", "multi"], help="The endpoint to test.")
    parser.add_argument("input", nargs="?", default=None, help="Input text or image file path.")
    parser.add_argument("--image", help="Optional image file path for multimodal request.")
    parser.add_argument("--model", help="Optional: Model alias to use.", default='clip')
    parser.add_argument("--prompt_name", help="Optional: prompt_name to pass (e.g., SearchQuery).")
    parser.add_argument("--dimensions", type=int, help="Optional: truncate dimension (e.g., 512, 256, 128).")

    # Add an optional argument for max_length with a default value
    parser.add_argument("--max_length", type=int, default=50, help="Optional: Specify the max_length for the captioning model.")

    args = parser.parse_args()

    # Determine which endpoint to test
    if args.endpoint == "text":
        if not args.input:
            print("Error: The 'text' endpoint requires input text.")
        else:
            test_text_endpoint(args.input, args.model, prompt_name=args.prompt_name, dimensions=args.dimensions)
    elif args.endpoint == "image":
        if not args.input:
            print("Error: The 'image' endpoint requires an image input file path.")
        else:
            test_image_endpoint(args.input, args.model, dimensions=args.dimensions)
    elif args.endpoint == "multi":
        test_multi_endpoint(args.input, args.image, args.model, prompt_name=args.prompt_name, dimensions=args.dimensions)
    elif args.endpoint == "caption":
        if not args.input:
            print("Error: The 'caption' endpoint requires an image input file path.")
        else:
            test_caption_endpoint(args.input, args.max_length)

