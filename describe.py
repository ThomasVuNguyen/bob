import os
import json
import boto3
import base64
from pathlib import Path

# Configuration variables - edit these as needed
BEDROCK_MODEL_ID = "us.meta.llama3-3-70b-instruct-v1:0"
# "us.anthropic.claude-sonnet-4-20250514-v1:0"
DEFAULT_IMAGE_PATH = "stick.jpg"
MAX_TOKENS = 1000
DESCRIPTION_PROMPT = "What would a normie think this object is? Just say what regular person would call this thing - no fancy words, just basic everyday language."

# Load AWS credentials from file
with open('credentials.json', 'r') as f:
    credentials = json.load(f)
    AWS_ACCESS_KEY_ID = credentials['AWS_ACCESS_KEY_ID']
    AWS_SECRET_ACCESS_KEY = credentials['AWS_SECRET_ACCESS_KEY']
    AWS_REGION = credentials['AWS_REGION']

def describe_image(image_path=None):
    """
    Generates a description of an image using AWS Bedrock.
    
    Args:
        image_path (str): Path to the image file (optional, uses DEFAULT_IMAGE_PATH if not provided)
        
    Returns:
        str: Generated description of the image
    """
    
    # Use default image path if none provided
    if image_path is None:
        image_path = DEFAULT_IMAGE_PATH
    
    # Set AWS credentials
    os.environ['AWS_ACCESS_KEY_ID'] = AWS_ACCESS_KEY_ID
    os.environ['AWS_SECRET_ACCESS_KEY'] = AWS_SECRET_ACCESS_KEY
    os.environ['AWS_DEFAULT_REGION'] = AWS_REGION
    
    # Initialize Bedrock client
    bedrock_runtime = boto3.client(
        service_name='bedrock-runtime',
        region_name=AWS_REGION
    )
    
    try:
        # Read and encode image
        with open(image_path, 'rb') as image_file:
            image_data = image_file.read()
            image_b64 = base64.b64encode(image_data).decode('utf-8')
        
        # Determine image format
        image_path_obj = Path(image_path)
        image_format = image_path_obj.suffix.lower().lstrip('.')
        if image_format == 'jpg':
            image_format = 'jpeg'
        
        # Prepare request body for Bedrock with image
        request_body = json.dumps({
            "anthropic_version": "bedrock-2023-05-31",
            "max_tokens": MAX_TOKENS,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image",
                            "source": {
                                "type": "base64",
                                "media_type": f"image/{image_format}",
                                "data": image_b64
                            }
                        },
                        {
                            "type": "text",
                            "text": DESCRIPTION_PROMPT
                        }
                    ]
                }
            ]
        })
        
        # Call Bedrock
        response = bedrock_runtime.invoke_model(
            body=request_body,
            modelId=BEDROCK_MODEL_ID,
            accept="application/json",
            contentType="application/json"
        )
        
        # Parse response
        response_body = json.loads(response.get('body').read())
        description = response_body['content'][0]['text']
        
        return description
        
    except Exception as e:
        return f"Error generating image description: {str(e)}"

if __name__ == "__main__":
    # Example usage
    import sys
    if len(sys.argv) > 1:
        image_path = sys.argv[1]
        description = describe_image(image_path)
        print("Generated Image Description:")
        print("=" * 50)
        print(description)
    else:
        print("Usage: python describe_codebase.py <image_path>")
        print("Example: python describe_codebase.py /path/to/image.jpg")