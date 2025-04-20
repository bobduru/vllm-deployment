import runpod
import time  
from vllm import LLM, SamplingParams


import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

hf_token = os.getenv("HUGGING_FACE_HUB_TOKEN")

if hf_token is None:
    raise ValueError("Missing HUGGING_FACE_HUB_TOKEN environment variable")

llm = LLM(model="google/gemma-3-1b-it")

sampling_params = SamplingParams(temperature=0.8, top_p=0.95)

def handler(event):
    """
    This function processes incoming requests to your Serverless endpoint.
    
    Args:
        event (dict): Contains the input data and request metadata
        
    Returns:
        Any: The result to be returned to the client
    """
    
    # Extract input data
    print(f"Worker Start")
    input = event['input']
    
    prompt = input.get('prompt')  

    print(f"Received prompt: {prompt}")

    output = llm.generate(prompt, sampling_params)
    

    
    return output[0].outputs[0].text

# Start the Serverless function when the script is run
if __name__ == '__main__':
    runpod.serverless.start({'handler': handler })