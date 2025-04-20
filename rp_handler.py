import runpod
import time  
from vllm import LLM, SamplingParams
import pandas as pd

import os
from dotenv import load_dotenv


def load_keywords(csv_path):
    """
    Load keywords and their categories from a CSV file.
    Assumes CSV has at least 'keyword' and 'category' columns.
    Returns:
        - keywords_dict: {keyword_lowercase: category}
        - keywords_context_str: formatted string for use in prompts
    """
    df = pd.read_csv(csv_path)

    # Build the keyword → category dictionary
    keywords_dict = {
        row["keyword"].strip().lower(): row["category"].strip()
        for _, row in df.iterrows()
        if pd.notna(row["keyword"]) and pd.notna(row["category"])
    }

    return keywords_dict

# Load environment variables from .env file
load_dotenv()

hf_token = os.getenv("HUGGING_FACE_HUB_TOKEN")

if hf_token is None:
    raise ValueError("Missing HUGGING_FACE_HUB_TOKEN environment variable")

llm = LLM(model="google/gemma-3-12b-it")

sampling_params = SamplingParams(temperature=0)



def handler(event):
    """
    This function processes incoming requests to your Serverless endpoint.
    
    Args:
        event (dict): Contains the input data and request metadata
        
    Returns:
        Any: The result to be returned to the client
    """
    keywords_dict = load_keywords("keywords.csv")
    print(keywords_dict)
    
    # Extract input data
    print(f"Worker Start")
    input = event['input']
    
    prompts = input.get('prompts')  

    print(f"Received prompts: {prompts}")

    outputs = llm.generate(prompts, sampling_params)
    output_texts = []
    # Process results and add labels to input objects
    for i, output in enumerate(outputs):
        generated_text = output.outputs[0].text.strip()
        output_texts.append(generated_text)
    

    
    return {
        "output_texts": output_texts,
        "keywords_dict": keywords_dict
    }

# Start the Serverless function when the script is run
if __name__ == '__main__':
    runpod.serverless.start({'handler': handler })