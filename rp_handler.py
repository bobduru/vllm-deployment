import runpod
import time  
from vllm import LLM, SamplingParams
import pandas as pd
import re
from collections import defaultdict
import traceback

import os
from dotenv import load_dotenv
from runpod import RunPodLogger


log = RunPodLogger()

def classify_list(model, sampling_params, input_list, labels, prompt):
        """Classify a large list by splitting into batches and calling classify_batch."""
        start_time = time.time()
    
        prompts = [prompt.format(text=entry["value"]) for entry in input_list]

        outputs = model.generate(prompts, sampling_params)
    
        end_time = time.time()
        classification_time = end_time - start_time
        print(f"Full classification took {classification_time:.2f} seconds")

        # Process results and add labels to input objects
        for i, output in enumerate(outputs):
            generated_text = output.outputs[0].text.strip()
            # Match against known labels
            matched_label = None
            for label in labels:
                if label.lower().startswith(generated_text.lower()):
                    matched_label = label
                    break
            
            # Add the matched label to the input object
            input_list[i]["label"] = matched_label or "Unknown"
            # input_list[i]["generated_text"] = generated_text
            # input_list[i]["output"] = output


        return {
            "results": input_list,
            "classification_time_seconds": classification_time
        }


def load_model():
    # Load environment variables from .env file
    load_dotenv()

    hf_token = os.getenv("HUGGING_FACE_HUB_TOKEN")

    if hf_token is None:
        raise ValueError("Missing HUGGING_FACE_HUB_TOKEN environment variable")

    # llm = LLM(model="google/gemma-3-12b-it")
    
    llm = LLM(
        model="ISTA-DASLab/gemma-3-12b-it-GPTQ-4b-128g",
        max_model_len=8046
    )

    sampling_params = SamplingParams(temperature=0, max_tokens=10)
    outputs = llm.generate("Hello world", sampling_params)
    print("TESTING")
    print(outputs)

    return llm


def get_labels_tokens(model, labels, only_first_token=False):
    # print("get_labels_first_token")
    valid_token_ids = []
    
    sampling_params = SamplingParams(temperature=0, max_tokens=1)
    for label in labels:
        outputs = model.generate(label, sampling_params)
        if only_first_token:
            #Index 1 because the first token is the beginning of the sentence token
            valid_token_ids.append(outputs[0].prompt_token_ids[1])
        else:
            valid_token_ids.extend(outputs[0].prompt_token_ids)

    print("valid_token_ids")
    print(valid_token_ids)
    return valid_token_ids


import requests

def get_data(url):
    """
    Make a GET request to the specified URL with authentication.
    
    Args:
        url (str): The URL to make the request to
        token (str): The authentication token
        
    Returns:
        dict: The JSON response data
        
    Raises:
        Exception: If the request fails
    """
    response = requests.get(url,)
    if response.status_code == 200:
        data = response.json()
        return data["prompt"], data["labels"]
    else:
        raise Exception(f"Request failed: {response.status_code}")



def handler(event):
    """
    This function processes incoming requests to your Serverless endpoint.
    
    Args:
        event (dict): Contains the input data and request metadata
        
    Returns:
        dict: Either contains the classification results or error information
    """
    try:

        
        # Validate input structure
        if not isinstance(event, dict) or 'input' not in event:
            return {"error": "Invalid event structure. Expected 'input' field."}

        input_data = event['input']
        if not isinstance(input_data, dict):
            return {"error": "Invalid input format. Expected dictionary."}

        if 'list_to_classify' not in input_data:
            return {"error": "Missing required field 'list_to_classify'."}

        list_to_classify = input_data['list_to_classify']
        if not isinstance(list_to_classify, list) or not list_to_classify:
            return {"error": "Invalid or empty list_to_classify. Expected non-empty list."}


        prompt, labels = get_data("http://209.97.142.66/prompt")


        # Initialize model if needed
        global model
        if "model" not in globals():
            log.info("Loading model")
            try:
                model = load_model()
            except ValueError as e:
                log.error(f"Failed to load model: {str(e)}")
                return {"error": f"Model initialization failed: {str(e)}"}

       

        # Load keywords and process request

        parameters = input_data.get('parameters', {})
        generation_tokens = parameters.get('generation_tokens', "label_restricted")  # Options: "restricted" or "free"
        return_prompt_template = parameters.get('return_prompt_template', False)
        


        sampling_params = None

        if generation_tokens == "label_restricted":
            log.info("Label restricted generation")
            log.info(labels)
            #tokenize the labels
            valid_token_ids = get_labels_tokens(model, labels, only_first_token=True)
            sampling_params = SamplingParams(temperature=0, max_tokens=1, allowed_token_ids=valid_token_ids)
        else:
            log.info("Free generation")
            sampling_params = SamplingParams(temperature=0, max_tokens=5)


        log.info(f"Received list to classify, length: {len(list_to_classify)}, first item: {list_to_classify[0]}")
        # log.info(f"Received keywords strategy: {keywords_strategy}")

        log.info(f"Classifying list")
        res = classify_list(model, sampling_params, list_to_classify, labels, prompt)
       

        if return_prompt_template:
            res["prompt_template"] = prompt
        
        return res

    except Exception as e:
        
        error_trace = traceback.format_exc()
        log.error(f"Unexpected error: {e}\n{error_trace}")
        return {
            "error": f"An unexpected error occurred: {str(e)}",
            "status": "error"
        }

# Start the Serverless function when the script is run
if __name__ == '__main__':
    runpod.serverless.start({'handler': handler })