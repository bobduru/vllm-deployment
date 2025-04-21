import runpod
import time  
from vllm import LLM, SamplingParams
import pandas as pd
import re
from collections import defaultdict


import os
from dotenv import load_dotenv
from runpod import RunPodLogger


log = RunPodLogger()

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

def find_matching_keywords(text, keywords_dict):
        """
        Check if any keywords (including multi-word) are present in the given text and return them with their categories.
        
        :param text: The input string to search within.
        :return: A list of tuples (matched_keyword, category).
        """
        text = text.lower()  # Normalize case
        matches = [(kw, keywords_dict[kw]) for kw in keywords_dict if re.search(r'\b' + re.escape(kw) + r'\b', text)]  
        return matches  # Return list of (keyword, category) tuples

def create_prompt(text, keywords_dict, keywords_strategy="find_manually", n_context_keywords_to_keep=None):
        prompt_prefix = (
            "You are a safety classifier that carefully analyzes text for potential risks. "
            "For every input, classify it strictly into one of the following categories:\n\n"
            "- Hate Speech: Any content that promotes, incites, or supports violence, discrimination, or hostility against individuals or groups based on attributes such as race, religion, gender, sexual orientation, disability, or nationality.\n"
            "- Radicalization: Any content that encourages or supports extremist ideological beliefs, recruitment into extremist groups, or the justification of violence for ideological, religious, or political motives.\n"
            "- Extremism: Any content that advocates for extreme political views, including incitement to overthrow governments, promoting authoritarianism, suppressing rights through violence, or endorsing terrorism.\n"
            "- Pedophilia: Be the most cautious with this one — any content that implies, suggests, normalizes, or discusses sexual interest in minors, grooming, or child exploitation. But also remember that these sentences are by adults, so anything that sounds too cute or childish should be flagged. Anything related to pornography (websites, videos, etc) should also be flagged. Better be too cautious than miss one.\n"
            "- Normal: If the text does not contain any of the above risks.\n\n"
            
        )

        labels = ["Hate Speech", "Radicalization", "Extremism", "Pedophilia", "Normal"]

    
        if keywords_strategy == "all_in_context" and keywords_dict:
            # Rebuild category → keywords mapping from the flat dictionary
            grouped_keywords = defaultdict(list)
            for keyword, category in keywords_dict.items():
                grouped_keywords[category].append(keyword)
        
            prompt_prefix += "For some more context here are keywords commonly used for the categories, you should flag a sentence with one of these keywords:\n"
            for label in labels:
                keywords = grouped_keywords.get(label, [])
                if keywords:
                    # Optionally limit the number of keywords
                    selected_keywords = (
                        keywords[:n_context_keywords_to_keep]
                        if n_context_keywords_to_keep is not None
                        else keywords
                    )
                    prompt_prefix += f"{label}: {', '.join(selected_keywords)}\n"
            prompt_prefix += "\n"
            

        # prompt_prefix += "Most of the time, the text will be web searches, so some of them can have weird characters or even be blank. "\
        #     "If you can't make sense of the text, and it doesn't look suspicious, just output 'Normal'.\n"
        prompt_prefix += "Instructions: Classify the following text in between <classify> tags and output only one of the labels : Hate Speech, Radicalization, Extremism, Pedophilia or Normal in between the <label> tags\n"
        prompt_prefix += "If you can't make sense of the text, and it doesn't look suspicious, just output <label>Normal</label>.\n"
        
        if keywords_strategy=="find_manually" and keywords_dict:
            matched_keywords = find_matching_keywords(text, keywords_dict)
        
            # If keywords were found, dynamically modify the prompt
            if matched_keywords:
                manual_keyword_context = "\n\nThese words were found in the message and are associated with risk categories, so be extra careful on this sentence:\n"
                manual_keyword_context += "\n".join([f"- {kw}: {cat}" for kw, cat in matched_keywords])
                
                prompt_prefix += manual_keyword_context

        prompt = prompt_prefix + "Text to classify: <classify>" + text + "</classify>\nLabel: <label>"
        return prompt


def classify_list(model, sampling_params, input_list, keywords_dict, keywords_strategy="find_manually", labels=["Hate Speech", "Radicalization", "Extremism", "Pedophilia", "Normal"]):
        """Classify a large list by splitting into batches and calling classify_batch."""
        start_time = time.time()
    
        prompts = [create_prompt(entry["value"], keywords_dict, keywords_strategy=keywords_strategy) for entry in input_list]

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

    llm = LLM(model="google/gemma-3-12b-it")


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
        try:
            log.info("Loading keywords")
            keywords_dict = load_keywords("keywords.csv")
        except Exception as e:
            log.error(f"Failed to load keywords: {str(e)}")
            return {"error": f"Failed to load keywords: {str(e)}"}

        labels = ["Hate Speech", "Radicalization", "Extremism", "Pedophilia", "Normal"]

       
        # Load keywords and process request

        parameters = input_data.get('parameters', {})
        generation_tokens = parameters.get('generation_tokens', "label_restricted")  # Options: "restricted" or "free"
        keywords_strategy = parameters.get('keywords_strategy', "all_in_context")
        return_prompt_template = parameters.get('return_prompt_template', False)
        


        sampling_params = None

        if generation_tokens == "label_restricted":
            log.info("Label restricted generation")
            #tokenize the labels
            valid_token_ids = get_labels_tokens(model, labels, only_first_token=True)
            sampling_params = SamplingParams(temperature=0, max_tokens=1, allowed_token_ids=valid_token_ids)
        else:
            log.info("Free generation")
            sampling_params = SamplingParams(temperature=0, max_tokens=1)


        # Validate keywords_strategy parameter
        valid_strategies = ["none", "all_in_context", "find_manually"]
        if keywords_strategy not in valid_strategies:
            log.error(f"Invalid keywords_strategy: {keywords_strategy}")
            return {
                "error": f"Invalid keywords_strategy. Must be one of: {', '.join(valid_strategies)}",
                "status": "error"
            }

        log.info(f"Received list to classify, length: {len(list_to_classify)}, first item: {list_to_classify[0]}")
        log.info(f"Received keywords strategy: {keywords_strategy}")

        log.info(f"Classifying list")
        res = classify_list(model, sampling_params, list_to_classify, keywords_dict, 
                          keywords_strategy=keywords_strategy, labels=labels)
       

        if return_prompt_template:
            prompt_template = create_prompt("{text to classify}", keywords_dict, keywords_strategy=keywords_strategy)
            log.info(f"Prompt template: {prompt_template}")
            res["prompt_template"] = prompt_template
        
        return res

    except Exception as e:
        print(e)
        log.error(f"Unexpected error in handler: {str(e)}")
        return {
            "error": f"An unexpected error occurred: {str(e)}",
            "status": "error"
        }

# Start the Serverless function when the script is run
if __name__ == '__main__':
    runpod.serverless.start({'handler': handler })