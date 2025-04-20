import os
import pandas as pd
import re
from collections import defaultdict
import time
from vllm import LLM, SamplingParams
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

class FlagClassifier:
    def __init__(self, keyword_csv_path):
        hf_token = os.getenv("HUGGING_FACE_HUB_TOKEN")
        if hf_token is None:
            raise ValueError("Missing HUGGING_FACE_HUB_TOKEN environment variable")
        self.model = LLM(model="google/gemma-3-12b-it")

        self.sampling_params = SamplingParams(temperature=0)

        self.keywords_dict = self.load_keywords(keyword_csv_path) if keyword_csv_path else {}


    def load_keywords(self, csv_path):
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

    def find_matching_keywords(self, text):
        """
        Check if any keywords (including multi-word) are present in the given text and return them with their categories.
        
        :param text: The input string to search within.
        :return: A list of tuples (matched_keyword, category).
        """
        text = text.lower()  # Normalize case
        matches = [(kw, self.keywords_dict[kw]) for kw in self.keywords_dict if re.search(r'\b' + re.escape(kw) + r'\b', text)]  
        return matches  # Return list of (keyword, category) tuples


    def create_prompt(self, text, keywords_strategy="find_manually", n_context_keywords_to_keep=None):
        prompt_prefix = (
            "You are a safety classifier that carefully analyzes each input message for potential risks. "
            "For every input, classify it strictly into one of the following categories:\n\n"
            "- Hate Speech: Any content that promotes, incites, or supports violence, discrimination, or hostility against individuals or groups based on attributes such as race, religion, gender, sexual orientation, disability, or nationality.\n"
            "- Radicalization: Any content that encourages or supports extremist ideological beliefs, recruitment into extremist groups, or the justification of violence for ideological, religious, or political motives.\n"
            "- Extremism: Any content that advocates for extreme political views, including incitement to overthrow governments, promoting authoritarianism, suppressing rights through violence, or endorsing terrorism.\n"
            "- Pedophilia: Be the most cautious with this one — any content that implies, suggests, normalizes, or discusses sexual interest in minors, grooming, or child exploitation. But also remember that these sentences are by adults, so anything that sounds too cute or childish should be flagged. Better be too cautious than miss one.\n"
            "- Normal: If the text does not contain any of the above risks.\n\n"
        )
        self.labels = ["Hate Speech", "Radicalization", "Extremism", "Pedophilia", "Normal"]

    
        if keywords_strategy == "all_in_context" and self.keywords_dict:
            # Rebuild category → keywords mapping from the flat dictionary
            grouped_keywords = defaultdict(list)
            for keyword, category in self.keywords_dict.items():
                grouped_keywords[category].append(keyword)
        
            prompt_prefix += "For some more context here are keywords commonly used for the categories, you should flag a sentence with one of these keywords:\n"
            for label in self.labels:
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
            

        prompt_prefix += "Instructions: Only output one of these labels without any additional text, formatting, or explanations.\n"

        if keywords_strategy=="find_manually" and self.keywords_dict:
            matched_keywords = self.find_matching_keywords(text)
        
            # If keywords were found, dynamically modify the prompt
            if matched_keywords:
                manual_keyword_context = "\n\nThese words were found in the message and are associated with risk categories, so be extra careful on this sentence:\n"
                manual_keyword_context += "\n".join([f"- {kw}: {cat}" for kw, cat in matched_keywords])
                
                prompt_prefix += manual_keyword_context

        prompt = prompt_prefix + f"Text to classify: {text}\nLabel:"
        return prompt

    def classify_list(self, input_list, keywords_strategy="find_manually"):
        """Classify a large list by splitting into batches and calling classify_batch."""
        start_time = time.time()
    
        prompts = [self.create_prompt(entry["value"], keywords_strategy=keywords_strategy) for entry in input_list]

        outputs = self.model.generate(prompts, self.sampling_params)
    
        end_time = time.time()
        print(f"Full classification took {end_time - start_time:.2f} seconds")

        # Process results and add labels to input objects
        for i, output in enumerate(outputs):
            generated_text = output.outputs[0].text.strip()
            print(f"Generated text: {generated_text!r}")
            # Match against known labels
            matched_label = None
            for label in self.labels:
                if generated_text.lower().startswith(label.lower()):
                    matched_label = label
                    break
            
            # Add the matched label to the input object
            input_list[i]["label"] = matched_label or "Unknown"

        return input_list