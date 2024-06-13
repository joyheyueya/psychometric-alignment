import copy
import numpy as np
import tiktoken
import torch
import os
import torch
import gc
from datasets import load_dataset
from peft import PeftModel
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
import pandas as pd
from datetime import datetime
import re
import numpy as np
from multiprocessing import Process
import pycountry

def calculate_age(x):
    birth_date = x[0]
    current_date = x[1]
    age = current_date.year - birth_date.year
    if (current_date.month, current_date.day) < (birth_date.month, birth_date.day):
        age -= 1
    return age

def try_convert_to_float(x, replace_all_char=True):
    '''
    Returns 0 if x cannot be converted to a float.
    '''
    try:
        if 'answer is' in x:
            x = x[x.index('answer is') + 9:]
        if replace_all_char:
            for c in copy.deepcopy(x):
                if not c.isdigit() and c != '.' and c != '-' and c != '/':
                    x = x.replace(c,'')
        else:
            x = x.replace(',','')
        return float(x)
    except Exception as e:
        print('cannot convert to float:', x)
        return 0.0
    
def check_correct(x):
    final_answer = x[0]
    ground_truth = x[1]
    try:
        if np.abs(float(final_answer) - float(ground_truth)) <= 1e-3:
            return 1
        else:
            return 0
    except Exception as e:
        return 0
    
def check_correct_letter(x):
    final_answer = str(x[0]).lower()
    ground_truth = str(x[1]).lower()

    return final_answer == ground_truth

def num_tokens_from_messages(messages, model="gpt-3.5-turbo-0613"):
    """Return the number of tokens used by a list of messages."""
    try:
        encoding = tiktoken.encoding_for_model(model)
    except KeyError:
        print("Warning: model not found. Using cl100k_base encoding.")
        encoding = tiktoken.get_encoding("cl100k_base")
    if model in {
        "gpt-3.5-turbo-0613",
        "gpt-3.5-turbo-16k-0613",
        "gpt-4-0314",
        "gpt-4-32k-0314",
        "gpt-4-0613",
        "gpt-4-32k-0613",
        }:
        tokens_per_message = 3
        tokens_per_name = 1
    elif model == "gpt-3.5-turbo-0301":
        tokens_per_message = 4  # every message follows <|start|>{role/name}\n{content}<|end|>\n
        tokens_per_name = -1  # if there's a name, the role is omitted
    elif "gpt-3.5-turbo" in model:
        print("Warning: gpt-3.5-turbo may update over time. Returning num tokens assuming gpt-3.5-turbo-0613.")
        return num_tokens_from_messages(messages, model="gpt-3.5-turbo-0613")
    elif "gpt-4" in model:
        print("Warning: gpt-4 may update over time. Returning num tokens assuming gpt-4-0613.")
        return num_tokens_from_messages(messages, model="gpt-4-0613")
    else:
        raise NotImplementedError(
            f"""num_tokens_from_messages() is not implemented for model {model}. See https://github.com/openai/openai-python/blob/main/chatml.md for information on how messages are converted to tokens."""
        )
    num_tokens = 0
    for message in messages:
        num_tokens += tokens_per_message
        for key, value in message.items():
            num_tokens += len(encoding.encode(value))
            if key == "name":
                num_tokens += tokens_per_name
    num_tokens += 3  # every reply is primed with <|start|>assistant<|message|>
    return num_tokens

def number_to_letter(number):
    mapping = {1: 'A', 2: 'B', 3: 'C', 4: 'D'}
    return mapping.get(number, None)

def extract_letter_within_box(x):
    matches = re.findall(r'\\boxed{([A-Za-z])}', x)
    if matches:
        return matches[-1]
    else:
        matches = re.findall(r'\\boxed{\\text{([A-Za-z])}}', x)
        if matches:
            return matches[-1]
        else:
            return ''

def extract_word_after_phrase(string, phrase):
    # Split the string by the phrase and take the part after it
    parts = string.split(phrase)
    if len(parts) > 1:
        # Split the remaining string by space and take the first word
        first_word = parts[1].strip().split()
        if len(first_word) > 0:
            first_word = first_word[0]
        else:
            first_word = ''
    else:
        first_word = ''
            
    return first_word

def fix_parsing_error(x):
    model_choice = extract_letter_within_box(x)
    if len(model_choice) == 1 and model_choice.isalpha():
        return model_choice
    elif len(model_choice) > 1:
        model_response = model_choice
    else:
        model_response = x
    
    matches = re.findall(r'\[\[([\s\S]*?)\]\]', model_response)
    if len(matches) >= 1:
        model_choice = matches[0]
    else:
        model_choice = np.nan
    if isinstance(model_choice, str):
        return model_choice
    else:
        matches = re.findall(r'\[([\s\S]*?)\]', model_response)
        if len(matches) >= 1:
            for m in matches:
                if len(m) == 1:
                    return m
            return matches[0]
        else:
            match = extract_word_after_phrase(model_response, "answer is:")
            for c in range(len(match)):
                if match[c].isalpha():
                    if c < len(match) - 1 and (not match[c+1].isalpha()):
                        return match[c]  
                    elif c == len(match) - 1:
                        return match[c]
            return np.nan    

def fix_parsing_error_given_response_and_choice(x):
    model_response = x[0]
    model_choice = x[1]
    if isinstance(model_response, str):
        if isinstance(model_choice, str):
            return model_choice
        else:
            matches = re.findall(r'\[([\s\S]*?)\]', model_response)
    #         print(matches)
            if len(matches) >= 1:
                return matches[0].upper()
            else:
                return np.nan    
    else:
        return model_choice

def extract_double_brackets(x):
    model_response = x   
    if isinstance(model_response, str):
        matches = re.findall(r'\[\[([\s\S]*?)\]\]', model_response)
        if len(matches) >= 1:
            model_choice = matches[0].upper()
        else:
            model_choice = ''
        return model_choice
    else:
        return ''

def convert_from_letter_to_num(x):
    letter = ''
    if not isinstance(x, str):
        return x
    for c in x:
        if c.isalpha():
            letter = c.lower()
            break
    if letter == 'a':
        return 1
    elif letter == 'b':
        return 2
    elif letter == 'c':
        return 3
    elif letter == 'd':
        return 4
    else:
        return np.nan

def extract_all_final_answers(text_list):
    answers = []
    for text in text_list:
        match = re.search(r"The answer is: (?:\\text{)?\(?([A-Z0-9]+)\)?\}?", text)
        if match:
            answers.append(match.group(1))
        else:
            answers.append(None)
    return answers

def clear_resources():
    torch.cuda.empty_cache()
    gc.collect()
    
def get_first_char(x):
    x = x.strip()
    if len(x) > 0:
        return x[0]
    else:
        return x
    
def map_binary_to_correctness(x):
    if x == 1:
        return "Correct"
    else:
        return "Incorrect"

def get_first_word(x):
    x = x.split('\n')
    if len(x) > 0:
        return x[0]
    else:
        return x
    
def create_folder(RESULTS_FOLDER):
    if not os.path.exists(RESULTS_FOLDER):
        os.makedirs(RESULTS_FOLDER)
        print("Folder created:", RESULTS_FOLDER)
    else:
        print("Folder already exists:", RESULTS_FOLDER)  
        
def clean_rationale(x):
    if len(x) > 0:
        if 'True answer:' in x:
            x = x.split('True answer:')[0]
            while x[-1:] == '\n':
                x = x[:-1]
            return x
        elif ']]' in x:
            return x.split(']]')[0] + ']]'
        else:
            return x
    else:
        return ''
    
def map_country_codes(code_group):
    names = []
    # Split codes if there are multiple country codes separated by '|'
    codes = code_group.split('|')
    for code in codes:
        try:
            country = pycountry.countries.get(alpha_2=code)
            if country:
                names.append(country.name)
            else:
                names.append("undefined")
        except Exception as e:
            names.append("undefined")

    # Join multiple country names with '|'
    full_name = '|'.join(names)

    return full_name