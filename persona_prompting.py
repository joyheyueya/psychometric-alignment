import openai
import os
import pandas as pd
import re
from datetime import datetime
from time import sleep
import matplotlib.pyplot as plt
import seaborn as sns
import random
import statsmodels.api as sm
from statsmodels.formula.api import ols
import numpy as np
from copy import deepcopy
import math
from statsmodels.stats.multicomp import MultiComparison
import warnings
warnings.filterwarnings('ignore')
import llm as lm
import memory
from utils import *
from utils_persona import *
import argparse
import torch
import gc
from vllm import LLM, SamplingParams
from multiprocessing import Process
import torch.multiprocessing as mp
mp.set_start_method('spawn', force=True)
DATA_DIR = ''

parser = argparse.ArgumentParser()
parser.add_argument('--openai_api_key', default=None, type=str)
parser.add_argument('--openai_api_org', default=None, type=str)
parser.add_argument('--date', type=str)
parser.add_argument('--test_file_path', type=str)
parser.add_argument('--prompt_method', type=str) # '_persona_CoT_structure' or '_persona' or '_persona_CoT'
parser.add_argument('--model_id', type=str)
parser.add_argument('--temperature', type=float) # for inference
parser.add_argument('--max_tokens', type=int) # for inference
parser.add_argument('--tp_size', default=4, type=int)
args = parser.parse_args()

test_file_path = args.test_file_path
DATE = args.date
prompt_method = args.prompt_method
TEMPERATURE = args.temperature
SETTING = 't' + str(TEMPERATURE).replace('.', '') + prompt_method
MODEL_ID = args.model_id
RESULTS_FOLDER = 'prompting_results/' + DATE + '_' + MODEL_ID.split('/')[-1]  + '_results_' + test_file_path.split('/')[-2] + '/'
MAX_TOKENS = args.max_tokens
REPETITION_PENALTY = 1
create_folder(DATA_DIR + RESULTS_FOLDER)
print('SETTING', SETTING)
print('MODEL_ID', MODEL_ID)

if 'gpt' in args.model_id and args.openai_api_key is not None:
    if args.openai_api_org == None:
        llm = lm.LLM(30,5,args.openai_api_key)
    else:
        llm = lm.LLM(30,5,args.openai_api_key,args.openai_api_org)
elif 'AWQ' in MODEL_ID:
    llm = LLM(model=MODEL_ID, quantization="awq", dtype="auto", trust_remote_code=True, gpu_memory_utilization=0.9, tensor_parallel_size=args.tp_size)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
else:
    llm = LLM(model=MODEL_ID, trust_remote_code=True, gpu_memory_utilization=0.9, tensor_parallel_size=args.tp_size, dtype='bfloat16')
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)

def get_vllm_inference(messages, llm, temperature, base_model_id, tokenizer):
    sampling_params = SamplingParams(max_tokens=MAX_TOKENS, 
                      temperature=temperature,
                      repetition_penalty=REPETITION_PENALTY) 
    sampling_params.stop = [tokenizer.eos_token]
    output_text_list = []    
    if 'instruct' not in base_model_id and 'Instruct' not in base_model_id and '-rl' not in base_model_id:
        for p_index in range(0, len(prompts), 1000):
            print(datetime.now())
            outputs = llm.generate(prompts=prompts[p_index:p_index+1000], sampling_params=sampling_params)
            generated_text = [output.outputs[0].text for output in outputs]
            output_text_list += generated_text  
    else:
        messages_list = [messages]

        # Avoid adding bos_token repeatedly
        prompt_token_ids = [tokenizer.apply_chat_template(messages, add_generation_prompt=True) for messages in messages_list]
        for p_index in range(0, len(messages_list), 1000):
            outputs = llm.generate(prompt_token_ids=prompt_token_ids[p_index:p_index+1000], sampling_params=sampling_params)
            generated_text = [output.outputs[0].text for output in outputs]
            output_text_list += generated_text        
    return output_text_list

    
    
results = pd.read_csv(test_file_path)

if 'countries' in results:
    results['countries_full'] = results['countries'].apply(lambda x: map_country_codes(x))

if 'duolingo' in test_file_path:
    results['persona'] = results.apply(create_persona_duolingo_first_pov, axis=1)
elif 'wordbank' in test_file_path:
    results['persona'] = results.apply(create_persona_wordbank_first_pov, axis=1)
elif 'eedi' in test_file_path:
    results['persona'] = results.apply(create_persona_3basic, axis=1)

results = results.sort_values(['UserId', 'DateAnswered'])

model_response_list = []
model_choice_list = []
memory_list = []
memory_len_list = []
NUM_QUESTIONS_SELECTED = results['QuestionId'].nunique()

last_user_id = None
for i in range(len(results)):
    if results.iloc[i]['UserId'] != last_user_id:
        mem = memory.Memory(3000)
        last_user_id = results.iloc[i]['UserId']
        print('last_user_id', last_user_id)
    persona = results.iloc[i]['persona']
    problem = results.iloc[i]['problem']
    if 'duolingo' in test_file_path:
        if prompt_method == '_persona_CoT_structure':
            user_message = f"""{persona}Given your characteristics, would you be able to recognize and spell the following word?
Word: {problem}
If yes, explain your reasoning and put the final answer in double square brackets (i.e., [[yes]]). If it is unlikely that you would be able to recognize or spell this word, explain your reasoning and put the final answer in double square brackets (i.e., [[no]]).
"""   
        elif prompt_method == '_persona_CoT':
            user_message = f"""{persona}Given your characteristics, would you be able to recognize and spell the following word?
Word: {problem}
Explain whether you think you would be able to recognize and spell this word and put the final answer (yes/no) in double square brackets (i.e., [[yes/no]]).
"""        
        elif prompt_method == '_persona':
            user_message = f"""{persona}Given your characteristics, would you be able to recognize and spell the following word?
Word: {problem}
Put the final answer (yes/no) in double square brackets (i.e., [[yes/no]]).
"""  

    elif 'wordbank' in test_file_path:
        if prompt_method == '_persona_CoT_structure':
            user_message = f"""{persona}Given your characteristics, would you be able to produce the following word?
Word: {problem}
If yes, explain your reasoning and put the final answer in double square brackets (i.e., [[yes]]). If it is unlikely that you would be able to produce this word, explain your reasoning and put the final answer in double square brackets (i.e., [[no]]).
"""   
        elif prompt_method == '_persona_CoT':
            user_message = f"""{persona}Given your characteristics, would you be able to produce the following word?
Word: {problem}
Explain whether you think you would be able to produce this word and put the final answer (yes/no) in double square brackets (i.e., [[yes/no]]).
"""        
        elif prompt_method == '_persona':
            user_message = f"""{persona}Given your characteristics, would you be able to produce the following word?
Word: {problem}
Put the final answer (yes/no) in double square brackets (i.e., [[yes/no]]).
"""  

    elif 'eedi' in test_file_path:
        if prompt_method == '_persona_CoT_structure':
            user_message = f"""{persona}Given your characteristics, would you be able to solve the following problem?
Problem: {problem}
If yes, explain your reasoning and put the final answer choice (a single letter) in double square brackets. If you are likely to struggle with this problem, give a plausible incorrect solution and put the final incorrect answer choice (a single letter) in double square brackets.
"""
        elif prompt_method == '_persona_CoT':
            user_message = f"""{persona}Given your characteristics, would you be able to solve the following problem?
Problem: {problem}
Explain whether you think you can solve this problem and put the final answer choice (a single letter) in double square brackets.
"""        
        elif prompt_method == '_persona':
            user_message = f"""{persona}Given your characteristics, would you be able to solve the following problem?
Problem: {problem}
Put the final answer choice (a single letter) in double square brackets.
"""
    mem.append([{"role": "user", "content": user_message}])
    if 'gpt' in args.model_id and args.openai_api_key is not None:
        model_response = llm.prompt(model_id=MODEL_ID, messages=mem.mem, temperature=TEMPERATURE)
    else:
        model_response = get_vllm_inference(mem.mem, llm, TEMPERATURE, MODEL_ID, tokenizer)[0]
    
    memory_list.append(''.join([str(item) for item in mem.mem]))
    memory_len_list.append(len(mem.start_indices))   

    mem.append({"role": "assistant", "content":  model_response}, to_last_trajectory=True)
    
    matches = re.findall(r'\[\[([\s\S]*?)\]\]', model_response)
    if len(matches) == 1:
        model_choice = matches[0].upper()
    else:
        model_choice = np.nan
    model_response_list.append(model_response)
    model_choice_list.append(model_choice)
    if i % 25 == 0:
        print(i)
        print(datetime.now())
    if (i+1) % NUM_QUESTIONS_SELECTED == 0:
        results_tmp = results.head(len(model_response_list))
        results_tmp['model_response'] = model_response_list
        results_tmp['model_choice'] = model_choice_list
        results_tmp['memory'] = memory_list
        results_tmp['memory_len'] = memory_len_list
        results_tmp['model_id'] = [MODEL_ID]*len(results_tmp)
        results_tmp['temperature'] = [TEMPERATURE]*len(results_tmp) 
        results_tmp.to_csv(DATA_DIR + RESULTS_FOLDER + SETTING + '_tmp.csv', index=False)

results = results.head(len(model_response_list)).copy()
results['model_response'] = model_response_list
results['model_choice'] = model_choice_list
results['memory'] = memory_list
results['memory_len'] = memory_len_list
results['model_id'] = [MODEL_ID]*len(results)
results['temperature'] = [TEMPERATURE]*len(results)
if 'duolingo' in test_file_path or 'wordbank' in test_file_path:
    results['ModelIsCorrect'] = results['model_choice'].apply(lambda x: x == 'YES')
else:
    results['model_answer_value'] = results['model_choice'].apply(lambda x: convert_from_letter_to_num(x))
    results['ModelIsCorrect'] = results['model_answer_value'] == results['CorrectAnswer']
results['ModelIsCorrect'] = results['ModelIsCorrect'].apply(lambda x: int(x))
results.to_csv(DATA_DIR + RESULTS_FOLDER + SETTING + '.csv', index=False)