import os
import argparse
import torch
import gc
from datasets import load_dataset
from peft import PeftModel
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from vllm import LLM, SamplingParams
import pandas as pd
from datetime import datetime
import re
import numpy as np
from multiprocessing import Process
import torch.multiprocessing as mp
mp.set_start_method('spawn', force=True)
from utils import *
import random
import warnings
warnings.filterwarnings('ignore')

parser = argparse.ArgumentParser()
parser.add_argument('--test_file_path', type=str)
parser.add_argument('--results_folder', type=str)
parser.add_argument('--max_tokens', type=int) # for inference
parser.add_argument('--tp_size', default=4, type=int)
args = parser.parse_args()

MAX_TOKENS = args.max_tokens
REPETITION_PENALTY = 1
RESULTS_FOLDER = args.results_folder
create_folder(RESULTS_FOLDER)

few_shot_df = pd.read_csv('data/few_shot_examples.csv')
if 'solution' not in few_shot_df.columns:
    few_shot_df['solution'] = few_shot_df['gpt4_solution']

p1 = few_shot_df.iloc[0]['problem']
s1 = few_shot_df.iloc[0]['solution']
p2 = few_shot_df.iloc[1]['problem']
s2 = few_shot_df.iloc[1]['solution']
p3 = few_shot_df.iloc[2]['problem']
s3 = few_shot_df.iloc[2]['solution']

def add_prompt_few_shot(sample):
    TEMPLATE = f"""### Question: {p1}\n### Answer: {s1}\n

### Question: {p2}\n### Answer: {s2}\n 

### Question: {p3}\n### Answer: {s3}\n

### Question: {sample}\n### Answer: 
"""
    prompt = TEMPLATE
    return prompt

def add_prompt(sample, persona):
    if len(persona) > 0:
        TEMPLATE = persona + '\nGiven your characteristics, would you be able to solve the following problem correctly?' + '\nProblem:' + sample + "\nIf yes, explain your reasoning and put the final answer choice (a single letter) within \\boxed{}. If it is likely that you would struggle with this problem, give a plausible incorrect solution and put the final incorrect answer choice (a single letter) within \\boxed{}. The final answer must be one of the four letters: A, B, C, or D."
    else:
        TEMPLATE = sample + "\nPlease reason step by step, and put your final answer in double square brackets (e.g., [[A/B/C/D]]). The final answer must be one of the four letters: A, B, C, or D."
    prompt = TEMPLATE
    return prompt
        
def get_vllm_inference(prompts, llm, temperature, base_model_id, tokenizer):
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
        messages_list = []
        for p in prompts:
            messages_list.append([{"role": "user", "content": p}])

        # Avoid adding bos_token repeatedly
        prompt_token_ids = [tokenizer.apply_chat_template(messages, add_generation_prompt=True) for messages in messages_list]
        for p_index in range(0, len(prompts), 1000):
            print(datetime.now())
            outputs = llm.generate(prompt_token_ids=prompt_token_ids[p_index:p_index+1000], sampling_params=sampling_params)
            generated_text = [output.outputs[0].text for output in outputs]
            output_text_list += generated_text        
    return output_text_list

def get_prompts(input_data, base_model_id):
    prompts = []
    if 'instruct' in base_model_id.lower() or '-rl' in base_model_id.lower():
        print('strong')
        for sample in input_data:
            text = add_prompt(sample, '')
            prompts.append(text)
    else:
        print('weak')
        for sample in input_data:
            text = add_prompt_few_shot(sample)
            prompts.append(text)
    return prompts

def run(base_model_id, temp_list, RESULTS_FOLDER):
    results = pd.read_csv(args.test_file_path)
    results = results.sort_values(['UserId', 'DateAnswered']).head(750) 
    print('Loading model', datetime.now())
    print('base_model_id', base_model_id)
    tokenizer = AutoTokenizer.from_pretrained(base_model_id)
    if 'AWQ' in base_model_id:
        llm = LLM(model=base_model_id, quantization="awq", dtype="auto", trust_remote_code=True, gpu_memory_utilization=0.9, tensor_parallel_size=args.tp_size)
    else:
        llm = LLM(model=base_model_id, trust_remote_code=True, gpu_memory_utilization=0.9, tensor_parallel_size=args.tp_size, dtype='bfloat16')
    print('finished loading model', datetime.now())  
    input_data = list(results['problem'])

    prompts = get_prompts(input_data, base_model_id)
    print('len(prompts)', len(prompts))

    for temperature in temp_list:
        output_text_list = get_vllm_inference(prompts, llm, temperature, base_model_id, tokenizer)

        results['temperature'] = [temperature]*len(results)
        results['repetition_penalty'] = [REPETITION_PENALTY]*len(results)
        results['inference_prompt'] = prompts
        results['model_response'] = output_text_list
        results['model_choice'] = results['model_response'].apply(lambda x: fix_parsing_error(x))
        results['model_answer_value'] = results['model_choice'].apply(lambda x: convert_from_letter_to_num(x))
        results['ModelIsCorrect'] = results['model_answer_value'] == results['CorrectAnswer']
        results['ModelIsCorrect'] = results['ModelIsCorrect'].apply(lambda x: int(x))
        results.to_csv(RESULTS_FOLDER + base_model_id.split('/')[-1] + '_t' + str(temperature).replace('.', '') + '.csv', index=False)
        
def main(): 
    model_list = [
        "mistralai/Mistral-7B-v0.1",
        "TheBloke/llemma_7b-AWQ",
        "TheBloke/llemma_34b-AWQ",
        "deepseek-ai/deepseek-math-7b-base",
        "deepseek-ai/deepseek-math-7b-instruct",
        "deepseek-ai/deepseek-math-7b-rl",
        "meta-llama/Meta-Llama-3-8B",
        "meta-llama/Meta-Llama-3-8B-Instruct",
        "meta-llama/Meta-Llama-3-70B",
        "meta-llama/Meta-Llama-3-70B-Instruct",     
    ]
    for base_model_id in model_list:
        p = Process(target=run, args=(base_model_id,[0, 0.7, 1], RESULTS_FOLDER))
        p.start()
        p.join()
        clear_resources()
        
if __name__ == "__main__":
    main()