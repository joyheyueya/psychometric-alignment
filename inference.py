import os
import argparse
import torch
import gc
from datasets import load_dataset
from peft import PeftModel
import torch
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
from utils_persona import *
import random
import warnings
warnings.filterwarnings('ignore')

parser = argparse.ArgumentParser()
parser.add_argument('--date', type=str)
parser.add_argument('--data_format', default='', type=str)
parser.add_argument('--base_model_id', type=str)
parser.add_argument('--test_file_path', type=str)
parser.add_argument('--input_dir', type=str)
parser.add_argument('--output_dir', type=str)
parser.add_argument('--adapter_path_id', type=str)
parser.add_argument('--input_response_column', type=str) # answer_choice or model_choice
parser.add_argument('--max_tokens', type=int)
parser.add_argument('--max_prior_question', type=int)
parser.add_argument('--tp_size', default=4, type=int)
parser.add_argument('--start_step', default=400, type=int)
parser.add_argument('--end_step', default=0, type=int)
parser.add_argument('--save_step', default=400, type=int)
parser.add_argument('--persona_template', default='first_pov', type=str) # 'first_pov' or 'third_pov'
parser.add_argument('--model_parent_dir', default="", type=str) #if non empty, ignore the start_step and end_step
args = parser.parse_args()

DATE = args.date
if 'duolingo' in args.test_file_path or 'wordbank' in args.test_file_path:
    DATA_FORMAT = '_binary'
else:
    DATA_FORMAT = args.data_format
BASE_MODEL_ID = args.base_model_id
INPUT_RESPONSE_COLUMN = args.input_response_column
MERGED_MODEL_DIR = args.input_dir + args.adapter_path_id + '/'
MAX_TOKENS = args.max_tokens
REPETITION_PENALTY = 1

tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_ID)

results = pd.read_csv(args.test_file_path)
if 'countries' in results:
    results['countries_full'] = results['countries'].apply(lambda x: map_country_codes(x))

if args.persona_template == 'first_pov':
    if 'duolingo' in args.test_file_path:
        results['persona'] = results.apply(create_persona_duolingo_first_pov, axis=1)
    elif 'wordbank' in args.test_file_path:
        results['persona'] = results.apply(create_persona_wordbank_first_pov, axis=1)
    else:
        results['persona'] = results.apply(create_persona_3basic, axis=1)
elif args.persona_template == 'third_pov':
    if 'duolingo' in args.test_file_path:
        results['persona'] = results[['countries', 'client']].apply(lambda x: create_persona_duolingo_third_pov(x), axis=1)
    else:
        results['persona'] = results[['age', 'Gender', 'PremiumPupil']].apply(lambda x: create_persona_3basic_thirdperson(x), axis=1)

if 'AnswerValue' in results.columns and 'CorrectAnswer' in results.columns:
    results['answer_choice'] = results['AnswerValue'].apply(lambda x: number_to_letter(x))
    results['correct_answer_choice'] = results['CorrectAnswer'].apply(lambda x: number_to_letter(x))
results['Correctness'] = results['IsCorrect'].apply(lambda x: map_binary_to_correctness(x))
results = results.sort_values(['UserId', 'DateAnswered'])
evaluation_df = results

def get_vllm_inference(prompts, llm, temperature, base_model_id):
    sampling_params = SamplingParams(max_tokens=MAX_TOKENS, 
                      temperature=temperature,
                      repetition_penalty=REPETITION_PENALTY) 
    sampling_params.stop = [tokenizer.eos_token]
    output_text_list = []    
    if 'instruct' not in base_model_id and 'Instruct' not in base_model_id:
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

def get_concatenated_string(input_df, output_df, DATA_FORMAT, input_response_column):
    if args.persona_template == 'first_pov':
        student_answer_indicator = 'Your answer'
    elif args.persona_template == 'third_pov':
        student_answer_indicator = 'Student answer'
    concatenated_string = ''
    if DATA_FORMAT == '':
        if len(input_df) > 0:
            concatenated_string = 'Question:\n' + '\nQuestion:\n'.join(input_df['problem'] + '\n' + student_answer_indicator + ':\n' + input_df[input_response_column] + '\nTrue answer:\n' + input_df['correct_answer_choice'])
            concatenated_string += '\nQuestion:\n' + output_df['problem'].values[0] + '\n' + student_answer_indicator + ':\n'
        else:
            concatenated_string = 'Question:\n' + output_df['problem'].values[0] + '\n' + student_answer_indicator + ':\n'
    elif DATA_FORMAT == '_binary':
        if len(input_df) > 0:
            if input_response_column == 'answer_choice':
                input_response_column = 'Correctness'
            concatenated_string = 'Question:\n' + '\nQuestion:\n'.join(input_df['problem'] + '\n' + student_answer_indicator + ':\n' + input_df[input_response_column])
            concatenated_string += '\nQuestion:\n' + output_df['problem'].values[0] + '\n' + student_answer_indicator + ':\n'
        else:
            concatenated_string = 'Question:\n' + output_df['problem'].values[0] + '\n' + student_answer_indicator + ':\n'
    return concatenated_string

def save_inference(llm, temperature, input_response_column, RESULTS_FOLDER, adapter_path=None):
    print('temperature',temperature)

    results = evaluation_df
    
    for n in range(1,51):
        print('n', n)

        results_sub = results.groupby('UserId').head(n).copy()
        results_single = results_sub.groupby('UserId').tail(1).copy()
        prompts = []

        for u in results_sub['UserId'].unique():
            sampled_df = results_sub[results_sub['UserId'] == u]
            output_df = sampled_df.tail(1)
            sampled_df = sampled_df.head(len(sampled_df) - 1)
            sampled_df = sampled_df.tail(min(len(sampled_df), args.max_prior_question))            
            concatenated_string = get_concatenated_string(sampled_df, output_df, DATA_FORMAT, input_response_column)
            concatenated_string = output_df.iloc[0]['persona'] + '\n' + concatenated_string
            prompts.append(concatenated_string)  

        print('len(prompts)', len(prompts))

        output_text_list = get_vllm_inference(prompts, llm, temperature, BASE_MODEL_ID)
        
        model_choice_list = [get_first_word(output) for output in output_text_list]
        answer_id_list = list(results_single['AnswerId'])
        tmp_df = pd.DataFrame()
        tmp_df['AnswerId'] = answer_id_list
        tmp_df['model_choice'] = model_choice_list
        tmp_df['prompt'] = prompts
        tmp_df['model_output'] = output_text_list
        if 'model_choice' in results:
            tmp_df_prev = results[['AnswerId','model_choice','prompt','model_output']].dropna(subset=['prompt'])
            tmp_df = pd.concat([tmp_df, tmp_df_prev])
        results = results[[c for c in results.columns if 'model_choice' not in c and 'prompt' not in c and 'model_output' not in c]].set_index('AnswerId').join(tmp_df.set_index('AnswerId')).reset_index()
        results = results.sort_values(['UserId', 'DateAnswered'])    

        results_tmp = results.copy()
        if adapter_path == None:
            results_tmp['model_id'] = [BASE_MODEL_ID.split('/')[-1]]*len(results_tmp)
        else:
            results_tmp['model_id'] = [BASE_MODEL_ID.split('/')[-1] + '_' + adapter_path.split('/')[-1]]*len(results_tmp)
        results_tmp['Temperature'] = [temperature]*len(results_tmp)
        results_tmp['repetition_penalty'] = [REPETITION_PENALTY]*len(results_tmp)

        if adapter_path == None:
            results_tmp.to_csv(RESULTS_FOLDER + DATE + '_' + BASE_MODEL_ID.split('/')[-1] + '_t' + str(temperature).replace('.', '') + '_rep' + str(REPETITION_PENALTY).replace('.', '') + '_test.csv', index=False)
        else:
            results_tmp.to_csv(RESULTS_FOLDER + DATE + '_' + BASE_MODEL_ID.split('/')[-1] + '_' + adapter_path.split('/')[-1] + '_t' + str(temperature).replace('.', '') + '_rep' + str(REPETITION_PENALTY).replace('.', '') + '_test.csv', index=False)
    
def run_model_iteration(model_dir, temp_list, RESULTS_FOLDER):
    print('Loading model', datetime.now())
    llm = LLM(model=model_dir, trust_remote_code=True, gpu_memory_utilization=0.9, tensor_parallel_size=args.tp_size, dtype='bfloat16')
    print('finished loading model', datetime.now())  
    
    for temp in temp_list:
        save_inference(llm, temp, INPUT_RESPONSE_COLUMN, RESULTS_FOLDER, adapter_path=model_dir.split('/')[-1]) 

def main():     
    if args.model_parent_dir == '':  
        if args.end_step == 0:
            end_step = args.start_step + 1
        else:
            end_step = args.end_step
        RESULTS_FOLDER = args.output_dir + DATE + '_results_' + args.adapter_path_id + '_condition_' + INPUT_RESPONSE_COLUMN + DATA_FORMAT + '/'
        create_folder(RESULTS_FOLDER)
        for checkpoint in range(args.start_step, end_step, args.save_step):
            adapter_path = "/checkpoint-" + str(checkpoint)
            model_dir = MERGED_MODEL_DIR + adapter_path.split('/')[-1]
            print('model_dir', model_dir)
            print('checkpoint', checkpoint)
            p = Process(target=run_model_iteration, args=(model_dir,[1],RESULTS_FOLDER))
            p.start()
            p.join()
            clear_resources()
    else:
        checkpoints = []
        for subdir in os.listdir(args.model_parent_dir):
            dir_path = os.path.join(args.model_parent_dir, subdir)

            if os.path.isdir(dir_path) and 'merge' in dir_path:
                for file in os.listdir(dir_path):
                    if file.startswith("checkpoint"):
                        checkpoints.append(os.path.join(dir_path, file)) 
        for model_dir in checkpoints:
            RESULTS_FOLDER = args.output_dir + args.model_parent_dir.split('/')[-1] + '/' + DATE + '_results_' + model_dir.split('/')[-2] + '_condition_' + INPUT_RESPONSE_COLUMN + DATA_FORMAT + '/'
            create_folder(RESULTS_FOLDER)
            print('model_dir', model_dir)
            p = Process(target=run_model_iteration, args=(model_dir,[1],RESULTS_FOLDER))
            p.start()
            p.join()
            clear_resources()
        
if __name__ == "__main__":
    main()