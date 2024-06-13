import openai
import argparse
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
import llm
from sklearn.model_selection import train_test_split
import json
from utils import *

parser = argparse.ArgumentParser()
parser.add_argument('--base_model_id', type=str)
parser.add_argument('--input_dir', type=str)
parser.add_argument('--adapter_path_id', type=str)
parser.add_argument('--start_step', type=int)
parser.add_argument('--end_step', default=0, type=int)
parser.add_argument('--save_step', default=400, type=int)
parser.add_argument('--merge_method', default="", type=str)
parser.add_argument('--model_parent_dir', default="", type=str) #if non empty, ignore the start_step and end_step
args = parser.parse_args()

BASE_MODEL_ID = args.base_model_id
eval_tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_ID, add_bos_token=True, trust_remote_code=True)
if args.end_step == 0:
    end_step = args.start_step + 1
else:
    end_step = args.end_step
    
if args.model_parent_dir == '':   
    for i in range(args.start_step, end_step, args.save_step):
        print(i)
        print(datetime.now())
        adapter_path = args.input_dir + args.adapter_path_id + "/checkpoint-" + str(i)
        model_dir = args.input_dir + args.adapter_path_id + args.merge_method + "_merged" + "/" + adapter_path.split('/')[-1] 
        print(adapter_path)
        print(model_dir)

        base_model = AutoModelForCausalLM.from_pretrained(BASE_MODEL_ID, device_map="auto")

        # Load and merge the PeftModel
        ft_model = PeftModel.from_pretrained(base_model, adapter_path)
        if args.merge_method == '_neg':
            state_dict = ft_model.state_dict()
            neg_dict = {k:-v for k,v in state_dict.items() if "lora_B" in k}
            state_dict.update(neg_dict)
            ft_model.load_state_dict(state_dict)
        ft_model = ft_model.merge_and_unload()

        # Save the merged model and tokenizer
        ft_model.save_pretrained(model_dir, safe_serialization=True)
        eval_tokenizer.save_pretrained(model_dir)

        del base_model, ft_model
        torch.cuda.empty_cache()
        gc.collect()
else:
    fourth_to_last_checkpoints = []
    for subdir in os.listdir(args.model_parent_dir):
        dir_path = os.path.join(args.model_parent_dir, subdir)
        if os.path.isdir(dir_path):
            checkpoints = []
            for file in os.listdir(dir_path):
                if file.startswith("checkpoint"):
                    checkpoints.append(file)

            # Sort the checkpoints
            checkpoints.sort(key=lambda x: int(x.split('-')[1]))

            # Check if there are enough checkpoints to select the fourth to last
            if len(checkpoints) >= 4:
                # Append the fourth to last checkpoint path to the list
                fourth_to_last_checkpoints.append(os.path.join(dir_path, checkpoints[-4]))

    for f in fourth_to_last_checkpoints:
        print(datetime.now())
        adapter_path = f
        model_dir = f.split('/checkpoint')[0] + args.merge_method + "_merged" + "/" + adapter_path.split('/')[-1] 
        print(adapter_path)
        print(model_dir)

        base_model = AutoModelForCausalLM.from_pretrained(BASE_MODEL_ID, device_map="auto")

        # Load and merge the PeftModel
        ft_model = PeftModel.from_pretrained(base_model, adapter_path)
        if args.merge_method == '_neg':
            state_dict = ft_model.state_dict()
            neg_dict = {k:-v for k,v in state_dict.items() if "lora_B" in k}
            state_dict.update(neg_dict)
            ft_model.load_state_dict(state_dict)
        ft_model = ft_model.merge_and_unload()

        # Save the merged model and tokenizer
        ft_model.save_pretrained(model_dir, safe_serialization=True)
        eval_tokenizer.save_pretrained(model_dir)

        del base_model, ft_model
        torch.cuda.empty_cache()
        gc.collect()        