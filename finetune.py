import os
import argparse
import torch
from datasets import load_dataset
from accelerate import FullyShardedDataParallelPlugin, Accelerator
from torch.distributed.fsdp.fully_sharded_data_parallel import FullOptimStateDictConfig, FullStateDictConfig
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig, EarlyStoppingCallback
import transformers
from datetime import datetime
import math
import wandb
from peft import prepare_model_for_kbit_training

parser = argparse.ArgumentParser()
parser.add_argument('--date', type=str)
parser.add_argument('--train_dataset', type=str)
parser.add_argument('--eval_dataset', type=str)
parser.add_argument('--tokenizer_max_length', type=int) # 1024 for eedi, 700 for duolingo
parser.add_argument('--base_model_id', type=str)
parser.add_argument('--input_dir', type=str)
parser.add_argument('--output_dir', type=str)
parser.add_argument('--wandb_project', type=str)
parser.add_argument('--wandb_key', type=str)
args = parser.parse_args()

wandb.login(key=args.wandb_key)
DATE = args.date
TOKENIZER_MAX_LENGTH = args.tokenizer_max_length 
BASE_MODEL_ID = args.base_model_id
INPUT_DIR = args.input_dir
OUTPUT_DIR =  args.output_dir + DATE + '/'

print(args.train_dataset)
print(args.eval_dataset)
train_dataset = load_dataset('json', data_files=INPUT_DIR + args.train_dataset, split='train')
eval_dataset = load_dataset('json', data_files=INPUT_DIR + args.eval_dataset, split='train')
print(train_dataset)
print(eval_dataset)

# Accelerator
fsdp_plugin = FullyShardedDataParallelPlugin(
    state_dict_config=FullStateDictConfig(offload_to_cpu=True, rank0_only=False),
    optim_state_dict_config=FullOptimStateDictConfig(offload_to_cpu=True, rank0_only=False),
)

accelerator = Accelerator(fsdp_plugin=fsdp_plugin)

wandb_project = args.wandb_project
if len(wandb_project) > 0:
    os.environ["WANDB_PROJECT"] = wandb_project
    
def formatting_func(row):
    return ("{instruction}\n{input}").format_map(row)

# Load Base Model
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16
)

model = AutoModelForCausalLM.from_pretrained(BASE_MODEL_ID, quantization_config=bnb_config, device_map="auto")

# Tokenization
tokenizer = AutoTokenizer.from_pretrained(
    BASE_MODEL_ID,
    padding_side="left",
    add_eos_token=True,
    add_bos_token=True,
)
tokenizer.pad_token = tokenizer.eos_token

def generate_and_tokenize_prompt(prompt, train_on_inputs=True):
    result = tokenizer(
        formatting_func(prompt),
        truncation=True,
        max_length=TOKENIZER_MAX_LENGTH,
        padding="max_length",
    )
    result["labels"] = result["input_ids"].copy()
    if not train_on_inputs:
        result["labels"] = [-100]*(len(result["input_ids"]) - 2) + result["input_ids"][-2:]

    return result

tokenized_train_dataset = train_dataset.map(generate_and_tokenize_prompt)
tokenized_val_dataset = eval_dataset.map(generate_and_tokenize_prompt)

print(tokenized_val_dataset[4]['input_ids'])
print(tokenized_val_dataset[4]['labels'])

# LoRA
model.gradient_checkpointing_enable()
model = prepare_model_for_kbit_training(model)

def print_trainable_parameters(model):
    """
    Prints the number of trainable parameters in the model.
    """
    trainable_params = 0
    all_param = 0
    for _, param in model.named_parameters():
        all_param += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()
    print(
        f"trainable params: {trainable_params} || all params: {all_param} || trainable%: {100 * trainable_params / all_param}"
    )
    
print(model)

from peft import LoraConfig, get_peft_model

config = LoraConfig(
    r=32,
    lora_alpha=64,
    target_modules=[
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
        "gate_proj",
        "up_proj",
        "down_proj",
        "lm_head",
    ],
    bias="none",
    lora_dropout=0.05,  # Conventional
    task_type="CAUSAL_LM",
)

model = get_peft_model(model, config)
print_trainable_parameters(model)

print('with the LoRA adapters added')
print(model)

# Train
if torch.cuda.device_count() > 1: # If more than 1 GPU
    model.is_parallelizable = True
    model.model_parallel = True
model = accelerator.prepare_model(model)

run_name = DATE + '_' + BASE_MODEL_ID.split('/')[-1] + '_' + args.train_dataset.split('/')[-1].split('.j')[0].split('train_answer_id_')[-1]
output_dir = OUTPUT_DIR + run_name

if len(train_dataset) < 5000:
    save_and_eval_step = 50
elif len(train_dataset) < 10000:
    save_and_eval_step = 100
elif len(train_dataset) < 50000:
    save_and_eval_step = 200
else:
    save_and_eval_step = 400

if 'duolingo' in args.input_dir or 'wordbank' in args.input_dir:
    if len(train_dataset) < 2500:
        save_and_eval_step = 5
    elif len(train_dataset) < 5000:
        save_and_eval_step = 10
    elif len(train_dataset) < 10000:
        save_and_eval_step = 50
    elif len(train_dataset) < 50000:
        save_and_eval_step = 100
    else:
        save_and_eval_step = 200

trainer = transformers.Trainer(
    model=model,
    train_dataset=tokenized_train_dataset,
    eval_dataset=tokenized_val_dataset,
    args=transformers.TrainingArguments(
        output_dir=output_dir,
        warmup_steps=1,
        per_device_train_batch_size=32,
        gradient_accumulation_steps=1,
        gradient_checkpointing=True,
        max_steps=20000,
        learning_rate=2.5e-5, # Want a small lr for finetuning
        bf16=True,
        optim="paged_adamw_8bit",
        logging_steps=save_and_eval_step,              # When to start reporting loss
        logging_dir="./logs",        # Directory for storing logs
        save_strategy="steps",       # Save the model checkpoint every logging step
        save_steps=save_and_eval_step,                # Save checkpoints every 50 steps
        evaluation_strategy="steps", # Evaluate the model every logging step
        eval_steps=save_and_eval_step,               # Evaluate and save checkpoints every 50 steps
        do_eval=True,                # Perform evaluation at the end of training
        load_best_model_at_end=True, 
        metric_for_best_model="eval_loss", 
        greater_is_better=False,
        report_to="wandb",           # Comment this out if you don't want to use weights & baises
        run_name=f"{run_name}-{datetime.now().strftime('%Y-%m-%d-%H-%M')}"          # Name of the W&B run (optional)
    ),
    data_collator=transformers.DataCollatorForLanguageModeling(tokenizer, mlm=False),
    callbacks=[EarlyStoppingCallback(
        early_stopping_patience=3,    # Number of evaluations with no improvement after which training will be stopped
        early_stopping_threshold=0.0  # Minimum change in the monitored metric to qualify as an improvement
    )]
)

model.config.use_cache = False  # silence the warnings. Please re-enable for inference!
trainer.train()