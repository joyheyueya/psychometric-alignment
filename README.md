# psychometric-alignment
Repository for the paper Psychometric Alignment: Capturing Human Knowledge Distributions via Language Models

## Downloading the datasets
All datasets (see details in `dataset_documentation.pdf`) are in a public Google Drive folder: [https://drive.google.com/drive/folders/1cFez1tATsCgXOGBxpR2oHAtuWUqPqlDT?usp=sharing](https://drive.google.com/drive/folders/1cFez1tATsCgXOGBxpR2oHAtuWUqPqlDT?usp=sharing). Unzip the `data.zip` file and put it in the current directory.

## Installation

1. Install conda
```bash
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
bash Miniconda3-latest-Linux-x86_64.sh
```
2. Create a conda environment
```bash
conda env create -f lm.yml
```
3. [optional] Create another conda envrionment for IRT. You only need to use this environment if you want to run IRT-related code.
```bash
conda env create -f pyr.yml
```

## Running the code for ensembling LMs
```bash
sh naive_ensemble_inference.sh
```

## Running the code for persona prompting
```bash
sh persona_prompting.sh
```
For models from openai, you need to provide your openai API key (and organization ID if needed). Here is an example:
```bash
python persona_prompting.py --openai_api_key='sk-***' --openai_api_org='org-***' --date='20240612' --test_file_path='data/eedi/test_data.csv' --prompt_method='_persona_CoT_structure' --model_id='gpt-3.5-turbo' --temperature=1 --max_tokens=400 
```

## Running the code for fine-tuning

1. Generate training and validation data by running `generate_train_val_data.ipynb`. You need to specify which dataset you want.
2. Create a wandb account and get your key: https://wandb.ai/site.
3. Fine-tune the model. Here is an example:
```bash
python finetune.py --date='20240612eedi_mistral' --train_dataset='20240612_data_with_train_answer_id_20240612_7.jsonl' --eval_dataset='20240612_data_with_val_answer_id_20240612_213.jsonl' --tokenizer_max_length=1024 --base_model_id='mistralai/Mistral-7B-v0.1' --input_dir='data/eedi/' --output_dir='../models/' --wandb_project='20240612eedi_mistral' --wandb_key='861d***'
```
4. Save the model.
```bash
python save_merged_lora.py --base_model_id='mistralai/Mistral-7B-v0.1' --model_parent_dir='../models/20240612eedi_mistral' --input_dir='' --adapter_path_id='' --start_step=0
```
5. Run inference.
```bash
python inference.py --date='20240612' --base_model_id='mistralai/Mistral-7B-v0.1' --model_parent_dir='../models/20240612eedi_mistral' --test_file_path='data/eedi/test_data.csv' --input_dir='' --output_dir='../results/' --adapter_path_id='' --input_response_column='model_choice' --max_tokens=10 --max_prior_question=10 --tp_size=1
``` 
