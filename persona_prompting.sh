#!/bin/bash
prompt_methods=('_persona_CoT_structure' '_persona_CoT' '_persona')
model_ids=('meta-llama/Meta-Llama-3-8B-Instruct' 'meta-llama/Meta-Llama-3-70B-Instruct' 'mistralai/Mistral-7B-Instruct-v0.2')
temperatures=(0.7 1)

for prompt_method in "${prompt_methods[@]}"; do
  for model_id in "${model_ids[@]}"; do
    for temperature in "${temperatures[@]}"; do
      python persona_prompting.py --date='20240612' \
        --test_file_path='data/eedi/test_data.csv' \
        --prompt_method="$prompt_method" \
        --model_id="$model_id" \
        --temperature="$temperature" \
        --max_tokens=400 \
        --tp_size=1
    done
  done
done

# for prompt_method in "${prompt_methods[@]}"; do
#   for model_id in "${model_ids[@]}"; do
#     for temperature in "${temperatures[@]}"; do
#       python persona_prompting.py --date='20240612' \
#         --test_file_path='data/wordbank/test_data.csv' \
#         --prompt_method="$prompt_method" \
#         --model_id="$model_id" \
#         --temperature="$temperature" \
#         --max_tokens=400 \
#         --tp_size=1
#     done
#   done
# done

# for prompt_method in "${prompt_methods[@]}"; do
#   for model_id in "${model_ids[@]}"; do
#     for temperature in "${temperatures[@]}"; do
#       python persona_prompting.py --date='20240612' \
#         --test_file_path='data/duolingo/test_data.csv' \
#         --prompt_method="$prompt_method" \
#         --model_id="$model_id" \
#         --temperature="$temperature" \
#         --max_tokens=400 \
#         --tp_size=1
#     done
#   done
# done

# # for openai models
# python persona_prompting.py --openai_api_key='sk-***' --openai_api_org='org-***' --date='20240612' --test_file_path='data/duolingo/test_data.csv' --prompt_method='_persona_CoT_structure' --model_id='gpt-3.5-turbo' --temperature=1 --max_tokens=400 