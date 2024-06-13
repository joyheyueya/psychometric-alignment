from openai import OpenAI
from time import sleep

class LLM():

    def __init__(self, wait_time, max_failed_attempt, api_key, organization=None):

        self.wait_time = wait_time
        self.max_failed_attempt = max_failed_attempt
        self.client = OpenAI(
            api_key=api_key,
            organization=organization
        )
 
    def prompt(self, model_id, messages, temperature, completion_prompt=None, max_tokens=300):
        num_attempt = 0
        output = ''
        while num_attempt < self.max_failed_attempt:
            try:
                if 'text-davinci' in model_id:
                    completion = self.client.chat.completions.create(
                          model=model_id,
                          prompt=completion_prompt,
                          max_tokens=max_tokens,
                          temperature=temperature
                        )
                    output = completion['choices'][0]['text']
                else:
                    chat_completion = self.client.chat.completions.create(
                        model=model_id, 
                        messages=messages, 
                        temperature=temperature)
                    output = chat_completion.choices[0].message.content
                num_attempt += 1
            except Exception as er:
                print(type(er))
                sleep(self.wait_time)
                continue
            break
        return output