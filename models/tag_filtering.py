import os
import backoff
import threading
import concurrent.futures
import openai
import anthropic

from openai import OpenAI, OpenAIError, RateLimitError

from models.prompt import TAG_FILTER_SYSTEM_PROMPT


openai_key = os.getenv('OPENAI_API_KEY')
oai_client = OpenAI(api_key=openai_key)

anthropic_key = os.getenv('ANTHROPIC_API_KEY')
ant_client = anthropic.Anthropic(api_key=anthropic_key)

@backoff.on_exception(backoff.expo, (OpenAIError, RateLimitError), max_tries=10)
def filter_tags_openai(tags: str, prompt: str):
    completion = oai_client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": TAG_FILTER_SYSTEM_PROMPT},
            {"role": "user", "content": f"Prompt: {prompt}\nTags: {tags}\nRelevant Tags:"}
        ],
        temperature=0.7,
    )
    return completion.choices[0].message.content

@backoff.on_exception(backoff.expo, (RateLimitError), max_tries=10)
def filter_tags_anthropic(tags: str, prompt: str):
    message = ant_client.messages.create(
        model="claude-3-5-haiku-20241022",
        temperature=0.0,
        max_tokens=50,
        system=TAG_FILTER_SYSTEM_PROMPT,
        messages=[
            {"role": "user", "content": f"Prompt: {prompt}\nTags: {tags}\nRelevant Tags:"}
        ]
    )
    return message.content[0].text

def process_tags_batch(prompts: list[dict], model='openai'):
    def worker(prompt_data):
        try:
            original_tags = prompt_data['tags']
            if model == 'openai':
                filtered_tags = filter_tags_openai(prompt_data['tags'], prompt_data['prompt'])
                return filtered_tags
            elif model == 'anthropic':
                filtered_tags = filter_tags_anthropic(prompt_data['tags'], prompt_data['prompt'])
                return filtered_tags
            # print(f'all_tags: {original_tags} | filtered_tags: {filtered_tags}')
        except Exception as e:
            print(f"Error processing prompt: {prompt_data['prompt']}: {e}")
            return e
    
    with concurrent.futures.ThreadPoolExecutor() as executor:
        results = list(executor.map(worker, prompts))
            
    return results
