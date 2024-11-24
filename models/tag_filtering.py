import os
import ell

from openai import OpenAI
from models.prompt import TAG_FILTER_SYSTEM_PROMPT


openai_key = os.getenv('OPENAI_API_KEY')

def filter_tags(tags: str, prompt: str):
    ell.init(store='./logdir', autocommit=True)

    @ell.simple(model="gpt-4o-mini", temperature=0, client=OpenAI(api_key=openai_key))
    def call_llm(tags: str, prompt: str):
        return [
            ell.system(TAG_FILTER_SYSTEM_PROMPT),
            ell.user(f'''Prompt: {prompt}
                     Tags: {tags}
                    ''')
        ]

    return call_llm(tags, prompt)