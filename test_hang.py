import os
os.environ['OPENAI_API_KEY'] = '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'
os.environ['OPENAI_BASE_URL'] = 'https://api.groq.com/openai/v1'
from mem0.llms.openai import OpenAILLM
from mem0.configs.llms.openai import OpenAIConfig

config = OpenAIConfig(model='llama-3.1-8b-instant')
llm = OpenAILLM(config)
print('Sending to Groq with json_object...')
try:
    res = llm.generate_response(messages=[{'role': 'user', 'content': 'Output {"test": 1} in JSON'}], response_format={'type': 'json_object'})
    print('Groq response:', res)
except Exception as e:
    print('Error:', e)
