import os
import logging
logging.basicConfig(level=logging.DEBUG)
os.environ['OPENAI_API_KEY'] = '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'
os.environ['OPENAI_BASE_URL'] = 'https://api.groq.com/openai/v1'
from mem0 import Memory
from interactive_rag import config

m = Memory.from_config(config)
print('Testing m.add()...')
m.add('What is my name', user_id='cli_user')
print('m.add() finished!')
