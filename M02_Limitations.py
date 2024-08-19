#%% packages
import os
from pprint import pprint
from langchain.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq
from langchain_openai import ChatOpenAI
from langchain.schema.output_parser import StrOutputParser
from dotenv import load_dotenv
load_dotenv()
# %% OpenAI models
# https://platform.openai.com/docs/models/overview
MODEL_NAME = 'gpt-3.5-turbo'
model = ChatOpenAI(model_name=MODEL_NAME,
                   temperature=0.5, # controls creativity
                   api_key=os.getenv('OPENAI_API_KEY'))

# %% GROQ models
# https://console.groq.com/docs/models
MODEL_NAME = 'llama-3.1-8b-instant'
model = ChatGroq(model_name=MODEL_NAME, 
                 api_key=os.getenv('GROQ_API_KEY'))

# %% Basic Example
res = model.invoke("What is a LangChain?")
# %%
pprint(res.content, width=50)

#%% gpt-3.5-turbo cutoff date 
# https://help.openai.com/en/articles/8555514-gpt-3-5-turbo-updates
# %%
res = model.invoke("What are the limitations of Large-Language-Models?")
# %%
pprint(res.content, width=50)
# %% Prompts and Chains
messages = [
    ("system", "You are a helpful AI assistent. Answer all questions to the best of your knowledge."),
    ("user", "What are the limitations of Large-Language-Models?"),
]

prompt = ChatPromptTemplate.from_messages(messages)
chain = prompt | model | StrOutputParser()
chain.invoke({})

# %% Pass Parameters to Chain
messages = [
    ("system", "You are a helpful AI assistent. Answer all questions to the best of your knowledge."),
    ("user", "Tell me a fun fact about {object}."),
]
prompt = ChatPromptTemplate.from_messages(messages)
chain = prompt | model | StrOutputParser()
res = chain.invoke({"object": "dogs"})
pprint(res, width=50)
# %%
