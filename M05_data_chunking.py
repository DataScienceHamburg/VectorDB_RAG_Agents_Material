#%% packages
from langchain.document_loaders import Docx2txtLoader
from langchain.text_splitter import CharacterTextSplitter, RecursiveCharacterTextSplitter
from langchain_experimental.text_splitter import SemanticChunker
from langchain_openai.embeddings import OpenAIEmbeddings
import matplotlib.pyplot as plt
from dotenv import load_dotenv, find_dotenv
load_dotenv(find_dotenv(usecwd=True))
# %%
loader = Docx2txtLoader("data/Vector Databases.docx")
pages = loader.load_and_split()
print(f'Loaded {len(pages)} pages from the Word Document.')


#%% Text Splitter
# Assumption: 1000 characters equal 250 tokens (average token length is 4)
text_splitter = CharacterTextSplitter(
    chunk_size = 1000,
    chunk_overlap  = 100,
    add_start_index = True,
)

recursive_text_splitter = RecursiveCharacterTextSplitter(
    chunk_size = 1000,
    chunk_overlap  = 100,
    add_start_index = True,
)



# %% Character Text Splitter
texts = text_splitter.split_documents(pages)
print(f'Number of Chunks after splitting: {len(texts)}')
# get the number of tokens in each chunk
chunks = [len(doc.page_content) / 4 for doc in texts]
print(f"Number of Tokens in each chunk: {chunks}")

# %% Recursive Character Text Splitter
texts = recursive_text_splitter.split_documents(pages)
print(f'Number of Chunks after splitting: {len(texts)}')
# get the number of tokens in each chunk
chunks = [len(doc.page_content) / 4 for doc in texts]
print(f"Number of Tokens in each chunk: {chunks}")


#%% visualize number of tokens in each chunk as barplot
plt.bar(range(len(chunks)), chunks)
plt.title('Number of Tokens in each chunk')
plt.show()
# %% Semantic Text Splitter
semantic_splitter = SemanticChunker(embeddings=OpenAIEmbeddings(), breakpoint_threshold_type="gradient")
texts = semantic_splitter.split_documents(pages)

# %%
texts

# %%
