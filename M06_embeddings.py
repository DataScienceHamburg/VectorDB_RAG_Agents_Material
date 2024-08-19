#%% packages
from langchain.document_loaders import Docx2txtLoader
from langchain_community.embeddings.sentence_transformer import SentenceTransformerEmbeddings
from langchain_openai.embeddings import OpenAIEmbeddings
from transformers import AutoTokenizer
from langchain.text_splitter import RecursiveCharacterTextSplitter
from dotenv import load_dotenv, find_dotenv
load_dotenv(find_dotenv(usecwd=True))
#%% Embeddings Model
# embeddings_model = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")

# %% OpenAI embeddings
# https://platform.openai.com/docs/guides/embeddings/embedding-models
embeddings_model = OpenAIEmbeddings(model='text-embedding-3-small')


# %%
tokenizer = AutoTokenizer.from_pretrained('sentence-transformers/all-minilm-l6-v2')

def tokens(text: str) -> int:
    return len(tokenizer.encode(text))

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size = 256,
    chunk_overlap  = 26,
    length_function = tokens,
    add_start_index = True,
)
# %%
loader = Docx2txtLoader("data/Vector Databases.docx")
pages = loader.load_and_split()

#%% Split the text with text_splitter
docs_texts = text_splitter.split_documents(pages)
# %%
texts = [doc.page_content for doc in docs_texts]
# %%
embeddings = embeddings_model.embed_documents(texts)
# %% for each chunk, calculate the embeddings
len(embeddings)
# %% get the length of vectors
len(embeddings[0])




# %%
