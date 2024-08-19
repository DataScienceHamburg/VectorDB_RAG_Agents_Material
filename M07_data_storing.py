#%% Packages
import os
from langchain.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings.sentence_transformer import SentenceTransformerEmbeddings
from langchain.vectorstores import Chroma
from langchain.schema import Document
from langchain_community.document_loaders import GutenbergLoader

#%% load the text and split it
def process_data(url: str, book_title: str) -> list[Document]:
    # load text
    loader = GutenbergLoader(url)
    data = loader.load()
    # split text
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size = 1000,
        chunk_overlap  = 100,
        add_start_index = True,
    )
    chunks = text_splitter.split_documents(data)
    # add metadata book_title
    for chunk in chunks:
        chunk.metadata["book_title"] = book_title
    return chunks

#%% Embeddings Model
embeddings_model = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")

# %% connect to the database
persistent_db_path = "db"
db_client = Chroma(persist_directory=persistent_db_path, embedding_function=embeddings_model)

#%% Text Import from Project Gutenberg
dracula_url = "https://www.gutenberg.org/cache/epub/345/pg345.txt"
dracula_chunks = process_data(url=dracula_url, book_title="Dracula")

#%% Text Import from Project Gutenberg
frankenstein_url = "https://www.gutenberg.org/cache/epub/84/pg84.txt"
frankenstein_chunks = process_data(url=frankenstein_url, book_title="Frankenstein")


# %% add the documents to the database
db_client.add_documents(dracula_chunks)

#%%
db_client.add_documents(frankenstein_chunks)
# %%
len(db_client.get()['ids'])

# %%
