#%% Packages
from langchain_community.document_loaders import TextLoader, YoutubeLoader, WikipediaLoader, PyPDFLoader, DirectoryLoader, UnstructuredFileLoader, Docx2txtLoader
from langchain.schema import Document
import os
from pprint import pprint

# %% Text Import from Markdown File
file_path = 'data/chromadb_cheatsheet.md'
loader = TextLoader(file_path)
docs_markdown = loader.load()

#%%
docs_markdown[0].page_content
# %% load youtube video transcript
base_video_url = 'https://www.youtube.com/watch?v='
video_id = 'dyO3lGJnY7I'  # video: Getting Up & Running With Chroma DB | Generative AI | Vector Database
video_url = f"{base_video_url}{video_id}"
loader = YoutubeLoader.from_youtube_url(
    video_url, 
    add_video_info=True, 
    language=['en', 'es'], 
    translation='en')

docs_youtube = loader.load()

#%% check the content
pprint(docs_youtube[0].page_content)

# %% wikipedia
loader = WikipediaLoader(query='Vector Database', 
                         lang='en', 
                         load_max_docs=1, 
                         doc_content_chars_max=100000)
docs_wikipedia = loader.load()

# %% get the first element
pprint(docs_wikipedia[0].page_content)
# %% RAG report
file_path = "data/Retrieval Augmented Generation.pdf"
loader = PyPDFLoader(file_path)
docs_pdf = loader.load()

#%% Word Document
file_path = "data/Vector Databases.docx"
loader = Docx2txtLoader(file_path)
docs_word = loader.load()

#%%
docs_word[0].page_content

# %% iterate over a complete folder
# get all files in folder
file_paths = [os.path.join('data', f) for f in os.listdir('data') if os.path.isfile(os.path.join('data', f))]
file_paths
#%% load all files
# supported file types:
# https://docs.unstructured.io/open-source/installation/full-installation
# pip install unstructured[all-docs]
docs_unstructured = []
for file_path in file_paths[:2]:
    print(file_path)
    loader = UnstructuredFileLoader(file_path)
    docs_unstructured.append(loader.load())

# %%
docs_unstructured

# %%
