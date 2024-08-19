
#%% Packages
from langchain.vectorstores import Chroma
from langchain_community.embeddings.sentence_transformer import SentenceTransformerEmbeddings
from pprint import pprint

#%% Embeddings Model
embeddings_model = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")


#%% connect to the database
persistent_db_path = "db"
db_client = Chroma(persist_directory=persistent_db_path, embedding_function=embeddings_model)

# %% get all objects
res = db_client.get()
# %% Default Retriever: Cosine Similarity
my_query = "Where does Dracula live?"
db_client.similarity_search(my_query)
# %%
# %% Similarity Score with Threshold
res = db_client.similarity_search_with_score(my_query, k=5)
found_docs= [doc[0] for doc in res]              
score = [doc[1] for doc in res]              

# %% Maximum Margin Relevance
# lambda_param: 0.0: only similarity, 1.0: only diversity
# k: number of results
db_client.max_marginal_relevance_search(my_query, 
                                        k=5, 
                                        lambda_param=0.5)

# %% Hybrid Search (Text + Metadata)
my_query = "Where does it play?"
filter_metadata =  {"book_title": "Frankenstein"}
res = db_client.similarity_search(my_query, 
                            filter=filter_metadata,
                            k=10)



# %%
pprint(res[0].page_content)
# %%
