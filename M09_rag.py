#%% Packages
from langchain.vectorstores import Chroma
from langchain_community.embeddings.sentence_transformer import SentenceTransformerEmbeddings
from pprint import pprint
from dotenv import load_dotenv
from langchain_groq import ChatGroq
load_dotenv()

#%% Embeddings Model
embeddings_model = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")

#%% connect to the database
persistent_db_path = "db"
db_client = Chroma(persist_directory=persistent_db_path, embedding_function=embeddings_model)


# %% LLM Setup
# available models: https://console.groq.com/docs/models
MODEL = "llama-3.1-8b-instant"
llm = ChatGroq(model=MODEL, temperature=0)

# %% RAG Chat Function
def rag_chat(user_query: str, k: int = 5):
    retrieved_docs = db_client.similarity_search(query=user_query, k=k)
    retrieved_docs_text = [doc.page_content for doc in retrieved_docs]
    retrieved_docs_text
    retrieved_docs_text_str = "\n".join(retrieved_docs_text)

    query_and_context = (
        "These docs can help you with your questions. If you have no answer, simply say 'I do not know'."
        f"Question: {user_query}\n"
        f"Relevant docs: {retrieved_docs_text_str}"
    )

    messages = [
        ("system", "You are an expert assistant providing information strictly based on the context provided to you. Your task is to answer questions or provide information only using the details given in the current context. Do not reference any external knowledge or information not explicitly mentioned in the context. If the context does not contain sufficient information to answer a question, clearly state that the information is not available in the provided context."),
        ("human", query_and_context)
    ]
    
    res = llm.invoke(messages)
    return res.content

#%% TESTING
# check if it only uses the provided context
user_query = "Where do the Olympic Games 2024 take place?"
rag_chat(user_query)
# %% try it out
user_query = "Where does Dracula live?"
rag_chat(user_query)
# %% RAG Chat with style
def rag_chat_add_style_language(user_query: str, k: int = 5, style: str = "formal", language: str = "english"):
    retrieved_docs = db_client.similarity_search(query=user_query, k=k)
    retrieved_docs_text = [doc.page_content for doc in retrieved_docs]
    retrieved_docs_text
    retrieved_docs_text_str = "\n".join(retrieved_docs_text)

    query_and_context = (
        "These docs can help you with your questions. If you have no answer, simply say 'I do not know'."
        f"Question: {user_query}\n"
        f"Relevant docs: {retrieved_docs_text_str}"
    )

    messages = [
        ("system", f"You are an expert assistant providing information strictly based on the context provided to you. Your task is to answer questions or provide information only using the details given in the current context. Do not reference any external knowledge or information not explicitly mentioned in the context. If the context does not contain sufficient information to answer a question, clearly state that the information is not available in the provided context. You should answer in a {style} style and in {language} language."),
        ("human", query_and_context)
    ]
    
    res = llm.invoke(messages)
    return res.content


# %%
user_query = "What happens after Dracula bites someone?"
# default of 5 docs does not hold the answer, try with 10
rag_chat_add_style_language(user_query, style="Shakespearean", language="english",  k=10)
# %%
