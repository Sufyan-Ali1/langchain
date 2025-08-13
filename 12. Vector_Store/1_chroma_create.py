from documents import *
from langchain_openai import OpenAIEmbeddings
from langchain.vectorstores import Chroma

docs = [doc1,doc2,doc3,doc4,doc5,doc6,doc7,doc8,doc9,doc10,doc11,doc12,doc13,doc14,doc15]

vector_store = Chroma(
    embedding_function= OpenAIEmbeddings(),
    persist_directory = "chroma_db",
    collection_name = 'sample'
)

vector_store.add_documents(docs)

vector_store.get(include = ['embeddings', 'documents', 'metadatas'])

vector_store.similarity_search(
    query="Who among these are a bowler",
    k=2
)

# score show distance
vector_store.similarity_search_with_score(
    query="Who among these are a bowler",
    k=2
)

vector_store.similarity_search_with_score(
    query="Who among these are a bowler",
    filter = {"team":"Karachi Kings"}, 
    k=2
)

vector_store.update_document(
    document_id="----",
    document = updated_document
)

vector_store.delete(ids=['-----'])