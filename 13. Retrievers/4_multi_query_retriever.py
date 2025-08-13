from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_openai import ChatOpenAI
from langchain.retrievers.multi_query import MultiQueryRetriever
from dotenv import load_dotenv
import os

load_dotenv()

doc1 = Document(
    page_content="The Great Wall of China stretches over 13,000 miles and was originally built to protect Chinese states from invasions."
)

doc2 = Document(
    page_content="Photosynthesis converts sunlight, water, and carbon dioxide into glucose and oxygen, sustaining most life on Earth."
)

doc3 = Document(
    page_content="Machine learning is a subset of artificial intelligence that allows systems to learn and improve from experience without being explicitly programmed."
) 

doc4 = Document(
    page_content="The Amazon Rainforest, often referred to as the 'lungs of the Earth,' produces over 20% of the world's oxygen."
)

doc5 = Document(
    page_content="The theory of relativity, developed by Albert Einstein, revolutionized our understanding of space, time, and gravity."
)



docs = [doc1,doc2,doc3,doc4,doc5]

vector_store = FAISS.from_documents(
    documents = docs
    embedding_function= OpenAIEmbeddings(),
    collection_name = 'sample'
)

# simple retriever
retriever = vector_store.as_retriever(
    search_type="mmr",
    search_kwargs={'k':2,"lambda_mult":1}#k=>top results, lambda_mult => relevance-diversity balance 0=>very diverse 1=> act as a simple rertriever
    )

# Multi Query Retriever
multi_query_retriever = MultiQueryRetriever.from_llm(
    retriever = vector_store.as_retriever(search_kwargs={"k":5}),
    llm =ChatOpenAI(model = os.getenv("DEEPSEEK_MODEL")) # to generate different version of queries
)

query = "What is Machine Learning"

results = multi_query_retriever.invoke(query)

for i , doc in enumerate(results):
    print(f"\n------------Result {i+1}---------------")
    print(f"Content:\n{doc.page_content}...")