from langchain_huggingface import HuggingFaceEmbeddings
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import json
import os

def load_embeddings(file_path = "4.Document_Similarity_Application/embeddings.json"):
    if os.path.exists(file_path):
        try:
            with open(file_path,"r") as f:
                try:
                    embeddings = json.load(f)
                except json.JSONDecodeError:
                    print("Error: The file is not a valid JSON.")
        except Exception as e:
            print(f"Error loading file: {e}")
    embedding_vectors = [item["embedding"] for item in embeddings]

    return embedding_vectors

def load_documents(file_path = '4.Document_Similarity_Application/document.json'):
    if os.path.exists(file_path):
        with open(file_path, 'r') as f:
            try:
                documents = json.load(f)
            except json.JSONDecodeError:
                print("Error: The file is not a valid JSON.")
    else:
        print(f"Error: File '{file_path}' not found.")
    return documents


def get_embeddings(query):
    embedding = HuggingFaceEmbeddings(model_name = "sentence-transformers/all-MiniLM-L6-v2")
    query_embedding = embedding.embed_query(query)
    return query_embedding


#input
query = input("Enter any query to find a document:\n")

query_embedding = get_embeddings(query)

embedding_vectors=load_embeddings()

similarities = cosine_similarity([query_embedding],embedding_vectors)

most_similar_index = np.argmax(similarities)
# load Documents
documents = load_documents()

print(f"Your Answer is in this Document:\nSimilarity : {int((similarities[0][most_similar_index])*100)}\n",documents[most_similar_index]['text'])