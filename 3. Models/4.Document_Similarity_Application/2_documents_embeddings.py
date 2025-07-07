from langchain_huggingface import HuggingFaceEmbeddings
import json
import os

file_path = '4.Document_Similarity_Application/document.json'

# Check if file exists
if os.path.exists(file_path):
    with open(file_path, 'r') as f:
        try:
            documents = json.load(f)
        except json.JSONDecodeError:
            print("Error: The file is not a valid JSON.")
else:
    print(f"Error: File '{file_path}' not found.")

embedding = HuggingFaceEmbeddings(model_name = "sentence-transformers/all-MiniLM-L6-v2")


embeddings = []
for document in documents:
    vector = embedding.embed_query(document["text"])
    embeddings.append({'id':document["id"], 'embedding' : vector})
    

file_path = "4.Document_Similarity_Application/embeddings.json"

# Save dictionary to JSON safely
try:
    with open(file_path, "w") as f:
        json.dump(embeddings, f, indent=2)
    print(f"Dictionary saved successfully to '{file_path}'")
except Exception as e:
    print(f"Error saving file: {e}")
