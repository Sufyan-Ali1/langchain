from langchain_huggingface import HuggingFaceEmbeddings

embedding = HuggingFaceEmbeddings(model_name = "sentence-transformers/all-MiniLM-L6-v2")

text ="Delhi is the capital of india"

docs = [
    "My name is sufyan Ali",
    "I am from Pakistan",
    "I Complete my Bachelor's from FAST NUCES LAHORE"
]

vector = embedding.embed_query(text)
vector = embedding.embed_documents(docs)

print(str(vector))