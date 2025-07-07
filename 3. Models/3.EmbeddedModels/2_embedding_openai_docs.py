from langchain_openai import OpenAIEmbeddings
from dotenv import load_dotenv

load_dotenv()

embedding = OpenAIEmbeddings(model ='text-embedding-3-large', dimension =32)

docs = [
    "My name is sufyan Ali",
    "I am from Pakistan",
    "I Complete my Bachelor's from FAST NUCES LAHORE"
]

result = embedding.embed_documents(docs)

print(str(result))