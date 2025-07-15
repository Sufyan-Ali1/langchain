from dotenv import load_dotenv
from huggingface_hub import InferenceClient
from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace
import os

load_dotenv()

# Create a custom client with a longer timeout
client = InferenceClient(
    model="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    token=os.getenv("HUGGINGFACE_API_KEY"),
    timeout=60  # ← Increase timeout to 60 seconds
)

# Pass this client to HuggingFaceEndpoint
llm = HuggingFaceEndpoint(
    repo_id="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    task="text-generation",
    client=client
)

# Wrap into Chat model
model = ChatHuggingFace(llm=llm)

# Invoke and print result
result = model.invoke("What is the capital of Pakistan?")
print(result.content)
