from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
import os

# Load environment variables
load_dotenv()

# No need to pass API key or base URL explicitly — it'll use the env vars
chatmodel = ChatOpenAI(
    model='deepseek/deepseek-v3-0324',  # Custom model from Novita
    temperature=0,
    max_tokens=100  # Use 'max_tokens', not 'max_completion_tokens'
)

# Invoke the model
result = chatmodel.invoke("What is the capital of Pakistan?")
print(result.content)  # .content gives just the text response
