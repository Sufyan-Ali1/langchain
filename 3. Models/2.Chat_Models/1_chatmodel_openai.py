from langchain_openai import ChatOpenAI
from dotenv import load_dotenv #for import environment variables

load_dotenv()

chatmodel=ChatOpenAI(model ='gpt-4',temperature=0,max_completion_tokens=10)

result = chatmodel.invoke("What is the capital of pakistan")

print(result)
