from langchain_openai import ChatOpenAI
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv
import os

load_dotenv()

model = ChatOpenAI(
    model = os.getenv("GEMMA_MODEL"),
    temperature = 1
)

prompt = PromptTemplate(
    template = "Suggest a catchy blog title about {topic}",
    input_variables = ["topic"]
)

chain = LLMChain(llm = model, prompt = prompt)

topic = "Cricket"

output = chain.run(topic)

print(output)