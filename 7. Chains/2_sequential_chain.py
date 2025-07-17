from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
import os

# Load environment variables from .env
load_dotenv()

template1 = PromptTemplate(
    template = "Write a detailed report on {topic}",
    input_variables= ['topic']
)
template2= PromptTemplate(
    template = "write a 5 line summary on the following text. /n{text}",
    input_variables = ['text']

)

# Initialize the ChatOpenAI model
model = ChatOpenAI(
    model=os.getenv("GEMMA_MODEL"),  # Custom model hosted via Novita
    temperature=1
)


parser = StrOutputParser()

chain = template1|model|parser|template2|model|parser

result = chain.invoke({"topic":"Cricket"})

print(result)

chain.get_graph().print_ascii()