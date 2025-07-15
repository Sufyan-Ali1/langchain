from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
# Load environment variables from .env
load_dotenv()

# Initialize the ChatOpenAI model
chatmodel = ChatOpenAI(
    model='google/gemma-3-27b-it',  # Custom model hosted via Novita
    temperature=1
)

template1 = PromptTemplate(
    template = "Write a detailed report on {topic}",
    input_variables= ['topic']
)
template2= PromptTemplate(
    template = "write a 5 line summary on the following text. /n{text}",
    input_variables = ['text']

)

parser = StrOutputParser()

chain = template1|chatmodel | parser | template2 |chatmodel |parser

result = chain.invoke({"topic":"black hole"})

print(result)
