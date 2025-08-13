from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain.schema.runnable import RunnableSequence
from dotenv import load_dotenv
import os
load_dotenv()

model = ChatOpenAI(
    model = os.getenv("DEEPSEEK_MODEL"),
    temperature = 1
)

prompt = PromptTemplate(
    template = "Write a joke about {topic}",
    input_variables = ['topic']
)

parser = StrOutputParser()

chain = RunnableSequence(prompt,model,parser)

result = chain.invoke({"topic":"AI"})

print(result)