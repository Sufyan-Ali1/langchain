from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain.schema.runnable import RunnableSequence,RunnableParallel
from dotenv import load_dotenv
import os

load_dotenv()

model = ChatOpenAI(
    model = os.getenv("DEEPSEEK_MODEL"),
    temperature = 1
)

prompt1 = PromptTemplate(
    template = "Generate a tweet about {topic}",
    input_variables = ['topic']
)

prompt2 = PromptTemplate(
    template = "Generate a linkedin post about {topic}",
    input_variables = ['topic']
)

parser = StrOutputParser()

parallel_chain = RunnableParallel({
    "tweet": RunnableSequence(prompt1,model,parser),
    "post" : RunnableSequence(prompt2,model,parser),
    })

result = parallel_chain.invoke({"topic":"AI"})

print(result)