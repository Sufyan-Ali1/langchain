from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain.schema.runnable import RunnableSequence,RunnableParallel,RunnablePassthrough
from dotenv import load_dotenv
import os
load_dotenv()

model = ChatOpenAI(
    model = os.getenv("DEEPSEEK_MODEL"),
    temperature = 1
)

prompt1 = PromptTemplate(
    template = "Write a joke about {topic}",
    input_variables = ['topic']
)

prompt2 = PromptTemplate(
    template = "Explain a joke {joke}",
    input_variables = ['joke']
)

parser = StrOutputParser()

joke_gen_chain = RunnableSequence(prompt1,model,parser)
parallel_chain = RunnableParallel({
    "joke":RunnablePassthrough(),
    "expanation":RunnableSequence(prompt2,model,parser)  
})
final_chain = RunnableSequence(joke_gen_chain,parallel_chain)
result = final_chain.invoke({"topic":"cricket"})

print(result)