from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain.schema.runnable import RunnableSequence,RunnableLambda,RunnablePassthrough,RunnableBranch
from dotenv import load_dotenv
import os
load_dotenv()


model = ChatOpenAI(
    model = os.getenv("DEEPSEEK_MODEL"),
    temperature = 1
)

prompt1 = PromptTemplate(
    template = "Write a detailed report on {topic}",
    input_variables = ['topic']
)

prompt2 = PromptTemplate(
    template = "Summarize this text {text}",
    input_variables = ['text']
)

parser = StrOutputParser()

report_gen_chain = RunnableSequence(prompt1,model,parser)

branch_chain = RunnableBranch(   
    (lambda x: len(x.split()) > 500, RunnableSequence(prompt2,model,parser)),
    RunnablePassthrough()
)

final_chain = RunnableSequence(report_gen_chain,branch_chain)
result = final_chain.invoke({"topic":"cricket"})

print(result)