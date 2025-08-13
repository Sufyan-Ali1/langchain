from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain.schema.runnable import RunnableSequence,RunnableParallel,RunnableLambda,RunnablePassthrough
from dotenv import load_dotenv
import os
load_dotenv()

def word_count(joke):
    return len(joke.split())

model = ChatOpenAI(
    model = os.getenv("DEEPSEEK_MODEL"),
    temperature = 1
)

prompt = PromptTemplate(
    template = "Write a joke about {topic}",
    input_variables = ['topic']
)

parser = StrOutputParser()

joke_gen_chain = RunnableSequence(prompt,model,parser)

parallel_chain = RunnableParallel({
    "joke":RunnablePassthrough(),
    "word count": RunnableLambda(word_count)
    # or "word count": RunnableLambda(lambda x : len(x.split()))
})

final_chain = RunnableSequence(joke_gen_chain,parallel_chain)
result = final_chain.invoke({"topic":"cricket"})

print(result)