from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain.schema.runnable import RunnableParallel
import os

# Load environment variables from .env
load_dotenv()

explanation_prompt = PromptTemplate(
    template = "Explain a {topic} in detail",
    input_variables= ['topic']
)
notes_prompt = PromptTemplate(
    template = "Generate short and simple notes the following text. \notes_prompt\notes_prompt\n{text}",
    input_variables = ['text']

)
mcqs_prompt = PromptTemplate(
    template = "Generate 5 tricky MCQ's from the following text. \n{text}",
    input_variables = ['text']

)
merge_prompt = PromptTemplate(
    template = "Merge the provided notes and quiz into a single document \nNotes -> {notes} \nQuiz{quiz}",
    input_variables = ['notes','quiz']

)
# Initialize the ChatOpenAI model
gemma_model = ChatOpenAI(
    model=os.getenv("GEMMA_MODEL"),  # Custom model hosted via Novita
    temperature=1
)

deepseek_model = ChatOpenAI(
    model=os.getenv("DEEPSEEK_MODEL"),  # Custom model hosted via Novita
    temperature=1
)

parser = StrOutputParser()

topic_chain = explanation_prompt|deepseek_model|parser

parallel_chain = RunnableParallel({
    'notes':notes_prompt|gemma_model|parser,# notes is the name of the chain
    "quiz":mcqs_prompt|deepseek_model|parser# quiz is the name of the chain
})

merg_chain = merge_prompt | gemma_model| parser

final_chain = topic_chain|parallel_chain|merg_chain

result = final_chain.invoke({"topic":"ML"})

print(result)