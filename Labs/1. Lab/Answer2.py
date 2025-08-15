from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv
import os

load_dotenv()

model = ChatOpenAI(
    model = os.getenv("Gemma_Model"),
    temperature = 0
)

text = """
Artificial Intelligence is rapidly transforming the world. From healthcare to transportation, AI is being integrated into all sectors to improve efficiency and decision-making.
"""

parser = StrOutputParser()

translate_prompt = PromptTemplate(
    template= "Translate this text into french\n\n{text}",
    input_variables= ["text"]
)

summary_prompt = PromptTemplate(
    template = "Summarize this french text in English\n\n{text}",
    input_variables = ["text"]
)

first_chain = translate_prompt | model | parser
second_chain = summary_prompt | model | parser

final_chain = first_chain  | second_chain

result = final_chain.invoke({"text":text})

print("Final Summary in English:\n")
print(result)