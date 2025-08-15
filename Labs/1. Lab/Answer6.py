from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from pydantic import BaseModel, Field
from langchain_core.output_parsers import PydanticOutputParser
from typing import List, Dict
import os
from datetime import datetime
import json

load_dotenv()

model = ChatOpenAI(model=os.getenv("DEEPSEEK_MODEL"), temperature=0)


def log(log_entry):
    log_entry["timestamp"] = datetime.utcnow().isoformat()
    with open(r"C:\Users\sufya\Desktop\Learning\Langchain\Labs\1. Lab\Answer6_chat_log.json", "a") as f:  # use "a" to append
        f.write(json.dumps(log_entry) + "\n")

class Output(BaseModel):
    user_message: str = Field(description="The original query or message provided by the user.")
    intent: List[str] = Field(description="A list of identified intents inferred from the user's query.")
    entities: Dict[str, str] = Field(description="Key-value pairs of named entities extracted from the user's query.")
    bot_response: str = Field(description="The AI-generated response to the user's query.")

outputparser = PydanticOutputParser(pydantic_object=Output)

query = "Book me a flight from Karachi to Lahore tomorrow morning."

prompt = PromptTemplate(
    template="Generate a Response for:\n{query}\n\nOutput structure should be:\n{format_instructions}",
    input_variables=["query"],
    partial_variables={"format_instructions": outputparser.get_format_instructions()}
)

chain = prompt | model | outputparser

response = chain.invoke({"query": query})


log(response.model_dump())

print(response.bot_response)
