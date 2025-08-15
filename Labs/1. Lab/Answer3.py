from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field
from typing import List,Optional
from dotenv import load_dotenv
import os
from datetime import datetime


load_dotenv()

model = ChatOpenAI(
    model = os.getenv("GEMMA_MODEL"),
    temperature = 0
)

class Legal(BaseModel):
    names: Optional[List[str]] = Field(description = "list of the names in paragraph")
    dates: Optional[List[datetime]] = Field(description="Give the list of dates in paragraph")
    legal_entites : Optional[List[str]] = Field(description = "list of the legal entites in paragraph")

parser = PydanticOutputParser(pydantic_object = Legal)

paragraph = """This Agreement is entered into on July 17, 2025, by and between Neuradev Studio (Private) Limited, a company duly incorporated under the laws of Pakistan with its registered office at Office #12, Tech Tower, Lahore, hereinafter referred to as the "Service Provider", and AlphaEdge Solutions LLC, a limited liability company organized and existing under the laws of the State of Delaware, with its principal place of business located at 450 Mission Street, San Francisco, CA 94105, hereinafter referred to as the "Client". Both parties collectively referred to as the "Parties", agree to be bound by the terms and conditions set forth herein.

"""

template = PromptTemplate(
    template = """
    Exteact these information from the Legal Paragraph

    {format_instruction}

    Legal Paragraph:

    {paragraph}
    """,
    input_variables = ["paragraph"],
    partial_variables = {"format_instruction":parser.get_format_instructions()}
)

chain = template | model | parser

result = chain.invoke({"paragraph":paragraph})
print(result)