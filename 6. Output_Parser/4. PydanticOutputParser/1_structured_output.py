from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field

# Load environment variables from .env
load_dotenv()

# Initialize the ChatOpenAI model
chatmodel = ChatOpenAI(
    model='google/gemma-3-27b-it',  # Custom model hosted via Novita
    temperature=1
)

class Person(BaseModel):
    name : str = Field(description = "Name of the Person")
    age : int = Field(gt = 18, description = "Age of the Person")
    city : str = Field(description = "Name of the City person Belongs to")

parser = PydanticOutputParser(pydantic_object = Person)


template = PromptTemplate(
    template = "Generate the name age and city of the fictional {topic} Person\n{format_instruction}",
    input_variables= ["topic"],
    partial_variables = {'format_instruction':parser.get_format_instructions()}
)

# ------------------ Method 1

# prompt = template.format({"topic":"Black hole"})

# result = chatmodel.invoke(prompt)

# final_result = parser.parse(result.content)

# ------------------ Method 2

chain = template | chatmodel |parser

final_result = chain.invoke({"topic":"Pakistani"})

print(final_result)
