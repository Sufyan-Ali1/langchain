from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser
# Load environment variables from .env
load_dotenv()

# Initialize the ChatOpenAI model
chatmodel = ChatOpenAI(
    model='google/gemma-3-27b-it',  # Custom model hosted via Novita
    temperature=1
)

parser = JsonOutputParser()

template = PromptTemplate(
    template = "Give me a name ,age and city of a fictional person \n{format_instruction}",
    input_variables= [],
    partial_variables = {'format_instruction':parser.get_format_instructions()}
)

# ------------------ Method 1

# prompt = template.format()

# result = chatmodel.invoke(prompt)

# final_result = parser.parse(result.content)

# ------------------ Method 2

chain = template | chatmodel |parser

final_result = chain.invoke({})

print(final_result)
print(type(final_result))