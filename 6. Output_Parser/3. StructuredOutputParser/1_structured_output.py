from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain.output_parsers import StructuredOutputParser, ResponseSchema

# Load environment variables from .env
load_dotenv()

# Initialize the ChatOpenAI model
chatmodel = ChatOpenAI(
    model='google/gemma-3-27b-it',  # Custom model hosted via Novita
    temperature=1
)

schema = [
    ResponseSchema(name ="fact_1",description = "Fact 1 about the topic"),
    ResponseSchema(name ="fact_2",description = "Fact 2 about the topic"),
    ResponseSchema(name ="fact_3",description = "Fact 3 about the topic")
]

parser = StructuredOutputParser.from_response_schemas(schema)


template = PromptTemplate(
    template = "Give me 3 fact about {topic} \n{format_instruction}",
    input_variables= ["topic"],
    partial_variables = {'format_instruction':parser.get_format_instructions()}
)

# ------------------ Method 1

# prompt = template.format({"topic":"Black hole"})

# result = chatmodel.invoke(prompt)

# final_result = parser.parse(result.content)

# ------------------ Method 2

chain = template | chatmodel |parser

final_result = chain.invoke({"topic":"Black hole"})

print(final_result)
print(type(final_result))