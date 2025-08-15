from langchain_openai import ChatOpenAI
from langchain.output_parsers import StructuredOutputParser,ResponseSchema
from langchain_core.prompts import PromptTemplate
from dotenv import load_dotenv
import os

load_dotenv()

job_description = """We are looking for passionate and driven Associate Software Engineers to join our team. This role is ideal for fresh graduates or individuals with up to 6 months of experience who are eager to learn, solve complex problems, and contribute to software development projects.

Requirements

Core Responsibilities

Assist in the design, development, and maintenance of software applications
Write clean, efficient, and well-documented code
Debug and resolve technical issues under the guidance of senior engineers
Participate in code reviews and contribute to software improvement
Work collaboratively with cross-functional teams to deliver high-quality software solutions
Stay updated with the latest industry trends and technologies

Qualification

Bachelor's degree in Computer Science, Software Engineering, or a related field
Proficiency in programming languages such as Java, Python, C#, or JavaScript
Understanding of databases, APIs, and software development life cycle (SDLC)
Strong analytical and problem-solving skills
Ability to work independently and in a team environment

"""


model = ChatOpenAI(
    model=os.getenv("GEMMA_MODEL"),
    temperature=1
)

sturucture = [
    ResponseSchema(name = "Required_Skills",description="Give me the list of required skills from the Job description"),
    ResponseSchema(name = "Experience Level",description ="From this job description Give me Experience Level that required"),
    ResponseSchema(name = "Summary", description = "Summary of the Jon")
]

parser = StructuredOutputParser.from_response_schemas(sturucture)

template = PromptTemplate(
    template = """
    Extract the Following information from the job description:

    {format_instruction}

    Job Descripton:

    {job_description}
    """,
    input_variables=["job_description"],
    partial_variables = {'format_instruction':parser.get_format_instructions()}
)



chain = template | model | parser

resul = chain.invoke({"job_description":job_description})

print(result)


