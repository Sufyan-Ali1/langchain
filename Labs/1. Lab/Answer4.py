from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import PydanticOutputParser, StrOutputParser
from langchain.schema.runnable import RunnableParallel, RunnableBranch, RunnableLambda
from dotenv import load_dotenv
from pydantic import Field, BaseModel
import os

load_dotenv()

# Load model
model = ChatOpenAI(
    model=os.getenv("DEEPSEEK_MODEL"),
    temperature=1
)

# Define Pydantic model
class Question(BaseModel):
    question: bool = Field(description="text is question or not")

# Output parsers
parser = PydanticOutputParser(pydantic_object=Question)
strparser = StrOutputParser()

# Prompt to determine if it's a question
prompt = PromptTemplate(
    template="{text}\n{format_instruction}",
    input_variables=['text'],
    partial_variables={'format_instruction': parser.get_format_instructions()}
)

# Summary prompt
summary_template = PromptTemplate(
    template="Summarize This Question:\n\nQuestion: {text}",
    input_variables=['text']
)

# Formal rewrite prompt
formal_template = PromptTemplate(
    template="Rewrite the following in formal tone:\n\n{text}",
    input_variables=['text']
)

# Routing logic using branching
branch_chain = RunnableBranch(
    (lambda x: x.question == True, summary_template | model | strparser),
    (lambda x: x.question == False, formal_template | model | strparser),
    RunnableLambda(lambda x: "Could not understand the text.")
)

# Combine chains
chain = prompt | model | parser
final_chain = chain | branch_chain


# Example 1: input is a question
input1 = {"text": "What are the benefits of using transformers in NLP?"}
print("Output (Question):", final_chain.invoke(input1))

# Example 2: input is a statement
input2 = {"text": "i want to apply for this job and hope you will consider my application"}
print("Output (Statement):", final_chain.invoke(input2))
