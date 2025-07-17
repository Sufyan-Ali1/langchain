from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain.schema.runnable import RunnableParallel,RunnableBranch,RunnableLambda
from langchain_core.output_parsers import PydanticOutputParser
from pydantic import BaseModel,Field
from typing import Literal
import os

# Load environment variables from .env
load_dotenv()
parser = StrOutputParser()


class Feedback(BaseModel):
    sentiment : Literal["positive","negative"] = Field(description="Sentiment of the Feedback")
pydantic_parser = PydanticOutputParser(pydantic_object=Feedback)

feedback_prompt = PromptTemplate(
    template = "Classify the sentiment of the following feedback text into positive or negative \n{feedback}\n{format_instruction}",
    input_variables= ['feedback'],
    partial_variables = {"format_instruction":pydantic_parser.get_format_instructions()}
)

# Initialize the ChatOpenAI model
model = ChatOpenAI(
    model=os.getenv("GEMMA_MODEL"),  # Custom model hosted via Novita
    temperature=1
)

classifier_chain = feedback_prompt|model|pydantic_parser

prompt2 = PromptTemplate(
    template = "Write an appropriate one line reponse to this positive Feedback\n{feedback}",
    input_variables = ["feedback"]
)
prompt3 = PromptTemplate(
    template = "Write an appropriate one line reponse to this Negative Feedback\n{feedback}",
    input_variables = ["feedback"]
)
branch_chain = RunnableBranch(
    #(Condition,Chain)
    (lambda x:x.sentiment == 'positive',prompt2|model|parser),
    (lambda x:x.sentiment == 'negative',prompt3|model|parser), 
    RunnableLambda(lambda x:"Could not find sentiment")
)

chain = classifier_chain|branch_chain

result = chain.invoke({"feedback":"This is a good smartphone"})
print(result)
chain.get_graph().print_ascii()