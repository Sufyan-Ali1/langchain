from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage
import requests
from dotenv import load_dotenv
import os

load_dotenv()
# Tool Creatiom
@tool
def multiply(a:int,b:int)->int:
    """Given 2 numbers a and b this tool returns their product"""
    return a*b

llm =  ChatOpenAI(
    model = os.getenv("DEEPSEEK_MODEL"),
    temperature = 1
)
# tool binding  
llm_with_tool = llm.bind_tools([multiply])

# Tool Calling
result = llm_with_tool.invoke("Hey how are you")

query = HumanMessage("Multiply 3 with 10")
messages = [query]
result1  = llm_with_tool.invoke(messages)
messages.append(result1)


# Tool executing
result2 = multiply.invoke(result1.tool_calls[0]['args'])#it returns only result

tool_message = multiply.invoke(result1.tool_calls[0])# but it return tool message
messages.append(tool_message)

result3=llm_with_tool.invoke(messages).content
print(result3)