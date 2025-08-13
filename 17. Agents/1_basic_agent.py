from langchain_openai import ChatOpenAI
from langchain_core.tools import tool
import requests
from langchain_community.tools import DuckDuckGoSearchRun
from dotenv import load_dotenv
import os
from langchain.agents import create_react_agent,AgentExecutor
from langchain import hub


load_dotenv()
# Tool
search_tool = DuckDuckGoSearchRun()

@tool
def get_weather_data(city:str)->str:
    """
    This Function fetches the currect weather data for a given city
    """
    url = f'https://api.weatherstack.com/current?access_key=4d1d8ae207a8c845a52df8a67bf3623e&query={city}'
    response = requests.get(url)

    return response.json()

# LLM
llm = ChatOpenAI(model=os.getenv("DEEPSEEK_MODEL"))

# Step 2 pull the react prompt from the langhchain hub (reAct=Reasoing + Action) reAct is a design pattern which is one of most famous
prompt =hub.pull("hwchase17/react")

# Step 3 Create the ReAct agent manuallty with the pulled prompt
agent = create_react_agent(
    llm=llm,
    tools=[search_tool,get_weather_data],
    prompt=prompt
)

# Step 4 Wrap it to AgentExecutor
agent_executor = AgentExecutor(
    agent=agent,
    tools=[search_tool,get_weather_data],
    verbose = True # agent jo bhi soch rha hga wo hamy dkhai bhi dega
)

res = agent_executor.invoke({"input": "Find the capital of Pakistan, then find it's current weather condition"})


print(res)
print(res["output"])