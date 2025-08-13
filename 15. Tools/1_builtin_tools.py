# 1st  Tool

# from langchain_community.tools import DuckDuckGoSearchRun

# search_tool = DuckDuckGoSearchRun()

# results = search_tool.invoke("PSL news")

# print(results)

# 2nd Tool

from langchain_community.tools import ShellTool

shell_tool = ShellTool()

results = shell_tool.invoke("dir")

print(results)
