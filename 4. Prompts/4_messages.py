from langchain_huggingface  import ChatHuggingFace, HuggingFacePipeline
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage


llm = HuggingFacePipeline.from_model_id(
    model_id = 'TinyLlama/TinyLlama-1.1B-Chat-v1.0',
    task = 'text-generation',
    pipeline_kwargs=dict(
        temperature = 0.5,
        max_new_tokens= 100
    )
)

model = ChatHuggingFace(llm = llm)

messages=[
    SystemMessage(content = "You are a helpful assistant"),
    HumanMessage(content = "Tell me about LangChain")

]


raw_output = model.invoke(messages)

if "<|assistant|>" in raw_output.content:
        reply = raw_output.content.split("<|assistant|>\n", 1)[-1].strip()
else:
        reply = raw_output.content.strip()

messages.append((AIMessage(content = reply)))

print(messages)