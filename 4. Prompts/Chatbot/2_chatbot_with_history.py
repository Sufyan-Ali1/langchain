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
chat_history = [
    SystemMessage(content = "You are helpful AI assistant")
]

while True:
    user_input = input("You : ")

    if user_input =="exit" or user_input =="Exit":
        break
    chat_history.append(HumanMessage(content = user_input))
    raw_output = model.invoke(chat_history)


    if "<|assistant|>" in raw_output.content:
        reply = raw_output.content.split("<|assistant|>\n", 1)[-1].strip()
    else:
        reply = raw_output.content.strip()

    chat_history.append((AIMessage(content = reply)))
    print(reply)


