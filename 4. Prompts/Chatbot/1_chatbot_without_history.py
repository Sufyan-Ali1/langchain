from langchain_huggingface  import ChatHuggingFace, HuggingFacePipeline
import streamlit as st

llm = HuggingFacePipeline.from_model_id(
    model_id = 'TinyLlama/TinyLlama-1.1B-Chat-v1.0',
    task = 'text-generation',
    pipeline_kwargs=dict(
        temperature = 0.5,
        max_new_tokens= 100
    )
)

model = ChatHuggingFace(llm = llm)

while True:
    if user_input =="exit" or user_input =="Exit":
        break
    user_input = input("You : ")
    result = model.invoke(user_input)
    print("AI : ",result.content)


# Problem : No Memory for Chatbot
