from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
from typing import TypedDict

# Load environment variables from .env
load_dotenv()

# Initialize the ChatOpenAI model
chatmodel = ChatOpenAI(
    model='google/gemma-3-27b-it',  # Custom model hosted via Novita
    temperature=1,
    max_tokens=100
)

# Define the structured output format
class Review(TypedDict):
    summary: str
    sentiment: str

# Wrap the model with structured output capability
structured_model = chatmodel.with_structured_output(Review)

# Input prompt
review_text = """I had the most incredible dining experience at this restaurant! 
The food was absolutely delicious - every dish was perfectly prepared and beautifully 
presented. Our server was attentive, knowledgeable, and made excellent recommendations. 
The atmosphere was elegant yet comfortable. I can't wait to come back and try more items 
from their menu. Definitely a new favorite spot!
"""

# Invoke the model
result = structured_model.invoke(review_text)

# Print the structured result
print(result)
