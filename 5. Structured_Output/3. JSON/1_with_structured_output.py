from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
from typing import TypedDict,Annotated,Optional
from pydantic import BaseModel,Field

# Load environment variables from .env
load_dotenv()

# Initialize the ChatOpenAI model
chatmodel = ChatOpenAI(
    model='google/gemma-3-27b-it',  # Custom model hosted via Novita
    temperature=1
)

# Define the structured output format
json_schema = {
    "title":"Review",
    "type":"object",
    "properties":{
        "key_themes":{
            "type":"array",
            "items": {
                "type":"string"
            },
            "description":"Write down all the key themes discussed in the review in a list"
        },
        "summary":{
            "type":"string",
            "description":"A brief summary of the review"
        },
        "sentiment":{
            "type":"string",
            "enum":["pos","neg"],
            "description":"Return sentiment of the review either negative,positive or neutral"
        },
        "pros":{
            "type":["array","null"],
            "items":{
                "type":"string"
            },
            "description":"Write down all the pros inside a list"
        }       
    },
    "required":["key_themes","summary","sentiment"]
}
# Wrap the model with structured output capability
structured_model = chatmodel.with_structured_output(json_schema)

# Input prompt
review_text = """I’ve been using the Samsung Galaxy A55 5G for about a month now, and honestly, I’m pleasantly surprised. I wasn’t expecting flagship-level performance, but for the price, it offers a great balance of features, design, and reliability.

The build feels premium with its aluminum frame and Gorilla Glass Victus+ front. The AMOLED screen is crisp and vibrant—even under bright sunlight. Everyday tasks like social media, browsing, and video streaming run flawlessly. Even light gaming is smooth.

Samsung’s One UI is clean and loaded with features, and I appreciate the regular software updates. The camera is good in daylight, and the night mode is usable but not perfect.

Pros:

Gorgeous 120Hz AMOLED Display : Smooth and colorful visuals.

Solid Build Quality : Feels like a premium device in hand.

Good Battery Life : Easily lasts a full day with moderate use.

Reliable Software Support : Samsung promises years of updates.

Decent Cameras : Great for casual photography.

Cons:

No Wireless Charging : A miss at this price point.

Average Low-Light Camera Performance : Night shots are grainy.

Bloatware Pre-installed : A few unnecessary apps out of the box.

No Charger in the Box : Just the cable, like many recent phones.
"""

# Invoke the model
result = structured_model.invoke(review_text)

# Print the structured result
print(result)
