from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser,PydanticOutputParser
from dotenv import load_dotenv
from pydantic import BaseModel,Field
import os
from typing import List

load_dotenv()

model = ChatOpenAI(
    model = os.getenv("GEMMA_MODEL"),
    temperature = 1.5
)

def tweet_score(tweet: str) -> float:
    """
    Custom scoring function to rank a tweet based on:
    - Length (ideal < 280 chars, reward for brevity)
    - Presence of hashtags (boost)
    - Presence of emojis (boost)
    - Starting with a strong word (boost)
    """
    max_len = 280
    score = 0

    # 1. Length bonus (shorter is better)
    length_penalty = min(len(tweet) / max_len, 1)
    score += 1 - length_penalty  # shorter = higher score

    # 2. Hashtag bonus
    if "#" in tweet:
        score += 0.2

    # 3. Emoji bonus
    emoji_chars = [c for c in tweet if ord(c) > 10000]
    if emoji_chars:
        score += 0.2

    # 4. Starts with strong words
    strong_starts = ["breaking", "alert", "🔥", "just in", "new", "update"]
    if tweet.lower().startswith(tuple(strong_starts)):
        score += 0.1

    return round(min(score, 1.0), 3)  # limit max score to 1.0

class Tweet(BaseModel):
    tweets : List[str] = Field(description="List of tweets")

parser = PydanticOutputParser(pydantic_object = Tweet)
strparser = StrOutputParser()
template = PromptTemplate(
    template ="Generate the Tweets from article Headlines \n{headlines} \n {format_instruction}",
    input_variables = ["headlines"],
    partial_variables = {"format_instruction":parser.get_format_instructions()}
)

chain = template | model | parser

result = chain.invoke({"headlines":"OpenAI Releases GPT-5 with Real-Time Web Access and Multimodal Capabilities"})


passed_tweets = [tweet for tweet in result.tweets if tweet_score(tweet)>0.60]

