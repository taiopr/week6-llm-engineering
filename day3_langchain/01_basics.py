import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage

load_dotenv()

# This wraps the OpenAI client — notice you don't pass messages manually
llm = ChatOpenAI(
    model="gpt-4o-mini",
    temperature=0.7,
    max_tokens=300
)

# Option 1: Pass a plain string
response = llm.invoke("What is a vector embedding?")
print("Plain string invoke:")
print(response.content)
print()

# Option 2: Pass structured messages — closer to raw API format
messages = [
    SystemMessage(content="You are a concise technical assistant."),
    HumanMessage(content="What is a vector embedding? One paragraph.")
]
response = llm.invoke(messages)
print("Structured messages invoke:")
print(response.content)
print()

# Inspect the full response object — notice what LangChain adds vs raw API
print("Response type:", type(response))
print("Response metadata:", response.response_metadata)