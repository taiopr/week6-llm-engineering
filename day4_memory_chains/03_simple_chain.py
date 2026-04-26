import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

load_dotenv()

llm = ChatOpenAI("model=gpt-40-mini", temperature=0.3)
parser = StrOutputParser()  # Extracts .content from the response object - gives you a plain string

# ── Single step chain ──────────────────────────────────────────

summarise_prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a precise technical summariser."),
    ("human", "Summarise the following in 3 bullet points:\n\n{text}")
])

single_chain = summarise_prompt | llm | parser

result = single_chain.invoke({"text": "LangChain is a framework for building LLM-powered applications. It provides abstractions for prompts, memory, chains, and agents. It integrates with many LLM providers and vector stores."})
print("Single chain output:")
print(result)
print()

# ── Two step chain ─────────────────────────────────────────────
# Step 1 output becomes Step 2 input
# RunnablePassthrough lets you pass data through unchanged

from langchain_core.runnables import RunnablePassthrough

translate_prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a translator. Translate input text to Spanish."),
    ("human", "{text}")
])

# Chain: summaise -> translate
# We need to map step1's string output into step2's expected dict format
two_step_chain = (
    summarise_prompt
    | llm
    | parser
    | (lambda text: {"text": text})  # Wrap string output as dict for next prompt
    | translate_prompt
    | llm
    | parser
)

result = two_step_chain.invoke({"text": "Machine learning models learn patterns from data. They require training exmamples and an optimisation objective. The training process adjusts model parameters to minimise a loss function. "})
print("Two step chain output (summarise -> translate):")
print(result)