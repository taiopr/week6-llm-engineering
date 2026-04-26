"""
RAW API vs LANGCHAIN — Direct comparison of the same operation.
Read both implementations. They produce identical results.
"""

import os
from dotenv import load_dotenv
from openai import OpenAI
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate

load_dotenv()

TOPIC = "vector embeddings"
AUDIENCE = "junior developer"

# ─────────────────────────────────────────────
# RAW API IMPLEMENTATION
# ─────────────────────────────────────────────

def explain_raw(topic: str, audience: str) -> str:
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        temperature=0.3,
        max_tokens=150,
        messages=[
            {"role": "system", "content": "You are a technical educator."},
            {"role": "user", "content": f"Explain {topic} to a {audience}. Under 80 words."}
        ]
    )
    return response.choices[0].message.content.strip()


# ─────────────────────────────────────────────
# LANGCHAIN IMPLEMENTATION
# ─────────────────────────────────────────────

def explain_langchain(topic: str, audience: str) -> str:
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.3, max_tokens=150)
    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a technical educator."),
        ("human", "Explain {topic} to a {audience}. Under 80 words.")
    ])
    chain = prompt | llm
    response = chain.invoke({"topic": topic, "audience": audience})
    return response.content.strip()


# ─────────────────────────────────────────────
# RUN BOTH AND COMPARE
# ─────────────────────────────────────────────

print("RAW API:")
print(explain_raw(TOPIC, AUDIENCE))

print("\nLANGCHAIN:")
print(explain_langchain(TOPIC, AUDIENCE))

print("\n--- What's different? ---")
print("""
Raw API:
  + You see the exact dict being sent to the API
  + No hidden system prompts or injected content
  + One dependency: openai
  + Debugging = print the request payload

LangChain:
  + PromptTemplate validates variables at definition time
  + LCEL pipe chains are composable and reusable
  + Swapping models = change one constructor argument
  - You don't see what's actually sent to the API without inspecting
  - One more abstraction layer when something breaks
  - Heavier dependency tree
""")