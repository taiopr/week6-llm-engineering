import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, AIMessage

load_dotenv()

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.7)

# This IS memory - a plain list of messages appended after each turn.
# ConversationBufferMemory was just a class wrapper around exactly this.
# Now you see the mechanism directly.
chat_history = []

def chat(user_input: str) -> str:
    global chat_history # Explicitly tell Python this refers to the module-level list
    # Add the new user messahe to history
    chat_history.append(HumanMessage(content=user_input))

    # Send the full history every time - this is what "memory" means
    response = llm.invoke(chat_history)

    # Store the model's reply so next turn includes it
    chat_history.append(AIMessage(content=response.content))

    return response.content

print("=== CONVERSATION WITH MEMORY ===\n")

# First turn
response1 = chat("My name is Marco and I'm learning about AI embeddings.")
print(f"Turn 1: {response1}\n")

# Second turn - model should remember the name and context
response2 = chat("What was i just telling you I was learning about?")
print(f"Turn 2: {response2}\n")

# Third turn - tests deeper context retention
response3 = chat("And what's my name?")
print(f"Turn 3: {response3}\n")

# Inspect what's actually stored in memory
print("=== MEMORY CONTENTS ===")
print(f"Number of messages stored: {len(chat_history)}")
for i, msg in enumerate(chat_history):
    print(f"   [{i}] {type(msg).__name__}: {msg.content[:80]}...")