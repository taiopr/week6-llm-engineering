import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage

load_dotenv()

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.3)

conversation_turns = [
    "My name is Marco. I'm a Houdini TD transitioning into AI engineering.",
    "I've been studying Python, APIs, and LLM integration for about 6 weeks.",
    "Today I'm learning about LangChain memory chains.",
    "I'm based in Barcelona and targetting Solutions Engineer roles.",
    "What do you know about me so far?",
]

# Test 1 - Buffer: keeps everything
def run_buffer_memory():
    """Keeps all messages - growing context, full recall."""
    print("\n" + "="*60)
    print("Strategy 1: Full Buffer (keep everything)")
    print("="*60)
    history = []
    for turn in conversation_turns:
        history.append(HumanMessage(content=turn))
        response = llm.invoke(history)
        history.append(AIMessage(content=response.content))
        print(f"User: {turn}")
        print(f"Assistant: {response.content[:150]}...\n")
    print(f"Total messages in memory: {len(history)}")

# Test 2 - Window: keeps last 2 exchanges only
# k=2 means keep last 2 human+AI pairs
def run_window_memory(k:int = 2):
    """Keeps only the last k human+AI pairs - fixed cost ceiling."""
    print("\n" + "="*60)
    print(f"Strategy 2: Window Memory (last {k} exchanges only)")
    print("="*60)
    full_history = []
    for turn in conversation_turns:
        full_history.append(HumanMessage(content=turn))

        # Slice: keep only the last k*2 messages (k pairs of human + AI)
        # Always send from the trimmed window
        window = full_history[-(k*2):]
        response = llm.invoke(window)
        full_history.append(AIMessage(content=response.content))

        print(f"User: {turn}")
        print(f"Assistant: {response.content[:150]}...\n")
        print(f"  [Messages sent this turn: {len(window)}]")
    print(f"Total messages stored: {len(full_history)}")

# Test 3 - Summary: compresses old messages with an LLM call
# This costs an extra API call but handles long conversations gracefully
def run_summary_memory():
    """"Compresses old messages into a summary - efficient for long conversations."""
    print("\n" + "="*60)
    print("Strategy 3: Summary Memory (LLM-compressed)")
    print("="*60)
    summary = ""
    recent = []

    for turn in conversation_turns:
        # Build context: summary of old turns + recent raw messages
        messages = []
        if summary:
            messages.append(SystemMessage(content=f"Summary of conversation so far: {summary}"))
        messages.extend(recent)
        messages.append(HumanMessage(content=turn))

        response = llm.invoke(messages)
        recent.append(HumanMessage(content=turn))
        recent.append(AIMessage(content=response.content))

        print(f"User: {turn}")
        print(f"Assistant: {response.content[:150]}...\n")

        # After every 2 exchanges, compress recent into summary
        if len(recent) >= 4:
            summary_prompt = [
                SystemMessage(content="Summarise this conversation excerpt concisely, preserving key facts."),
                *recent
            ]
            summary_response = llm.invoke(summary_prompt)
            summary = summary_response.content
            recent = [] # Clear recent after summarising
            print(f" [Summary updated: {summary[:100]}...]\n")


run_buffer_memory()
run_window_memory(k=2)
run_summary_memory()