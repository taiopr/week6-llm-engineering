import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.tools import tool
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langgraph.prebuilt import create_react_agent

load_dotenv()

# --- Tools defined BEFORE they are referenced ---

@tool
def get_weather(city: str) -> str:
    """Get the current weather for a given city. Returns temperature and conditions."""
    weather_data = {
        "barcelona": "22°C, sunny with light breeze",
        "london": "14°C, overcast with chance of rain",
        "new york": "18°C, partly cloudy",
        "tokyo": "26°C, humid and clear",
    }
    return weather_data.get(city.lower(), f"Weather data not available for {city}")


@tool
def calculate(expression: str) -> str:
    """
    Evaluate a mathematical expression and return the result.
    Input should be a valid Python math expression like '15 * 24' or '100 / 4'.
    """
    try:
        result = eval(expression, {"__builtins__": {}}, {})
        return f"Result: {result}"
    except Exception as e:
        return f"Error evaluating expression: {str(e)}"


@tool
def search_knowledge_base(query: str) -> str:
    """Search an internal knowledge base for information about AI concepts."""
    kb = {
        "embeddings": "Embeddings are numerical vector representations of text that capture semantic meaning.",
        "rag": "RAG (Retrieval Augmented Generation) combines vector search with LLM generation.",
        "temperature": "Temperature controls randomness in LLM output sampling.",
        "langchain": "LangChain is a framework for building LLM-powered applications.",
    }
    for key, value in kb.items():
        if key in query.lower():
            return value
    return "No relevant information found in knowledge base."


# --- Agent construction AFTER tools are defined ---

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
tools = [get_weather, calculate, search_knowledge_base]

# create_react_agent is the current LangGraph replacement for AgentExecutor
agent = create_react_agent(llm, tools)

# --- Run it ---

test_queries = [
    "What's the weather in Barcelona?",
    "What is 347 multiplied by 28?",
    "What is RAG in AI?",
    "What's the weather in Tokyo and what is 15 squared?",
]

for query in test_queries:
    print(f"\n{'='*60}")
    print(f"Query: {query}")
    print("-" * 60)
    result = agent.invoke({"messages": [{"role": "user", "content": query}]})
    # LangGraph returns messages list — the last message is the final answer
    print(f"Final answer: {result['messages'][-1].content}")