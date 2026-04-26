from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from dotenv import load_dotenv

load_dotenv()

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.3)

# Define a reusable template with named variables
# {topic} and {level} are placeholders filled at call time
template = ChatPromptTemplate.from_messages([
    ("system", "You are a technical educator. Adjust complexity for the audience level specified."),
    ("human", "Explain {topic} to someone at a {level} level. Keep it under 100 words.")
])

# The pipe operator (|) is LCEL — LangChain Expression Language
# It chains: template → llm
# Read it as: "format the prompt, then send it to the model"
chain = template | llm

# Invoke with variable substitution
response = chain.invoke({
    "topic": "cosine similarity",
    "level": "beginner"
})
print("Beginner explanation:")
print(response.content)
print()

response = chain.invoke({
    "topic": "cosine similarity",
    "level": "senior ML engineer"
})
print("Senior explanation:")
print(response.content)