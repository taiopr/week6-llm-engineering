# LangChain vs Raw Python - Document Processing Pipeline

## What This Repo Contains

Two implementations of the same three-step document pipeline:
1. Summarise a document
2. Analyse the summary
3. Generate an action plan from the analysis

Both include a conversational Q&A session with memory grounded in the outputs.

## The Pipeline

1. Raw document
2. Summarise
3. Analyse
4. Action Plan
5. Q&A

## Implementation Xomparison

### Lines of Code
- LangChain version: X lines
- Raw Python verison: Y lines

### Dependencies
- LangChain version: openai, langchain, langchain-openai, langchain-core
- Raw Python version: openai, python-dotenv

### What LangChain Abstracts
| Operation | LangChain | Raw Python |
|---|---|---|
| Prompt definition | ChatPromptTemplate | Function returning list[dict] |
| API call | llm.invoke() | client.chat.completions.create() |
| Output extraction | StrOutputParser | response.choices[0].message.content |
| Step chaining | LCEL pipe + lambda | Sequential function calls |
| Memory | HumanMessage/AIMessage objects | Plain dicts with role/content keys |

### Key Finding: Memory Is Identical

The most important observation: memory management is mechanically identical in both versions.
Both maintain a list that grows with each turn and send the full list on every API call.
LangChain's message objects (HumanMessage, AIMessage) are wrappers around the same
role/content structure the raw API uses.

### When to Use Each

**Raw API:**
- Production systems requiring full visibility into requests
- When debugging needs to be straightforward
- Minimal dependency surface
- Teams unfamiliar with LangChain

**LangChain:**
- Rapid prototyping
- RAG pipelines using LangChain's retrival ecosystem
- Projects that benefit from pre-built integrations
- When LCEL composability reduces significant boilerplate