# Week 6 - LLM Engineering Foundations

## Overview

Week 6 covers the three foundational pillars of LLM engineering: semantic representation, output control, and system architecture. By the end of this week the core building blocks of a production RAG system are in place - embeddings, retrieval, sampling parameters, prompt chaining, memory, and the tradeoff between API calls and frameworks abstractions.

Each day builds on the previous. The embeddings from Day 1 feed the retrieval layer in week 7. The parameter intuition from Day 2 informs every prompt pipeline built after it. The chain and memory patterns from Days 3 and 4 become the orchestration layer of the full RAG system.

---

## Repository Structure

```
week6/
├── day1_embeddings/
│   ├── embeddings.py
│   ├── snippets.py
│   └── embeddings_cache.json
│
├── day2_parameters/
│   ├── temperature_experiment.py
│   └── results/
│       └── raw_results.json
│
├── day3_langchain/
│   ├── 01_basics.py
│   ├── 02_prompt_templates.py
│   ├── 03_agent.py
│   └── 04_comparison.py
│
├── day4_memory_chains/
│   ├── 01_memory_basic.py
│   ├── 02_memory_types.py
│   ├── 03_simple_chain.py
│   ├── 04_document_chain.py
│   └── sample_document.txt
│
├── day5_comparison/
│   ├── langchain_version.py
│   ├── raw_version.py
│   ├── comparison_notes.md
│   └── sample_document.txt
│
└── README.md
```

---

## Day 1 - Embeddings

### What Was Learned

A text embedding is a fixed-length list of point numbers that represents the semantic meaning of a piece of text in high-dimensional space. OpenAIs's `text-embedding-3-small` model produces vectors of 1536 dimensions. The key property: texts with similar meanings produce vectors that point in similar directions, regardless of wether they share any words.

This is the foundation of semantic search and RAG. Instead of matching keywords, you match menaing.

### Cosine Similarity

To compare two embedding vectors, cosine similarity measures the angle between them rather than the distance. Direction encodes meaning - magnitude does not. The formula: 

```
cosine_similarity(A, B) = (A · B) / (||A|| * ||B||)
```

Implemented from scratch in pure Python with no external libraries:

```python
def dot_product(a, b):
    return sum(x * y for x, y in zip(a, b))
 
def magnitude(v):
    return sum(x ** 2 for x in v) ** 0.5
 
def cosine_similarity(a, b):
    return dot_product(a, b) / (magnitude(a) * magnitude(b))
```

Result ranges from -1 (opposite) to 1 (identical). In practice with text embeddings, similar content scores 0.7-0.95 and unrelated content scores below 0.3.

### What Was Built

- 10 text snippets across 4 semantic clusters: ML/AI, VFX/3D, cooking, finance
- Embedding generation via OpenAI API - all 10 in a single batch call
- Cosine similarity implemented from scratch - no numpy, no sklearn
- Query ranking system: embed a natural language query, rank all snippets by similarity score
- JSON caching layer: embeddings saved to disk on first run, loaded from cache on subsequent runs

### Key Bug Fixed

Cache logic was placed outside `if __name__ == "__main__"` and after an unconditional API call. Both errors were corrected - cache check must come first and must be inside the main guard.

### Files

| File | Purpose |
|---|---|
| `snippets.py` | 10 text snippets in 4 semantic clusters |
| `embeddings.py` | Embedding generation, cosine similarity, query ranking, caching |
| `embeddings_cache.json` | Cached vectors — avoids redundant API calls |
 
---

## Day 2 - Sampling Parameters

### What Was Learned

LLMs generate text one token at a time by sampling from a probability distribution over the entire vocabulary. Sampling parameters control how that distribution is used - they don't change what the model knows, only how it decides.

### Temperature

Reshapes the probability distribution before sampling.

| Value | Effect | Use Case |
|---|---|---|
| 0.0 | Deterministic — always picks highest probability token | Classification, extraction, structured output |
| 0.2–0.4 | Very focused, minimal variation | Code generation, factual Q&A |
| 0.7–1.0 | Natural variation, balanced | General chat, summarisation |
| 1.2–1.5 | Creative divergence, still coherent | Brainstorming, creative writing |
| 2.0+ | Often incoherent | Rarely useful |

### top-p (Nucleus Sampling)

Instead of reshaping the distribution, truncates it. The model keeps only the smallest set of tokens whose cumulative probability reaches `top_p`and samples only from that nucleus.

- `top_p = 0.1` -> sample from top 10% of probability mass only
- `top_p = 1.0` -> sample from full distribution

**Temperature vs top_p:** temperature changes the shape of the distribution. top_p changes the size of the candidate pool. Use one or the other - not both simultaneously.

### max_tokens

Hard ceiling on output length. The model stops generating when it hits the limit, even mid-sentence. Check `finish_reason` on every response in production:

- `"stop"` -> model finished naturally
- `"length"` -> hit the ceiling, output was runcated

Truncation is a silent failure if unchecked.

### What Was Built

- Temperature experiment: same prompt run 10 times at 5 different temperatures (0.0, 0.5, 1.0, 1.5, 2.0)
- Variance analyser: counts unique responses, uniqueness ratio, identical runs
- top_p experiment: temperature fixed at 1.0, top_p varied across 5 values
- max_tokens truncation test: observing `finish_reason` at limits of 10, 30, 100, 500 tokens
- All results saved to `results/raw_results.json`

### Files

| File | Purpose
|---|---|
| `temperature_experiment.py` | Full parameter experiment with variance analysis |
| `results/raw_results.json` | Raw output data from all 50 runs |

---

## Day 3 -  LangChain: Abstraction vs Control

### What Was Learned

LangChain wraps raw API operations in reusable abstractions. Every abstraction maps directly to something the raw API does:

| LangChain | Raw API Equivalent |
|---|---|
| `ChatOpenAI` | `client.chat.completions.create()` |
| `ChatPromptTemplate` | f-string formatting into a messages list |
| `@tool` decorator | Hand-written JSON schema |
| `create_react_agent` | Manual tool-call while loop |
| Message objects | Plain dicts with role/content keys |

### LCEL - LangChain Expression Language

the `|`pipe operator chains components declaratively:

```python
chain = prompt | llm | parser
```

Nothing executes at this line. The chain runs when `.invoke()` is called. Each component's output becomes the next component's input.

### What Was Built

- `01_basics.py` - LangChain model interface, plain string and structured message invocation
- `02_prompt_templates,py` - ChatPromptTemplate with variable substitution, LCEL chain
- `03_agent.py` - Tool-calling agent with three tools (weather, calculator, knowledge base) rebuilt using `create_react_agent` from LangGraph
- `04_comparison.py` - Side by side: same operation in raw API and LangChain

### Production Issue Encountered

`AgentExecuter` and `create_openai_tools_agent` were removed in the installed LangChain version. Fixed by switching to `create_react_agent` from `langgraph.prebuilt` - the current canonicalaproach. Response format changed from `result['output']` to `result['messages'][-1].content`.

### The Honest Tradeoff

**Use raw API when:** production system, full request visibility needed, minimal dependencies, straightforward debugging required.

**Use LangChain when:** prototyping quickly, building RAG pipelines using its retrieval ecosystem, pre-built integrations save significant work.

### Files

| File | Purpose |
|---|---|
| `01_basics.py` | LangChain model interface |
| `02_prompt_templates.py` | PromptTemplate + LCEL chain |
| `03_agent.py` | Tool-calling agent with LangGraph |
| `04_comparison.py` | Raw API vs LangChain side by side |
 
---

## day 4 - Memory and Chains

### What Was Learned

**Memory is not a model feature. It is a message management by the application.**

Every LLM call is stateless. Conversation memory is mantained by appending messages to a list and sending the full list on every API call. The three memory strategies are three different answers to: *which messages do I include?*

| Strategy | Mechanism | Recall | Cost|
|---|---|---|---|
| Full Buffer | Send all messages every time | Perfect | Grows without bound |
| Window (k=n) | Send last n exchanges only | Partial - forgets old turns | Fixed ceiling |
| Summary | Compress old turns + send recent raw | Good - facts preserved | Extra API call per compression |

### Chains

A chain is a sequence of operations where each step's output feeds the next. In LCEL:

```python
chain = prompt | llm | parser
```

When chaining two prompts, an adapter is required between them because the parser outputs a plain string but the next prompt expects a dict:

```python
chain = (
    step1_prompt | llm | parser
    | (lambda text: {"text": text})   # adapter
    | step2_prompt | llm | parser
)
```

### Three-Step Document Pipeline

```
Raw document
    │
    ▼ Step 1 — Summarise (extract facts, preserve numbers)
    │
    ▼ Step 2 — Analyse (risks, quick wins, opportunities)
    │
    ▼ Step 3 — Action Plan (P0/P1/P2 tasks with owners and definitions of done)
    │
    ▼ Saved report + interactive Q&A with memory
```

Each step receives the previous step's output - not the original document. By step 3 the model reasons over a clean analysis, not raw text.

### Production Issue Encountered

`langchain.memory` and `langchain.chains` were removed in the installed version. `ConversationChain` and `ConversationBufferMemory` no longer exist. Resolved by implementing memory as a plain message list - which is exactly what those classes were doing internally. The fix revealed the mechanism more clearly than the abstraction did.

### Files

| File | Purpose |
|---|---|
| `01_memory_basic.py` | Plain list memory implementation |
| `02_memory_types.py` | Three memory strategies compared |
| `03_simple_chain.py` | LCEL single and two-step chains |
| `04_document_chain.py` | Three-step document pipeline with Q&A |
| `sample_document.txt` | Customer feedback report used as input |
 
---

## Day 5 - Raw Python vs LangChain: Full Comparison

### What Was Built

The exact same three-step document processing pipeline rebuilt in raw Python - no LangChain imports. The two implementations produce identical outputs. The comparison makes visible exactly what the framework abstracts.

### The Core Difference

**Prompt definition:**
```python
# LangChain
step1_prompt = ChatPromptTemplate.from.messages([
    ("system", "..."),
    ("human", "...{document}...")
])

# Raw Python
def build_step1_messages(document: str) -> list[dict]:
    return [
        {"role": "system", "content": "..."},
        {"role": "user", "content": f"...{document}..."}
    ]
```

**Step chaining:**
```python
# LangChain - LCEL with adapter lambdas
full_chain = step1_chain | (lambda s: {"summary": s}) | step2_chain

# Raw Python - explicit sequential calls
summary = call_llm(build_step1_messages(document))
analysis = call_llm(build_step2_messages(summary))
```

**Memory - mechanically identical:**
```python
# LangChain
chat_history = [SystemMessage(content=context)]
chat_history.append(HumanMessage(content=str(user_input)))

# Raw Python
chat_history = [{"role": "system", "content": context}]
chat_history.append({"role": "user", "content": user_input})
```

Memory is a list append in both cases. LangChain's message objects serialise to the same dict structure the raw API receives.

### Key Findings

1. **Memory is identical.** Both versions append to a list and send the full list on every call. LangChain's `HumanMessage`/`AIMessage` objects are wrappers around the same `role`/`content` dict structure.

2. **The adapter lambda exists because of LCEL's type system.** Raw Python passes strings directly between functions - no adapter needed. LangChain requires the lambda because prompt templates expect dicts, not strings.

3. **Raw Python is more debuggable.** When something breaks, the traceback points directly to the API call. LangChain tracebacks pass through multiple abstraction layers before reaching the actual error.

4. **Dependencies:** LangChain version requires `langchain`, `langchain-openai`, `langchain-core`, `langgrapg`. Raw version requires only `openai` and `python-dotenv`.

### Files

| File | Purpose |
|---|---|
| `langchain_version.py` | Day 4 pipeline - LangChain implementation |
| `raw_version.py` | Same pipeline - raw OpenAI SDK only |
| `comparison_notes.md` | Structures written comparison |

---

## Setup

### requirements

```bash
pip install openai langchain langchain-openai langchain-core langgraph python-dotenv
```

### Environment

Create a `.env` file in the root directory:

```
OPENAI_API_KEY=your_key_here
```

### Running Any File

```bash
# Always run from inside the day's folder to avoid path issues
cd day1_embeddings
python embeddings.py

cd day2_parameters
python temperature_experiment.py

cd day3_langchain
python 01_basics.py

cd day4_memory_chains
python 04_document_chain.py

cd day5_comparison
python raw_version.py
python langchain_version.py
```

---

## Core Concepts - Quick Reference

| Concept | One Line Summary |
|---|---|
| Embedding | A vector of floats representing the semantic meaning of text |
| Cosine similarity | Measures the angle between two vectors - direction encodes meaning |
| Temperature | Controls how the probability distributionis sampled - lower = more deterministic |
| top_p | Truncates the candidate token pool to the top p% of probability mass |
| max_tokens | Hard ceiling on output length - check finish_reason for truncation |
| Memory | A list of messages sent in full on every API call - no model feature |
| LCEL chain | Components wired with pipe operator - executes on .invoke()|
| Adapter lambda | Converts string output to dict input between LCEL chain steps |
| RAG | Retrive relevant content via embeddings, then generate answer from the context |

---

Test
↓
This test
