import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage

load_dotenv()

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.3)
parser = StrOutputParser()


# ── Step 1: Summarise ──────────────────────────────────────────
# Reduce the raw document to its essential content
# Output: a concise structured summary

step1_prompt = ChatPromptTemplate.from_messages([
    ("system", """You are a precise document analyst
Your job is to extract the key facts from a document
Be concise. Preserve specific numbers, names, and metrics
Do not add interpretation - only extract what is stated."""),
    ("human", """Summarise this document. Include:
- Main topic
- Key positive findings (with numbers if present)
- Key negative findings (with numbers if present)
- Any urgent items
     
Document:
{document}""")
])


# ── Step 2: Analyse ────────────────────────────────────────────
# Take the summary and produce structured analysis
# Output: categorised problems and opportunities

step2_prompt = ChatPromptTemplate.from_messages([
    ("system", """You are a product analyst specialising in customer feedback
You receive summaries and identify patterns, risks, and opportunities
Be specific. Reference the data points from the summary."""),
    ("human", """Analyse this summary. Identify:
1. Critical risks (things that could cause churn or legal issues)
2. Quick wins (things that could be fixed fast for high impact)
3. Strategic opportunities (longer-term product improvements)
4. Overall health assessment (one sentence)
     
Summary:
{summary}""")
])


# ── Step 3: Action Items ───────────────────────────────────────
# Take the analysis and produce concrete, prioritised next steps
# Output: actionable plan a team can execute

step3_prompt = ChatPromptTemplate.from_messages([
    ("system", """You are a product manager writing a sprint planning document
You convert analysis into specific, actionable tasks
Each task must have: a priority (P0/P1/P2), an owner role, and a clear definition of done. """),
    ("human", """Convert this analysis into an action plan
Format each action as:
[Priority] | [Owner Role] | [Action] | [Definition of Done]

Analysis:
{analysis}""")
])



# ── Wire the chain ─────────────────────────────────────────────
step1_chain = step1_prompt | llm | parser
step2_chain = step2_prompt | llm | parser
step3_chain = step3_prompt | llm | parser

# Full pipeline - each lambda adapts string output to the next prompt's expected key
#full_chain = (
#    step1_chain
#    | (lambda summary: {"summary": summary})
#    | step2_chain
#    | (lambda analysis: {"analysis": analysis})
#    | step3_chain
#)


# ── Run it ─────────────────────────────────────────────────────

def process_document(document: str) -> dict:
    """
    Run the document through all three steps.
    Returns intermediate outputs so you can inspect each stage.
    """
    print("Processing document...\n")

    # Run each step individually first so we can inspect intermediate outputs
    print("-- Step 1: Summarising --")
    summary = step1_chain.invoke({"document": document})
    print(summary)
    print()

    print("-- Step 2: Analysing --")
    analysis = step2_chain.invoke({"summary": summary})
    print(analysis)
    print()

    print("-- Step 3: Generating Action Items --")
    action_plan = step3_chain.invoke({"analysis": analysis})
    print(action_plan)
    print()

    return {
        "summary": summary,
        "analysis": analysis,
        "action_plan": action_plan
    }

    # Save the full report
def save_report(result: dict, path: str = "processed_report.txt") -> None:
    with open(path, "w") as f:
        f.write("=== STEP 1: SUMMARY ===\n\n")
        f.write(result["summary"])
        f.write("\n\n=== STEP 2: ANALYSIS ===\n\n")
        f.write(result["analysis"])
        f.write("\n\n=== STEP 3: ACTION PLAN ===\n\n")
        f.write(result["action_plan"])

    print(f"\nFull report saved to {path}")


# ── Q&A session with memory ────────────────────────────────────

def qa_session(result: dict) -> None:
    """
    After processing, allow conversational Q&A grounded in the outputs.
    Memory keeps the conversation coherent across turns.
    """
    # Inject the processed results from the three processing outputs as context into the system message
    context = f"""You are a helpful assisstant.
You have already analysed a customer feedback document.
Here are the outputs of that analysis:

SUMMARY:
{result['summary']}

ANALYSIS:
{result['analysis']}

ACTION PLAN:
{result['action_plan']}

Answer questions based on this analysis. Be specific and reference the data."""
    
    # Seed memory with the context as the first system message
    # This means every API call includes the full analysis as background
    chat_history = [SystemMessage(content=context)]

    print("\n" + "=" * 60)
    print("Q&A SESSION — ask anything about the document analysis")
    print("Type 'quit' to exit")
    print("=" * 60 + "\n")

    while True:
        user_input = input("You: ").strip()

        if not user_input:
            continue
        if user_input.lower() in ("quit", "exit", "q"):
            print("Ending Q&A session.")
            break

        # Append user message to history
        chat_history.append(HumanMessage(content=str(user_input)))

        # Send full history - model sees system context + all prior turns
        response = llm.invoke(chat_history)

        # Append model reply so next turn includes it
        chat_history.append(AIMessage(content=str(response.content)))

        print(f"\nAssistant: {response.content}\n")

# ── Main ───────────────────────────────────────────────────────

if __name__ == "__main__":
    # Load the document
    import os
    base_dir = os.path.dirname(os.path.abspath(__file__))
    with open(os.path.join(base_dir, "sample_document.txt"), "r") as f:
        document = f.read()

    result = process_document(document)
    save_report(result)
    qa_session(result)