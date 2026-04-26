import os
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

MODEL = "gpt-4o-mini"
TEMPERATURE = 0.3


# ── Core API call function ─────────────────────────────────────
# This is what LangChain's llm.invoke() was doing internally.
# One function, used by all three steps and the Q&A session.

def call_llm(messages: list[dict], temperature: float = TEMPERATURE) -> str:
    """
    Send a list of messages to the OpenAI API.
    Returns the response text as a plain string.
    This replaces: llm.invoke() | StrOutputParser()
    """

    response = client.chat.completions.create(
        model = MODEL,
        temperature=temperature,
        messages=messages
    )
    return response.choices[0].message.content.strip()


# ── Prompt builders ────────────────────────────────────────────
# These replace ChatPromptTemplate.
# Plain functions that take variables and return a messages list.
# Exactly what ChatPromptTemplate.from_messages().format_messages() produced.

def build_step1_messages(document: str) -> list[dict]:
    return[
        {
            "role": "system",
            "content": (
                "You are a precise document analyst. "
                "Your job is to extract key facts from a document. "
                "Be concise. Preserve specific numbers, names, and metrics. "
                "Do not add interpretation - only extract what is stated."
            )
        },
        {
            "role": "user",
            "content": (
                f"Summarise this document. Include:\n"
                f"- Main topic\n"
                f"- Key positive findings (with numbers if present)\n"
                f"- Key negative findings (with numbers if present)\n"
                f"- Any urgent items\n\n"
                f"Document:\n{document}"
            )
        }
    ]


def build_step2_messages(summary: str) -> list[dict]:
    return [
        {
            "role": "system",
            "content": (
                "You are a product analyst specialising in customer feedback. "
                "You receive summaries and identify patterns, risks, and opportunities. "
                "Be specific. Reference the data points from the summary. "
            )
        },
        {
            "role": "user",
            "content": (
                f"Analyse this summary. Identify:\n"
                f"1. Critical risks (things that could cause churn or legal issues)\n"
                f"2. Quick wins (things that could be fixed fast for high impact)\n"
                f"3. Strategic opportunities (longer-term product improvements)\n"
                f"4. Overall health assessment (one sentence)\n\n"
                f"Summary:\n{summary}"
            )
        }
    ]


def build_step3_messages(analysis: str) -> list[dict]:
    return[
        {
            "role": "system",
            "content": (
                "You are a product manager writing a sprint planning document. "
                "You convert analysis into specific, actionable tasks. "
                "Each task must have: a priority (P0/P1/P2), an owner role, "
                "and a clear definition of done."
            )
        },
        {
            "role": "user",
            "content": (
                f"Convert this analysis into an action plan.\n"
                f"Format each action as:\n"
                f"[Priority] | [Owner Role] | [Action] | [Definition of Done]\n\n"
                f"Analysis:\n{analysis}"
            )
        }
    ]


# ── Three step pipeline ────────────────────────────────────────
# This replaces the LCEL chain and the lambda adapters.
# Explicit sequential calls - you can see exactly what flows where.

def process_document(document: str) -> dict:
    """
    Run the document through all three steps sequentially.
    Each step receives the preivious step's string output directly.
    No framework needed - just three function calls in order.
    """
    print("=" * 60)
    print("PROCESSING DOCUMENT")
    print("=" * 60)

    print("\n-- Step 1: Summarising --\n")
    summary = call_llm(build_step1_messages(document))
    print(summary)

    print("\n-- Step 2: Analysing --\n")
    analysis = call_llm(build_step2_messages(summary))
    print(analysis)

    print("\n-- Step 3: Generating Action Items --\n")
    action_plan = call_llm(build_step3_messages(analysis))
    print(action_plan)

    return {
        "summary": summary,
        "analysis": analysis,
        "action_plan": action_plan
    }


# ── Save report ────────────────────────────────────────────────
# Identical to LangChain version - this had nothing to do with the framework.

def save_report(result: dict, path: str = "processed_report_raw.txt") -> None:
    base_dir = os.path.dirname(os.path.abspath(__file__))
    full_path = os.path.join(base_dir, path)
    with open(full_path, "w") as f:
        f.write("=== STEP 1: SUMMARY ===\n\n")
        f.write(result["summary"])
        f.write("\n\n=== STEP 2: ANALYSIS ===\n\n")
        f.write(result["analysis"])
        f.write("\n\n=== STEP 3: ACTION PLAN ===\n\n")
        f.write(result["action_plan"])
    print(f"\nFull report saved to {full_path}")


# ── Q&A session with memory ────────────────────────────────────
# This replaces ConversationChain + ConversationBuffetMemory
# Memory is a plain list of dicts - same mechanism, no abstraction

def qa_session(result: dict) -> None:
    """
    Interactive Q&A grounded in the document analysis.
    Memory = a list of message dicts that grows with each turn.
    This is identical in mechanism to the LangChain version -
    LangChain's memory classes were doing exactly this internally
    """
    context = (
        f"You are a helpful assistant.\n"
        f"You have already analysed a customer feedback document. \n"
        f"Here are the outputs of that analysis:\n\n"
        f"SUMMARY:\n{result['summary']}\n\n"
        f"ANALYSIS:\n{result['analysis']}\n\n"
        f"ACTION PLAN:\n{result['action_plan']}\n\n"
        f"Answer questions based on this analysis only. "
        f"Be specific and reference the data."
    ) 

    # Memory list - seeded with system context, same as LangChain version
    # Difference: dicts instead of LangChain message objects
    # The API receives the same payload either way
    chat_history = [
        {"role": "system", "content": context}
    ]

    print("\n" + "=" * 60)
    print("Q&A SESSION -ask anything about the document analysis")
    print("Type 'quit' to exit")
    print("=" * 60 + "\n")

    while True:
        user_input = input("You: ").strip()

        if not user_input:
            continue
        if user_input.lower() in ("quit", "exit", "q"):
            print("Ending Q&A session.")
            break

        # Append user message - plain dict, not HumanMessage object
        chat_history.append({"role": "user", "content": user_input})

        # Call API with full history
        response = call_llm(chat_history, temperature=0.5)

        # Append assistant reply
        chat_history.append({"role": "assistant", "content": response})

        print(f"\nAssistant: {response}\n")


# ── Main ───────────────────────────────────────────────────────

if __name__ == "__main__":
    base_dir = os.path.dirname(os.path.abspath(__file__))
    doc_path = os.path.join(base_dir, "simple_document.txt")

    with open(doc_path, "r") as f:
        document = f.read()

    result = process_document(document)
    save_report(result)
    qa_session(result)