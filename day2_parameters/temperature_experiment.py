import os
import json
import time
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# The prompt is deliberately open-ended — this maximises observable variance.
# A closed factual question ("what is 2+2") would show almost no variance.
PROMPT = "Describe what happens when a raindrop hits a still pond. Two sentences."

TEMPERATURES = [0.0, 0.5, 1.0, 1.5, 2.0]
RUNS_PER_TEMP = 10


def run_completion(prompt: str, temperature: float) -> str:
    """
    Single API call. Returns the text content only.
    We use max_tokens=100 — plenty for two sentences, prevents runaway output.
    """
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
        temperature=temperature,
        max_tokens=100
    )
    return response.choices[0].message.content.strip()


def run_experiment(prompt: str, temperatures: list[float], runs: int) -> dict:
    """
    For each temperature, collect `runs` completions.
    Returns a dict: { temperature_value: [list of responses] }
    """
    results = {}

    for temp in temperatures:
        print(f"\nTemperature {temp} — running {runs} completions...")
        responses = []

        for i in range(runs):
            response = run_completion(prompt, temp)
            responses.append(response)
            print(f"  Run {i+1:02d}: {response[:80]}...")
            time.sleep(0.3)  # Avoid rate limit on rapid sequential calls

        results[temp] = responses

    return results


def analyse_variance(results: dict) -> None:
    """
    For each temperature, print:
    - All unique responses (deduped)
    - How many of the 10 runs were identical to the first run
    - A rough uniqueness ratio
    """
    print("\n" + "=" * 70)
    print("VARIANCE ANALYSIS")
    print("=" * 70)

    for temp, responses in results.items():
        unique = set(responses)
        identical_to_first = sum(1 for r in responses if r == responses[0])
        uniqueness_ratio = len(unique) / len(responses)

        print(f"\n--- Temperature: {temp} ---")
        print(f"Unique responses:     {len(unique)} / {len(responses)}")
        print(f"Identical to run 1:   {identical_to_first} / {len(responses)}")
        print(f"Uniqueness ratio:     {uniqueness_ratio:.1f}")
        print("Responses:")
        for j, r in enumerate(responses, 1):
            print(f"  [{j:02d}] {r}")


def save_results(results: dict, path: str = "results/raw_results.json") -> None:
    os.makedirs("results", exist_ok=True)
    # JSON keys must be strings
    serialisable = {str(k): v for k, v in results.items()}
    with open(path, "w") as f:
        json.dump(serialisable, f, indent=2)
    print(f"\nResults saved to {path}")


if __name__ == "__main__":
    print(f"Prompt: '{PROMPT}'")
    print(f"Running {RUNS_PER_TEMP} completions × {len(TEMPERATURES)} temperatures "
          f"= {RUNS_PER_TEMP * len(TEMPERATURES)} total API calls\n")

    results = run_experiment(PROMPT, TEMPERATURES, RUNS_PER_TEMP)
    analyse_variance(results)
    save_results(results)