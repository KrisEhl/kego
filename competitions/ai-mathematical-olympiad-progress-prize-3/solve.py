"""AIMO PP3 Baseline: Tool-Integrated Reasoning (TIR) solver.

Drop-in model via LLM_BASE_URL env var — same code runs locally (MLX/Ollama)
and on Kaggle (vLLM/SGLang serving 32B+ models on H100s).

Local setup (MLX, QwQ-32B ~19GB):
    mlx_lm.server --model mlx-community/QwQ-32B-4bit --port 8080
    LLM_BASE_URL=http://localhost:8080/v1 python solve.py --mode eval

Local setup (Ollama):
    ollama run qwq:32b
    LLM_BASE_URL=http://localhost:11434/v1 LLM_MODEL=qwq:32b python solve.py --mode eval

Kaggle notebook (vLLM / SGLang server already started):
    import sys; sys.path.append('/kaggle/input/aimo3-scripts')
    from solve import predict
    inference_server = AIMO3InferenceServer(predict)
    inference_server.run_local_gateway()

Evaluate on reference problems:
    python solve.py --mode eval [--limit N] [--n N] [--tir]

Solve a single problem:
    python solve.py --mode solve --problem "What is 2+2?"
"""

import argparse
import os
import re
import subprocess
import sys
import tempfile
from collections import Counter
from pathlib import Path

import pandas as pd
from openai import InternalServerError, OpenAI

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

BASE_URL = os.environ.get("LLM_BASE_URL", "http://localhost:8080/v1")
MODEL = os.environ.get("LLM_MODEL", "mlx-community/QwQ-32B-4bit")
N_SAMPLES = int(os.environ.get("N_SAMPLES", "4"))
MAX_TIR_STEPS = int(os.environ.get("MAX_TIR_STEPS", "4"))
CODE_TIMEOUT = int(os.environ.get("CODE_TIMEOUT", "15"))
API_KEY = os.environ.get("OPENAI_API_KEY", "local")

DATA_DIR = (
    Path(os.environ.get("KEGO_PATH_DATA", Path(__file__).resolve().parents[2] / "data"))
    / "ai"
    / "ai-mathematical-olympiad-progress-prize-3"
)

SYSTEM_PROMPT = """\
You are an expert math competition solver. Solve the problem step by step.

When you need to compute something, write Python code in a ```python block.
You will see the output and can continue reasoning.

End your response with \\boxed{N} where N is your final integer answer (0–99999).
"""

# ---------------------------------------------------------------------------
# Code execution (sandboxed subprocess)
# ---------------------------------------------------------------------------

# Allowed imports — restrict to math/science libs to reduce execution risk
_IMPORT_WHITELIST = {
    "math",
    "cmath",
    "decimal",
    "fractions",
    "numbers",
    "itertools",
    "functools",
    "collections",
    "heapq",
    "sympy",
    "numpy",
    "scipy",
    "string",
    "re",
    "sys",
}


def _is_safe(code: str) -> bool:
    """Rough safety check: block obviously dangerous patterns."""
    danger = [
        r"\bos\b",
        r"\bsubprocess\b",
        r"\bshutil\b",
        r"\bopen\b",
        r"\beval\b",
        r"\bexec\b",
        r"__import__",
        r"importlib",
        r"socket",
        r"urllib",
        r"requests",
        r"http",
    ]
    for pattern in danger:
        if re.search(pattern, code):
            return False
    return True


def execute_python(code: str) -> str:
    """Execute Python code in a subprocess. Returns stdout or error string."""
    if not _is_safe(code):
        return "[blocked: potentially unsafe code]"

    # Prefix with safe math imports
    preamble = "from math import *\nimport sympy\n"
    full_code = preamble + code

    with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
        f.write(full_code)
        tmpfile = f.name

    try:
        result = subprocess.run(
            [sys.executable, tmpfile],
            capture_output=True,
            text=True,
            timeout=CODE_TIMEOUT,
        )
        output = result.stdout
        if result.returncode != 0 and result.stderr:
            output += f"\n[error]: {result.stderr[:300]}"
        return output.strip() or "(no output)"
    except subprocess.TimeoutExpired:
        return f"[timeout after {CODE_TIMEOUT}s]"
    except Exception as e:
        return f"[error: {e}]"
    finally:
        os.unlink(tmpfile)


# ---------------------------------------------------------------------------
# Answer extraction
# ---------------------------------------------------------------------------


def extract_code_blocks(text: str) -> list[str]:
    """Extract Python code blocks from model response."""
    return re.findall(r"```python\n(.*?)```", text, re.DOTALL)


def extract_answer(text: str) -> int | None:
    r"""Extract the final integer answer from model text.

    Tries in order:
    1. \boxed{N}
    2. "The answer is N" / "answer: N"
    3. Last standalone integer in valid range
    """
    # \boxed{N}
    for m in re.finditer(r"\\boxed\{(\d+)\}", text):
        val = int(m.group(1))
        if 0 <= val <= 99999:
            return val

    # "answer is N" / "= N" at end
    for m in re.finditer(
        r"(?:answer\s+is|answer:|final\s+answer[:\s]+|=\s*)(\d+)",
        text,
        re.IGNORECASE,
    ):
        val = int(m.group(1))
        if 0 <= val <= 99999:
            return val

    return None


# ---------------------------------------------------------------------------
# API call wrapper (handles Ollama 500 tool-call parse bug)
# ---------------------------------------------------------------------------


def _chat(
    client: OpenAI, messages: list, temperature: float, max_tokens: int = 4096
) -> str:
    """Call chat completions, returning the reply text.

    Passes tool_choice=none to prevent models from entering tool-call mode.
    Falls back to a retry without that param if the server rejects it.
    """
    kwargs = {
        "model": MODEL,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
    }
    try:
        # Ollama ≥0.3 and vLLM support tool_choice via extra_body
        response = client.chat.completions.create(
            **kwargs, extra_body={"tool_choice": "none"}
        )
    except (InternalServerError, Exception):
        # Fall back without extra_body — some servers reject unknown params
        try:
            response = client.chat.completions.create(**kwargs)
        except InternalServerError as e:
            # Ollama gpt-oss bug: model output parsed as tool call → return partial text
            msg = str(e)
            raw_match = re.search(r"raw='(.*?)', err=", msg, re.DOTALL)
            if raw_match:
                return raw_match.group(1).replace("\\n", "\n")
            return ""
    return response.choices[0].message.content or ""


# ---------------------------------------------------------------------------
# TIR solver: one deterministic solution with code execution loop
# ---------------------------------------------------------------------------


def solve_tir(problem: str, client: OpenAI, verbose: bool = False) -> int | None:
    """Solve one problem with Tool-Integrated Reasoning.

    Interleaves model reasoning with Python code execution until the model
    provides a final \\boxed{} answer or MAX_TIR_STEPS is reached.
    """
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": problem},
    ]

    for step in range(MAX_TIR_STEPS):
        reply = _chat(client, messages, temperature=0.0)
        messages.append({"role": "assistant", "content": reply})

        if verbose:
            print(f"  [TIR step {step + 1}] {reply[:120]}...")

        answer = extract_answer(reply)
        code_blocks = extract_code_blocks(reply)

        if not code_blocks:
            # No code to run — take answer if present, or let model try again
            if answer is not None:
                return answer
            # Ask model to give a concrete answer
            messages.append(
                {
                    "role": "user",
                    "content": "Please give your final answer as \\boxed{N}.",
                }
            )
            continue

        # Execute last code block and feed output back
        output = execute_python(code_blocks[-1])
        if verbose:
            print(f"    [code output] {output[:100]}")

        messages.append({"role": "user", "content": f"Code output:\n{output}"})

        if answer is not None:
            return answer

    # Extract from all assistant turns combined
    all_text = "\n".join(m["content"] for m in messages if m["role"] == "assistant")
    return extract_answer(all_text)


# ---------------------------------------------------------------------------
# Voting solver: N independent samples → majority vote
# ---------------------------------------------------------------------------


def solve_with_voting(
    problem: str, client: OpenAI, n: int = N_SAMPLES, verbose: bool = False
) -> int:
    """Generate N solutions with TIR and return the majority-vote answer.

    Falls back to 0 if no valid answer is extracted from any sample.
    """
    answers: list[int] = []

    for i in range(n):
        if verbose:
            print(f"  [sample {i + 1}/{n}]")
        # Temperature 0 for first sample, 0.7 for the rest (diversity)
        temp = 0.0 if i == 0 else 0.7

        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": problem},
        ]

        for step in range(MAX_TIR_STEPS):
            reply = _chat(client, messages, temperature=temp)
            messages.append({"role": "assistant", "content": reply})

            answer = extract_answer(reply)
            code_blocks = extract_code_blocks(reply)

            if code_blocks:
                output = execute_python(code_blocks[-1])
                messages.append({"role": "user", "content": f"Code output:\n{output}"})

            if not code_blocks:
                if answer is not None:
                    break
                # Prompt for final answer
                messages.append(
                    {"role": "user", "content": "Give your final answer as \\boxed{N}."}
                )
            elif answer is not None:
                break

        # Extract from all assistant turns
        all_text = "\n".join(m["content"] for m in messages if m["role"] == "assistant")
        ans = extract_answer(all_text)
        if ans is not None:
            answers.append(ans)
            if verbose:
                print(f"    → {ans}")
        else:
            if verbose:
                print("    → (no answer extracted)")

    if not answers:
        return 0

    counter = Counter(answers)
    winner, count = counter.most_common(1)[0]
    if verbose and len(counter) > 1:
        print(f"  vote: {dict(counter)} → {winner} ({count}/{len(answers)})")
    return winner


# ---------------------------------------------------------------------------
# Kaggle gateway predict function
# ---------------------------------------------------------------------------


_client: OpenAI | None = None


def _get_client() -> OpenAI:
    global _client
    if _client is None:
        _client = OpenAI(base_url=BASE_URL, api_key=API_KEY)
    return _client


def predict(df: "pd.DataFrame") -> "pd.DataFrame":
    """Kaggle gateway predict endpoint.

    Receives a polars/pandas DataFrame with one row (columns: id, problem).
    Returns a DataFrame with columns: id, answer.
    """
    import polars as pl

    client = _get_client()
    row = df.row(0, named=True) if hasattr(df, "row") else df.iloc[0].to_dict()
    problem_id = row["id"]
    problem = row["problem"]

    answer = solve_with_voting(problem, client, n=N_SAMPLES)
    return pl.DataFrame({"id": [problem_id], "answer": [answer]})


# ---------------------------------------------------------------------------
# CLI: eval / solve
# ---------------------------------------------------------------------------


def _run_eval(args: "argparse.Namespace", client: OpenAI) -> None:
    df = pd.read_csv(DATA_DIR / "reference.csv")
    if args.limit:
        df = df.head(args.limit)

    print(f"Evaluating {len(df)} problems (n_samples={args.n}, tir={args.tir})\n")

    correct = 0
    results = []

    for _, row in df.iterrows():
        pid = row["id"]
        problem = row["problem"]
        true_ans = int(row["answer"])

        print(f"[{pid}] {problem[:80]}...")
        if args.tir:
            pred = solve_tir(problem, client, verbose=args.verbose) or 0
        else:
            pred = solve_with_voting(problem, client, n=args.n, verbose=args.verbose)

        ok = pred == true_ans
        correct += int(ok)
        results.append({"id": pid, "predicted": pred, "true": true_ans, "correct": ok})
        print(f"  → predicted={pred}, true={true_ans}  {'✓' if ok else '✗'}\n")

    acc = correct / len(df)
    print(f"{'=' * 50}")
    print(f"Accuracy: {correct}/{len(df)} = {acc:.1%}")

    out_path = Path(__file__).parent / "eval_results.csv"
    pd.DataFrame(results).to_csv(out_path, index=False)
    print(f"Results saved to {out_path}")


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="AIMO PP3 TIR solver")
    parser.add_argument(
        "--mode",
        choices=["eval", "solve"],
        default="eval",
        help="eval: benchmark on reference.csv; solve: single problem",
    )
    parser.add_argument("--problem", default=None, help="Problem text (solve mode)")
    parser.add_argument(
        "--n",
        type=int,
        default=N_SAMPLES,
        help=f"Number of voting samples (default: {N_SAMPLES})",
    )
    parser.add_argument(
        "--tir", action="store_true", help="Use single deterministic TIR"
    )
    parser.add_argument(
        "--limit", type=int, default=None, help="Limit to first N problems"
    )
    parser.add_argument("--verbose", "-v", action="store_true", help="Print TIR steps")
    args = parser.parse_args()

    client = OpenAI(base_url=BASE_URL, api_key=API_KEY)

    if args.mode == "solve":
        problem = args.problem or input("Problem: ")
        if args.tir:
            answer = solve_tir(problem, client, verbose=args.verbose) or 0
        else:
            answer = solve_with_voting(problem, client, n=args.n, verbose=args.verbose)
        print(f"\nAnswer: {answer}")

    elif args.mode == "eval":
        _run_eval(args, client)


if __name__ == "__main__":
    main()
