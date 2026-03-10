"""Local eval harness for AIMO PP3 using AIME 2024/2025 datasets.

Connects to a running inference server (vLLM / LMDeploy / Ollama) via OpenAI-compatible API.
Measures maj@N accuracy which proxies the Kaggle LB score.

Usage:
    # Start inference server first, e.g.:
    #   vllm serve Qwen/QwQ-32B-AWQ --port 8080 --gpu-memory-utilization 0.92
    #   lmdeploy serve api_server Qwen/QwQ-32B-AWQ --tp 1 --server-port 8080

    LLM_BASE_URL=http://localhost:8080/v1 python eval_aime.py
    LLM_BASE_URL=http://localhost:8080/v1 python eval_aime.py --dataset aime_2024 --n 8
    LLM_BASE_URL=http://localhost:8080/v1 python eval_aime.py --dataset both --n 8 --limit 10
"""

import argparse
import csv
import os
import re
import subprocess
import sys
import tempfile
import time
from collections import Counter
from datetime import datetime
from pathlib import Path

from openai import OpenAI

# ---------------------------------------------------------------------------
# Config (override via env vars)
# ---------------------------------------------------------------------------

LLM_BASE_URL = os.getenv("LLM_BASE_URL", "http://localhost:8080/v1")
LLM_MODEL = os.getenv("LLM_MODEL", "")  # auto-detected from server if empty

SYSTEM_PROMPT = """\
You are an expert math competition solver. Solve the problem step by step.

When you need to compute something, write Python code in a ```python block.
You will see the output and can continue reasoning.

End your response with \\boxed{N} where N is your final integer answer (0-99999).
"""

MAX_TIR_STEPS = 6
MAX_TOKENS = 7500
RESULTS_DIR = Path(__file__).parent / "eval_results"

# ---------------------------------------------------------------------------
# TIR solver (mirrors notebook logic)
# ---------------------------------------------------------------------------


def execute_python(code: str, timeout: int = 15) -> str:
    for pattern in [
        r"\bos\b",
        r"\bsubprocess\b",
        r"\bopen\b",
        r"__import__",
        r"socket",
        r"urllib",
        r"requests",
    ]:
        if re.search(pattern, code):
            return "[blocked: unsafe code]"
    with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
        f.write("from math import *\nimport sympy\n" + code)
        tmp = f.name
    try:
        r = subprocess.run(
            [sys.executable, tmp], capture_output=True, text=True, timeout=timeout
        )
        out = r.stdout
        if r.returncode != 0 and r.stderr:
            out += f"\n[err]: {r.stderr[:300]}"
        return (out.strip() or "(no output)")[:2000]
    except subprocess.TimeoutExpired:
        return f"[timeout after {timeout}s]"
    finally:
        os.unlink(tmp)


def extract_code_blocks(text: str) -> list[str]:
    return re.findall(r"```python\n(.*?)```", text, re.DOTALL)


def extract_answer(text: str) -> int | None:
    for m in re.finditer(r"\\boxed\{(\d+)\}", text):
        v = int(m.group(1))
        if 0 <= v <= 99999:
            return v
    for m in re.finditer(
        r"(?:answer\s+is|answer:|final\s+answer[:\s]+|=\s*)(\d+)", text, re.IGNORECASE
    ):
        v = int(m.group(1))
        if 0 <= v <= 99999:
            return v
    return None


def _get_model(client: OpenAI, model_override: str) -> str:
    if model_override:
        return model_override
    models = client.models.list()
    return models.data[0].id


def _chat(client: OpenAI, model: str, messages: list, temperature: float) -> str:
    r = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=temperature,
        max_tokens=MAX_TOKENS,
    )
    return r.choices[0].message.content or ""


def solve_tir(
    problem: str, client: OpenAI, model: str, temperature: float = 0.0
) -> int | None:
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": problem},
    ]
    for _ in range(MAX_TIR_STEPS):
        reply = _chat(client, model, messages, temperature)
        messages.append({"role": "assistant", "content": reply})
        code_blocks = extract_code_blocks(reply)
        if code_blocks:
            out = execute_python(code_blocks[-1])
            messages.append({"role": "user", "content": f"Code output:\n{out}"})
        if not code_blocks:
            ans = extract_answer(reply)
            if ans is not None:
                return ans
            messages.append(
                {"role": "user", "content": "Give your final answer as \\boxed{N}."}
            )
        elif extract_answer(reply) is not None:
            break
    all_text = "\n".join(m["content"] for m in messages if m["role"] == "assistant")
    return extract_answer(all_text)


def solve_with_voting(
    problem: str, client: OpenAI, model: str, n: int = 8
) -> tuple[int | None, dict]:
    answers = []
    for i in range(n):
        temp = 0.0 if i == 0 else 0.7
        ans = solve_tir(problem, client, model, temperature=temp)
        if ans is not None:
            answers.append(ans)
        print(f"    sample {i + 1}/{n} → {ans}", flush=True)
    if not answers:
        return None, {}
    counter = Counter(answers)
    winner, _ = counter.most_common(1)[0]
    return winner, dict(counter)


# ---------------------------------------------------------------------------
# Dataset loading
# ---------------------------------------------------------------------------


def load_aime(year: int) -> list[dict]:
    """Load AIME problems from HuggingFace. Returns list of {problem, answer}."""
    from datasets import load_dataset

    if year == 2025:
        ds = load_dataset("MathArena/aime_2025", split="train")
        # Fields: problem, answer (int)
        return [
            {
                "id": f"aime25_{i}",
                "problem": row["problem"],
                "answer": int(row["answer"]),
            }
            for i, row in enumerate(ds)
        ]
    elif year == 2024:
        ds = load_dataset("HuggingFaceH4/aime_2024", split="train")
        # Fields: problem, answer (str or int)
        return [
            {
                "id": f"aime24_{i}",
                "problem": row["problem"],
                "answer": int(row["answer"]),
            }
            for i, row in enumerate(ds)
        ]
    else:
        raise ValueError(f"Unknown year: {year}")


def load_reference() -> list[dict]:
    """Load competition reference.csv (10 problems)."""
    import pandas as pd

    ref_path = (
        Path(__file__).parent.parent.parent
        / "data"
        / "ai-mathematical-olympiad-progress-prize-3"
        / "reference.csv"
    )
    if not ref_path.exists():
        raise FileNotFoundError(f"Reference CSV not found: {ref_path}")
    df = pd.read_csv(ref_path)
    return [
        {"id": row["id"], "problem": row["problem"], "answer": int(row["answer"])}
        for _, row in df.iterrows()
    ]


# ---------------------------------------------------------------------------
# Eval loop
# ---------------------------------------------------------------------------


def run_eval(
    problems: list[dict], client: OpenAI, model: str, n: int, tag: str
) -> None:
    from tqdm import tqdm

    RESULTS_DIR.mkdir(exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = RESULTS_DIR / f"eval_{tag}_{timestamp}.csv"

    correct = 0
    rows = []

    print(
        f"\nEvaluating {len(problems)} problems | model={model} | n={n} | tag={tag}\n"
    )
    t0 = time.time()

    pbar = tqdm(problems, unit="problem", dynamic_ncols=True)
    for i, p in enumerate(pbar):
        pid = p["id"]
        true_ans = p["answer"]
        pbar.set_description(f"{pid} | correct {correct}/{i}")
        print(f"\n[{i + 1}/{len(problems)}] {pid} | true={true_ans}", flush=True)

        pred, vote_dist = solve_with_voting(p["problem"], client, model, n=n)
        is_correct = pred == true_ans
        if is_correct:
            correct += 1

        elapsed = time.time() - t0
        pbar.set_postfix(acc=f"{correct}/{i + 1}", elapsed=f"{elapsed / 60:.1f}m")
        print(
            f"  vote: {vote_dist} → pred={pred} | {'✓' if is_correct else '✗'} | {elapsed:.0f}s elapsed\n",
            flush=True,
        )

        rows.append(
            {
                "id": pid,
                "predicted": pred,
                "true": true_ans,
                "correct": is_correct,
                "votes": str(vote_dist),
            }
        )

    # Summary
    acc = correct / len(problems)
    total_time = time.time() - t0
    print("=" * 60)
    print(f"maj@{n} accuracy: {correct}/{len(problems)} = {acc:.1%}")
    print(
        f"Total time: {total_time / 60:.1f} min ({total_time / len(problems):.0f}s/problem)"
    )
    print(f"Results: {out_path}")

    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(
            f, fieldnames=["id", "predicted", "true", "correct", "votes"]
        )
        writer.writeheader()
        writer.writerows(rows)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(description="Local AIME eval for AIMO PP3")
    parser.add_argument(
        "--dataset",
        choices=["aime_2025", "aime_2024", "both", "reference"],
        default="aime_2025",
    )
    parser.add_argument(
        "--n", type=int, default=8, help="Samples per problem (majority vote)"
    )
    parser.add_argument("--limit", type=int, default=0, help="Max problems (0 = all)")
    parser.add_argument(
        "--offset", type=int, default=0, help="Skip first N problems (resume)"
    )
    args = parser.parse_args()

    client = OpenAI(base_url=LLM_BASE_URL, api_key="local")
    model = _get_model(client, LLM_MODEL)
    print(f"Server: {LLM_BASE_URL} | Model: {model}")

    if args.dataset == "aime_2025":
        problems = load_aime(2025)
        tag = "aime2025"
    elif args.dataset == "aime_2024":
        problems = load_aime(2024)
        tag = "aime2024"
    elif args.dataset == "both":
        problems = load_aime(2024) + load_aime(2025)
        tag = "aime2024+2025"
    else:
        problems = load_reference()
        tag = "reference"

    if args.offset:
        problems = problems[args.offset :]
    if args.limit:
        problems = problems[: args.limit]

    run_eval(problems, client, model, n=args.n, tag=tag)


if __name__ == "__main__":
    main()
