"""Standalone GDPVal evaluation runner for EvoSkill.

Usage:
    python src/evaluation/gdpval_runner.py \
        --cache-dir /path/to/gdpval/hf_cache \
        --workspace-dir outputs/gdpval_workspaces \
        --output results/gdpval_gemma4_31b.pkl \
        --max-concurrent 4 \
        --limit 20
"""
from __future__ import annotations

import asyncio
import pickle
from collections import Counter
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


@dataclass
class GDPValResult:
    index: int
    task_id: str
    sector: str
    occupation: str
    prompt: str
    expected_extensions: list[str]
    prediction: str | None        # raw final_answer text from agent
    generated_files: list[str]    # files found in submission/
    correct: bool | None
    score: float | None
    error: str | None
    details: dict[str, Any] = field(default_factory=dict)


def _load_dataset(cache_dir: str, limit: int | None = None) -> list[dict]:
    from datasets import load_dataset

    ds = load_dataset("openai/gdpval", split="train", cache_dir=cache_dir)
    if limit:
        ds = ds.select(range(min(limit, len(ds))))
    return list(ds)


def _materialize_reference_files(
    relative_paths: list[str],
    cache_dir: str,
    workspace_dir: Path,
) -> list[Path]:
    """Download/copy reference files from HF cache into workspace."""
    from huggingface_hub import hf_hub_download

    local_paths: list[Path] = []
    for rel_path in relative_paths:
        p = Path(rel_path)
        # Already absolute on disk?
        if p.is_absolute() and p.exists():
            local_paths.append(p)
            continue
        try:
            local = hf_hub_download(
                repo_id="openai/gdpval",
                repo_type="dataset",
                filename=rel_path,
                cache_dir=cache_dir,
                local_files_only=True,
            )
            local_paths.append(Path(local))
        except Exception:
            # Try network download as fallback
            try:
                local = hf_hub_download(
                    repo_id="openai/gdpval",
                    repo_type="dataset",
                    filename=rel_path,
                    cache_dir=cache_dir,
                )
                local_paths.append(Path(local))
            except Exception as exc:
                print(f"  [warn] Could not materialize {rel_path}: {exc}")
    return local_paths


def _build_prompt(
    row: dict,
    reference_paths: list[Path],
    submission_dir: Path,
    max_rubric_chars: int = 4000,
) -> str:
    prompt_text = str(row.get("prompt", "")).strip()
    rubric = str(row.get("rubric_pretty", "")).strip()
    if len(rubric) > max_rubric_chars:
        rubric = rubric[:max_rubric_chars] + "\n[rubric truncated]"

    ref_block = (
        "\n".join(f"- {p.name}: {p}" for p in reference_paths)
        if reference_paths
        else "- No reference files materialized locally."
    )
    urls = row.get("reference_file_urls") or []
    url_block = "\n".join(f"- {u}" for u in urls) if urls else "- None provided."

    deliverable_files = row.get("deliverable_files") or []
    extensions = [Path(f).suffix.lower() for f in deliverable_files if Path(f).suffix]
    ext_block = ", ".join(extensions) if extensions else "unspecified"

    return (
        f"You are solving one GDPVal benchmark task. "
        f"This benchmark expects deliverable files, not just a text answer.\n\n"
        f"Task:\n{prompt_text}\n\n"
        f"Local reference files:\n{ref_block}\n\n"
        f"Reference file URLs:\n{url_block}\n\n"
        f"Create the deliverable files inside this submission directory:\n{submission_dir}\n\n"
        f"Expected deliverable file types:\n{ext_block}\n\n"
        f"Rubric preview:\n{rubric or 'Unavailable'}\n\n"
        f"Leave the created files in the submission directory. "
        f"When you are done, return a short manifest listing the file names you created, "
        f"one per line, and nothing else."
    )


def _score(submission_dir: Path, expected_extensions: list[str]) -> tuple[bool | None, float | None, list[str]]:
    generated = sorted(
        str(p.relative_to(submission_dir))
        for p in submission_dir.rglob("*")
        if p.is_file()
    )
    if not expected_extensions:
        return None, None, generated

    exp_counter = Counter(expected_extensions)
    gen_counter = Counter(Path(f).suffix.lower() for f in generated if Path(f).suffix)
    matched = sum(min(gen_counter[ext], cnt) for ext, cnt in exp_counter.items())
    total = sum(exp_counter.values())
    score = matched / total if total else None
    correct = matched == total if total else None
    return correct, score, generated


async def _run_one(
    index: int,
    row: dict,
    workspace_root: Path,
    cache_dir: str,
    agent_options: Any,
    semaphore: asyncio.Semaphore,
) -> GDPValResult:
    from src.schemas import AgentResponse
    from src.agent_profiles.base import Agent

    task_id = str(row.get("task_id") or f"gdpval-{index:04d}")
    workspace_dir = workspace_root / task_id
    submission_dir = workspace_dir / "submission"
    submission_dir.mkdir(parents=True, exist_ok=True)

    ref_paths = _materialize_reference_files(
        row.get("reference_files") or [], cache_dir, workspace_dir
    )
    deliverable_files = row.get("deliverable_files") or []
    expected_extensions = [Path(f).suffix.lower() for f in deliverable_files if Path(f).suffix]
    prompt = _build_prompt(row, ref_paths, submission_dir)

    async with semaphore:
        try:
            agent = Agent(agent_options, AgentResponse)
            trace = await agent.run(prompt)
            prediction = str(trace.output.final_answer) if trace.output else None
            error = None
        except Exception as exc:
            prediction = None
            trace = None
            error = str(exc)

    correct, score, generated_files = _score(submission_dir, expected_extensions)

    return GDPValResult(
        index=index,
        task_id=task_id,
        sector=str(row.get("sector") or ""),
        occupation=str(row.get("occupation") or ""),
        prompt=prompt,
        expected_extensions=expected_extensions,
        prediction=prediction,
        generated_files=generated_files,
        correct=correct,
        score=score,
        error=error,
        details={
            "workspace_dir": str(workspace_dir),
            "submission_dir": str(submission_dir),
            "reference_file_paths": [str(p) for p in ref_paths],
        },
    )


async def evaluate_gdpval(
    cache_dir: str,
    workspace_dir: str,
    output_path: str,
    model: str | None = None,
    max_concurrent: int = 4,
    limit: int | None = None,
    resume: bool = True,
) -> list[GDPValResult]:
    from src.agent_profiles.gdpval_agent import get_gdpval_agent_options

    run_ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    workspace_root = Path(workspace_dir) / run_ts
    workspace_root.mkdir(parents=True, exist_ok=True)
    out_path = Path(output_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    rows = _load_dataset(cache_dir, limit=limit)
    print(f"GDPVal: {len(rows)} tasks loaded")

    # Resume: skip already-completed task_ids
    existing: list[GDPValResult] = []
    done_ids: set[str] = set()
    if resume and out_path.exists():
        try:
            with open(out_path, "rb") as f:
                existing = pickle.load(f)
            done_ids = {r.task_id for r in existing if r.error is None}
            print(f"Resuming: {len(done_ids)} already done")
        except Exception:
            pass

    agent_options = get_gdpval_agent_options(model=model)
    semaphore = asyncio.Semaphore(max_concurrent)

    tasks = [
        _run_one(i, row, workspace_root, cache_dir, agent_options, semaphore)
        for i, row in enumerate(rows)
        if str(row.get("task_id") or f"gdpval-{i:04d}") not in done_ids
    ]

    results: list[GDPValResult] = list(existing)
    for coro in asyncio.as_completed(tasks):
        r = await coro
        results.append(r)
        with open(out_path, "wb") as f:
            pickle.dump(results, f)
        status = "✓" if r.correct else ("?" if r.correct is None else "✗")
        print(f"  [{status}] {r.task_id[:20]:20s}  score={r.score}  files={r.generated_files}")

    return results


def print_summary(results: list[GDPValResult]) -> None:
    total = len(results)
    errors = sum(1 for r in results if r.error)
    scored = [r for r in results if r.correct is not None]
    correct = sum(1 for r in scored if r.correct)
    mean_score = sum(r.score for r in scored if r.score is not None) / max(len(scored), 1)

    print(f"\n{'=' * 50}")
    print(f"Total tasks:  {total}")
    print(f"Errors:       {errors}")
    print(f"Scored:       {len(scored)}")
    print(f"Correct:      {correct} / {len(scored)} ({correct / max(len(scored), 1) * 100:.1f}%)")
    print(f"Mean score:   {mean_score:.3f}")

    sector_acc: dict[str, list[float]] = {}
    for r in scored:
        sector_acc.setdefault(r.sector, []).append(float(r.correct or 0))
    print("\nBy sector:")
    for sector, vals in sorted(sector_acc.items()):
        print(f"  {sector[:40]:40s}  {sum(vals):.0f}/{len(vals)}  ({sum(vals)/len(vals)*100:.0f}%)")


if __name__ == "__main__":
    import argparse
    from src.agent_profiles import set_sdk, set_vllm_config

    parser = argparse.ArgumentParser(description="Run GDPVal evaluation with EvoSkill agent")
    parser.add_argument("--cache-dir", required=True, help="HuggingFace cache dir for openai/gdpval")
    parser.add_argument("--workspace-dir", default="outputs/gdpval_workspaces", help="Root dir for task workspaces")
    parser.add_argument("--output", default="results/gdpval_eval.pkl", help="Output pickle path")
    parser.add_argument("--model", default=None, help="Model override (e.g. local HF path)")
    parser.add_argument("--max-concurrent", type=int, default=4)
    parser.add_argument("--limit", type=int, default=None, help="Limit number of tasks")
    parser.add_argument("--no-resume", action="store_true", help="Disable resume from existing results")
    parser.add_argument("--sdk", default="vllm", choices=["claude", "vllm"], help="SDK backend (default: vllm)")
    parser.add_argument("--vllm-base-url", default="http://localhost:8765/v1", help="vLLM server base URL")
    parser.add_argument("--vllm-max-tokens", type=int, default=8192, help="Max tokens for vLLM generation")
    parser.add_argument("--vllm-context-length", type=int, default=131072, help="vLLM model context window size")
    args = parser.parse_args()

    set_sdk(args.sdk)
    if args.sdk == "vllm":
        set_vllm_config(
            base_url=args.vllm_base_url,
            model_name=args.model or "",
            max_tokens=args.vllm_max_tokens,
            context_length=args.vllm_context_length,
        )
        print(f"[SDK] vLLM backend: {args.vllm_base_url}, model: {args.model}, max_tokens: {args.vllm_max_tokens}")
    else:
        print(f"[SDK] {args.sdk} backend")

    results = asyncio.run(evaluate_gdpval(
        cache_dir=args.cache_dir,
        workspace_dir=args.workspace_dir,
        output_path=args.output,
        model=args.model,
        max_concurrent=args.max_concurrent,
        limit=args.limit,
        resume=not args.no_resume,
    ))
    print_summary(results)
