#!/usr/bin/env python3
"""Run self-improving agent loop on the GDPVal benchmark.

GDPVal tasks require creating deliverable files (xlsx, pptx, csv, md …).
Scoring is manifest-based: the agent's FINAL_ANSWER lists the files it
created; we compare their extensions against the expected file types.

Usage (vLLM backend):
    python scripts/run_loop_gdpval.py \\
        --vllm-base-url http://127.0.0.1:8765/v1 \\
        --model /path/to/model \\
        --cache-dir /path/to/gdpval/hf_cache \\
        --workspace-dir outputs/gdpval_loop_workspaces

Usage (Claude backend, for testing):
    python scripts/run_loop_gdpval.py \\
        --model claude-opus-4-5-20251101
"""

import argparse
import asyncio
from pathlib import Path


# ── Workspace helpers ──────────────────────────────────────────────────────────

def _prepare_loop_data(
    cache_dir: str,
    workspace_root: Path,
    limit: int | None = None,
) -> tuple[dict[str, list[tuple[str, str]]], list[tuple[str, str, str]]]:
    """Load GDPVal dataset and build train_pools + val_data for the loop.

    Returns:
        train_pools: {sector: [(question, ground_truth), ...]}
        val_data:    [(question, ground_truth, sector), ...]

    Where:
        question     = full prepared prompt (submission_dir baked in)
        ground_truth = comma-separated expected extensions, e.g. ".xlsx,.md"
    """
    from datasets import load_dataset
    from huggingface_hub import hf_hub_download

    ds = load_dataset("openai/gdpval", split="train", cache_dir=cache_dir)
    if limit:
        ds = ds.select(range(min(limit, len(ds))))

    rows = list(ds)

    # Build (sector, question, ground_truth) triples
    triples: list[tuple[str, str, str]] = []
    for i, row in enumerate(rows):
        task_id = str(row.get("task_id") or f"gdpval-{i:04d}")
        sector = str(row.get("sector") or "general")

        # Create workspace
        submission_dir = workspace_root / task_id / "submission"
        submission_dir.mkdir(parents=True, exist_ok=True)

        # Materialize reference files
        ref_paths: list[Path] = []
        for rel_path in (row.get("reference_files") or []):
            p = Path(rel_path)
            if p.is_absolute() and p.exists():
                ref_paths.append(p)
                continue
            try:
                local = hf_hub_download(
                    repo_id="openai/gdpval",
                    repo_type="dataset",
                    filename=rel_path,
                    cache_dir=cache_dir,
                    local_files_only=True,
                )
                ref_paths.append(Path(local))
            except Exception:
                try:
                    local = hf_hub_download(
                        repo_id="openai/gdpval",
                        repo_type="dataset",
                        filename=rel_path,
                        cache_dir=cache_dir,
                    )
                    ref_paths.append(Path(local))
                except Exception:
                    pass

        # Expected extensions (ground truth for scoring)
        deliverable_files = row.get("deliverable_files") or []
        expected_exts = [Path(f).suffix.lower() for f in deliverable_files if Path(f).suffix]
        ground_truth = ",".join(expected_exts)

        # Build prepared prompt
        ref_block = (
            "\n".join(f"- {p.name}: {p}" for p in ref_paths)
            if ref_paths else "- No reference files materialized locally."
        )
        urls = row.get("reference_file_urls") or []
        url_block = "\n".join(f"- {u}" for u in urls) if urls else "- None provided."
        ext_block = ", ".join(expected_exts) if expected_exts else "unspecified"
        rubric = str(row.get("rubric_pretty") or "").strip()
        if len(rubric) > 4000:
            rubric = rubric[:4000] + "\n[rubric truncated]"

        prompt_text = str(row.get("prompt") or "").strip()
        question = (
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
            f"one per line, wrapped in <FINAL_ANSWER>...</FINAL_ANSWER> tags."
        )

        triples.append((sector, question, ground_truth))

    # Stratified split: ~18% train, ~12% val per sector
    from collections import defaultdict
    import random

    by_sector: dict[str, list[tuple[str, str]]] = defaultdict(list)
    for sector, question, ground_truth in triples:
        by_sector[sector].append((question, ground_truth))

    train_pools: dict[str, list[tuple[str, str]]] = {}
    val_data: list[tuple[str, str, str]] = []

    rng = random.Random(42)
    for sector, items in by_sector.items():
        items_shuffled = list(items)
        rng.shuffle(items_shuffled)
        n_train = max(1, int(len(items_shuffled) * 0.18))
        n_val = max(1, int(len(items_shuffled) * 0.12))
        train_pools[sector] = items_shuffled[:n_train]
        val_data.extend((q, a, sector) for q, a in items_shuffled[n_train:n_train + n_val])

    return train_pools, val_data


# ── CLI ────────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run self-improving agent loop on GDPVal")
    parser.add_argument("--mode", default="skill_only", choices=["skill_only", "prompt_only"])
    parser.add_argument("--max-iterations", type=int, default=20)
    parser.add_argument("--frontier-size", type=int, default=3)
    parser.add_argument("--no-improvement-limit", type=int, default=5)
    parser.add_argument("--concurrency", type=int, default=4)
    parser.add_argument("--failure-samples", type=int, default=3)
    parser.add_argument("--no-cache", action="store_true")
    parser.add_argument("--no-reset-feedback", action="store_true")
    parser.add_argument("--continue", dest="continue_loop", action="store_true")
    parser.add_argument("--limit", type=int, default=None, help="Limit dataset size (for testing)")
    parser.add_argument(
        "--cache-dir",
        default="/hkfs/work/workspace/scratch/lmu_eqm3765-skill/SkillEvolve/data/benchmarks/gdpval",
        help="HuggingFace cache dir for openai/gdpval",
    )
    parser.add_argument(
        "--workspace-dir",
        default="outputs/gdpval_loop_workspaces",
        help="Root dir for per-task submission workspaces",
    )
    parser.add_argument("--model", default="claude-opus-4-5-20251101")
    parser.add_argument("--vllm-base-url", default=None, help="Enable vLLM mode")
    parser.add_argument("--vllm-max-tokens", type=int, default=8192)
    parser.add_argument("--vllm-context-length", type=int, default=131072)
    return parser.parse_args()


async def main(args: argparse.Namespace) -> None:
    # ── Configure SDK ──────────────────────────────────────────────────────────
    if args.vllm_base_url:
        from src.agent_profiles import set_sdk, set_vllm_config
        set_sdk("vllm")
        set_vllm_config(
            base_url=args.vllm_base_url,
            model_name=args.model,
            max_tokens=args.vllm_max_tokens,
            context_length=args.vllm_context_length,
        )
        print(f"[Config] vLLM: {args.vllm_base_url}, model: {args.model}")
    else:
        print(f"[Config] Claude backend, model: {args.model}")

    # ── Prepare workspaces & loop data ────────────────────────────────────────
    from datetime import datetime, timezone
    workspace_root = Path(args.workspace_dir) / datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    print(f"[Config] Workspace root: {workspace_root}")
    print("[Data] Loading GDPVal dataset and preparing workspaces …")

    train_pools, val_data = _prepare_loop_data(
        cache_dir=args.cache_dir,
        workspace_root=workspace_root,
        limit=args.limit,
    )

    sectors = sorted(train_pools.keys())
    total_train = sum(len(v) for v in train_pools.values())
    print(f"[Data] Sectors ({len(sectors)}): {', '.join(sectors)}")
    print(f"[Data] Train: {total_train}  Val: {len(val_data)}")

    # ── Imports (after SDK configured) ────────────────────────────────────────
    from src.loop import SelfImprovingLoop, LoopConfig, LoopAgents
    from src.agent_profiles import (
        Agent,
        skill_proposer_options,
        prompt_proposer_options,
        skill_generator_options,
        prompt_generator_options,
    )
    from src.agent_profiles.gdpval_agent import make_gdpval_agent_options
    from src.agent_profiles.skill_generator import get_project_root
    from src.evaluation.gdpval_scorer import score_gdpval
    from src.registry import ProgramManager
    from src.schemas import (
        AgentResponse,
        SkillProposerResponse,
        PromptProposerResponse,
        ToolGeneratorResponse,
        PromptGeneratorResponse,
    )

    base_options = make_gdpval_agent_options(model=args.model)

    agents = LoopAgents(
        base=Agent(base_options, AgentResponse),
        skill_proposer=Agent(skill_proposer_options, SkillProposerResponse),
        prompt_proposer=Agent(prompt_proposer_options, PromptProposerResponse),
        skill_generator=Agent(skill_generator_options, ToolGeneratorResponse),
        prompt_generator=Agent(prompt_generator_options, PromptGeneratorResponse),
    )
    manager = ProgramManager(cwd=get_project_root())

    config = LoopConfig(
        max_iterations=args.max_iterations,
        frontier_size=args.frontier_size,
        no_improvement_limit=args.no_improvement_limit,
        concurrency=args.concurrency,
        evolution_mode=args.mode,
        failure_sample_count=args.failure_samples,
        categories_per_batch=args.failure_samples,
        cache_enabled=not args.no_cache,
        reset_feedback=not args.no_reset_feedback,
        continue_mode=args.continue_loop,
    )

    print(f"[Loop] evolution_mode={args.mode}, max_iter={args.max_iterations}")
    loop = SelfImprovingLoop(config, agents, manager, train_pools, val_data, scorer=score_gdpval)
    result = await loop.run()

    print(f"\nBest: {result.best_program} ({result.best_score:.2%})")
    print(f"Frontier: {result.frontier}")


if __name__ == "__main__":
    args = parse_args()
    asyncio.run(main(args))
