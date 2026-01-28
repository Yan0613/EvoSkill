#!/usr/bin/env python3
"""Run full evaluation on OfficeQA dataset."""
import argparse
import asyncio
from pathlib import Path

import pandas as pd

from src.agent_profiles import Agent, base_agent_options, make_base_agent_options
from src.evaluation.eval_full import evaluate_full, load_results
from src.schemas import AgentResponse


async def main():
    parser = argparse.ArgumentParser(description="Evaluate agent on OfficeQA dataset")
    parser.add_argument(
        "--output",
        "-o",
        type=Path,
        default=Path("results/eval_results.pkl"),
        help="Output pkl file path",
    )
    parser.add_argument(
        "--max-concurrent",
        "-c",
        type=int,
        default=8,
        help="Max concurrent evaluations",
    )
    parser.add_argument(
        "--no-resume",
        action="store_true",
        help="Don't resume from existing results (start fresh)",
    )
    parser.add_argument(
        "--difficulty",
        "-d",
        choices=["all", "easy", "hard"],
        default="all",
        help="Filter by difficulty",
    )
    parser.add_argument(
        "--num-samples",
        "-n",
        type=int,
        default=None,
        help="Limit to first N samples (default: all)",
    )
    parser.add_argument(
        "--model",
        "-m",
        type=str,
        choices=["opus", "sonnet", "haiku"],
        default=None,
        help="Model for base agent (default: opus)",
    )
    args = parser.parse_args()

    # Load dataset
    # dataset_path = Path(
    #     "~/mnt/shared-resources-sentient-research/salah_resources/datasets/officeqa/"
    # ).expanduser()
    dataset_path = Path(
        "~/officeqa/"
    ).expanduser()
    data = pd.read_csv(dataset_path / "officeqa.csv")

    # Filter by difficulty if requested
    if args.difficulty != "all":
        data = data[data["difficulty"] == args.difficulty]

    # Limit to num_samples if specified
    if args.num_samples is not None:
        data = data.head(args.num_samples)

    print(f"Dataset: {len(data)} samples ({args.difficulty})")

    # Prepare items with index
    items = [(i, row["question"], row["answer"]) for i, row in data.iterrows()]

    # Create agent and run
    agent_options = make_base_agent_options(model=args.model) if args.model else base_agent_options
    agent = Agent(agent_options, AgentResponse)

    model_info = f" (model: {args.model})" if args.model else " (model: opus)"
    print(f"Agent configured{model_info}")

    results = await evaluate_full(
        agent=agent,
        items=items,
        output_path=args.output,
        max_concurrent=args.max_concurrent,
        resume=not args.no_resume,
    )

    # Summary
    all_results = load_results(args.output)
    successful = [r for r in all_results if r.error is None]
    failed = [r for r in all_results if r.error is not None]

    print(f"\n{'='*50}")
    print(f"Total completed: {len(all_results)}/{len(data)}")
    print(f"Successful: {len(successful)}")
    print(f"Failed: {len(failed)}")
    if failed:
        print(f"Failed indices: {[r.index for r in failed]}")
    print(f"Results saved to: {args.output}")


if __name__ == "__main__":
    asyncio.run(main())
