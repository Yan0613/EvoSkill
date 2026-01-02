#!/usr/bin/env python3
"""Run self-improving agent loop."""

import argparse
import asyncio

import pandas as pd

from src.loop import SelfImprovingLoop, LoopConfig, LoopAgents
from src.agent_profiles import (
    Agent,
    base_agent_options,
    skill_proposer_options,
    prompt_proposer_options,
    skill_generator_options,
    prompt_generator_options,
)
from src.agent_profiles.skill_generator import get_project_root
from src.registry import ProgramManager
from src.schemas import (
    AgentResponse,
    SkillProposerResponse,
    PromptProposerResponse,
    ToolGeneratorResponse,
    PromptGeneratorResponse,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run self-improving agent loop")
    parser.add_argument(
        "--mode",
        type=str,
        choices=["skill_only", "prompt_only"],
        default="skill_only",
        help="Evolution mode: 'skill_only' or 'prompt_only' (default: skill_only)",
    )
    parser.add_argument(
        "--max-iterations",
        type=int,
        default=20,
        help="Maximum number of improvement iterations (default: 20)",
    )
    parser.add_argument(
        "--frontier-size",
        type=int,
        default=3,
        help="Number of top-performing programs to keep (default: 3)",
    )
    parser.add_argument(
        "--no-improvement-limit",
        type=int,
        default=5,
        help="Stop after this many iterations without improvement (default: 5)",
    )
    parser.add_argument(
        "--concurrency",
        type=int,
        default=4,
        help="Number of concurrent evaluations (default: 4)",
    )
    parser.add_argument(
        "--no-cache",
        action="store_true",
        help="Disable run caching",
    )
    parser.add_argument(
        "--no-reset-feedback",
        action="store_true",
        help="Don't reset feedback history on start",
    )
    return parser.parse_args()


async def main(args: argparse.Namespace):
    data = pd.read_csv('.dataset/train_set.csv')

    train = data.sample(20, random_state=42)
    val = data.drop(train.index).sample(5, random_state=77)

    train_data = [(row.question, row.ground_truth) for _, row in train.iterrows()]
    val_data = [(row.question, row.ground_truth) for _, row in val.iterrows()]

    agents = LoopAgents(
        base=Agent(base_agent_options, AgentResponse),
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
        cache_enabled=not args.no_cache,
        reset_feedback=not args.no_reset_feedback,
    )

    print(f"Running loop with evolution_mode={args.mode}")
    loop = SelfImprovingLoop(config, agents, manager, train_data, val_data)
    result = await loop.run()

    print(f"Best: {result.best_program} ({result.best_score:.2%})")
    print(f"Frontier: {result.frontier}")


if __name__ == "__main__":
    args = parse_args()
    asyncio.run(main(args))
