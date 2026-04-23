#!/usr/bin/env python3
"""Run full evaluation on OfficeQA dataset."""

import asyncio
from pathlib import Path
from typing import Literal, Optional

import pandas as pd
from pydantic import Field
from pydantic_settings import (
    BaseSettings,
    SettingsConfigDict,
)

from src.agent_profiles import (
    Agent,
    base_agent_options,
    make_base_agent_options,
    set_sdk,
    set_hf_config,
<<<<<<< Updated upstream
    set_vllm_config,
=======
>>>>>>> Stashed changes
)
from src.evaluation import score_answer
from src.evaluation.eval_full import evaluate_full, load_results
from src.schemas import AgentResponse


class EvalSettings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        cli_parse_args=True,
    )
    output: Path = Field(
        default=Path("results/eval_results.pkl"), description="Output pkl file path"
    )
    max_concurrent: int = Field(default=8, description="Max concurrent evaluations")
    resume: bool = Field(default=True, description="Resume from existing results")
    difficulty: Literal["all", "easy", "hard"] = Field(
        default="all", description="Filter by difficulty"
    )
    num_samples: Optional[int] = Field(
        default=None, description="Limit to first N samples"
    )
    model: Optional[str] = Field(
        default="claude-opus-4-5-20251101",
        description="Model for base agent (opus, sonnet, haiku)",
    )
    dataset_path: Path = Field(
        default=Path("~/officeqa/officeqa.csv").expanduser(),
        description="Path to OfficeQA dataset CSV",
    )
<<<<<<< Updated upstream
    sdk: Literal["claude", "opencode", "huggingface", "vllm"] = Field(
        default="claude",
        description="SDK to use: 'claude', 'opencode', 'huggingface', or 'vllm'",
=======
    sdk: Literal["claude", "opencode", "huggingface"] = Field(
        default="claude",
        description="SDK to use: 'claude', 'opencode', or 'huggingface'",
>>>>>>> Stashed changes
    )
    hf_model: str = Field(
        default="Qwen/Qwen3-4B",
        description="HuggingFace model name or local path (used when sdk=huggingface)",
    )
    hf_max_new_tokens: int = Field(
        default=512,
        description="Max new tokens for HuggingFace model generation",
    )
    hf_enable_thinking: bool = Field(
        default=False,
        description="Enable thinking mode for HuggingFace models that support it",
<<<<<<< Updated upstream
    )
    vllm_base_url: str = Field(
        default="http://localhost:8000/v1",
        description="vLLM server base URL (used when sdk=vllm)",
    )
    vllm_max_tokens: int = Field(
        default=2048,
        description="Max tokens for vLLM generation",
    )
    vllm_context_length: int = Field(
        default=131072,
        description="vLLM model context window size",
=======
>>>>>>> Stashed changes
    )


async def main(settings: EvalSettings):
    set_sdk(settings.sdk)
    if settings.sdk == "huggingface":
        set_hf_config(
            model_name=settings.hf_model,
            max_new_tokens=settings.hf_max_new_tokens,
            enable_thinking=settings.hf_enable_thinking,
        )
        print(f"[SDK] HuggingFace backend: {settings.hf_model}")
<<<<<<< Updated upstream
    elif settings.sdk == "vllm":
        set_vllm_config(
            base_url=settings.vllm_base_url,
            model_name=settings.model,
            max_tokens=settings.vllm_max_tokens,
            context_length=settings.vllm_context_length,
        )
        print(f"[SDK] vLLM backend: {settings.vllm_base_url}, model: {settings.model}, context: {settings.vllm_context_length}, max_tokens: {settings.vllm_max_tokens}")
=======
>>>>>>> Stashed changes
    else:
        print(f"[SDK] {settings.sdk} backend, model: {settings.model}")

    # Load dataset
    data = pd.read_csv(settings.dataset_path)

    # Filter by difficulty if requested
    if settings.difficulty != "all":
        data = data[data["difficulty"] == settings.difficulty]

    # Limit to num_samples if specified
    if settings.num_samples is not None:
        data = data.head(settings.num_samples)

    print(f"Dataset: {len(data)} samples ({settings.difficulty})")

    # Prepare items with index
    items = [
        (int(i), str(row["question"]), str(row["answer"])) for i, row in data.iterrows()
    ]

    # Create agent and run
    agent_options = (
        make_base_agent_options(model=settings.model)
        if settings.model
        else base_agent_options
    )
    agent = Agent(agent_options, AgentResponse)

    model_info = f" (model: {settings.model})" if settings.model else " (model: opus)"
    print(f"Agent configured{model_info}")

    await evaluate_full(
        agent=agent,
        items=items,
        output_path=settings.output,
        max_concurrent=settings.max_concurrent,
        resume=settings.resume,
    )

    # Summary
    all_results = load_results(settings.output)
    successful = [r for r in all_results if r.error is None]
    failed = [r for r in all_results if r.error is not None]

    # Compute accuracy across 6 tolerances
    tolerances = [0.05, 0.01, 0.001, 0.1, 0.0, 0.025]
    final_scores = []
    for r in successful:
        try:
            predicted = r.trace.output.final_answer
            score = sum(
                score_answer(ground_truth=r.ground_truth, predicted=predicted, tolerance=tol)
                for tol in tolerances
            ) / len(tolerances)
            final_scores.append(score)
        except Exception:
            pass

    print(f"\n{'=' * 50}")
    print(f"Total completed: {len(all_results)}/{len(data)}")
    print(f"Successful: {len(successful)}")
    print(f"Failed: {len(failed)}")
    if failed:
        print(f"Failed indices: {[r.index for r in failed]}")
    if final_scores:
        accuracy = sum(final_scores) / len(final_scores)
        exact = sum(1 for s in final_scores if s == 1.0)
        print(f"Accuracy (avg over 6 tolerances): {accuracy:.4f} ({accuracy:.2%})")
        print(f"Exact match (score=1.0): {exact}/{len(final_scores)} ({exact/len(final_scores):.2%})")
    print(f"Results saved to: {settings.output}")


if __name__ == "__main__":
    settings = EvalSettings()
    asyncio.run(main(settings))
