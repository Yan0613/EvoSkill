"""Feedback Descent: Open-Ended Text Optimization via Pairwise Comparison."""

from .feedback_descent import (
    FeedbackDescent,
    EvaluationResult,
    FeedbackEntry,
    FeedbackDescentResult,
    Proposer,
    Evaluator,
)

# Lazy import: api module has heavy dependencies (llm_sandbox, claude_agent_sdk, etc.)
# Only import when explicitly requested to allow huggingface/opencode SDK usage
def __getattr__(name):
    if name in ("EvoSkill", "EvalRunner", "EvalSummary"):
        from .api import EvoSkill, EvalRunner, EvalSummary
        globals()["EvoSkill"] = EvoSkill
        globals()["EvalRunner"] = EvalRunner
        globals()["EvalSummary"] = EvalSummary
        return globals()[name]
    raise AttributeError(f"module 'src' has no attribute {name!r}")


__all__ = [
    "FeedbackDescent",
    "EvaluationResult",
    "FeedbackEntry",
    "FeedbackDescentResult",
    "Proposer",
    "Evaluator",
    "EvoSkill",
    "EvalRunner",
    "EvalSummary",
]
