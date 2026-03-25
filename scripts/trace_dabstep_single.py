#!/usr/bin/env python3
"""
Trace a single DABStep sample with vLLM backend.
Records full input, output, tool calls, and timing to a JSON file.

Usage:
    python3 scripts/trace_dabstep_single.py \
        --vllm-base-url http://29.206.1.136:8765/v1 \
        --output results/trace_dabstep_single.json
"""
import argparse
import asyncio
import json
import time
from pathlib import Path

import pandas as pd

# ── Patch vLLM client to intercept all API calls ──────────────────────────────
_TRACE_LOG: list[dict] = []   # global trace collector


def _patch_openai_client():
    """Monkey-patch AsyncOpenAI.chat.completions.create to log every call."""
    try:
        from openai.resources.chat import completions as _comp_mod
        _orig_create = _comp_mod.AsyncCompletions.create

        async def _traced_create(self_inner, **kwargs):
            call_idx = len(_TRACE_LOG)
            entry: dict = {
                "turn": call_idx + 1,
                "request": {
                    "model": kwargs.get("model"),
                    "messages": _safe_copy(kwargs.get("messages", [])),
                    "tools": kwargs.get("tools"),
                    "tool_choice": kwargs.get("tool_choice"),
                    "parallel_tool_calls": kwargs.get("parallel_tool_calls"),
                    "max_tokens": kwargs.get("max_tokens"),
                    "temperature": kwargs.get("temperature"),
                },
                "response": None,
                "error": None,
                "duration_ms": None,
            }
            _TRACE_LOG.append(entry)
            t0 = time.monotonic()
            try:
                resp = await _orig_create(self_inner, **kwargs)
                entry["duration_ms"] = int((time.monotonic() - t0) * 1000)
                entry["response"] = _serialize_response(resp)
                return resp
            except Exception as e:
                entry["duration_ms"] = int((time.monotonic() - t0) * 1000)
                entry["error"] = str(e)
                raise

        _comp_mod.AsyncCompletions.create = _traced_create
        print("[Tracer] OpenAI AsyncCompletions.create patched ✓")
    except Exception as e:
        print(f"[Tracer] WARNING: could not patch OpenAI client: {e}")


def _safe_copy(obj):
    """Deep-copy an object, truncating very long strings to avoid huge JSON."""
    MAX_STR = 8000
    if isinstance(obj, str):
        return obj if len(obj) <= MAX_STR else obj[:MAX_STR] + f"... [truncated {len(obj)-MAX_STR} chars]"
    if isinstance(obj, list):
        return [_safe_copy(x) for x in obj]
    if isinstance(obj, dict):
        return {k: _safe_copy(v) for k, v in obj.items()}
    return obj


def _serialize_response(resp) -> dict:
    """Convert an OpenAI ChatCompletion response to a plain dict."""
    try:
        choices = []
        for c in resp.choices:
            msg = c.message
            tool_calls = None
            if msg.tool_calls:
                tool_calls = [
                    {
                        "id": tc.id,
                        "type": tc.type,
                        "function": {
                            "name": tc.function.name,
                            "arguments": tc.function.arguments,
                        },
                    }
                    for tc in msg.tool_calls
                ]
            choices.append({
                "index": c.index,
                "finish_reason": c.finish_reason,
                "message": {
                    "role": msg.role,
                    "content": _safe_copy(msg.content or ""),
                    "tool_calls": tool_calls,
                },
            })
        usage = None
        if resp.usage:
            usage = {
                "prompt_tokens": resp.usage.prompt_tokens,
                "completion_tokens": resp.usage.completion_tokens,
                "total_tokens": resp.usage.total_tokens,
            }
        return {
            "id": resp.id,
            "model": resp.model,
            "choices": choices,
            "usage": usage,
        }
    except Exception as e:
        return {"_serialize_error": str(e)}


# ── Main ───────────────────────────────────────────────────────────────────────

PROMPT_TEMPLATE = """You are an expert data analyst and you will answer factoid questions by loading and referencing the files/documents listed below.
You have these files available:
{context_files}

Here is the question you need to answer:
{question}

Here are the guidelines you must follow when answering the question above:
{guidelines}
"""


async def main():
    parser = argparse.ArgumentParser(description="Trace a single DABStep sample with vLLM")
    parser.add_argument("--dataset", type=Path, default=Path(".dataset/dabstep_dev.csv"))
    parser.add_argument("--data-dir", type=str, default=".dataset/DABstep-data/data/context")
    parser.add_argument("--sample-index", type=int, default=0, help="Row index in CSV to evaluate (default: 0)")
    parser.add_argument("--model", type=str, default="Qwen/Qwen2.5-7B-Instruct")
    parser.add_argument("--vllm-base-url", type=str, default="http://29.206.1.136:8765/v1")
    parser.add_argument("--vllm-max-tokens", type=int, default=8192)
    parser.add_argument("--vllm-context-length", type=int, default=32768)
    parser.add_argument("--output", type=Path, default=Path("results/trace_dabstep_single.json"))
    args = parser.parse_args()

    # Patch OpenAI client BEFORE importing agent code
    _patch_openai_client()

    # Now import agent code
    import sys
    sys.path.insert(0, str(Path(__file__).parent.parent))

    from src.agent_profiles import set_sdk, set_vllm_config, make_dabstep_agent_options, Agent
    from src.schemas import AgentResponse

    set_sdk("vllm")
    set_vllm_config(
        base_url=args.vllm_base_url,
        model_name=args.model,
        max_tokens=args.vllm_max_tokens,
        context_length=args.vllm_context_length,
    )
    print(f"[Config] vLLM: {args.vllm_base_url}, model: {args.model}")

    # Load dataset
    data = pd.read_csv(args.dataset)
    row = data.iloc[args.sample_index]
    task_id = row["task_id"]
    question = row["question"]
    answer = row["answer"]
    guidelines = row["guidelines"]
    level = row.get("level", "unknown")

    print(f"\n[Sample] task_id={task_id}, level={level}")
    print(f"[Question] {question}")
    print(f"[Ground Truth] {answer}")

    # Build context files list
    data_dir = Path(args.data_dir).resolve()
    if data_dir.exists():
        context_file_names = sorted(f.name for f in data_dir.iterdir() if f.is_file())
        context_files_text = "\n".join(f"- {data_dir / name}" for name in context_file_names)
    else:
        context_files_text = "(no context files available)"

    prompt = PROMPT_TEMPLATE.format(
        context_files=context_files_text,
        question=question,
        guidelines=guidelines,
    )

    # Create agent
    agent_options_factory = make_dabstep_agent_options(model=args.model, data_dir=args.data_dir)
    agent = Agent(agent_options_factory, AgentResponse)

    # Run agent
    print("\n[Running agent...]")
    t_start = time.monotonic()
    trace = await agent.run(prompt)
    total_ms = int((time.monotonic() - t_start) * 1000)
    print(f"[Done] {total_ms}ms, turns={trace.num_turns}")

    # Score
    from src.evaluation.dabstep_scorer import question_scorer
    final_answer = str(trace.output.final_answer) if trace.output and trace.output.final_answer else ""
    score = question_scorer(final_answer, str(answer))
    print(f"\n[Result]")
    print(f"  Final Answer : {final_answer!r}")
    print(f"  Ground Truth : {answer!r}")
    print(f"  Correct      : {score}")
    print(f"  Is Error     : {trace.is_error}")
    if trace.parse_error:
        print(f"  Parse Error  : {trace.parse_error}")

    # Build output JSON
    output_data = {
        "meta": {
            "task_id": int(task_id),
            "level": str(level),
            "model": args.model,
            "vllm_base_url": args.vllm_base_url,
            "vllm_max_tokens": args.vllm_max_tokens,
            "vllm_context_length": args.vllm_context_length,
            "total_duration_ms": total_ms,
            "num_turns": trace.num_turns,
            "is_error": trace.is_error,
            "parse_error": trace.parse_error,
        },
        "input": {
            "question": question,
            "guidelines": guidelines,
            "context_files": context_file_names if data_dir.exists() else [],
            "full_prompt": _safe_copy(prompt),
        },
        "output": {
            "final_answer": final_answer,
            "ground_truth": str(answer),
            "correct": bool(score),
            "raw_model_output": _safe_copy(trace.result or ""),
        },
        "tool_call_trace": _TRACE_LOG,
        "summary": {
            "total_api_calls": len(_TRACE_LOG),
            "tool_calls_made": sum(
                len(entry["response"]["choices"][0]["message"]["tool_calls"] or [])
                for entry in _TRACE_LOG
                if entry["response"] and entry["response"].get("choices")
            ),
            "total_prompt_tokens": sum(
                (entry["response"].get("usage") or {}).get("prompt_tokens", 0)
                for entry in _TRACE_LOG
                if entry["response"]
            ),
            "total_completion_tokens": sum(
                (entry["response"].get("usage") or {}).get("completion_tokens", 0)
                for entry in _TRACE_LOG
                if entry["response"]
            ),
        },
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(output_data, f, ensure_ascii=False, indent=2)

    print(f"\n[Saved] {args.output}")
    print(f"[Summary] API calls={output_data['summary']['total_api_calls']}, "
          f"tool_calls={output_data['summary']['tool_calls_made']}, "
          f"prompt_tokens={output_data['summary']['total_prompt_tokens']}, "
          f"completion_tokens={output_data['summary']['total_completion_tokens']}")


if __name__ == "__main__":
    asyncio.run(main())
