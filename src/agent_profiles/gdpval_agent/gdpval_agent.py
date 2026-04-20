from __future__ import annotations

from typing import Any, Union


GDPVAL_VLLM_SYSTEM = (
    "You are an expert knowledge worker who creates professional deliverable files. "
    "You have access to tools (Read, Bash, Write) to read reference files and produce output files. "
    "Always read the reference files and rubric before creating any deliverable.\n\n"
    "## CRITICAL TOOL-CALLING RULES — you MUST follow these exactly:\n\n"
    "1. ALWAYS use the OpenAI function-calling API to invoke tools. "
    "NEVER write tool calls as raw text or XML in your response content.\n\n"
    "2. Call ONE tool at a time. After each tool call, WAIT for the result "
    "before deciding what to do next.\n\n"
    "3. Do NOT include your final answer in the same turn as a tool call. "
    "First create the files, then give your answer in a separate turn.\n\n"
    "4. When you have finished creating all deliverable files, output ONLY the file manifest "
    "wrapped in <FINAL_ANSWER>...</FINAL_ANSWER> tags — one file name per line, nothing else."
)


def get_gdpval_agent_options(model: str | None = None) -> Union[Any, dict]:
    from src.agent_profiles.sdk_config import is_claude_sdk, is_vllm_sdk
    from src.agent_profiles.hf_tools import HF_TOOLS

    if is_vllm_sdk():
        return {
            "system": GDPVAL_VLLM_SYSTEM,
            "tools": HF_TOOLS,
            "model_id": model or "",
            "backend": "vllm",
        }

    if not is_claude_sdk():
        return {
            "system": GDPVAL_VLLM_SYSTEM,
            "tools": HF_TOOLS,
            "model_id": model or "",
            "backend": "",
        }

    from claude_agent_sdk import ClaudeAgentOptions
    from src.agent_profiles.skill_generator import get_project_root
    from src.schemas import AgentResponse

    options = ClaudeAgentOptions(
        system_prompt={
            "type": "text",
            "text": (
                "You are an expert knowledge worker who creates professional deliverable files. "
                "Read all reference files carefully, follow the rubric, and create the required "
                "output files in the submission directory. Return only the file manifest."
            ),
        },
        output_format={"type": "json_schema", "schema": AgentResponse.model_json_schema()},
        allowed_tools=["Read", "Write", "Bash", "Glob", "Grep"],
        setting_sources=["user", "project"],
        permission_mode="acceptEdits",
        cwd=get_project_root(),
        max_buffer_size=10 * 1024 * 1024,
    )
    if model:
        options.model = model
    return options


def make_gdpval_agent_options(model: str | None = None):
    def factory():
        return get_gdpval_agent_options(model=model)
    return factory
