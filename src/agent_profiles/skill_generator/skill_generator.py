import os
from pathlib import Path
from typing import TYPE_CHECKING, Any, Union
if TYPE_CHECKING:
    from claude_agent_sdk import ClaudeAgentOptions

from src.schemas import ToolGeneratorResponse
from src.agent_profiles.skill_generator.prompt import SKILL_GENERATOR_SYSTEM_PROMPT
from src.agent_profiles.sdk_config import is_claude_sdk, is_vllm_sdk

def get_project_root() -> str:
    """Get the project root directory by looking for pyproject.toml."""
    current = Path(__file__).resolve()
    for parent in current.parents:
        if (parent / "pyproject.toml").exists():
            return str(parent)
    # Fallback: go up 3 levels from current file (src/agent_profiles/skill_generator/)
    return str(current.parent.parent.parent)


skill_generator_system_prompt = {
    "type": "preset",
    "preset": "claude_code",
    "append": SKILL_GENERATOR_SYSTEM_PROMPT.strip()
}

skill_generator_output_format = {
    "type": "json_schema",
    "schema": ToolGeneratorResponse.model_json_schema()
}

# Default available tools for skill generator
SKILL_GENERATOR_TOOLS = ["Read", "Write", "Bash", "Glob", "Grep", "Edit", "WebFetch", "WebSearch", "TodoWrite", "BashOutput", "Skill"]


def _make_skill_generator_options() -> Union[Any, dict]:
    """Factory function that creates agent options for skill generator.

    Returns ClaudeAgentOptions for Claude SDK or dict for HF/vLLM backends.
    """
    if not is_claude_sdk():
        # HuggingFace / vLLM path: return a plain dict
        # Reuse the same system prompt as Claude SDK, prefixed with tool-calling rules
        from src.agent_profiles.hf_tools import HF_TOOLS
        tool_calling_preamble = (
            "## CRITICAL TOOL-CALLING RULES\n\n"
            "1. ALWAYS use the OpenAI function-calling API to invoke tools. "
            "NEVER write tool calls as raw text or XML in your response content.\n\n"
            "2. Call ONE tool at a time. After each tool call, WAIT for the result "
            "before deciding what to do next.\n\n"
            "3. When you have your final answer, wrap it in <FINAL_ANSWER>...</FINAL_ANSWER> tags.\n\n"
        )
        vllm_system = tool_calling_preamble + SKILL_GENERATOR_SYSTEM_PROMPT.strip()
        return {
            "system": vllm_system,
            "tools": HF_TOOLS,
            "model_id": "",
            "backend": "vllm" if is_vllm_sdk() else "",
            "output_format": skill_generator_output_format,
        }

    from claude_agent_sdk import ClaudeAgentOptions
    return ClaudeAgentOptions(
        output_format=skill_generator_output_format,
        system_prompt=skill_generator_system_prompt,
        setting_sources=["user", "project"],
        allowed_tools=SKILL_GENERATOR_TOOLS,
        permission_mode='acceptEdits',
        cwd=get_project_root(),
    )


skill_generator_options = _make_skill_generator_options
