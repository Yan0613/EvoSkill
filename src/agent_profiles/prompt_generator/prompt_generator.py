from typing import TYPE_CHECKING, Any, Union
if TYPE_CHECKING:
    from claude_agent_sdk import ClaudeAgentOptions
from src.schemas import PromptGeneratorResponse
from src.agent_profiles.prompt_generator.prompt import PROMPT_GENERATOR_SYSTEM_PROMPT
from src.agent_profiles.skill_generator import get_project_root
from src.agent_profiles.sdk_config import is_claude_sdk, is_vllm_sdk


PROMPT_GENERATOR_TOOLS = ["Read", "Bash", "Glob", "Grep", "WebFetch", "WebSearch", "TodoWrite", "BashOutput"]


prompt_generator_system_prompt = {
    "type": "preset",
    "preset": "claude_code",
    "append": PROMPT_GENERATOR_SYSTEM_PROMPT.strip()
}

prompt_generator_output_format = {
    "type": "json_schema",
    "schema": PromptGeneratorResponse.model_json_schema()
}


def _make_prompt_generator_options() -> Union[Any, dict]:
    """Factory function that creates agent options for prompt generator.

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
        vllm_system = tool_calling_preamble + PROMPT_GENERATOR_SYSTEM_PROMPT.strip()
        return {
            "system": vllm_system,
            "tools": HF_TOOLS,
            "model_id": "",
            "backend": "vllm" if is_vllm_sdk() else "",
            "output_format": prompt_generator_output_format,
        }

    from claude_agent_sdk import ClaudeAgentOptions
    return ClaudeAgentOptions(
        output_format=prompt_generator_output_format,
        system_prompt=prompt_generator_system_prompt,
        allowed_tools=PROMPT_GENERATOR_TOOLS,
        cwd=get_project_root(),
    )


prompt_generator_options = _make_prompt_generator_options
