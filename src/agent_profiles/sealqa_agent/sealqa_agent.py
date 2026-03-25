from pathlib import Path
from typing import TYPE_CHECKING, Any, Union
if TYPE_CHECKING:
    from claude_agent_sdk import ClaudeAgentOptions
from src.schemas import AgentResponse
from src.agent_profiles.skill_generator import get_project_root
from src.agent_profiles.sdk_config import is_claude_sdk, is_vllm_sdk, is_huggingface_sdk


SEALQA_AGENT_TOOLS = ["Read", "Write", "Bash", "Glob", "Grep", "Edit", "WebFetch", "WebSearch", "TodoWrite", "BashOutput", "Skill"]

# Path to the prompt file (read at runtime)
PROMPT_FILE = Path(__file__).parent / "prompt.txt"


def get_sealqa_agent_options(model: str | None = None) -> Union[Any, dict]:
    """
    Factory function that creates agent options with the current prompt.

    Reads prompt.txt from disk each time, allowing dynamic updates
    without restarting the Python process.

    Args:
        model: Model to use (e.g., "opus", "sonnet"). If None, uses SDK default.
    """
    if not is_claude_sdk():
        # HuggingFace / vLLM path: return a plain dict with Python tool implementations
        from src.agent_profiles.hf_tools import HF_TOOLS
        # NOTE: Do NOT use prompt.txt here if it contains <tool_call> XML examples —
        # those confuse vLLM models into outputting raw XML instead of using the OpenAI tools API.
        # Use a concise system prompt without any XML tool-call examples.
        if is_vllm_sdk():
            vllm_system = (
                "You are a knowledgeable assistant with broad expertise. "
                "You have access to tools (Read, Bash, Grep, Glob) to look up information. "
                "Always use tools to gather information before answering — never guess or rely on training knowledge. "
                "Answer questions accurately and concisely.\n\n"
                "## CRITICAL TOOL-CALLING RULES — you MUST follow these exactly:\n\n"
                "1. ALWAYS use the OpenAI function-calling API to invoke tools. "
                "NEVER write tool calls as raw text or XML in your response content. "
                "Do NOT output <tool_call>...</tool_call> XML tags in your message text — "
                "the framework will NOT execute them. Only structured function calls work.\n\n"
                "2. Call ONE tool at a time. After each tool call, WAIT for the result "
                "before deciding what to do next. Do NOT call multiple tools in a single turn.\n\n"
                "3. Do NOT include your final answer in the same turn as a tool call. "
                "First call the tool, wait for the result, then give your answer in a separate turn.\n\n"
                "4. When you have enough information to answer, output your final answer "
                "wrapped in <FINAL_ANSWER>...</FINAL_ANSWER> tags. Do NOT use any other format."
            )
            return {
                "system": vllm_system,
                "tools": HF_TOOLS,
                "model_id": model or "",
                "backend": "vllm",
            }
        else:
            # HuggingFace backend
            prompt_text = PROMPT_FILE.read_text().strip() if PROMPT_FILE.exists() else ""
            return {
                "system": prompt_text,
                "tools": HF_TOOLS,
                "model_id": model or "",
            }

    from claude_agent_sdk import ClaudeAgentOptions
    # Read prompt from disk
    prompt_text = PROMPT_FILE.read_text().strip()

    system_prompt = {
        "type": "preset",
        "preset": "claude_code",
        "append": prompt_text
    }

    output_format = {
        "type": "json_schema",
        "schema": AgentResponse.model_json_schema()
    }

    options = ClaudeAgentOptions(
        system_prompt=system_prompt,
        output_format=output_format,
        allowed_tools=SEALQA_AGENT_TOOLS,
        setting_sources=["user", "project"],
        permission_mode='acceptEdits',
        cwd=get_project_root(),
        max_buffer_size=10 * 1024 * 1024,  # 10MB buffer (default is 1MB)
    )

    if model:
        options.model = model

    return options


def make_sealqa_agent_options(model: str | None = None):
    """Create a factory function for sealqa agent options with a specific model.

    Args:
        model: Model to use (e.g., "opus", "sonnet"). If None, uses SDK default.

    Returns:
        A callable that returns ClaudeAgentOptions configured with the model.
    """
    def factory():
        return get_sealqa_agent_options(model=model)
    return factory


# For backward compatibility, expose the factory as the options
sealqa_agent_options = get_sealqa_agent_options
