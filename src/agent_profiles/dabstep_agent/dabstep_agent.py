from pathlib import Path
from typing import TYPE_CHECKING, Any, Union
if TYPE_CHECKING:
    from claude_agent_sdk import ClaudeAgentOptions
from src.schemas import AgentResponse
from src.agent_profiles.skill_generator import get_project_root
from src.agent_profiles.sdk_config import is_claude_sdk


DABSTEP_AGENT_TOOLS = ["Read", "Write", "Bash", "Glob", "Grep", "Edit", "WebFetch", "WebSearch", "TodoWrite", "BashOutput", "Skill"]

# Path to the prompt file (read at runtime)
PROMPT_FILE = Path(__file__).parent / "prompt.txt"


def get_dabstep_agent_options(model: str | None = None, data_dir: str | None = None) -> Union[Any, dict]:
    """
    Factory function that creates agent options with the current prompt.

    Reads prompt.txt from disk each time, allowing dynamic updates
    without restarting the Python process.

    Args:
        model: Model to use (e.g., "opus", "sonnet"). If None, uses SDK default.
        data_dir: Path to the data directory to add. If None, no extra dirs are added.
    """
    if not is_claude_sdk():
        # HuggingFace / vLLM path: return a plain dict with Python tool implementations
        from src.agent_profiles.sdk_config import is_vllm_sdk
        from src.agent_profiles.hf_tools import HF_TOOLS

        # Read prompt from disk (same file as Claude path for consistency)
        prompt_text = PROMPT_FILE.read_text().strip() if PROMPT_FILE.exists() else ""
        data_dir_line = f"Data directory: {data_dir}\n\n" if data_dir else ""
        vllm_system = f"{data_dir_line}{prompt_text}"
        return {
            "system": vllm_system,
            "tools": HF_TOOLS,
            "model_id": model or "",
            "backend": "vllm" if is_vllm_sdk() else "",
        }

    # Claude SDK path — identical to original
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

    add_dirs = []
    if data_dir:
        add_dirs.append(data_dir)

    options = ClaudeAgentOptions(
        system_prompt=system_prompt,
        output_format=output_format,
        allowed_tools=DABSTEP_AGENT_TOOLS,
        setting_sources=["user", "project"],
        permission_mode='acceptEdits',
        add_dirs=add_dirs,
        cwd=get_project_root(),
        max_buffer_size=10 * 1024 * 1024,  # 10MB buffer (default is 1MB)
    )

    if model:
        options.model = model

    return options


def make_dabstep_agent_options(model: str | None = None, data_dir: str | None = None):
    """Create a factory function for dabstep agent options with a specific model.

    Args:
        model: Model to use (e.g., "opus", "sonnet"). If None, uses SDK default.
        data_dir: Path to the data directory to add. If None, no extra dirs are added.

    Returns:
        A callable that returns ClaudeAgentOptions configured with the model and data_dir.
    """
    def factory() -> Union[Any, dict]:
        return get_dabstep_agent_options(model=model, data_dir=data_dir)
    return factory


# For backward compatibility, expose the factory as the options
dabstep_agent_options = get_dabstep_agent_options
