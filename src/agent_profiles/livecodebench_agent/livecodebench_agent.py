from typing import Any, Union

from src.agent_profiles.sdk_config import is_claude_sdk, is_vllm_sdk, is_huggingface_sdk
from src.agent_profiles.skill_generator import get_project_root
from src.schemas import AgentResponse


# Use full tool suite for LiveCodeBench (agent can use tools to test/debug)
LIVECODEBENCH_AGENT_TOOLS = [
    "Read",
    "Write",
    "Bash",
    "Glob",
    "Grep",
    "Edit",
    "WebFetch",
    "WebSearch",
    "TodoWrite",
    "BashOutput",
    "Skill",
]

# NOTE: Question formatting (in livecodebench_format.py) matches Artificial Analysis.
# However, we use default Claude Code system prompts and tools for better performance.
# Reference: https://artificialanalysis.ai/benchmarks/livecodebench


def get_livecodebench_agent_options(
    model: str | None = None,
) -> Union[Any, dict[str, Any]]:
    """
    Factory function that creates agent options for LiveCodeBench evaluation.

    Returns ClaudeAgentOptions for Claude SDK or dict for OpenCode SDK.
    Uses default system prompts and full tool access.

    Args:
        model: Model to use (e.g., "opus", "sonnet"). If None, uses SDK default.
    """
    if is_claude_sdk():
        from claude_agent_sdk import ClaudeAgentOptions

        # Use default claude_code preset (no custom append)
        system_prompt = {"type": "preset", "preset": "claude_code"}
        output_format = {
            "type": "json_schema",
            "schema": AgentResponse.model_json_schema(),
        }

        options = ClaudeAgentOptions(
            system_prompt=system_prompt,
            output_format=output_format,
            allowed_tools=LIVECODEBENCH_AGENT_TOOLS,
            setting_sources=["user", "project"],
            permission_mode="acceptEdits",
            cwd=get_project_root(),
            max_buffer_size=10 * 1024 * 1024,  # 10MB buffer (default is 1MB)
        )

        if model:
            options.model = model

        return options
    else:
        # HuggingFace / vLLM path: return a plain dict with Python tool implementations
        from src.agent_profiles.hf_tools import HF_TOOLS

        # System prompt for vLLM: force Bash tool usage to test/debug code
        VLLM_SYSTEM = (
            "You are an expert software engineer and competitive programmer.\n\n"
            "## CRITICAL TOOL-CALLING RULES — you MUST follow these exactly:\n\n"
            "1. ALWAYS use the OpenAI function-calling API to invoke tools. "
            "NEVER write tool calls as raw text or XML in your response content. "
            "Do NOT output <tool_call>...</tool_call> XML tags in your message text — "
            "the framework will NOT execute them. Only structured function calls work.\n\n"
            "2. Call ONE tool at a time. After each tool call, WAIT for the result "
            "before deciding what to do next. Do NOT call multiple tools in a single turn.\n\n"
            "3. Do NOT include your final answer in the same turn as a tool call. "
            "First call the tool, wait for the result, then give your answer in a separate turn.\n\n"
            "4. When you have your final answer, wrap it in <FINAL_ANSWER>...</FINAL_ANSWER> tags. "
            "Do NOT use any other format (no bare FINAL_ANSWER prefix, no markdown code block only).\n\n"
            "## MANDATORY WORKFLOW — follow these steps in order:\n\n"
            "### Step 1: Analyze the problem\n"
            "- Read the constraints carefully (especially the upper bound of input values).\n"
            "- If the range is large (e.g. up to 10^9 or 10^18), a brute-force loop is IMPOSSIBLE "
            "— you MUST use an efficient algorithm (e.g. math formula, digit DP, binary search).\n\n"
            "### Step 2: Write your solution using the Write tool (ONE tool call)\n"
            "- Call Write to save your Python solution to `/tmp/sol.py`.\n"
            "- The solution must read from stdin and write to stdout.\n"
            "- WAIT for the Write result before proceeding.\n\n"
            "### Step 3: Test each sample input using the Bash tool (ONE call per test)\n"
            "- Call Bash once per sample input: `echo '<input>' | python3 /tmp/sol.py`\n"
            "- WAIT for each Bash result before running the next test.\n"
            "- If any test fails, fix the code (Write again) and re-test.\n"
            "- Do NOT skip this step even if you think your solution is correct.\n\n"
            "### Step 4: Submit your final answer\n"
            "- Only after ALL sample tests pass, output: <FINAL_ANSWER>```python\n<your solution>\n```</FINAL_ANSWER>\n\n"
            "## IMPORTANT RULES:\n"
            "- You MUST call Write and Bash tools before giving your final answer.\n"
            "- NEVER submit a solution without testing it first.\n"
            "- If your solution times out on any test, it means your algorithm is too slow — rewrite it."
        )

        # 为vLLM后端提供专门的配置（提供Bash工具用于代码测试）
        if is_vllm_sdk():
            return {
                "system": VLLM_SYSTEM,
                "tools": {k: HF_TOOLS[k] for k in ("Bash", "Write")},  # Bash + Write for code testing
                "model_id": model or "",
                "backend": "vllm"
            }
        else:
            # HuggingFace后端
            return {
                "system": "",  # Use default system prompt
                "tools": HF_TOOLS,
                "model_id": model or "",
            }


def make_livecodebench_agent_options(model: str | None = None):
    """Create a factory function for LiveCodeBench agent options with a specific model.

    Args:
        model: Model to use (e.g., "opus", "sonnet"). If None, uses SDK default.

    Returns:
        A callable that returns ClaudeAgentOptions configured with the model.
    """

    def factory() -> Union[Any, dict[str, Any]]:
        return get_livecodebench_agent_options(model=model)

    return factory


# For backward compatibility, expose the factory as the options
livecodebench_agent_options = get_livecodebench_agent_options
