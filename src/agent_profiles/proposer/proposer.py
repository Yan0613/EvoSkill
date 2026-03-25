from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from claude_agent_sdk import ClaudeAgentOptions
from src.schemas import ProposerResponse
from src.agent_profiles.proposer.prompt import PROPOSER_SYSTEM_PROMPT
from src.agent_profiles.skill_generator import get_project_root



PROPOSER_TOOLS = ["Read", "Bash", "Glob", "Grep", "WebFetch", "WebSearch", "TodoWrite", "BashOutput"]


proposer_system_prompt = {
    "type": "preset",
    "preset": "claude_code",
    "append": PROPOSER_SYSTEM_PROMPT.strip()
}
proposer_output_format = {
            "type": "json_schema",
            "schema": ProposerResponse.model_json_schema()
        }


def _make_proposer_options():
    from claude_agent_sdk import ClaudeAgentOptions
    return ClaudeAgentOptions(
        output_format=proposer_output_format,
        system_prompt=proposer_system_prompt,
        allowed_tools=PROPOSER_TOOLS,
        cwd=get_project_root(),
    )


proposer_options = _make_proposer_options