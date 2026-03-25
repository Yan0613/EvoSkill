from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from claude_agent_sdk import ClaudeAgentOptions
from src.schemas import SkillProposerResponse
from src.agent_profiles.skill_proposer.prompt import SKILL_PROPOSER_SYSTEM_PROMPT
from src.agent_profiles.skill_generator import get_project_root


SKILL_PROPOSER_TOOLS = [
    "Read",
    "Bash",
    "Glob",
    "Grep",
    "WebFetch",
    "WebSearch",
    "TodoWrite",
    "BashOutput",
]


skill_proposer_system_prompt = {
    "type": "preset",
    "preset": "claude_code",
    "append": SKILL_PROPOSER_SYSTEM_PROMPT.strip(),
}

skill_proposer_output_format = {
    "type": "json_schema",
    "schema": SkillProposerResponse.model_json_schema(),
}


def _make_skill_proposer_options():
    from claude_agent_sdk import ClaudeAgentOptions
    return ClaudeAgentOptions(
        output_format=skill_proposer_output_format,
        system_prompt=skill_proposer_system_prompt,
        allowed_tools=SKILL_PROPOSER_TOOLS,
        cwd=get_project_root(),
    )


skill_proposer_options = _make_skill_proposer_options
