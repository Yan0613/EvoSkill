import asyncio
import json
import logging
from typing import TYPE_CHECKING, Any, Callable, Generic, Optional, Type, TypeVar, Union

from pydantic import BaseModel, ValidationError

from .sdk_config import is_claude_sdk, is_huggingface_sdk, is_vllm_sdk

logger = logging.getLogger(__name__)

T = TypeVar("T", bound=BaseModel)

# Import ClaudeAgentOptions at module level for type hints only
if TYPE_CHECKING:
    from claude_agent_sdk import ClaudeAgentOptions as ClaudeAgentOptionsType
else:
    ClaudeAgentOptionsType = Any

# Type alias for options that can be static or dynamically generated
# Supports both ClaudeAgentOptions and dict (for opencode)
OptionsProvider = Union[
    ClaudeAgentOptionsType,
    dict[str, Any],
    Callable[[], Union[ClaudeAgentOptionsType, dict[str, Any]]],
]


class AgentTrace(BaseModel, Generic[T]):
    """Metadata and output from an agent run."""

    # From first message (SystemMessage)
    uuid: str
    session_id: str
    model: str
    tools: list[str]

    # From last message (ResultMessage)
    duration_ms: int
    total_cost_usd: float
    num_turns: int
    usage: dict[str, Any]
    result: str
    is_error: bool

    # The validated structured output (None if parsing failed)
    output: Optional[T] = None

    # Error info when output parsing fails
    parse_error: Optional[str] = None
    raw_structured_output: Optional[Any] = None

    # Full response list for debugging
    messages: list[Any]

    class Config:
        arbitrary_types_allowed = True

    def summarize(
        self,
        head_chars: int = 60_000,
        tail_chars: int = 60_000,
    ) -> str:
        """
        Create a summary of the trace for passing to downstream agents.

        - On success: returns full trace
        - On failure (parse_error): truncates to head + tail to avoid context exhaustion

        Args:
            head_chars: Characters to keep from start (only used on failure)
            tail_chars: Characters to keep from end (only used on failure)
        """
        # Build the core info
        lines = [
            f"Model: {self.model}",
            f"Turns: {self.num_turns}",
            f"Duration: {self.duration_ms}ms",
            f"Is Error: {self.is_error}",
        ]

        if self.parse_error:
            lines.append(f"Parse Error: {self.parse_error}")

        if self.output:
            lines.append(f"Output: {self.output}")

        # Convert result to string
        result_str = str(self.result) if self.result else ""

        # Only truncate on failure
        if self.parse_error and len(result_str) > (head_chars + tail_chars):
            truncated_middle = len(result_str) - head_chars - tail_chars
            lines.append(f"\n## Result (truncated, {truncated_middle:,} chars omitted)")
            lines.append(f"### Start:\n{result_str[:head_chars]}")
            lines.append(f"\n[... {truncated_middle:,} characters truncated ...]\n")
            lines.append(f"### End:\n{result_str[-tail_chars:]}")
        else:
            lines.append(f"\n## Full Result\n{result_str}")

        return "\n".join(lines)


class Agent(Generic[T]):
    """Simple wrapper for running Claude agents.

    Args:
        options: Either a ClaudeAgentOptions instance (static) or a callable
                 that returns ClaudeAgentOptions (dynamic, called on each run).
        response_model: Pydantic model for structured output validation.
    """

    TIMEOUT_SECONDS = 1200  # 20 minutes
    MAX_RETRIES = 3
    INITIAL_BACKOFF = 30  # seconds

    def __init__(self, options: OptionsProvider, response_model: Type[T]):
        self._options = options
        self.response_model = response_model

    def _get_options(self) -> Union[ClaudeAgentOptionsType, dict[str, Any]]:
        """Get options, calling the provider if it's a callable."""
        if callable(self._options):
            return self._options()
        return self._options

    async def _execute_hf_query(self, query: str, system_text: str, tools: dict | None = None) -> list[Any]:
        """Execute query using HuggingFace Transformers (open-source models), with optional tool calling."""
        import asyncio
        import re
        from .sdk_config import get_hf_config

        hf_cfg = get_hf_config()

        # Lazy-load the HF runner (singleton per process)
        runner = _get_hf_runner(hf_cfg)

        # Run inference in a thread pool to avoid blocking the event loop
        loop = asyncio.get_event_loop()
        raw_output = await loop.run_in_executor(
            None,
            lambda: runner.generate(
                system_text, query,
                hf_cfg["enable_thinking"],
                tools=tools,
            ),
        )

        # Extract <FINAL_ANSWER> tag (XML format) or bare "FINAL_ANSWER" keyword
        # (Qwen2.5-7B often emits: "...FINAL_ANSWER```python...```" without XML tags,
        #  and the keyword may appear anywhere in the content, not just at the start)
        match = re.search(
            r'<FINAL_ANSWER>\s*(.*?)\s*</FINAL_ANSWER>', raw_output,
            re.DOTALL | re.IGNORECASE,
        )
        if match:
            final_answer = match.group(1).strip()
        else:
            # Fallback: find bare FINAL_ANSWER keyword and take everything after it
            bare_match = re.search(r'\bFINAL_ANSWER\b\s*', raw_output, re.IGNORECASE)
            if bare_match:
                final_answer = raw_output[bare_match.end():].strip()
            else:
                final_answer = raw_output.strip()

        # Build a fake structured output compatible with AgentResponse
        structured_output = {
            "final_answer": final_answer,
            "reasoning": raw_output,
        }

        fake_result = _HFResult(
            raw_output=raw_output,
            structured_output=structured_output,
            model_name=hf_cfg["model_name"],
        )
        fake_system = _HFSystem(model_name=hf_cfg["model_name"])
        return [fake_system, fake_result]

    # Removed: single-tool constraint. vLLM now supports parallel tool calls
# parallel_tool_calls=False: force the model to call one tool at a time and
        # wait for the result before issuing the next call. This prevents the common
        # failure where the model fires Read + Bash concurrently — the Bash command
        # is constructed before the Read result is available, causing column-name
        # guessing and wrong file paths.
    VLLM_SINGLE_TOOL_INSTRUCTION = ""  # kept for compatibility, no longer injected

    VLLM_FINAL_ANSWER_SUFFIX = (
        "\n\nIMPORTANT: At the end of your response, you MUST output your final answer in this exact format:\n"
        "<FINAL_ANSWER>your answer here</FINAL_ANSWER>\n"
        "Rules:\n"
        "- For factual/numeric questions: put only the value (e.g. <FINAL_ANSWER>42</FINAL_ANSWER>)\n"
        "- For code questions: put the complete code inside the tag (e.g. <FINAL_ANSWER>```python\n...\n```</FINAL_ANSWER>)\n"
        "- Do NOT omit the <FINAL_ANSWER> tag. It is required."
    )

    @staticmethod
    def _repair_json(raw: str) -> dict | None:
        """Attempt to repair and parse malformed JSON from model output.

        Tries multiple strategies in order:
        1. Direct json.loads
        2. json-repair library (if available)
        3. Manual bracket-completion heuristics
        Returns parsed dict or None on failure.
        """
        import json
        # Strategy 1: direct parse
        try:
            return json.loads(raw)
        except json.JSONDecodeError:
            pass
        # Strategy 2: json-repair library
        try:
            import json_repair  # type: ignore
            repaired = json_repair.repair_json(raw)
            return json.loads(repaired)
        except Exception:
            pass
        # Strategy 3: escape raw control characters inside JSON strings.
        # Qwen2.5-7B often generates Write(content="...code...") where the
        # code contains literal newlines/tabs (not \n/\t escapes), which are
        # invalid JSON control characters. A simple state machine escapes them.
        def _escape_ctrl(s: str) -> str:
            out, in_str, esc = [], False, False
            for ch in s:
                if esc:
                    out.append(ch); esc = False
                elif ch == '\\' and in_str:
                    out.append(ch); esc = True
                elif ch == '"':
                    out.append(ch); in_str = not in_str
                elif in_str and ch == '\n':
                    out.append('\\n')
                elif in_str and ch == '\r':
                    out.append('\\r')
                elif in_str and ch == '\t':
                    out.append('\\t')
                else:
                    out.append(ch)
            return ''.join(out)
        try:
            return json.loads(_escape_ctrl(raw))
        except json.JSONDecodeError:
            pass
        # Strategy 4: bracket completion heuristics
        for suffix in ["", "}", '"}', "}}", '"}}', "}}}"]:
            try:
                return json.loads(raw + suffix)
            except json.JSONDecodeError:
                continue
        return None

    @staticmethod
    def _parse_ndjson(raw: str) -> list[dict]:
        """Parse one or more JSON objects from a string (NDJSON / concatenated JSON).

        Qwen2.5 sometimes generates multiple JSON objects inside a single
        <tool_call> block (e.g. two tool calls concatenated), which causes
        json.loads to raise "Extra data".  This method uses raw_decode to
        extract all valid JSON objects from the string sequentially.
        """
        import json
        decoder = json.JSONDecoder()
        results = []
        s = raw.strip()
        idx = 0
        while idx < len(s):
            # Skip whitespace between objects
            while idx < len(s) and s[idx] in ' \t\n\r':
                idx += 1
            if idx >= len(s):
                break
            try:
                obj, end_idx = decoder.raw_decode(s, idx)
                if isinstance(obj, dict):
                    results.append(obj)
                idx = end_idx
            except json.JSONDecodeError:
                # Try _repair_json on the remainder as a last resort
                remainder = s[idx:]
                repaired = Agent._repair_json(remainder)
                if repaired is not None:
                    results.append(repaired)
                break
        return results

    @staticmethod
    def _build_skills_system_prompt() -> str:
        """Scan .claude/skills/ and inject skill list into system prompt.

        This mirrors Claude SDK's setting_sources=["user", "project"] behavior:
        Claude Code automatically reads .claude/skills/ at startup and makes
        skill names + descriptions available to the model. We replicate this
        by injecting the skill catalog into the system prompt so the model
        knows to call the Skill tool with the right names.
        """
        import os

        # Locate .claude/skills/ relative to project root
        current = os.path.abspath(__file__)
        skills_dir = None
        for _ in range(10):
            parent = os.path.dirname(current)
            candidate = os.path.join(parent, ".claude", "skills")
            if os.path.isdir(candidate):
                skills_dir = candidate
                break
            if current == parent:
                break
            current = parent

        if skills_dir is None:
            return ""

        skill_dirs = sorted([
            d for d in os.listdir(skills_dir)
            if os.path.isdir(os.path.join(skills_dir, d))
        ])
        if not skill_dirs:
            return ""

        lines = [
            "\n\n## Available Skills",
            "You have access to reusable skills via the Skill tool. "
            "Call Skill with the skill name to read its full instructions before starting complex tasks.",
            "Available skills:",
        ]
        for skill_name in skill_dirs:
            skill_file = os.path.join(skills_dir, skill_name, "SKILL.md")
            desc = ""
            if os.path.exists(skill_file):
                try:
                    with open(skill_file, "r", encoding="utf-8") as f:
                        for line in f:
                            line = line.strip()
                            if line.startswith("description:"):
                                desc = line[len("description:"):].strip()
                                break
                except Exception:
                    pass
            lines.append(f"  - **{skill_name}**: {desc}" if desc else f"  - **{skill_name}**")
        lines.append(
            "\nAlways check available skills before starting complex analysis tasks "
            "by calling: Skill(name=\"<skill_name>\")"
        )
        return "\n".join(lines)

    async def _execute_vllm_query(self, query: str, system_text: str, tools: dict | None = None, output_format: dict | None = None) -> list[Any]:
        """Execute query using vLLM OpenAI-compatible server, with OpenAI tools API.

        Uses the standard OpenAI tools API (tool_choice="auto") for tool calling.
        Malformed JSON in tool_call arguments is repaired client-side via _repair_json()
        before dispatching, so vLLM's hermes_tool_parser JSONDecodeError is bypassed:
        we pass tools through the API but handle argument parsing ourselves.

        Skills are auto-injected into the system prompt (mirroring Claude's
        setting_sources=["user", "project"] behavior) so the model knows which
        skills are available without needing to call Skill(name="list") first.

        Args:
            output_format: Optional dict with {"type": "json_schema", "schema": {...}}
                          for structured output. When provided, the FINAL_ANSWER
                          instruction includes the JSON schema so the model knows
                          exactly what fields to output.
        """
        import json
        import re
        from .sdk_config import get_vllm_config

        vllm_cfg = get_vllm_config()

        # Build system prompt: base + skills catalog + single-tool rule + FINAL_ANSWER instruction
        full_system = (system_text or "")
        if tools and "Skill" in tools:
            # Auto-inject skill catalog so model knows available skills upfront
            # (mirrors Claude's setting_sources=["user", "project"] behavior)
            full_system = full_system + self._build_skills_system_prompt()
        if tools:
            # Only inject FINAL_ANSWER suffix when tools are used (agentic tasks).
            # For pure generation tasks (tools=None, e.g. LiveCodeBench), the model
            # should output code directly without XML tags.
            if output_format and output_format.get("type") == "json_schema":
                # Structured output: inject JSON schema into FINAL_ANSWER instruction
                schema = output_format.get("schema", {})
                schema_str = json.dumps(schema, indent=2)
                full_system = full_system + (
                    "\n\nIMPORTANT: At the end of your response, you MUST output your final answer "
                    "as a JSON object inside <FINAL_ANSWER>...</FINAL_ANSWER> tags.\n"
                    "The JSON MUST conform to this exact schema:\n"
                    f"```json\n{schema_str}\n```\n"
                    "Rules:\n"
                    "- Output ONLY valid JSON inside the <FINAL_ANSWER> tags, no other text.\n"
                    "- All required fields MUST be present.\n"
                    "- Do NOT omit the <FINAL_ANSWER> tag. It is required."
                )
            else:
                full_system = full_system + self.VLLM_FINAL_ANSWER_SUFFIX

        # Build OpenAI-format tool definitions from tools dict.
        # vLLM with --tool-call-parser hermes will inject these into the chat
        # template automatically (Qwen2.5 uses <tools>...</tools> XML in system
        # prompt) and parse the model's <tool_call> responses server-side.
        openai_tools = None
        if tools:
            openai_tools = [
                {
                    "type": "function",
                    "function": {
                        "name": name,
                        "description": spec.get("description", "") if isinstance(spec, dict) else "",
                        "parameters": spec.get("parameters", {"type": "object", "properties": {}}) if isinstance(spec, dict) else {},
                    },
                }
                for name, spec in tools.items()
            ]
        # For large-context models (e.g. Qwen2.5-72B with 128K context), no query
        # truncation is needed. We only truncate if the query genuinely exceeds the
        # model's context window (using the configured context_length).
        context_length = vllm_cfg.get("context_length", 131072)
        reserved_output = vllm_cfg.get("max_tokens", 8192)
        # Rough estimate: 1 token ≈ 3 chars (conservative for CJK/code content)
        max_input_chars = (context_length - reserved_output) * 3
        system_chars = len(full_system)
        query_budget = max_input_chars - system_chars - 500  # 500 chars overhead
        if query_budget < 2000:
            query_budget = 2000  # always allow at least a reasonable query
        truncated_query = query
        if len(query) > query_budget:
            half = query_budget // 2
            truncated_query = (
                query[:half]
                + f"\n... [query truncated {len(query) - query_budget} chars to fit context window] ...\n"
                + query[-half:]
            )
            logger.warning(
                f"[vLLM] Query truncated: {len(query)} -> {len(truncated_query)} chars "
                f"(context_length={context_length}, max_tokens={reserved_output})"
            )

        messages = [
            {"role": "system", "content": full_system},
            {"role": "user", "content": truncated_query},
        ]

        from openai import AsyncOpenAI
        client = AsyncOpenAI(
            base_url=vllm_cfg["base_url"],
            api_key=vllm_cfg["api_key"],
        )

        raw_output = ""
        num_turns = 0
        MAX_TOOL_TURNS = 30

        while num_turns < MAX_TOOL_TURNS:
            num_turns += 1

            # Use OpenAI tools API when tools are available.
            # vLLM's hermes_tool_parser handles Qwen2.5's <tool_call> format
            # server-side. If it fails (e.g. model generates multiple JSON
            # objects in one <tool_call> tag), we fall back to client-side
            # parsing in Case 2 below.
            create_kwargs: dict = {
                "model": vllm_cfg["model_name"],
                "messages": messages,
                "max_tokens": vllm_cfg["max_tokens"],
                "temperature": 0,
            }
            if openai_tools:
                create_kwargs["tools"] = openai_tools
                create_kwargs["tool_choice"] = "auto"
                create_kwargs["parallel_tool_calls"] = False

            try:
                response = await client.chat.completions.create(**create_kwargs)
            except Exception as api_err:
                # If the API itself fails, retry without tools so we still get
                # a final answer.
                logger.warning(f"[vLLM] tools API call failed ({api_err}), retrying without tools")
                fallback_kwargs = {k: v for k, v in create_kwargs.items() if k not in ("tools", "tool_choice")}
                try:
                    response = await client.chat.completions.create(**fallback_kwargs)
                except Exception as api_err2:
                    logger.warning(f"[vLLM] fallback API call also failed ({api_err2}), aborting")
                    break

            choice = response.choices[0]
            msg = choice.message
            content = msg.content or ""

            # ── Case 1: model returned structured tool_calls (hermes parser succeeded) ──
            if msg.tool_calls:
                # When parallel_tool_calls=False is set, enforce it client-side too:
                # small models (e.g. Qwen2.5-7B) sometimes still return multiple
                # tool_calls in one turn despite the flag. Execute only the first one
                # so the model sees each result before deciding the next step.
                # (For models that correctly return one tool call at a time this has
                # no effect since len(msg.tool_calls) == 1 anyway.)
                tool_calls_this_turn = msg.tool_calls[:1] if openai_tools else msg.tool_calls
                messages.append({
                    "role": "assistant",
                    "content": content,
                    "tool_calls": [
                        {
                            "id": tc.id,
                            "type": "function",
                            "function": {
                                "name": tc.function.name,
                                "arguments": tc.function.arguments,
                            },
                        }
                        for tc in tool_calls_this_turn
                    ],
                })

                async def _exec_tc(tc):
                    fn_name = tc.function.name
                    raw_args = tc.function.arguments or "{}"
                    fn_args = self._repair_json(raw_args)
                    if fn_args is None:
                        tool_result_str = f"[Error: could not parse arguments for {fn_name}: {raw_args!r}]"
                    else:
                        if isinstance(fn_args, str):
                            fn_args = {}
                        tool_result = await _dispatch_tool_call(fn_name, fn_args, tools)
                        tool_result_str = str(tool_result)
                    return tc.id, tool_result_str

                # Run all tool calls concurrently
                tc_results = await asyncio.gather(*[_exec_tc(tc) for tc in tool_calls_this_turn])
                for tc_id, tool_result_str in tc_results:
                    messages.append({
                        "role": "tool",
                        "tool_call_id": tc_id,
                        "content": tool_result_str,
                    })
                # Compress early tool results if messages history is getting too large
                messages = await _compress_messages_history(messages, vllm_cfg)
                continue  # next turn

            # ── Case 2: hermes parser failed → parse <tool_call> tags from content ──
            # hermes_tool_parser raises JSONDecodeError when Qwen2.5 generates
            # non-standard JSON (e.g. unescaped newlines/backticks in code strings,
            # or multiple JSON objects in one <tool_call> tag). vLLM falls back to
            # putting raw content in msg.content with msg.tool_calls empty.
            # We parse <tool_call> tags client-side and execute them.
            #
            # NOTE: We execute the tool even when FINAL_ANSWER also appears in the
            # same content. Small models (esp. 7B) often write tool call + premature
            # FINAL_ANSWER in one shot without waiting for the actual tool result.
            # The "answer" is based on predicted (not real) output — for code tasks
            # (LiveCodeBench) this means untested code. Executing the tool and letting
            # the model re-answer with real results is always more correct.
            # We strip the premature FINAL_ANSWER from the appended assistant message
            # so the model is not anchored to its unverified prediction.
            tool_call_tags = re.findall(
                r'<tool_call>\s*(.*?)\s*</tool_call>', content, re.DOTALL
            )
            if tool_call_tags and tools:
                any_dispatched = False
                # Strip premature FINAL_ANSWER from content before appending:
                # the model predicted an answer without having seen tool results yet.
                # Keeping it in the message would anchor the model to a wrong answer.
                content_for_msg = re.sub(
                    r'\s*<FINAL_ANSWER>.*?</FINAL_ANSWER>', '', content,
                    flags=re.DOTALL | re.IGNORECASE,
                ).strip()
                messages.append({"role": "assistant", "content": content_for_msg})
                tool_results = []
                # Collect all valid (fn_name, fn_args) pairs first.
                # NOTE: We only execute the FIRST valid tool call per turn to enforce
                # sequential tool use. Small models (e.g. Qwen2.5-7B) often emit all
                # tool calls + FINAL_ANSWER in one shot without waiting for results.
                # Executing them concurrently causes race conditions (e.g. Bash runs
                # before Write finishes). By executing one at a time we force the model
                # to observe each result before proceeding.
                pending_calls = []
                for raw_tc_json in tool_call_tags:
                    # Handle NDJSON: multiple JSON objects in one <tool_call> block
                    parsed_calls = self._parse_ndjson(raw_tc_json)
                    for tc_parsed in parsed_calls:
                        if not isinstance(tc_parsed, dict):
                            continue
                        fn_name = tc_parsed.get("name") or tc_parsed.get("function", {}).get("name", "")
                        fn_args = tc_parsed.get("arguments") or tc_parsed.get("parameters") or {}
                        if isinstance(fn_args, str):
                            fn_args = self._repair_json(fn_args) or {}
                        if not fn_name or fn_name not in tools:
                            logger.warning(f"[vLLM] Unknown tool in fallback: {fn_name!r}")
                            continue
                        logger.info(f"[vLLM] Fallback tool call: {fn_name}({fn_args})")
                        pending_calls.append((fn_name, fn_args))
                        # Only execute the first valid tool call per turn.
                        # The model will see the result and decide the next step.
                        break
                    if pending_calls:
                        break

                if pending_calls:
                    any_dispatched = True
                    fn_name, fn_args = pending_calls[0]
                    tool_result = await _dispatch_tool_call(fn_name, fn_args, tools)
                    tool_results = [(fn_name, str(tool_result))]

                if any_dispatched:
                    # Use user-role message for fallback results (no tool_call_id available)
                    combined = "\n\n".join(
                        f"[Tool result: {name}]\n{result}"
                        for name, result in tool_results
                    )
                    messages.append({"role": "user", "content": combined})
                    logger.debug(
                        f"[vLLM] Fallback: dispatched 1 tool call (first of {len(tool_call_tags)} tags) "
                        f"from <tool_call> tags, total messages: {len(messages)}"
                    )
                    # Compress early tool results if messages history is getting too large
                    messages = await _compress_messages_history(messages, vllm_cfg)
                    continue  # next turn

            # ── Final answer: no tool calls ──

            raw_output = content
            break

        # ── Max turns exhausted: force a final-answer turn ──
        # If the model kept calling tools for all MAX_TOOL_TURNS turns, raw_output
        # is still "". The while loop exits via the condition (not via break), so
        # no content was ever captured. Send one final no-tools request so the model
        # can output its best answer based on the tool results it has already seen.
        if not raw_output:
            messages.append({
                "role": "user",
                "content": (
                    "You have used all available tool turns. "
                    "Based on everything you have done so far, output your best final answer now. "
                    "You MUST wrap it in <FINAL_ANSWER>...</FINAL_ANSWER> tags."
                ),
            })
            try:
                forced_response = await client.chat.completions.create(
                    model=vllm_cfg["model_name"],
                    messages=messages,
                    max_tokens=vllm_cfg["max_tokens"],
                    temperature=0,
                )
                raw_output = forced_response.choices[0].message.content or ""
                logger.info(
                    f"[vLLM] Max turns exhausted — forced final-answer turn produced "
                    f"{len(raw_output)} chars"
                )
            except Exception as forced_err:
                logger.warning(f"[vLLM] Forced final-answer turn failed: {forced_err}")

        # Extract <FINAL_ANSWER> tag (XML format) or bare "FINAL_ANSWER" keyword
        # (Qwen2.5-7B often emits: "...<tool_call>...</tool_call>\nFINAL_ANSWER```python...```"
        #  without XML tags, and the keyword may appear anywhere in the content)
        match = re.search(
            r'<FINAL_ANSWER>\s*(.*?)\s*</FINAL_ANSWER>', raw_output,
            re.DOTALL | re.IGNORECASE,
        )
        if match:
            final_answer = match.group(1).strip()
        else:
            # Fallback: find bare FINAL_ANSWER keyword and take everything after it
            bare_match = re.search(r'\bFINAL_ANSWER\b\s*', raw_output, re.IGNORECASE)
            if bare_match:
                final_answer = raw_output[bare_match.end():].strip()
            else:
                final_answer = raw_output.strip()

        structured_output = {
            "final_answer": final_answer,
            "reasoning": raw_output,
        }

        fake_result = _HFResult(
            raw_output=raw_output,
            structured_output=structured_output,
            model_name=vllm_cfg["model_name"],
        )
        fake_result.num_turns = num_turns
        fake_system = _HFSystem(model_name=vllm_cfg["model_name"])
        return [fake_system, fake_result]
    async def _execute_query(self, query: str) -> list[Any]:
        """Execute a single query attempt."""

        if is_huggingface_sdk() or is_vllm_sdk():
            # HuggingFace / vLLM path: do NOT call _get_options() to avoid
            # importing claude_agent_sdk. Extract system prompt and tools safely.
            system_text = _safe_extract_system_from_provider(self._options)
            tools = _safe_extract_tools_from_provider(self._options)
            if is_vllm_sdk():
                output_format = _safe_extract_output_format_from_provider(self._options)
                return await self._execute_vllm_query(query, system_text, tools, output_format=output_format)
            else:
                return await self._execute_hf_query(query, system_text, tools)

        options = self._get_options()

        if is_claude_sdk():
            # Claude SDK path
            from claude_agent_sdk import ClaudeAgentOptions, ClaudeSDKClient

            # Convert dict to ClaudeAgentOptions if needed
            if isinstance(options, dict):
                claude_opts = ClaudeAgentOptions(
                    system_prompt=options.get("system"),
                    allowed_tools=list(options.get("tools", {}).keys())
                    if options.get("tools")
                    else [],
                    output_format=options.get("format"),
                    setting_sources=["user", "project"],
                    permission_mode="acceptEdits",
                )
                if "model_id" in options and "claude" in options["model_id"].lower():
                    claude_opts.model = options["model_id"]
                options = claude_opts

            async with ClaudeSDKClient(options) as client:
                await client.query(query)
                return [msg async for msg in client.receive_response()]
        else:
            # OpenCode SDK path
            from opencode_ai import AsyncOpencode

            if not isinstance(options, dict):
                raise TypeError(
                    f"OpenCode SDK requires dict options, got {type(options)}"
                )

            # Start opencode server if needed
            import subprocess
            import time

            try:
                # Quick check if server is running
                client = AsyncOpencode(base_url="http://127.0.0.1:4096")
                await client.session.create(extra_body={})
            except Exception:
                # Start server
                subprocess.Popen(
                    ["opencode", "serve", "--port", "4096", "--hostname", "127.0.0.1"],
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                    start_new_session=True,
                )
                time.sleep(2)
                client = AsyncOpencode(base_url="http://127.0.0.1:4096")

            session = await client.session.create(extra_body={})

            extra_body = {}
            if "format" in options:
                extra_body["format"] = options["format"]

            message = await client.session.chat(
                id=session.id,
                model_id=options.get("model_id", "zai-org/GLM-5"),
                provider_id=options.get("provider_id", "togetherai"),
                parts=[{"type": "text", "text": query}],
                system=options.get("system"),
                mode=options.get("mode", "build"),
                tools=options.get("tools", {}),
                extra_body=extra_body if extra_body else None,
            )

            # Return as single-item list for consistency with Claude SDK
            return [message]

    async def _run_with_retry(self, query: str) -> list[Any]:
        """Execute query with timeout and exponential backoff retry."""
        last_error: Exception | None = None
        backoff = self.INITIAL_BACKOFF

        for attempt in range(self.MAX_RETRIES):
            try:
                async with asyncio.timeout(self.TIMEOUT_SECONDS):
                    return await self._execute_query(query)
            except asyncio.TimeoutError:
                last_error = TimeoutError(
                    f"Query timed out after {self.TIMEOUT_SECONDS}s"
                )
                logger.warning(
                    f"Attempt {attempt + 1}/{self.MAX_RETRIES} timed out. Retrying in {backoff}s..."
                )
            except Exception as e:
                last_error = e
                # For vLLM/HF: 400 context-too-long errors will never succeed on retry — fail fast
                if is_huggingface_sdk() or is_vllm_sdk():
                    err_str = str(e)
                    if "400" in err_str and ("context length" in err_str or "input_tokens" in err_str or "BadRequestError" in err_str):
                        logger.warning(
                            f"Attempt {attempt + 1}/{self.MAX_RETRIES} failed with non-retryable 400 error, giving up immediately: {e}"
                        )
                        break
                logger.warning(
                    f"Attempt {attempt + 1}/{self.MAX_RETRIES} failed: {e}. Retrying in {backoff}s..."
                )

            if attempt < self.MAX_RETRIES - 1:
                await asyncio.sleep(backoff)
                backoff *= 2  # Exponential backoff

        raise last_error if last_error else RuntimeError("All retries exhausted")

    async def run(self, query: str) -> AgentTrace[T]:
        messages = await self._run_with_retry(query)

        if is_huggingface_sdk() or is_vllm_sdk():
            # HuggingFace / vLLM path: messages = [_FakeSystem, _FakeResult]
            first = messages[0]   # _FakeSystem
            last = messages[-1]   # _FakeResult

            output = None
            parse_error = None
            raw_structured_output = last.structured_output

            if raw_structured_output is not None:
                try:
                    output = self.response_model.model_validate(raw_structured_output)
                except (ValidationError, json.JSONDecodeError, TypeError) as e:
                    # Fallback: the vLLM path wraps output as {"final_answer": "...", "reasoning": "..."}.
                    # The model may have put valid JSON for the response_model inside final_answer.
                    # Try to parse final_answer as JSON and validate against response_model.
                    fallback_parsed = False
                    if isinstance(raw_structured_output, dict) and "final_answer" in raw_structured_output:
                        fa = raw_structured_output["final_answer"]
                        if isinstance(fa, str) and fa.strip():
                            # Strip any residual XML tags (e.g. </FINAL_ANSWER>) from the value
                            import re as _re
                            cleaned = _re.sub(r'</?FINAL_ANSWER[^>]*>', '', fa).strip()
                            # Try to extract JSON from the string (may be wrapped in markdown code blocks)
                            json_match = _re.search(r'\{[\s\S]*\}', cleaned)
                            if json_match:
                                parsed = self._repair_json(json_match.group(0))
                                if parsed is not None:
                                    try:
                                        output = self.response_model.model_validate(parsed)
                                        raw_structured_output = parsed
                                        fallback_parsed = True
                                        logger.info(
                                            f"[vLLM] Successfully parsed response_model from final_answer JSON"
                                        )
                                    except (ValidationError, TypeError):
                                        pass
                    if not fallback_parsed:
                        parse_error = f"{type(e).__name__}: {str(e)}"
            else:
                parse_error = "No structured output returned"

            if is_vllm_sdk():
                from .sdk_config import get_vllm_config
                model_name = get_vllm_config()["model_name"]
            else:
                from .sdk_config import get_hf_config
                model_name = get_hf_config()["model_name"]
            return AgentTrace(
                uuid=first.data.get("uuid", "hf-uuid"),
                session_id=last.session_id,
                model=model_name,
                tools=[],
                duration_ms=last.duration_ms,
                total_cost_usd=last.total_cost_usd,
                num_turns=last.num_turns,
                usage=last.usage,
                result=last.result,
                is_error=last.is_error or parse_error is not None,
                output=output,
                parse_error=parse_error,
                raw_structured_output=raw_structured_output,
                messages=messages,
            )
        elif is_claude_sdk():
            # Claude SDK: messages list with SystemMessage, AssistantMessage, ResultMessage
            first = messages[0]
            last = messages[-1]

            # Try to parse structured output
            output = None
            parse_error = None
            raw_structured_output = last.structured_output

            if raw_structured_output is not None:
                try:
                    output = self.response_model.model_validate(raw_structured_output)
                except (ValidationError, json.JSONDecodeError, TypeError) as e:
                    parse_error = f"{type(e).__name__}: {str(e)}"
            else:
                parse_error = (
                    "No structured output returned (context limit likely exceeded)"
                )

            return AgentTrace(
                uuid=first.data.get("uuid"),
                session_id=last.session_id,
                model=first.data.get("model"),
                tools=first.data.get("tools", []),
                duration_ms=last.duration_ms,
                total_cost_usd=last.total_cost_usd,
                num_turns=last.num_turns,
                usage=last.usage,
                result=last.result,
                is_error=last.is_error or parse_error is not None,
                output=output,
                parse_error=parse_error,
                raw_structured_output=raw_structured_output,
                messages=messages,
            )
        else:
            # OpenCode SDK: single AssistantMessage with extra fields
            message = messages[0]

            # Extract structured output from info dict (extra field)
            output = None
            parse_error = None
            raw_structured_output = None

            if hasattr(message, "info") and message.info:
                raw_structured_output = message.info.get("structured")

            if raw_structured_output is not None:
                try:
                    output = self.response_model.model_validate(raw_structured_output)
                except (ValidationError, json.JSONDecodeError, TypeError) as e:
                    parse_error = f"{type(e).__name__}: {str(e)}"
            else:
                parse_error = (
                    "No structured output returned (context limit likely exceeded)"
                )

            # Extract text from parts (extra field)
            result_text = ""
            if hasattr(message, "parts"):
                for part in message.parts:
                    if isinstance(part, dict) and part.get("type") == "text":
                        result_text += part.get("text", "")

            # Get metadata from info dict
            info = message.info if hasattr(message, "info") else {}
            usage = info.get("tokens", {}) if info else {}
            cost = info.get("cost", 0.0) if info else 0.0

            options = self._get_options()
            model_name = (
                options.get("model_id", "unknown")
                if isinstance(options, dict)
                else "unknown"
            )
            tools = (
                list(options.get("tools", {}).keys())
                if isinstance(options, dict) and options.get("tools")
                else []
            )

            return AgentTrace(
                uuid=message.session_id or "unknown",
                session_id=message.session_id or "unknown",
                model=model_name,
                tools=tools,
                duration_ms=0,
                total_cost_usd=cost,
                num_turns=1,
                usage=usage,
                result=result_text,
                is_error=parse_error is not None,
                output=output,
                parse_error=parse_error,
                raw_structured_output=raw_structured_output,
                messages=messages,
            )


# ─────────────────────────────────────────────────────────────────────────────
# HuggingFace Transformers helpers (used by Agent._execute_hf_query)
# ─────────────────────────────────────────────────────────────────────────────

_hf_runner_instance = None
_hf_runner_model_name: str | None = None


class _HFResult:
    """Picklable fake ResultMessage for HuggingFace SDK path."""
    __slots__ = ("session_id", "duration_ms", "total_cost_usd", "num_turns",
                 "usage", "result", "is_error", "structured_output")

    def __init__(self, raw_output: str, structured_output: dict, model_name: str):
        self.session_id = "hf-session"
        self.duration_ms = 0
        self.total_cost_usd = 0.0
        self.num_turns = 1
        self.usage = {}
        self.result = raw_output
        self.is_error = False
        self.structured_output = structured_output


class _HFSystem:
    """Picklable fake SystemMessage for HuggingFace SDK path."""
    __slots__ = ("data",)

    def __init__(self, model_name: str):
        self.data = {"uuid": "hf-uuid", "model": model_name, "tools": []}


def _safe_extract_system_from_provider(options_provider: Any) -> str:
    """Safely extract system prompt text from an options provider.

    For HF/vLLM backends the factory returns a plain dict (no claude_agent_sdk import),
    so we can safely call it to get the dict and extract the system key.
    """
    # If it's a plain string, return directly
    if isinstance(options_provider, str):
        return options_provider

    # If it's a dict, extract system key
    if isinstance(options_provider, dict):
        return options_provider.get("system", "") or ""

    # If it's a callable (factory function), call it to get the options dict
    if callable(options_provider):
        try:
            result = options_provider()
            if isinstance(result, dict):
                return result.get("system", "") or ""
        except Exception:
            pass

        # Fallback: try to find a PROMPT_FILE in the module of the factory
        try:
            import inspect
            source_module = inspect.getmodule(options_provider)
            if source_module and hasattr(source_module, "PROMPT_FILE"):
                prompt_file = source_module.PROMPT_FILE
                if hasattr(prompt_file, "read_text") and prompt_file.exists():
                    return prompt_file.read_text().strip()
        except Exception:
            pass

    return ""


def _safe_extract_tools_from_provider(options_provider: Any) -> dict | None:
    """Safely extract tools dict from an options provider.

    For HF/vLLM backends the factory returns a plain dict (no claude_agent_sdk import),
    so we can safely call it to get the dict and extract the tools key.

    Returns a dict of {tool_name: tool_spec} or None if no tools found.
    """
    if options_provider is None:
        return None

    # If it's a dict, extract tools key
    if isinstance(options_provider, dict):
        return options_provider.get("tools") or None

    # If it's a callable, call it to get the options dict
    if callable(options_provider):
        try:
            result = options_provider()
            if isinstance(result, dict):
                return result.get("tools") or None
        except Exception:
            pass

    return None


def _safe_extract_output_format_from_provider(options_provider: Any) -> dict | None:
    """Safely extract output_format dict from an options provider.

    For vLLM backends, the factory may return a dict with an 'output_format' key
    containing {"type": "json_schema", "schema": {...}}. This is used to inject
    the JSON schema into the FINAL_ANSWER instruction so the model knows exactly
    what structured output to produce.

    Returns a dict like {"type": "json_schema", "schema": {...}} or None.
    """
    if options_provider is None:
        return None

    # If it's a dict, extract output_format key
    if isinstance(options_provider, dict):
        return options_provider.get("output_format") or None

    # If it's a callable, call it to get the options dict
    if callable(options_provider):
        try:
            result = options_provider()
            if isinstance(result, dict):
                return result.get("output_format") or None
        except Exception:
            pass

    return None


async def _summarize_content_with_llm(content: str, vllm_cfg: dict) -> str:
    """Use vLLM to produce a concise summary of a tool result message.

    Falls back to head-truncation if the LLM call fails, so the agentic loop
    is never blocked by a summarization error.

    Args:
        content: The raw tool result text to summarize.
        vllm_cfg: vLLM config dict (base_url, api_key, model_name, max_tokens).

    Returns:
        A short summary string (≤ 300 chars on failure fallback, or LLM output).
    """
    prompt = (
        "Summarize the following tool output in 2-4 concise sentences. "
        "Preserve key facts, numbers, file names, and error messages. "
        "Do NOT include any preamble — output only the summary.\n\n"
        f"Tool output:\n{content}"
    )
    try:
        from openai import AsyncOpenAI
        client = AsyncOpenAI(
            base_url=vllm_cfg["base_url"],
            api_key=vllm_cfg["api_key"],
        )
        response = await client.chat.completions.create(
            model=vllm_cfg["model_name"],
            messages=[{"role": "user", "content": prompt}],
            max_tokens=256,
            temperature=0,
        )
        summary = response.choices[0].message.content or ""
        return f"[LLM summary of {len(content)} chars]\n{summary.strip()}"
    except Exception as exc:
        logger.warning(f"[vLLM] LLM summarization failed ({exc}), falling back to truncation")
        return f"[truncated {len(content)} chars] {content[:300]}..."


async def _compress_messages_history(messages: list, vllm_cfg: dict) -> list:
    """Compress early tool result messages when the conversation history grows too large.

    Uses LLM summarization instead of blind truncation so that key information
    (numbers, file names, error details) is preserved in compressed messages.

    Strategy:
    - Keep system message (index 0) and user message (index 1) intact.
    - Keep the last KEEP_RECENT turns (8 messages) fully intact.
    - For older tool/user messages in the middle, summarize large content via LLM.

    Args:
        messages: Current messages list.
        vllm_cfg: vLLM config dict with context_length and max_tokens.

    Returns:
        Possibly compressed messages list.
    """
    context_length = vllm_cfg.get("context_length", 131072)
    max_tokens = vllm_cfg.get("max_tokens", 8192)
    # Budget: (context_length - max_tokens) tokens, 1 token ≈ 3 chars (conservative)
    budget_chars = (context_length - max_tokens) * 3

    # Quick check: if total size is within budget, nothing to do
    total_chars = sum(len(str(m.get("content") or "")) for m in messages)
    if total_chars <= budget_chars:
        return messages

    # Keep the last 4 turns (8 messages: 4 assistant + 4 tool) fully intact
    KEEP_RECENT = 8
    # Messages to potentially compress: everything between [2, len-KEEP_RECENT)
    # Index 0 = system, index 1 = user query — never compress these
    compress_end = max(2, len(messages) - KEEP_RECENT)

    compressed = list(messages)  # shallow copy
    for i in range(2, compress_end):
        msg = compressed[i]
        role = msg.get("role", "")
        content = msg.get("content") or ""
        # Only compress large tool results and intermediate user messages
        if role in ("tool", "user") and i > 1 and len(content) > 500:
            summary = await _summarize_content_with_llm(content, vllm_cfg)
            compressed[i] = dict(msg)  # copy to avoid mutating original
            compressed[i]["content"] = summary
            logger.debug(
                f"[vLLM] Summarized message[{i}] role={role}: "
                f"{len(content)} -> {len(summary)} chars"
            )

    new_total = sum(len(str(m.get("content") or "")) for m in compressed)
    if new_total < total_chars:
        logger.info(
            f"[vLLM] History compressed via LLM summarization: {total_chars} -> {new_total} chars "
            f"(budget={budget_chars})"
        )
    return compressed


async def _dispatch_tool_call(fn_name: str, fn_args: dict, tools: dict | None) -> Any:
    """Dispatch a tool call by name, executing the registered tool function.

    tools dict format: {name: {"fn": callable, "description": str, "parameters": dict}}
    If the tool has no "fn" key, returns a placeholder string.
    """
    if not tools or fn_name not in tools:
        return f"[Tool '{fn_name}' not found]"

    spec = tools[fn_name]
    fn = spec.get("fn") if isinstance(spec, dict) else None

    if fn is None:
        return f"[Tool '{fn_name}' has no implementation]"

    try:
        import asyncio
        if asyncio.iscoroutinefunction(fn):
            result = await fn(**fn_args)
        else:
            loop = asyncio.get_event_loop()
            result = await loop.run_in_executor(None, lambda: fn(**fn_args))
        return result
    except Exception as e:
        return f"[Tool '{fn_name}' error: {e}]"


def _get_hf_runner(hf_cfg: dict) -> "HFRunner":
    """Return a cached HFRunner, reloading if the model name changed."""
    global _hf_runner_instance, _hf_runner_model_name
    if _hf_runner_instance is None or _hf_runner_model_name != hf_cfg["model_name"]:
        _hf_runner_instance = HFRunner(
            model_name=hf_cfg["model_name"],
            max_new_tokens=hf_cfg["max_new_tokens"],
            device=hf_cfg["device"],
        )
        _hf_runner_model_name = hf_cfg["model_name"]
    return _hf_runner_instance


def _extract_system_text(options: Any) -> str:
    """Extract a plain-text system prompt from ClaudeAgentOptions or dict options."""
    if options is None:
        return ""
    # dict-style options (opencode format)
    if isinstance(options, dict):
        return options.get("system", "") or ""
    # ClaudeAgentOptions: system_prompt can be str or dict
    sp = getattr(options, "system_prompt", None)
    if sp is None:
        return ""
    if isinstance(sp, str):
        return sp
    if isinstance(sp, dict):
        # {"type": "preset", "preset": "claude_code", "append": "..."}
        return sp.get("append", "") or sp.get("text", "") or ""
    return str(sp)


class HFRunner:
    """Lazy-loaded HuggingFace Transformers model runner (singleton)."""

    SYSTEM_SUFFIX = (
        "\n\nIMPORTANT: You must end your response with your final answer in this exact format:\n"
        "<FINAL_ANSWER>your answer here</FINAL_ANSWER>\n"
        "Keep the final answer concise - just the value, number, or short phrase requested."
    )

    def __init__(self, model_name: str, max_new_tokens: int = 512, device: str = "auto"):
        import os
        # Ensure proxy is set for model download
        # TODO: replace this or delete it
        os.environ.setdefault("HTTP_PROXY", "proxy1")
        os.environ.setdefault("HTTPS_PROXY", "peoxy2")
        os.environ.setdefault("HF_TOKEN", "youtoken")

        from transformers import AutoModelForCausalLM, AutoTokenizer

        logger.info(f"[HFRunner] Loading model: {model_name}")
        self.model_name = model_name
        self.max_new_tokens = max_new_tokens

        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name, trust_remote_code=True
        )
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype="auto",
            device_map=device,
            trust_remote_code=True,
        )
        self.model.eval()
        logger.info(f"[HFRunner] Model loaded on: {next(self.model.parameters()).device}")

    def generate(
        self,
        system_prompt: str,
        user_query: str,
        enable_thinking: bool = False,
        tools: dict | None = None,
    ) -> str:
        """Run inference and return the raw model output string.

        If tools are provided, uses transformers tool-calling pipeline (agentic loop).
        Otherwise falls back to plain text generation.
        """
        import torch

        if tools:
            return self._generate_with_tools(system_prompt, user_query, enable_thinking, tools)

        # Append FINAL_ANSWER instruction to system prompt
        full_system = (system_prompt or "") + self.SYSTEM_SUFFIX

        messages = [
            {"role": "system", "content": full_system},
            {"role": "user", "content": user_query},
        ]

        # Apply chat template
        if hasattr(self.tokenizer, "apply_chat_template") and self.tokenizer.chat_template:
            try:
                text = self.tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True,
                    enable_thinking=enable_thinking,
                )
            except TypeError:
                # Some models don't support enable_thinking
                text = self.tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True,
                )
        else:
            text = f"<|system|>\n{full_system}\n<|user|>\n{user_query}\n<|assistant|>\n"

        inputs = self.tokenizer(text, return_tensors="pt").to(self.model.device)

        with torch.no_grad():
            output_ids = self.model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                do_sample=False,
                temperature=None,
                top_p=None,
                pad_token_id=self.tokenizer.eos_token_id,
            )

        new_tokens = output_ids[0][inputs["input_ids"].shape[1]:]
        return self.tokenizer.decode(new_tokens, skip_special_tokens=True).strip()

    def _generate_with_tools(  # noqa: C901
        self,
        system_prompt: str,
        user_query: str,
        enable_thinking: bool,
        tools: dict,
    ) -> str:
        """Agentic tool-calling loop using transformers chat template tool support."""
        import json
        import re
        import torch

        # Build transformers-format tool list
        tf_tools = [
            {
                "type": "function",
                "function": {
                    "name": name,
                    "description": spec.get("description", ""),
                    "parameters": spec.get("parameters", {"type": "object", "properties": {}}),
                },
            }
            for name, spec in tools.items()
        ]

        full_system = (system_prompt or "") + self.SYSTEM_SUFFIX
        messages = [
            {"role": "system", "content": full_system},
            {"role": "user", "content": user_query},
        ]

        MAX_TOOL_TURNS = 30
        for _ in range(MAX_TOOL_TURNS):
            # Apply chat template with tools
            try:
                text = self.tokenizer.apply_chat_template(
                    messages,
                    tools=tf_tools,
                    tokenize=False,
                    add_generation_prompt=True,
                    enable_thinking=enable_thinking,
                )
            except TypeError:
                text = self.tokenizer.apply_chat_template(
                    messages,
                    tools=tf_tools,
                    tokenize=False,
                    add_generation_prompt=True,
                )

            inputs = self.tokenizer(text, return_tensors="pt").to(self.model.device)
            with torch.no_grad():
                output_ids = self.model.generate(
                    **inputs,
                    max_new_tokens=self.max_new_tokens,
                    do_sample=False,
                    temperature=None,
                    top_p=None,
                    pad_token_id=self.tokenizer.eos_token_id,
                )
            new_tokens = output_ids[0][inputs["input_ids"].shape[1]:]
            raw = self.tokenizer.decode(new_tokens, skip_special_tokens=False).strip()

            # Try to parse tool call from output
            # Transformers models typically output JSON tool calls in a special format
            tool_call_match = re.search(
                r'<tool_call>\s*(.*?)\s*</tool_call>', raw, re.DOTALL
            )
            if not tool_call_match:
                # No tool call — this is the final answer
                return self.tokenizer.decode(new_tokens, skip_special_tokens=True).strip()

            # Parse and execute tool call
            try:
                tc = json.loads(tool_call_match.group(1))
                fn_name = tc.get("name", "")
                fn_args = tc.get("arguments", tc.get("parameters", {}))
                if isinstance(fn_args, str):
                    fn_args = json.loads(fn_args)
            except (json.JSONDecodeError, KeyError):
                # Can't parse tool call, treat as final answer
                return self.tokenizer.decode(new_tokens, skip_special_tokens=True).strip()

            # Execute tool synchronously
            spec = tools.get(fn_name, {})
            fn = spec.get("fn") if isinstance(spec, dict) else None
            if fn is not None:
                try:
                    tool_result = fn(**fn_args)
                except Exception as e:
                    tool_result = f"[Tool error: {e}]"
            else:
                tool_result = f"[Tool '{fn_name}' not found]"

            # Append assistant tool call + tool result to messages
            # Use clean decoded text (skip_special_tokens=True) for the assistant message
            clean_raw = self.tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
            messages.append({"role": "assistant", "content": clean_raw})
            messages.append({
                "role": "tool",
                "name": fn_name,
                "content": str(tool_result),
            })

        # Exceeded max turns — return last raw output (new_tokens set in last iteration)
        try:
            return self.tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
        except UnboundLocalError:
            return "[Max tool turns exceeded with no output]"
