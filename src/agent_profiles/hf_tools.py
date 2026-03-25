"""
Common tool implementations for HuggingFace / vLLM backend.

These replace the Claude SDK built-in tools (Read, Bash, Grep, Glob, Write)
with plain Python implementations that can be registered in the tools dict
and dispatched by the agentic loop in base.py.
"""

from __future__ import annotations

import subprocess


# Default max lines to read at once — prevents large files from flooding the context.
# Claude's built-in Read tool also paginates large files rather than dumping everything.
DEFAULT_READ_LIMIT = 250


def _read_file(path: str, offset: int = 0, limit: int = 0) -> str:
    """Read a file and return its contents.

    To avoid flooding the model context window, at most DEFAULT_READ_LIMIT lines
    are returned by default (matching Claude's built-in Read tool behaviour).
    Pass limit=-1 to read the entire file (use with caution on large files).

    Special case: for large CSV/JSON files (>1000 lines), when no explicit limit
    is given (limit=0) and offset=0, only the first line (header) is returned
    with a hint to use Bash+pandas for analysis. This prevents 250 rows of raw
    data from flooding the context.
    """
    # Large-file extensions that should never be dumped into context
    _LARGE_FILE_EXTS = {".csv", ".tsv", ".json", ".jsonl", ".ndjson"}

    import os
    try:
        with open(path, "r", encoding="utf-8", errors="replace") as f:
            lines = f.readlines()
        total_lines = len(lines)

        # Auto-cap large data files to header-only when no explicit limit given
        _ext = os.path.splitext(path)[1].lower()
        if limit == 0 and offset == 0 and _ext in _LARGE_FILE_EXTS and total_lines > 1000:
            header = lines[0] if lines else ""
            return (
                header.rstrip("\n")
                + f"\n\n[Large file ({total_lines} lines): only the header is shown. "
                f"Use Bash with pandas/python to analyze data, e.g.:\n"
                f"  python3 -c \"import pandas as pd; df=pd.read_csv('{path}'); ...\"]"
            )

        if offset:
            lines = lines[offset:]
        # limit=0  → apply default cap
        # limit=-1 → read all (explicit override)
        # limit>0  → honour caller's explicit limit
        if limit == 0:
            effective_limit = DEFAULT_READ_LIMIT
        elif limit == -1:
            effective_limit = None
        else:
            effective_limit = limit
        if effective_limit is not None:
            truncated = len(lines) > effective_limit
            lines = lines[:effective_limit]
        else:
            truncated = False
        result = "".join(lines)
        if truncated:
            shown_end = offset + effective_limit
            result += (
                f"\n[... file truncated: showing lines {offset+1}-{shown_end} of {total_lines} total. "
                f"Use offset={shown_end} to read the next chunk, or use Bash with pandas/grep for large files.]"
            )
        return result
    except FileNotFoundError:
        return f"[Error: file not found: {path}]"
    except Exception as e:
        return f"[Error reading {path}: {e}]"


def _bash(command: str, timeout: int = 30) -> str:
    """Execute a shell command and return stdout + stderr."""
    try:
        result = subprocess.run(
            command,
            shell=True,
            capture_output=True,
            text=True,
            timeout=timeout,
        )
        output = result.stdout
        if result.stderr:
            output += "\n[stderr]\n" + result.stderr
        return output.strip() or "[No output]"
    except subprocess.TimeoutExpired:
        return f"[Error: command timed out after {timeout}s]"
    except Exception as e:
        return f"[Error running command: {e}]"


def _grep(pattern: str, path: str = ".", include: str = "", flags: str = "") -> str:
    """Search for a pattern in files using grep."""
    import os
    # Block grep on large data files to prevent context overflow
    _BLOCKED_EXTS = {".csv", ".tsv", ".jsonl", ".ndjson"}
    _target_ext = os.path.splitext(path)[1].lower()
    if _target_ext in _BLOCKED_EXTS:
        return (
            f"[Grep blocked on '{path}': Grep must NOT be used on CSV/TSV/JSONL files "
            f"because it dumps raw rows into context and overflows the context window. "
            f"Use Bash+pandas instead, e.g.:\n"
            f"  python3 -c \"import pandas as pd; df=pd.read_csv('{path}'); "
            f"print(df[df['<column>']=='{pattern}'])\"]"
        )
    cmd = ["grep", "-rn"]
    if include:
        cmd += [f"--include={include}"]
    if flags:
        cmd += flags.split()
    cmd += [pattern, path]
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=15)
        return result.stdout.strip() or "[No matches found]"
    except Exception as e:
        return f"[Error: {e}]"


def _glob(pattern: str, path: str = ".") -> str:
    """List files matching a glob pattern."""
    import glob as _glob
    import os
    full_pattern = os.path.join(path, pattern) if not pattern.startswith("/") else pattern
    matches = _glob.glob(full_pattern, recursive=True)
    if not matches:
        return "[No files matched]"
    return "\n".join(sorted(matches))


def _write_file(path: str, content: str) -> str:
    """Write content to a file."""
    import os
    try:
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            f.write(content)
        return f"[Written {len(content)} bytes to {path}]"
    except Exception as e:
        return f"[Error writing {path}: {e}]"


def _skill(name: str = "") -> str:
    """Read a skill file from .claude/skills/<name>/SKILL.md.

    If name is empty or "list", returns a list of all available skills.
    Otherwise reads and returns the content of the specified skill's SKILL.md.
    This mirrors the behavior of Claude SDK's built-in Skill tool.
    """
    import os
    import glob as _glob_mod

    # Locate the project root by searching upward for pyproject.toml
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
        return "[Error: could not locate .claude/skills directory]"

    # List mode: return all available skills with their descriptions
    if not name or name.strip().lower() in ("", "list"):
        skill_dirs = sorted([
            d for d in os.listdir(skills_dir)
            if os.path.isdir(os.path.join(skills_dir, d))
        ])
        if not skill_dirs:
            return "[No skills available]"
        lines = ["Available skills:"]
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
            lines.append(f"  - {skill_name}: {desc}" if desc else f"  - {skill_name}")
        return "\n".join(lines)

    # Read specific skill
    skill_file = os.path.join(skills_dir, name.strip(), "SKILL.md")
    if not os.path.exists(skill_file):
        # Try case-insensitive match
        try:
            all_skills = os.listdir(skills_dir)
            match = next((s for s in all_skills if s.lower() == name.strip().lower()), None)
            if match:
                skill_file = os.path.join(skills_dir, match, "SKILL.md")
        except Exception:
            pass

    if not os.path.exists(skill_file):
        available = ", ".join(sorted([
            d for d in os.listdir(skills_dir)
            if os.path.isdir(os.path.join(skills_dir, d))
        ]))
        return f"[Error: skill '{name}' not found. Available skills: {available}]"

    try:
        with open(skill_file, "r", encoding="utf-8") as f:
            return f.read()
    except Exception as e:
        return f"[Error reading skill '{name}': {e}]"


# ─────────────────────────────────────────────────────────────────────────────
# Tools dict — format: {name: {"fn": callable, "description": str, "parameters": dict}}
# ─────────────────────────────────────────────────────────────────────────────

HF_TOOLS: dict = {
    "Read": {
        "fn": _read_file,
        "description": (
            f"Read the contents of a file at the given path. "
            f"Returns at most {DEFAULT_READ_LIMIT} lines by default. "
            f"USE CASES: (1) Read README/manual files in full (pass limit=-1). "
            f"(2) Read CSV header only (pass limit=1) to get exact column names before writing any Bash command. "
            f"(3) Read small config/JSON files. "
            f"DO NOT use Read to analyze large CSV data — use Bash+pandas instead."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": "Absolute or relative path to the file"},
                "offset": {"type": "integer", "description": "Line offset to start reading from (0-indexed)"},
                "limit": {"type": "integer", "description": f"Max lines to read (default {DEFAULT_READ_LIMIT}; pass -1 to read entire file)"},
            },
            "required": ["path"],
        },
    },
    "Bash": {
        "fn": _bash,
        "description": (
            "Execute a shell command and return its output. "
            "IMPORTANT: Before using column names in any command, you MUST know the exact column names "
            "(from the pre-loaded CSV headers in context, or from Read with limit=1). "
            "NEVER guess column names or use positional indices like $1, $2. "
            "For ANY analysis on CSV/TSV/JSONL files (counting, filtering, aggregation, lookup), "
            "you MUST use Bash+pandas — do NOT use Grep or Read on these files: "
            "python3 -c \"import pandas as pd; df=pd.read_csv('file.csv'); print(df['col'].value_counts())\""
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "command": {"type": "string", "description": "Shell command to execute"},
                "timeout": {"type": "integer", "description": "Timeout in seconds (default 30)"},
            },
            "required": ["command"],
        },
    },
    "Grep": {
        "fn": _grep,
        "description": (
            "Search for a regex pattern in files. "
            "WARNING: NEVER use Grep on large data files (.csv, .tsv, .jsonl) — "
            "it dumps all matching raw rows into context and will overflow the context window. "
            "For CSV/TSV/JSONL data, ALWAYS use Bash+pandas instead: "
            "python3 -c \"import pandas as pd; df=pd.read_csv('file.csv'); print(df[df['col']=='value'])\". "
            "Grep is only suitable for small text/config/code files (e.g. .md, .py, .json config files)."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "pattern": {"type": "string", "description": "Regex pattern to search for"},
                "path": {"type": "string", "description": "Directory or file to search in"},
                "include": {"type": "string", "description": "File glob filter, e.g. '*.py'"},
                "flags": {"type": "string", "description": "Extra grep flags, e.g. '-i'"},
            },
            "required": ["pattern"],
        },
    },
    "Glob": {
        "fn": _glob,
        "description": "List files matching a glob pattern.",
        "parameters": {
            "type": "object",
            "properties": {
                "pattern": {"type": "string", "description": "Glob pattern, e.g. '**/*.csv'"},
                "path": {"type": "string", "description": "Base directory to search in"},
            },
            "required": ["pattern"],
        },
    },
    "Write": {
        "fn": _write_file,
        "description": "Write content to a file (creates parent directories if needed).",
        "parameters": {
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": "Path to write to"},
                "content": {"type": "string", "description": "Content to write"},
            },
            "required": ["path", "content"],
        },
    },
    "Skill": {
        "fn": _skill,
        "description": (
            "Read a skill (reusable workflow/guide) from the project's .claude/skills directory. "
            "Call with name='' or name='list' to list all available skills and their descriptions. "
            "Call with a specific skill name (e.g. name='brainstorming') to read its full SKILL.md content. "
            "Always check available skills before starting complex analysis tasks."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "name": {
                    "type": "string",
                    "description": (
                        "Name of the skill to read (e.g. 'brainstorming', 'data-extraction-verification'). "
                        "Leave empty or pass 'list' to list all available skills."
                    ),
                },
            },
            "required": [],
        },
    },
}
