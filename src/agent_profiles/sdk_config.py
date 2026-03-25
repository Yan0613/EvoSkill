"""SDK configuration and selection logic.

This module provides a global setting to choose between claude-agent-sdk, opencode-ai,
and huggingface transformers (open-source models).
"""

from typing import Literal

SDKType = Literal["claude", "opencode", "huggingface", "vllm"]

# Global SDK selection (can be overridden via CLI arguments)
_current_sdk: SDKType = "claude"

# Global HuggingFace model config (set when sdk == "huggingface")
_hf_model_name: str = "Qwen/Qwen3-4B"
_hf_max_new_tokens: int = 512
_hf_device: str = "auto"
_hf_enable_thinking: bool = False

# Global vLLM config (set when sdk == "vllm")
_vllm_base_url: str = "http://localhost:8000/v1"
_vllm_model_name: str = "Qwen/Qwen3-4B"
_vllm_max_tokens: int = 8192
_vllm_api_key: str = "EMPTY"
_vllm_context_length: int = 131072  # model's max context window (e.g. 128K for 72B)


def set_sdk(sdk: SDKType) -> None:
    """Set the current SDK to use globally."""
    global _current_sdk
    if sdk not in ("claude", "opencode", "huggingface", "vllm"):
        raise ValueError(f"Invalid SDK type: {sdk}. Must be 'claude', 'opencode', 'huggingface', or 'vllm'")
    _current_sdk = sdk


def set_hf_config(
    model_name: str = "Qwen/Qwen3-4B",
    max_new_tokens: int = 512,
    device: str = "auto",
    enable_thinking: bool = False,
) -> None:
    """Configure HuggingFace model settings (only used when sdk == 'huggingface')."""
    global _hf_model_name, _hf_max_new_tokens, _hf_device, _hf_enable_thinking
    _hf_model_name = model_name
    _hf_max_new_tokens = max_new_tokens
    _hf_device = device
    _hf_enable_thinking = enable_thinking


def get_hf_config() -> dict:
    """Get current HuggingFace model configuration."""
    return {
        "model_name": _hf_model_name,
        "max_new_tokens": _hf_max_new_tokens,
        "device": _hf_device,
        "enable_thinking": _hf_enable_thinking,
    }


def set_vllm_config(
    base_url: str = "http://localhost:8000/v1",
    model_name: str = "Qwen/Qwen3-4B",
    max_tokens: int = 8192,
    api_key: str = "EMPTY",
    context_length: int = 131072,
) -> None:
    """Configure vLLM server settings (only used when sdk == 'vllm')."""
    global _vllm_base_url, _vllm_model_name, _vllm_max_tokens, _vllm_api_key, _vllm_context_length
    _vllm_base_url = base_url
    _vllm_model_name = model_name
    _vllm_max_tokens = max_tokens
    _vllm_api_key = api_key
    _vllm_context_length = context_length


def get_vllm_config() -> dict:
    """Get current vLLM server configuration."""
    return {
        "base_url": _vllm_base_url,
        "model_name": _vllm_model_name,
        "max_tokens": _vllm_max_tokens,
        "api_key": _vllm_api_key,
        "context_length": _vllm_context_length,
    }


def get_sdk() -> SDKType:
    """Get the currently configured SDK."""
    return _current_sdk


def is_claude_sdk() -> bool:
    """Check if claude-agent-sdk is the current SDK."""
    return _current_sdk == "claude"


def is_opencode_sdk() -> bool:
    """Check if opencode-ai is the current SDK."""
    return _current_sdk == "opencode"


def is_huggingface_sdk() -> bool:
    """Check if HuggingFace Transformers is the current SDK."""
    return _current_sdk == "huggingface"


def is_vllm_sdk() -> bool:
    """Check if vLLM (OpenAI-compatible server) is the current SDK."""
    return _current_sdk == "vllm"
