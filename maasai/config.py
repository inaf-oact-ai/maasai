from __future__ import annotations

import os
from dataclasses import dataclass, field


@dataclass(slots=True)
class LiteLLMProxySettings:
	base_url: str = field(default_factory=lambda: os.getenv("LITELLM_PROXY_BASE_URL", "http://localhost:4000"))
	api_key: str = field(default_factory=lambda: os.getenv("LITELLM_PROXY_API_KEY", "sk-local-dev"))
	timeout_seconds: float = field(default_factory=lambda: float(os.getenv("LITELLM_TIMEOUT_SECONDS", "180")))
	alias_small: str = field(default_factory=lambda: os.getenv("LITELLM_ALIAS_SMALL", "astro-small"))
	alias_medium: str = field(default_factory=lambda: os.getenv("LITELLM_ALIAS_MEDIUM", "astro-medium"))
	alias_large: str = field(default_factory=lambda: os.getenv("LITELLM_ALIAS_LARGE", "astro-large"))
	alias_multimodal: str = field(default_factory=lambda: os.getenv("LITELLM_ALIAS_MULTIMODAL", "astro-multimodal"))


@dataclass(slots=True)
class WorkflowSettings:
	max_approval_iterations: int = field(default_factory=lambda: int(os.getenv("MAASAI_MAX_APPROVAL_ITERATIONS", "3")))
	strict_english_only: bool = field(default_factory=lambda: os.getenv("MAASAI_STRICT_ENGLISH_ONLY", "true").lower() == "true")
	pii_policy: str = field(default_factory=lambda: os.getenv("MAASAI_PII_POLICY", "block"))


@dataclass(slots=True)
class Settings:
	litellm: LiteLLMProxySettings = field(default_factory=LiteLLMProxySettings)
	workflow: WorkflowSettings = field(default_factory=WorkflowSettings)


SETTINGS = Settings()
