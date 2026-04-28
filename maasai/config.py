from __future__ import print_function
from __future__ import annotations

##################################################
###          MODULE IMPORT
##################################################
# - STANDARD MODULES
import os
from dataclasses import dataclass, field

# - MAASAI MODULES
from .rag import PlannerRAGSettings

###################################################
###     HELPER METHODS
###################################################
def _env_bool(name: str, default: bool) -> bool:
	value = os.getenv(name)
	if value is None:
		return default
	return value.strip().lower() in {"1", "true", "yes", "y", "on"}


def _env_list(name: str, default: str) -> list[str]:
	return [
		item.strip()
		for item in os.getenv(name, default).split(",")
		if item.strip()
	]

###################################################
###     CONFIG CLASSES
###################################################
#@dataclass(slots=True)
#class LiteLLMProxySettings:
#	base_url: str = field(default_factory=lambda: os.getenv("LITELLM_PROXY_BASE_URL", "http://localhost:4000"))
#	api_key: str = field(default_factory=lambda: os.getenv("LITELLM_PROXY_API_KEY", "sk-local-dev"))
#	timeout_seconds: float = field(default_factory=lambda: float(os.getenv("LITELLM_TIMEOUT_SECONDS", "180")))
#	alias_small: str = field(default_factory=lambda: os.getenv("LITELLM_ALIAS_SMALL", "astro-small"))
#	alias_medium: str = field(default_factory=lambda: os.getenv("LITELLM_ALIAS_MEDIUM", "astro-medium"))
#	alias_large: str = field(default_factory=lambda: os.getenv("LITELLM_ALIAS_LARGE", "astro-large"))
#	alias_multimodal: str = field(default_factory=lambda: os.getenv("LITELLM_ALIAS_MULTIMODAL", "astro-multimodal"))

@dataclass(slots=True)
class LiteLLMSettings:
	"""LiteLLM runtime settings.

	Model aliases and model routing are intentionally not defined here.
	They are read from the LiteLLM YAML config passed with --config_litellm.
	"""
	timeout_seconds: float = field(
		default_factory=lambda: float(os.getenv("LITELLM_TIMEOUT_SECONDS", "180"))
	)
	num_retries: int = field(
		default_factory=lambda: int(os.getenv("LITELLM_NUM_RETRIES", "0"))
	)
	retry_after: float = field(
		default_factory=lambda: float(os.getenv("LITELLM_RETRY_AFTER", "0"))
	)

@dataclass(slots=True)
class LLMSettings:
	"""Generic LLM call settings used by MAASAI nodes."""

	timeout_seconds: float = field(
		default_factory=lambda: float(os.getenv("LLM_TIMEOUT_SECONDS", "20"))
	)

@dataclass(slots=True)
class WorkflowSettings:
	max_approval_iterations: int = field(
		default_factory=lambda: int(os.getenv("MAASAI_MAX_APPROVAL_ITERATIONS", "3"))
	)
	strict_english_only: bool = field(
		default_factory=lambda: _env_bool("MAASAI_STRICT_ENGLISH_ONLY", True)
	)
	pii_policy: str = field(
		default_factory=lambda: os.getenv("MAASAI_PII_POLICY", "block")
	)

@dataclass(slots=True)
class RuntimeSettings:
	"""Non-model runtime settings."""

	mlflow_enabled: bool = field(
		default_factory=lambda: _env_bool("MAASAI_MLFLOW_ENABLED", True)
	)
	print_graph: bool = field(
		default_factory=lambda: _env_bool("MAASAI_PRINT_GRAPH", True)
	)

@dataclass(slots=True)
class Settings:
	litellm: LiteLLMSettings = field(default_factory=LiteLLMSettings)
	llm: LLMSettings = field(default_factory=LLMSettings)
	workflow: WorkflowSettings = field(default_factory=WorkflowSettings)
	rag: PlannerRAGSettings = field(default_factory=PlannerRAGSettings)
	runtime: RuntimeSettings = field(default_factory=RuntimeSettings)

SETTINGS = Settings()
