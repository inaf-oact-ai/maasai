from __future__ import print_function
from __future__ import annotations

##################################################
###          MODULE IMPORT
##################################################
# - STANDARD MODULES
from dataclasses import dataclass
from typing import Any

# - LLANGCHAIN MODULES
from langchain_litellm import ChatLiteLLMRouter

# - MAASAI MODULES
from maasai import logger

##################################################
###          HELPER CLASS
##################################################
@dataclass(frozen=True)
class CapabilityRequirements:
	"""Optional capability constraints for selecting a logical alias."""

	tool_required: bool = False
	structured_output_required: bool = False
	prefer_local: bool = False
	allow_commercial: bool = True

##################################################
###          ROUTER CLASS
##################################################
class ModelRouter:
	"""MAASAI logical router on top of an embedded LiteLLM Router.

	Responsibilities:
	1. Select a logical alias (e.g. ``model-small``) based on workflow stage,
	   complexity, and optional capability requirements.
	2. Return a cached ``ChatLiteLLMRouter`` instance for that alias, so it can
	   be passed directly to LangChain / LangGraph agents.
	3. Leave actual deployment-level balancing to LiteLLM's ``Router``, which
	   selects among all configured backends registered under the same alias.

	Expected config layout (loaded from YAML):

	{
		"model_list": [...],
		"routing_policy": {
			"default_alias": "model-small",
			"task_to_alias": {...},
			"fallbacks": {...},
			"provider_preferences": {...}
		}
	}
	"""

	def __init__(
		self,
		litellm_router: Any,
		config: dict[str, Any],
		default_temperature: float = 0.0,
		default_max_tokens: int | None = None,
	) -> None:
		self.router = litellm_router
		self.config = config
		self.routing_policy: dict[str, Any] = config.get("routing_policy", {})
		self.model_list: list[dict[str, Any]] = config.get("model_list", [])
		self.default_temperature = default_temperature
		self.default_max_tokens = default_max_tokens

		self.models_by_alias: dict[str, list[dict[str, Any]]] = self._index_models(self.model_list)
		self._llm_cache: dict[tuple[str, float, int | None], ChatLiteLLMRouter] = {}

	def _index_models(self, model_list: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
		by_alias: dict[str, list[dict[str, Any]]] = {}
		for entry in model_list:
			alias = entry.get("model_name")
			if not alias:
				continue
			by_alias.setdefault(alias, []).append(entry)
		return by_alias

	def _metadata_for_alias(self, alias: str) -> list[dict[str, Any]]:
		entries = self.models_by_alias.get(alias, [])
		out: list[dict[str, Any]] = []
		for entry in entries:
			meta = entry.get("model_info", {}).get("metadata", {})
			if isinstance(meta, dict):
				out.append(meta)
		return out

	def _alias_exists(self, alias: str) -> bool:
		return alias in self.models_by_alias and len(self.models_by_alias[alias]) > 0

	def _alias_supports_capabilities(
		self,
		alias: str,
		req: CapabilityRequirements,
	) -> bool:
		metadata_list = self._metadata_for_alias(alias)
		if not metadata_list:
			# No metadata -> permissive fallback
			return True

		def entry_ok(meta: dict[str, Any]) -> bool:
			if req.tool_required and meta.get("tool_calling") is not True:
				return False
			if req.structured_output_required and meta.get("structured_output") is not True:
				return False
			if req.prefer_local and meta.get("cost_tier") != "local":
				return False
			if not req.allow_commercial and meta.get("cost_tier") == "commercial":
				return False
			return True

		return any(entry_ok(meta) for meta in metadata_list)

	def _stage_default_alias(self, stage: str) -> str:
		task_to_alias = self.routing_policy.get("task_to_alias", {})
		default_alias = self.routing_policy.get("default_alias", "model-small")
		return task_to_alias.get(stage, default_alias)

	def _stage_fallback_aliases(self, stage: str) -> list[str]:
		fallbacks = self.routing_policy.get("fallbacks", {})
		values = fallbacks.get(stage, [])
		if isinstance(values, str):
			return [values]
		if isinstance(values, list):
			return [v for v in values if isinstance(v, str)]
		return []

	def _preferred_local_stages(self) -> set[str]:
		prefs = self.routing_policy.get("provider_preferences", {})
		values = prefs.get("prefer_local_for", [])
		return set(values) if isinstance(values, list) else set()

	def _commercial_allowed_stages(self) -> set[str]:
		prefs = self.routing_policy.get("provider_preferences", {})
		values = prefs.get("allow_commercial_for", [])
		return set(values) if isinstance(values, list) else set()

	def _complexity_override(self, stage: str, complexity: str) -> str | None:
		if complexity != "complex":
			return None

		hard_stage_map = {
			"planner": "model-large",
			"supervisor": "model-large",
			"aggregation": "model-large",
			"final_response": "model-large",
		}
		alias = hard_stage_map.get(stage)
		if alias and self._alias_exists(alias):
			return alias
		return None

	def _build_requirements(
		self,
		stage: str,
		*,
		tool_required: bool,
		structured_output_required: bool,
	) -> CapabilityRequirements:
		prefer_local = stage in self._preferred_local_stages()
		allow_commercial = stage in self._commercial_allowed_stages() or not prefer_local
		logger.info(f"router::_build_requirements(): stage={stage}, allow_commercial={allow_commercial}")
		print("self._preferred_local_stages()")
		print(self._preferred_local_stages())
		
		return CapabilityRequirements(
			tool_required=tool_required,
			structured_output_required=structured_output_required,
			prefer_local=prefer_local,
			allow_commercial=allow_commercial,
		)

	def pick_alias(
		self,
		*,
		stage: str,
		complexity: str = "simple",
		tool_required: bool = False,
		structured_output_required: bool = False,
	) -> str:
		"""Pick the best logical alias for a workflow stage."""

		req = self._build_requirements(
			stage,
			tool_required=tool_required,
			structured_output_required=structured_output_required,
		)
		print("router::pick_alias(): req")
		print(req)

		candidates: list[str] = []

		override = self._complexity_override(stage, complexity)
		if override:
			candidates.append(override)

		stage_default = self._stage_default_alias(stage)
		
		candidates.append(stage_default)

		candidates.extend(self._stage_fallback_aliases(stage))

		default_alias = self.routing_policy.get("default_alias", "model-small")
		logger.info(f"router::pick_alias(): stage_default={stage_default}, default_alias={default_alias}")
		
		candidates.append(default_alias)
		
		print("router::pick_alias(): candidates")
		print(candidates)

		seen: set[str] = set()
		ordered_candidates = [c for c in candidates if c and not (c in seen or seen.add(c))]

		for alias in ordered_candidates:
			if not self._alias_exists(alias):
				continue
			if self._alias_supports_capabilities(alias, req):
				return alias

		for alias in ordered_candidates:
			if self._alias_exists(alias):
				return alias

		raise ValueError("No valid model alias found in routing policy/model_list.")

	def get_llm(
		self,
		*,
		stage: str,
		complexity: str = "simple",
		tool_required: bool = False,
		structured_output_required: bool = False,
		temperature: float | None = None,
		max_tokens: int | None = None,
		**kwargs: Any,
	) -> ChatLiteLLMRouter:
		"""Return a cached ChatLiteLLMRouter for the selected alias.

		This is the main method you should use before passing the model to
		LangChain / LangGraph agent constructors.
		"""

		alias = self.pick_alias(
			stage=stage,
			complexity=complexity,
			tool_required=tool_required,
			structured_output_required=structured_output_required,
		)
		logger.info(f"router::get_llm(): stage={stage}, complexity={complexity}, alias={alias}")

		final_temperature = self.default_temperature if temperature is None else temperature
		final_max_tokens = self.default_max_tokens if max_tokens is None else max_tokens

		cache_key = (alias, final_temperature, final_max_tokens)
		if cache_key not in self._llm_cache:
			llm_kwargs: dict[str, Any] = {
				"router": self.router,
				"model_name": alias,
				"temperature": final_temperature,
			}
			if final_max_tokens is not None:
				llm_kwargs["max_tokens"] = final_max_tokens
			llm_kwargs.update(kwargs)

			self._llm_cache[cache_key] = ChatLiteLLMRouter(**llm_kwargs)

		return self._llm_cache[cache_key]

	async def ainvoke(
		self,
		*,
		stage: str,
		messages: list[Any],
		complexity: str = "simple",
		tool_required: bool = False,
		structured_output_required: bool = False,
		temperature: float | None = None,
		max_tokens: int | None = None,
		**kwargs: Any,
	) -> Any:
		"""Convenience async invoke wrapper."""
		llm = self.get_llm(
			stage=stage,
			complexity=complexity,
			tool_required=tool_required,
			structured_output_required=structured_output_required,
			temperature=temperature,
			max_tokens=max_tokens,
		)
		return await llm.ainvoke(messages, **kwargs)

	def invoke(
		self,
		*,
		stage: str,
		messages: list[Any],
		complexity: str = "simple",
		tool_required: bool = False,
		structured_output_required: bool = False,
		temperature: float | None = None,
		max_tokens: int | None = None,
		**kwargs: Any,
	) -> Any:
		"""Convenience sync invoke wrapper."""
		llm = self.get_llm(
			stage=stage,
			complexity=complexity,
			tool_required=tool_required,
			structured_output_required=structured_output_required,
			temperature=temperature,
			max_tokens=max_tokens,
		)
		return llm.invoke(messages, **kwargs)

	def debug_selection(
		self,
		*,
		stage: str,
		complexity: str = "simple",
		tool_required: bool = False,
		structured_output_required: bool = False,
	) -> dict[str, Any]:
		"""Return debugging info for why an alias was selected."""
		alias = self.pick_alias(
			stage=stage,
			complexity=complexity,
			tool_required=tool_required,
			structured_output_required=structured_output_required,
		)
		return {
			"stage": stage,
			"complexity": complexity,
			"tool_required": tool_required,
			"structured_output_required": structured_output_required,
			"selected_alias": alias,
			"available_aliases": sorted(self.models_by_alias.keys()),
			"selected_alias_entries": self.models_by_alias.get(alias, []),
		}
