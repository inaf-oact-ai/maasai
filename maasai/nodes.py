from __future__ import print_function
from __future__ import annotations

##################################################
###          MODULE IMPORT
##################################################
# - STANDARD MODULES
import re
from typing import Any
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeoutError

# - LLM/LANGCHAIN MODULES
from langchain.messages import HumanMessage
from langgraph.types import interrupt

# - MAASAI MODULES
from .agents import AgentFactory
from .config import Settings
from .guardrails import detect_pii, is_probably_english, is_scientific_or_astronomy_related, wrap_guardrail_response
from .rag import PromptRAG
from .assets import _prepare_asset, _asset_field
from .schemas import FinalAnswer
from .schemas import PreparedAsset
from .schemas import PromptAssessment
from .schemas import OptimizedPrompt
from .schemas import ApprovalDecision
from .schemas import TaskPlan
from .schemas import PlanStep
from .schemas import StepResult
from .state import GraphState
from .context import NodeContext
from maasai import logger

##################################################
###          HELPER METHODS
##################################################
def _extract_text(messages: list[Any]) -> str:
	""" Extract text from input messages """
	chunks: list[str] = []
	for message in messages:
		content = getattr(message, "content", message)
		if isinstance(content, str):
			chunks.append(content)
		elif isinstance(content, list):
			for item in content:
				if isinstance(item, dict) and item.get("type") == "text":
					chunks.append(str(item.get("text", "")))
	return "\n".join(part for part in chunks if part).strip()

def _strip_rewrite_metadata_sections(text: str) -> str:
	"""Remove duplicated metadata sections from rewritten_prompt."""
	if not text:
		return text

	markers = [
		"\n**Assumptions**:",
		"\nAssumptions:",
		"\n**Open Questions**:",
		"\nOpen Questions:",
		"\n**Rationale**:",
		"\nRationale:",
	]

	cut_positions = [
		text.find(marker)
		for marker in markers
		if text.find(marker) != -1
	]

	if not cut_positions:
		return text.strip()

	return text[:min(cut_positions)].strip()

def _strip_references_section(text: str) -> str:
	"""Remove model-generated reference/bibliography sections from final answer body."""
	if not text:
		return text

	markers = [
		"\n## References",
		"\n# References",
		"\n**References**",
		"\nReferences:",
		"\n---\n**References**",
		"\n## Bibliography",
		"\n# Bibliography",
		"\n**Bibliography**",
		"\nBibliography:",
		"\n## Works Cited",
		"\n# Works Cited",
		"\n**Works Cited**",
		"\nWorks Cited:",
	]

	positions = [
		text.find(marker)
		for marker in markers
		if text.find(marker) != -1
	]

	if not positions:
		return text.strip()

	return text[:min(positions)].rstrip("- \n")
	
def _normalize_inline_citations(text: str) -> str:
	"""Normalize repeated numeric inline citations such as [1, 1] -> [1]."""
	if not text:
		return text

	def repl(match: re.Match[str]) -> str:
		raw = match.group(1)
		parts = [
			part.strip()
			for part in raw.split(",")
			if part.strip().isdigit()
		]

		if not parts:
			return match.group(0)

		unique = []
		for part in parts:
			if part not in unique:
				unique.append(part)

		return "[" + ", ".join(unique) + "]"

	return re.sub(r"\[([0-9,\s]+)\]", repl, text)

#def _invoke_with_timeout(agent, payload, timeout_s: float):
#	with ThreadPoolExecutor(max_workers=1) as executor:
#		future = executor.submit(agent.invoke, payload)
#		return future.result(timeout=timeout_s)

def _invoke_with_timeout(agent, payload, timeout_s: float):
	executor = ThreadPoolExecutor(max_workers=1)
	future = executor.submit(agent.invoke, payload)
	try:
		return future.result(timeout=timeout_s)
	except FuturesTimeoutError:
		future.cancel()
		raise
	finally:
		executor.shutdown(wait=False, cancel_futures=True)

def _build_intake_prompt(
	text: str,
	prepared_assets: list[Any] | None = None,
) -> str:
	""" Create prompt for intake agent """
	prepared_assets = prepared_assets or []

	lines = [
		"USER INPUT:",
		text or "<empty>",
	]

	if prepared_assets:
		lines.extend([
			"",
			"ATTACHMENTS:",
		])
		
		for idx, asset in enumerate(prepared_assets, start=1):
			path = _asset_field(asset, "path", "")
			kind = _asset_field(asset, "kind", "unknown")
			notes = _asset_field(asset, "notes", [])
			original_mime_type = _asset_field(asset, "original_mime_type", None)
			preview_mime_type = _asset_field(asset, "preview_mime_type", None)

			lines.append(f"- Asset {idx}:")
			lines.append(f"  kind: {kind}")
			lines.append(f"  path: {path}")
			if original_mime_type:
				lines.append(f"  original_mime_type: {original_mime_type}")
			if preview_mime_type:
				lines.append(f"  preview_mime_type: {preview_mime_type}")
			if notes:
				lines.append(f"  notes: {notes}")

	else:
		lines.extend([
			"",
			"ATTACHMENTS: none",
		])

	lines.extend([
		"",
		"Provide your structured decision.",
	])

	return "\n".join(lines)


def _build_intake_message_content(
	text: str,
	prepared_assets: list[PreparedAsset],
) -> list[dict[str, Any]]:
	content = [{"type": "text", "text": _build_intake_prompt(text, prepared_assets)}]

	for asset in prepared_assets:
		preview_mime_type = getattr(asset, "preview_mime_type", None)
		if asset.base64_data and preview_mime_type:
			content.append({
				"type": "image_url",
				"image_url": {
					"url": f"data:{preview_mime_type};base64,{asset.base64_data}"
				},
			})

	return content


def _build_assessment_prompt(
	raw_user_text: str,
	prepared_assets: list[Any] | None = None,
	intake_decision: Any | None = None,
) -> str:
	"""Create prompt for assessment agent."""
	prepared_assets = prepared_assets or []

	lines = [
		"USER REQUEST:",
		raw_user_text or "<empty>",
	]

	if intake_decision is not None:
		lines.extend([
			"",
			"INTAKE SUMMARY:",
			f"- accepted: {getattr(intake_decision, 'accepted', None)}",
			f"- domain_ok: {getattr(intake_decision, 'domain_ok', None)}",
			f"- reason: {getattr(intake_decision, 'reason', None)}",
		])

	if prepared_assets:
		lines.extend([
			"",
			"ATTACHMENTS:",
		])
		for idx, asset in enumerate(prepared_assets, start=1):
			lines.append(f"- Asset {idx}:")
			lines.append(f"  kind: {_asset_field(asset, 'kind', 'unknown')}")
			lines.append(f"  path: {_asset_field(asset, 'path', '')}")
			notes = _asset_field(asset, 'notes', [])
			if notes:
				lines.append(f"  notes: {notes}")
	else:
		lines.extend([
			"",
			"ATTACHMENTS: none",
		])
	
	lines.extend([
		"",
		"ASSESSMENT INSTRUCTIONS:",
		"- needs_rewrite must be True only if downstream execution should not proceed without rewriting.",
		"- rewrite_would_help should be True if the request is understandable but could be improved.",
		"- executable_as_is should be True if a useful direct answer or action is possible already.",
		"- Identify missing details and non-blocking ambiguities separately.",
		"- If rewrite is useful or required, provide a rewrite_goal describing the intended improvement.",
		"",
		"WORKER ROUTING RULES:",
		"- suggested_worker must be one of: general, image, catalog, literature, step-dependent.",
		"- Use suggested_worker='literature' when the request asks for papers, references, citations, inline references, bibliography, DOI/arXiv identifiers, publication context, literature review, related work, state of the art, or introduction/background sections with references.",
		"- Use suggested_worker='literature' for scientific writing tasks that require references, e.g. introduction sections, related-work sections, background sections, abstracts with citations, or literature-grounded summaries.",
		"- Use suggested_worker='image' when the request asks to inspect, classify, detect, segment, measure, or analyze objects/sources/morphology in attached images or FITS files.",
		"- Use suggested_worker='catalog' when the request asks for catalog lookup, cross-matching, counterpart search, source metadata, coordinates, cone search, source tables, survey tables, or CAESAR queries.",
		"- Use suggested_worker='general' only for conceptual explanations, workflow design, coding help, mathematical reasoning, or synthesis tasks that do not require image data, catalog/database access, or literature citations.",
		"- If requires_planning=True and the task needs multiple specialist capabilities, set suggested_worker='step-dependent' and populate required_workers with the executable workers likely needed.",
		"- required_workers must contain only executable workers: general, image, catalog, literature.",
		"- Do not put 'step-dependent' inside required_workers.",
		"",
		"TASK TYPE RULES:",
		"- For reference-grounded writing, use task_type='literature-grounded-writing'.",
		"- For paper/reference search or summaries, use task_type='literature-review'.",
		"- For attached image or FITS analysis, use task_type='image-analysis'.",
		"- For catalog/counterpart/coordinate/database queries, use task_type='catalog-query'.",
		"- For multi-step tasks requiring multiple specialist workers, use task_type='multi-step-specialist-analysis'.",
		"- For conceptual methodology, coding, or workflow design without required citations, use task_type='workflow-design', 'coding', or 'explanation'.",
	])

	return "\n".join(lines)

def _build_rewrite_prompt(
	raw_user_text: str,
	assessment: PromptAssessment,
	prepared_assets: list[Any] | None = None,
) -> str:
	"""Create prompt for rewrite agent."""
	prepared_assets = prepared_assets or []

	lines = [
		"ORIGINAL USER REQUEST:",
		raw_user_text or "<empty>",
		"",
		"ASSESSMENT SUMMARY:",
		f"- needs_rewrite: {assessment.needs_rewrite}",
		f"- rewrite_would_help: {assessment.rewrite_would_help}",
		f"- executable_as_is: {assessment.executable_as_is}",
		f"- complexity: {assessment.complexity}",
		f"- requires_planning: {assessment.requires_planning}",
		f"- task_type: {assessment.task_type}",
		f"- suggested_worker: {assessment.suggested_worker}",
		f"- required_workers: {assessment.required_workers}",
		f"- missing_details: {assessment.missing_details}",
		f"- ambiguities: {assessment.ambiguities}",
		f"- rewrite_goal: {assessment.rewrite_goal}",
	]

	if prepared_assets:
		lines.extend([
			"",
			"ATTACHMENTS:",
		])
		for idx, asset in enumerate(prepared_assets, start=1):
			lines.append(f"- Asset {idx}:")
			lines.append(f"  kind: {_asset_field(asset, 'kind', 'unknown')}")
			lines.append(f"  path: {_asset_field(asset, 'path', '')}")
			notes = _asset_field(asset, "notes", [])
			if notes:
				lines.append(f"  notes: {notes}")
	else:
		lines.extend([
			"",
			"ATTACHMENTS: none",
		])

	lines.extend([
		"",
		"REWRITE INSTRUCTIONS:",
		"- Rewrite the original request into a clearer execution-ready task.",
		"- Preserve the user's intent.",
		"- Do not invent unavailable data or facts.",
		"- If assumptions are needed, make them explicit in the assumptions field.",
		"- If unresolved issues remain, include them in open_questions.",
		"- Do not include assumptions inside rewritten_prompt; use the assumptions field.",
		"- Do not include open questions inside rewritten_prompt; use the open_questions field.",
		"- rewritten_prompt must be a clean task specification only.",
		"- The rewritten prompt should be suitable for downstream planning or direct execution.",
	])

	return "\n".join(lines)


def _default_optimized_prompt_from_state(state: GraphState) -> OptimizedPrompt:
	"""Create a default optimized prompt when no rewrite was needed."""
	raw_user_text = state.get("raw_user_text", "")

	return OptimizedPrompt(
		rewritten_prompt=raw_user_text,
		assumptions=[],
		open_questions=[],
		rationale="Prompt was assessed as executable without rewrite.",
	)

	
def _serialize_rag_docs_for_planner(docs: list[Any]) -> list[dict[str, Any]]:
	"""Convert retrieved RAG docs into serializable planner context.

	Keep full metadata so retrieved chunks can be cited in the plan,
	final answer, and debug output.
	"""
	items: list[dict[str, Any]] = []

	for doc in docs:
		metadata = dict(getattr(doc, "metadata", {}) or {})

		items.append({
			"doc_id": str(getattr(doc, "doc_id", "")),
			"title": str(getattr(doc, "title", "")),
			"text": str(getattr(doc, "text", "")),
			"score": getattr(doc, "score", None),
			"collection": getattr(doc, "collection", None) or metadata.get("collection"),
			"metadata": metadata,
		})

	return items

def _build_planner_prompt(
	execution_prompt: str,
	assessment: PromptAssessment | None = None,
	prepared_assets: list[Any] | None = None,
	rag_context: list[dict[str, Any]] | None = None,
) -> str:
	"""Create prompt for planner agent."""
	prepared_assets = prepared_assets or []
	rag_context = rag_context or []

	lines = [
		"EXECUTION PROMPT:",
		execution_prompt or "<empty>",
	]

	if assessment is not None:
		lines.extend([
			"",
			"PROMPT ASSESSMENT:",
			f"- complexity: {assessment.complexity}",
			f"- requires_planning: {assessment.requires_planning}",
			f"- task_type: {assessment.task_type}",
			f"- suggested_worker: {assessment.suggested_worker}",
			f"- required_workers: {assessment.required_workers}",
			f"- missing_details: {assessment.missing_details}",
			f"- ambiguities: {assessment.ambiguities}",
		])

	
	if prepared_assets:
		lines.extend([
			"",
			"ATTACHMENTS:",
		])
		for idx, asset in enumerate(prepared_assets, start=1):
			lines.append(f"- Asset {idx}:")
			lines.append(f"  kind: {_asset_field(asset, 'kind', 'unknown')}")
			lines.append(f"  path: {_asset_field(asset, 'path', '')}")
			notes = _asset_field(asset, "notes", [])
			if notes:
				lines.append(f"  notes: {notes}")
	else:
		lines.extend([
			"",
			"ATTACHMENTS: none",
		])

	lines.extend([
		"",
		"AVAILABLE WORKERS:",
		"- general: conceptual explanations, scientific reasoning, code guidance, workflow design.",
		"- image: astronomical image/FITS analysis and image-derived morphology tasks.",
		"- catalog: catalog queries, cross-matching, source metadata, survey table analysis.",
		"- literature: literature search, paper summaries, reference discovery.",
	])

	lines.extend([
		"",
		"WORKER ROUTING RULES:",
		"- Use literature worker for tasks asking for papers, references, citations, inline references, bibliography, literature reviews, related work, state-of-the-art discussion, or introduction/background sections with references.",
		"- Use image worker for attached image/FITS inspection, object/source classification, morphology, segmentation, source detection, photometry, or image-derived measurements.",
		"- Use catalog worker for catalog queries, cross-matching, counterpart searches, coordinates, cone searches, source metadata, survey tables, or CAESAR-backed lookup.",
		"- Use general worker for conceptual explanations, workflow design, coding guidance, and synthesis when no specialist evidence source is required.",
		"- If suggested_worker is 'step-dependent', assign workers per step using required_workers as guidance.",
		"- Do not assign all steps to general unless no specialist worker is needed.",
		"- required_workers is a hint about which executable workers should appear in the plan; each PlanStep.worker must still be one of general, image, catalog, or literature.",
	])
	
	if rag_context:
		lines.extend([
			"",
			"OPTIONAL PLANNING CONTEXT FROM RAG:",
			"Use this only to improve the execution plan. Do not treat it as final scientific evidence.",
		])
		for idx, item in enumerate(rag_context, start=1):
			metadata = item.get("metadata", {}) or {}
			source_bits = []

			for key in [
				"collection",
				"doctype",
				"title",
				"paper_title",
				"document_title",
				"authors",
				"first_author",
				"year",
				"journal",
				"doi",
				"arxiv_id",
				"arxiv_abs_url",
				"url",
				"file_name",
				"page_label",
			]:
				value = item.get(key) or metadata.get(key)
				if value:
					source_bits.append(f"{key}: {value}")

			lines.extend([
				f"[{idx}] {item.get('title') or item.get('doc_id') or 'Untitled'}",
				f"doc_id: {item.get('doc_id')}",
				f"collection: {item.get('collection')}",
				f"score: {item.get('score')}",
			])

			if source_bits:
				lines.append("source_metadata: " + "; ".join(str(x) for x in source_bits))

			lines.extend([
				"excerpt:",
				item.get("text", ""),
				"",
			])
			
	else:
		lines.extend([
			"",
			"OPTIONAL PLANNING CONTEXT FROM RAG: none",
		])

	lines.extend([
		"",
		"PLANNING INSTRUCTIONS:",
		"- Produce an ordered execution plan.",
		"- Use the fewest steps needed.",
		"- Assign one worker to each step.",
		"- Do not include final scientific answers in the plan.",
		"- The plan should describe what downstream workers should do.",
		"- If the task requires multiple specialist capabilities, create separate steps for those capabilities rather than collapsing everything into a general step.",
	])

	return "\n".join(lines)


def _has_meaningful_rag_source(item: dict[str, Any]) -> bool:
	metadata = item.get("metadata", {}) or {}

	if item.get("text"):
		return True

	for key in [
		"title",
		"paper_title",
		"document_title",
		"file_name",
		"file_path",
		"doi",
		"arxiv_id",
		"arxiv_abs_url",
		"url",
	]:
		if item.get(key) or metadata.get(key):
			return True

	return False

def _citation_from_rag_item(index: int, item: dict[str, Any]) -> dict[str, Any]:
	"""Build a structured citation record from a serialized RAG item."""
	metadata = item.get("metadata", {}) or {}

	return {
		"index": index,
		"doc_id": item.get("doc_id"),
		"title": (
			item.get("title")
			or metadata.get("title")
			or metadata.get("paper_title")
			or metadata.get("document_title")
		),
		"collection": item.get("collection") or metadata.get("collection"),
		"score": item.get("score"),
		"doctype": metadata.get("kind") or metadata.get("doctype"),
		"file_name": metadata.get("file_name"),
		"file_path": metadata.get("file_path") or metadata.get("filepath"),
		"page_label": metadata.get("page_label"),
		"year": metadata.get("year") or metadata.get("pub_year"),
		"authors": metadata.get("authors") or metadata.get("author"),
		"first_author": metadata.get("first_author"),
		"journal": metadata.get("journal"),
		"volume": metadata.get("volume"),
		"pages": metadata.get("pages"),
		"doi": metadata.get("doi"),
		"url": metadata.get("url") or metadata.get("download_url"),
		"metadata": metadata,
	}


def _collect_citations_from_results(
	results: list[StepResult],
) -> list[dict[str, Any]]:
	"""Collect unique citations from worker evidence."""
	citations: list[dict[str, Any]] = []
	seen: set[str] = set()

	for result in results:
		for item in result.evidence or []:
			if not _has_meaningful_rag_source(item):
				continue

			metadata = item.get("metadata", {}) or {}

			key = (
				metadata.get("doi")
				or metadata.get("arxiv_id")
				or metadata.get("url")
				or metadata.get("filepath")
				or item.get("title")
				or item.get("doc_id")
			)

			if key and str(key) in seen:
				continue

			if key:
				seen.add(str(key))

			citations.append(
				_citation_from_rag_item(
					index=len(citations) + 1,
					item=item,
				)
			)

	return citations

def _extract_agent_text_old(response: Any) -> str:
	"""Extract the last text response from a LangChain/LangGraph agent response."""
	if isinstance(response, dict):
		messages = response.get("messages", [])
		if messages:
			last_message = messages[-1]
			content = getattr(last_message, "content", last_message)
			if isinstance(content, str):
				return content
			return str(content)

		if "output" in response:
			return str(response["output"])

		if "structured_response" in response:
			return str(response["structured_response"])

	return str(response)


def _extract_agent_text(response: Any) -> str:
	"""Extract visible text from a LangChain/LangGraph agent response.

	Some chat backends return structured content blocks such as:
	[
		{"type": "thinking", "thinking": "..."},
		{"type": "text", "text": "..."}
	]

	Only user-visible text blocks should be returned.
	"""
	def _content_to_text(content: Any) -> str:
		if isinstance(content, str):
			return content

		if isinstance(content, list):
			text_chunks = []
			for item in content:
				if isinstance(item, dict):
					if item.get("type") == "text":
						text = item.get("text", "")
						if text:
							text_chunks.append(str(text))
					elif "text" in item and item.get("type") != "thinking":
						text = item.get("text", "")
						if text:
							text_chunks.append(str(text))
				elif isinstance(item, str):
					text_chunks.append(item)

			return "\n".join(text_chunks).strip()

		if isinstance(content, dict):
			if content.get("type") == "text":
				return str(content.get("text", ""))
			if "text" in content and content.get("type") != "thinking":
				return str(content.get("text", ""))

		return str(content)

	if isinstance(response, dict):
		messages = response.get("messages", [])
		if messages:
			last_message = messages[-1]
			content = getattr(last_message, "content", last_message)
			return _content_to_text(content)

		if "output" in response:
			return _content_to_text(response["output"])

		if "structured_response" in response:
			return _content_to_text(response["structured_response"])

	content = getattr(response, "content", response)
	return _content_to_text(content)


def _serialize_rag_docs_for_worker(docs: list[Any]) -> list[dict[str, Any]]:
	"""Convert retrieved RAG docs into serializable worker evidence."""
	items: list[dict[str, Any]] = []

	for doc in docs:
		metadata = dict(getattr(doc, "metadata", {}) or {})

		items.append({
			"doc_id": str(getattr(doc, "doc_id", "")),
			"title": str(getattr(doc, "title", "")),
			"text": str(getattr(doc, "text", "")),
			"score": getattr(doc, "score", None),
			"collection": getattr(doc, "collection", None) or metadata.get("collection"),
			"metadata": metadata,
		})

	return items


def _build_worker_prompt(
	plan: TaskPlan,
	step: PlanStep,
	state: GraphState,
	previous_results: list[StepResult],
	worker_rag_context: list[dict[str, Any]] | None = None,
) -> str:
	"""Build a worker-specific execution prompt for one plan step."""
	execution_prompt = state.get("execution_prompt", "")
	prepared_assets = state.get("prepared_assets", [])
	planner_rag_context = state.get("planner_rag_context", []) or []
	worker_rag_context = worker_rag_context or []

	lines = [
		"TASK OBJECTIVE:",
		plan.objective,
		"",
		"APPROVED EXECUTION PROMPT:",
		execution_prompt or "<empty>",
		"",
		"CURRENT PLAN STEP:",
		f"- id: {step.id}",
		f"- title: {step.title}",
		f"- worker: {step.worker}",
		f"- description: {step.description}",
		f"- inputs: {step.inputs}",
		f"- expected_output: {step.expected_output}",
	]

	if prepared_assets:
		lines.extend([
			"",
			"AVAILABLE ATTACHMENTS:",
		])
		for idx, asset in enumerate(prepared_assets, start=1):
			lines.append(f"- Asset {idx}:")
			lines.append(f"  kind: {_asset_field(asset, 'kind', 'unknown')}")
			lines.append(f"  path: {_asset_field(asset, 'path', '')}")
			notes = _asset_field(asset, "notes", [])
			if notes:
				lines.append(f"  notes: {notes}")
	else:
		lines.extend([
			"",
			"AVAILABLE ATTACHMENTS: none",
		])

	if previous_results:
		lines.extend([
			"",
			"PREVIOUS STEP RESULTS:",
		])
		for result in previous_results:
			lines.extend([
				f"- {result.step_id} [{result.worker}] {result.status}",
				f"  output: {result.output or result.error or '<empty>'}",
			])
	else:
		lines.extend([
			"",
			"PREVIOUS STEP RESULTS: none",
		])

	if planner_rag_context:
		lines.extend([
			"",
			"PLANNER RAG CONTEXT:",
			"This context was used to plan the task. Use it only as supporting context, not as final evidence unless independently relevant.",
		])
		for idx, item in enumerate(planner_rag_context, start=1):
			lines.extend([
				f"[{idx}] {item.get('title') or item.get('doc_id') or 'Untitled'}",
				f"collection: {item.get('collection')}",
				"excerpt:",
				item.get("text", ""),
				"",
			])
	else:
		lines.extend([
			"",
			"PLANNER RAG CONTEXT: none",
		])



	if worker_rag_context:
		lines.extend([
			"",
			"LITERATURE RAG CONTEXT:",
			"Use only these retrieved literature passages as citation evidence. "
			"Bibliography entries must correspond to retrieved evidence items, not to papers merely mentioned inside those passages. "
			"Quote or cite them when relevant, but do not invent claims beyond the provided text.",
		])

		for idx, item in enumerate(worker_rag_context, start=1):
			metadata = item.get("metadata", {}) or {}

			source_bits = []
			for key in [
				"title",
				"paper_title",
				"document_title",
				"authors",
				"first_author",
				"year",
				"journal",
				"doi",
				"arxiv_id",
				"arxiv_abs_url",
				"url",
				"page_label",
			]:
				value = item.get(key) or metadata.get(key)
				if value:
					source_bits.append(f"{key}: {value}")

			lines.extend([
				f"[L{idx}] {item.get('title') or item.get('doc_id') or 'Untitled'}",
				f"doc_id: {item.get('doc_id')}",
				f"collection: {item.get('collection')}",
				f"score: {item.get('score')}",
			])

			if source_bits:
				lines.append("source_metadata: " + "; ".join(str(x) for x in source_bits))

			lines.extend([
				"excerpt:",
				item.get("text", ""),
				"",
			])
	else:
		if step.worker == "literature":
			lines.extend([
				"",
				"LITERATURE RAG CONTEXT: none",
			])

	lines.extend([
		"",
		"EXECUTION INSTRUCTIONS:",
		"- Execute only the current plan step.",
		"- Use available tools if they are needed and available.",
		"- Do not fabricate observations, catalog values, measurements, citations, or tool outputs.",
		"- Do not include a separate References or Bibliography section in the answer body unless explicitly requested; structured citations are attached separately.",
		"- If the step cannot be completed, explain why and what information/tool is missing.",
		"- Return a concise but complete result for this step.",
	])

	return "\n".join(lines)


def _build_aggregation_prompt(
	plan: TaskPlan,
	results: list[StepResult],
	state: GraphState,
	citations: list[dict[str, Any]] | None = None,
) -> str:
	"""Build the prompt consumed by the aggregator agent."""
	
	citations = citations or []
	
	lines = [
		"TASK OBJECTIVE:",
		plan.objective,
		"",
		"APPROVED EXECUTION PROMPT:",
		state.get("execution_prompt", "") or "<empty>",
		"",
		"PLAN RATIONALE:",
		plan.rationale,
		"",
		"STEP RESULTS:",
	]

	for result in results:
		lines.extend([
			f"- step_id: {result.step_id}",
			f"  worker: {result.worker}",
			f"  status: {result.status}",
			f"  output: {result.output or '<empty>'}",
			f"  error: {result.error or '<none>'}",
			f"  evidence_count: {len(result.evidence or [])}",
		])

		if result.evidence:
			lines.append("  evidence:")
			for idx, item in enumerate(result.evidence, start=1):
				metadata = item.get("metadata", {}) or {}
				lines.extend([
					f"    - evidence_id: {idx}",
					f"      title: {item.get('title') or metadata.get('title') or metadata.get('paper_title')}",
					f"      doc_id: {item.get('doc_id')}",
					f"      collection: {item.get('collection')}",
					f"      score: {item.get('score')}",
					f"      doi: {metadata.get('doi')}",
					f"      arxiv_id: {metadata.get('arxiv_id')}",
					f"      url: {metadata.get('url') or metadata.get('arxiv_abs_url')}",
				])


	if citations:
		lines.extend([
			"",
			"DEDUPLICATED CITATIONS:",
			"Use these citation indices for inline references and the bibliography. "
			"Do not create bibliography entries that are not listed here.",
		])

		for citation in citations:
			authors = citation.get("authors") or []
			if isinstance(authors, list):
				authors_text = ", ".join(str(author) for author in authors)
			else:
				authors_text = str(authors)

			lines.extend([
				f"[{citation.get('index')}]",
				f"title: {citation.get('title')}",
				f"authors: {authors_text}",
				f"year: {citation.get('year')}",
				f"journal: {citation.get('journal')}",
				f"volume: {citation.get('volume')}",
				f"pages: {citation.get('pages')}",
				f"doi: {citation.get('doi')}",
				f"url: {citation.get('url')}",
				"",
			])
	else:
		lines.extend([
			"",
			"DEDUPLICATED CITATIONS: none",
		])

	lines.extend([
		"",
		"AGGREGATION INSTRUCTIONS:",
		"- Produce the final user-facing answer.",
		"- Synthesize the successful worker outputs.",
		"- Mention failed or incomplete steps as caveats.",
		"- Do not invent evidence or results that are not present in the step outputs.",
		"- Keep the response clear and scientifically cautious.",
		"- Preserve uncertainty: distinguish hard requirements, recommendations, and illustrative defaults.",
		"- Do not add a separate References or Bibliography section to the answer body unless explicitly requested; FinalAnswer.citations will carry the structured citation list.",
		"- Do not include a References, Bibliography, or Works Cited section in the answer body unless the user explicitly requested one; structured citations are printed separately.",
		"- Avoid repeated citation markers such as [1, 1]; cite each source only once per claim.",
		"- Use cautious wording: say 'recommended' or 'preferred' unless the evidence supports a hard requirement.",
		"- If DEDUPLICATED CITATIONS are provided, use only those citation indices for inline references.",
		"- Do not create bibliography entries for evidence chunks or papers that are not listed in DEDUPLICATED CITATIONS.",
		"- If multiple evidence chunks come from the same paper, cite the single deduplicated citation index.",
	])

	return "\n".join(lines)

def _format_task_plan(plan: TaskPlan) -> str:
	"""Format a task plan for human-readable logging."""
	lines = [
		"",
		"=== TASK PLAN ===",
		f"Objective: {plan.objective}",
		f"Requires RAG context: {plan.requires_rag_context}",
		f"Rationale: {plan.rationale}",
		"",
		"Steps:",
	]

	for step in plan.steps:
		lines.extend([
			f"- {step.id} [{step.worker}] {step.title}",
			f"  description: {step.description}",
			f"  inputs: {step.inputs}",
			f"  expected_output: {step.expected_output}",
		])

	lines.append("=================")

	return "\n".join(lines)


def _correct_suggested_worker(
	text: str,
	prepared_assets: list[Any] | None,
	assessment: PromptAssessment | None,
) -> str:
	"""Correct obvious worker-routing mistakes made by the assessment model."""
	prepared_assets = prepared_assets or []
	text_l = (text or "").lower()

	literature_triggers = [
		"inline reference",
		"inline references",
		"with references",
		"with citations",
		"add references",
		"add citations",
		"references",
		"citations",
		"bibliography",
		"literature review",
		"related work",
		"state of the art",
		"introduction section",
		"background section",
		"papers",
		"doi",
		"arxiv",
	]

	if any(trigger in text_l for trigger in literature_triggers):
		return "literature"

	has_image_asset = any(
		_asset_field(asset, "kind", "unknown") in {"image", "fits"}
		for asset in prepared_assets
	)

	image_triggers = [
		"classify objects",
		"classify sources",
		"detect objects",
		"detect sources",
		"segment",
		"segmentation",
		"morphology",
		"inspect image",
		"inspect the image",
		"analyze image",
		"analyze the image",
		"photometry",
		"source detection",
	]

	if has_image_asset and any(trigger in text_l for trigger in image_triggers):
		return "image"

	catalog_triggers = [
		"catalog",
		"catalogue",
		"cross-match",
		"crossmatch",
		"counterpart",
		"counterparts",
		"cone search",
		"source metadata",
		"ra dec",
		"radec",
		"coordinates",
		"arcsec",
		"arcmin",
		"caesar",
		"simbad",
		"ned",
		"vizier",
	]

	if any(trigger in text_l for trigger in catalog_triggers):
		return "catalog"

	if assessment is not None and assessment.suggested_worker in {
		"general",
		"image",
		"catalog",
		"literature",
	}:
		return assessment.suggested_worker

	if assessment is not None and assessment.suggested_worker == "step-dependent":
		required_workers = getattr(assessment, "required_workers", []) or []

		# If the assessment says step-dependent but planning is not used for some reason,
		# choose the first specialist worker as a safe direct fallback.
		for candidate in ["literature", "image", "catalog", "general"]:
			if candidate in required_workers:
				return candidate

	return "general"


def _default_step_for_worker(worker: str) -> PlanStep:
	"""Create a worker-specific default step for direct single-step execution."""

	if worker == "literature":
		return PlanStep(
			id="step-1",
			title="Retrieve literature evidence and write with citations",
			description=(
				"Use literature RAG to retrieve relevant publications/passages, then answer "
				"the approved prompt using only retrieved evidence for inline references and "
				"bibliography entries."
			),
			worker="literature",
			inputs=["execution_prompt", "literature_rag_context"],
			expected_output=(
				"A literature-grounded response with inline citations and a bibliography "
				"containing only sources supported by retrieved evidence."
			),
		)

	if worker == "image":
		return PlanStep(
			id="step-1",
			title="Analyze attached astronomical image data",
			description=(
				"Use the attached image assets to perform the requested image analysis task. "
				"Report limitations if the required image content, metadata, or calibration "
				"information are missing."
			),
			worker="image",
			inputs=["execution_prompt", "prepared_assets"],
			expected_output=(
				"An image-analysis result grounded in the attached data, with caveats for "
				"missing metadata or unsupported measurements."
			),
		)

	if worker == "catalog":
		return PlanStep(
			id="step-1",
			title="Query catalog or source metadata",
			description=(
				"Use catalog/database tools to perform the requested lookup, counterpart search, "
				"cross-match, coordinate-based query, source metadata retrieval, or survey-table "
				"analysis."
			),
			worker="catalog",
			inputs=["execution_prompt"],
			expected_output=(
				"A catalog-grounded result with retrieved source metadata, query results, "
				"or a clear explanation of missing query parameters/tools."
			),
		)

	return PlanStep(
		id="step-1",
		title="Direct scientific execution",
		description=(
			"Answer the execution prompt directly using scientific reasoning. "
			"Do not fabricate unavailable observations, data products, citations, or tool results."
		),
		worker="general",
		inputs=["execution_prompt"],
		expected_output="A complete response to the approved execution prompt.",
	)

##################################################
###          GRAPH NODES
##################################################
def intake_triage(state: GraphState, ctx: NodeContext) -> dict[str, Any]:
	""" User input triage operations """
	messages = state.get("messages", [])
	raw_user_text = _extract_text(messages)
	logger.info(f"--> intake_triage(): raw_user_text={raw_user_text}")

	# 1. detect files / image paths from state
	# state may include e.g. input_files, image_paths, attachments
	attachments = state.get("attachments", [])

	# 2. preprocess assets
	prepared_assets = []
	for item in attachments:
		prepared_assets.append(
			_prepare_asset(item, ctx)
		)
		
	valid_assets = [a for a in prepared_assets if getattr(a, "is_valid", False)]
	invalid_assets = [a for a in prepared_assets if not getattr(a, "is_valid", False)]
	
	if invalid_assets:
		logger.warning(f"Received {len(invalid_assets)}/{len(prepared_assets)} attachments, return failure!")
		attachment_errors = []
		for asset in invalid_assets:
			attachment_errors.append({
				"path": getattr(asset, "path", None),
				"kind": getattr(asset, "kind", None),
				"error": getattr(asset, "error", "Invalid attachment"),
			})
		
		return {
			"raw_user_text": raw_user_text,
			"multimodal": len(prepared_assets) > 0,
			"prepared_assets": prepared_assets,
			"language_precheck_ok": None,
			"language_model_ok": None,
			"language_ok": None,
			"pii_precheck_detected": None,
			"pii_model_detected": None,
			"pii_detected": None,
			"domain_ok": None,
			"intake_reason": "Invalid attachments received",
			"attachment_errors": attachment_errors,
			"status": "invalid_attachments",
			"route_reason": "One or more attachments could not be opened or converted.",
		}
	
	logger.info("--> intake_triage(): prepared_assets")
	print("prepared_assets")
	print(prepared_assets)

	# 3. cheap deterministic checks first
	#language_ok = True if not ctx.settings.workflow.strict_english_only else is_probably_english(raw_user_text)
	language_precheck_ok = (
		True if not ctx.settings.workflow.strict_english_only
		else is_probably_english(raw_user_text)
	)

	logger.info("--> intake_triage(): language_precheck_ok")
	print("language_precheck_ok")
	print(language_precheck_ok)
	
	pii_reasons = detect_pii(raw_user_text)
	logger.info("--> intake_triage(): detect_pii")
	print("pii_reasons")
	print(pii_reasons)
	
	#pii_detected = bool(pii_reasons)
	pii_precheck_detected = bool(pii_reasons)

	# 4. domain/scope decision from text + prepared image summaries
	message_content = _build_intake_message_content(
		text=raw_user_text,
		prepared_assets=prepared_assets,
	)
	
	logger.info("--> intake_triage(): message_content")
	print("message_content")
	print(message_content)

	# - Invoke agent
	logger.info("--> intake_triage(): Invoking intake_agent on prompt ...")
	#decision = ctx.agents.intake_agent.invoke({
	#	"messages": [HumanMessage(content=message_content)]
	#})["structured_response"]

	timeout_s = ctx.settings.llm.timeout_seconds
	try:
		response = _invoke_with_timeout(
			ctx.agents.intake_agent,
			{"messages": [HumanMessage(content=message_content)]},
			timeout_s=timeout_s,
		)
		decision = response["structured_response"]

	except FuturesTimeoutError:
		logger.exception("intake_agent timed out")
		return {
			"raw_user_text": raw_user_text,
			"multimodal": len(prepared_assets) > 0,
			"prepared_assets": prepared_assets,
			"language_precheck_ok": language_precheck_ok,
			"language_model_ok": None,
			"language_ok": language_precheck_ok,
			"pii_precheck_detected": pii_precheck_detected,
			"pii_model_detected": None,
			"pii_detected": pii_precheck_detected,
			"domain_ok": None,
			"intake_reason": "The intake model did not respond before the timeout.",
			"status": "blocked_intake",
			"route_reason": "The intake model is unavailable or unreachable.",
		}

	except Exception as exc:
		logger.exception("intake_agent failed")
		return {
			"raw_user_text": raw_user_text,
			"multimodal": len(prepared_assets) > 0,
			"prepared_assets": prepared_assets,
			"language_precheck_ok": language_precheck_ok,
			"language_model_ok": None,
			"language_ok": language_precheck_ok,
			"pii_precheck_detected": pii_precheck_detected,
			"pii_model_detected": None,
			"pii_detected": pii_precheck_detected,
			"domain_ok": None,
			"intake_reason": str(exc),
			"status": "blocked_intake",
			"route_reason": f"Intake model invocation failed: {exc}",
		}


	print("decision")
	print(decision)
	
	language_model_ok= getattr(decision, "language_ok", None)
	pii_model_detected= getattr(decision, "pii_detected", None)
	language_ok= language_model_ok if language_model_ok is not None else language_precheck_ok
	pii_detected= pii_model_detected if pii_model_detected is not None else pii_precheck_detected

	accepted = (
		language_ok
		and (not pii_detected or ctx.settings.workflow.pii_policy != "block")
		and decision.accepted
	)
	
	print("accepted")
	print(accepted)
	
	#print("strict_english_only")
	#print(ctx.settings.workflow.strict_english_only)
	#print("language_precheck_ok before return")
	#print(language_precheck_ok)

	if not accepted:
		return {
			"raw_user_text": raw_user_text,
			"multimodal": len(prepared_assets) > 0,
			"prepared_assets": prepared_assets,
			"intake_decision": decision,
			"language_precheck_ok": language_precheck_ok,
			"language_model_ok": language_model_ok,
			"language_ok": language_ok,
			"pii_precheck_detected": pii_precheck_detected,
			"pii_model_detected": pii_model_detected,
			"pii_detected": pii_detected,
			"domain_ok": getattr(decision, "domain_ok", None),
			"intake_reason": getattr(decision, "reason", None),
			"status": "blocked_intake",
			"route_reason": wrap_guardrail_response(decision.reason),
		}
		
	return {
		"raw_user_text": raw_user_text,
		"multimodal": len(prepared_assets) > 0,
		"prepared_assets": prepared_assets,
		"intake_decision": decision,
		"language_precheck_ok": language_precheck_ok,
		"language_model_ok": language_model_ok,
		"language_ok": language_ok,
		"pii_precheck_detected": pii_precheck_detected,
		"pii_model_detected": pii_model_detected,
		"pii_detected": pii_detected,
		"domain_ok": getattr(decision, "domain_ok", None),
		"intake_reason": getattr(decision, "reason", None),
		"approval_iterations": state.get("approval_iterations", 0),
		"max_approval_iterations": state.get("max_approval_iterations", ctx.settings.workflow.max_approval_iterations),
		"status": "running",
	}
	

def assess_prompt(state: GraphState, ctx: NodeContext) -> dict[str, Any]:
	"""Assess whether an accepted request is ready for downstream execution."""
	raw_user_text = state.get("raw_user_text", "")
	prepared_assets = state.get("prepared_assets", [])
	intake_decision = state.get("intake_decision")

	assessment_prompt = _build_assessment_prompt(
		raw_user_text=raw_user_text,
		prepared_assets=prepared_assets,
		intake_decision=intake_decision,
	)

	logger.info("--> assess_prompt(): invoking assessment_agent ...")

	response = ctx.agents.assessment_agent.invoke({
		"messages": [HumanMessage(content=assessment_prompt)]
	})
	assessment: PromptAssessment = response["structured_response"]

	logger.info(
		"---> assess_prompt(): "
		f"needs_rewrite={assessment.needs_rewrite}, "
		f"rewrite_would_help={assessment.rewrite_would_help}, "
		f"executable_as_is={assessment.executable_as_is}, "
		f"complexity={assessment.complexity}, "
		f"requires_planning={assessment.requires_planning}, "
		f"task_type={assessment.task_type}, "
		f"suggested_worker={assessment.suggested_worker}, "
		f"required_workers={assessment.required_workers}"
	)

	return {
		"prompt_assessment": assessment,
		"status": "needs_rewrite" if assessment.needs_rewrite else "running",
	}


def rewrite_prompt(state: GraphState, ctx: NodeContext) -> dict[str, Any]:
	"""Rewrite an underspecified but accepted request into a clearer execution-ready prompt."""
	raw_user_text = state.get("raw_user_text", "")
	prepared_assets = state.get("prepared_assets", [])
	assessment: PromptAssessment | None = state.get("prompt_assessment")

	if assessment is None:
		return {
			"status": "blocked_intake",
			"route_reason": "Cannot rewrite prompt because prompt assessment is missing.",
			"intake_reason": "Missing prompt assessment before rewrite.",
		}

	rewrite_request = _build_rewrite_prompt(
		raw_user_text=raw_user_text,
		assessment=assessment,
		prepared_assets=prepared_assets,
	)

	logger.info("--> rewrite_prompt(): invoking optimizer_agent ...")

	response = ctx.agents.optimizer_agent.invoke({
		"messages": [HumanMessage(content=rewrite_request)]
	})
	optimized: OptimizedPrompt = response["structured_response"]

	logger.info(
		"---> rewrite_prompt(): "
		f"rewritten_prompt={optimized.rewritten_prompt!r}, "
		f"assumptions={optimized.assumptions}, "
		f"open_questions={optimized.open_questions}"
	)

	return {
		"optimized_prompt": optimized,
		"status": "running",
	}


def approval_node(state: GraphState, ctx: NodeContext) -> dict[str, Any]:
	"""Ask the user to approve, revise, or reject the prompt before execution."""
	_ = ctx

	candidate = state.get("optimized_prompt")
	if candidate is None:
		candidate = _default_optimized_prompt_from_state(state)

	assessment = state.get("prompt_assessment")

	decision_payload = interrupt({
		"type": "prompt_approval",
		"candidate_prompt": candidate.rewritten_prompt,
		"assumptions": candidate.assumptions,
		"open_questions": candidate.open_questions,
		"rationale": candidate.rationale,
		"assessment": assessment.model_dump() if assessment else None,
		"iteration": state.get("approval_iterations", 0),
		"max_iterations": state.get("max_approval_iterations", 3),
		"instructions": (
			"Approve the rewritten prompt to continue, request revision with feedback, "
			"or reject to stop the workflow."
		),
	})

	decision = ApprovalDecision.model_validate(decision_payload)

	update: dict[str, Any] = {
		"approval_decision": decision,
		"approval_iterations": state.get("approval_iterations", 0) + 1,
	}

	if decision.decision == "approve":
		update["approved_prompt"] = candidate.rewritten_prompt
		update["status"] = "approved"
	elif decision.decision == "revise":
		update["status"] = "awaiting_approval"
	else:
		update["status"] = "rejected_by_user"

	return update

def refine_from_feedback(state: GraphState, ctx: NodeContext) -> dict[str, Any]:
	"""Revise the optimized prompt using user feedback."""
	decision = state.get("approval_decision")
	candidate = state.get("optimized_prompt")
	raw_user_text = state.get("raw_user_text", "")
	assessment = state.get("prompt_assessment")

	base_prompt = candidate.rewritten_prompt if candidate else raw_user_text
	feedback_text = (
		decision.feedback
		if decision and decision.feedback
		else "Please improve clarity, specificity, and execution-readiness."
	)

	lines = [
		"ORIGINAL USER REQUEST:",
		raw_user_text or "<empty>",
		"",
		"CURRENT REWRITTEN PROMPT:",
		base_prompt or "<empty>",
		"",
		"USER FEEDBACK:",
		feedback_text,
	]

	if assessment is not None:
		lines.extend([
			"",
			"ASSESSMENT CONTEXT:",
			f"- task_type: {assessment.task_type}",
			f"- suggested_worker: {assessment.suggested_worker}",
			f"- required_workers: {assessment.required_workers}",
			f"- missing_details: {assessment.missing_details}",
			f"- ambiguities: {assessment.ambiguities}",
			f"- rewrite_goal: {assessment.rewrite_goal}",
		])

	lines.extend([
		"",
		"REVISION INSTRUCTIONS:",
		"- Revise the prompt according to the user feedback.",
		"- Preserve the original scientific intent.",
		"- Do not invent unavailable data or facts.",
		"- Keep assumptions only in the assumptions field.",
		"- Keep unresolved questions only in the open_questions field.",
	])

	logger.info("--> refine_from_feedback(): invoking optimizer_agent ...")

	response = ctx.agents.optimizer_agent.invoke({
		"messages": [HumanMessage(content="\n".join(lines))]
	})
	optimized: OptimizedPrompt = response["structured_response"]

	optimized = optimized.model_copy(
		update={
			"rewritten_prompt": _strip_rewrite_metadata_sections(
				optimized.rewritten_prompt
			)
		}
	)

	return {
		"optimized_prompt": optimized,
		"status": "running",
	}

def give_up(state: GraphState, ctx: NodeContext) -> dict[str, Any]:
	"""Stop when approval/rewrite loop cannot converge."""
	_ = ctx

	return {
		"status": "rejected_after_iterations",
		"route_reason": wrap_guardrail_response(
			"The system could not converge on an approved prompt within the allowed number of iterations."
		),
	}
	

def prepare_prompt(state: GraphState, ctx: NodeContext) -> dict[str, Any]:
	"""Prepare the final prompt to be consumed by downstream execution nodes."""
	_ = ctx

	approved_prompt = state.get("approved_prompt")
	optimized_prompt = state.get("optimized_prompt")
	raw_user_text = state.get("raw_user_text", "")

	if approved_prompt:
		execution_prompt = approved_prompt
	elif optimized_prompt:
		execution_prompt = optimized_prompt.rewritten_prompt
	else:
		execution_prompt = raw_user_text

	return {
		"execution_prompt": execution_prompt,
		"status": "prepared",
	}

def planner_or_default(state: GraphState, ctx: NodeContext) -> dict[str, Any]:
	"""Create a task plan, either directly or using the planner agent."""
	execution_prompt = state.get("execution_prompt") or state.get("approved_prompt") or state.get("raw_user_text", "")
	assessment: PromptAssessment | None = state.get("prompt_assessment")
	prepared_assets = state.get("prepared_assets", [])
	primary_worker = _correct_suggested_worker(
		text=execution_prompt,
		prepared_assets=prepared_assets,
		assessment=assessment,
	)

	requires_planner = False
	if assessment is not None:
		requires_planner = (
			assessment.requires_planning
			or assessment.complexity == "complex"
		)

	planner_rag_enabled = bool(state.get("planner_rag_enabled", False))
	planner_rag_k = int(state.get("planner_rag_k", 5) or 5)
	rag_context: list[dict[str, str]] = []
	required_workers = []
	if assessment is not None:
		required_workers = getattr(assessment, "required_workers", []) or []

	skip_planner_rag = "literature" in required_workers

	if requires_planner:
		if planner_rag_enabled and ctx.rag is not None and not skip_planner_rag:
		
			logger.info(
				f"--> planner_or_default(): retrieving planner RAG context, k={planner_rag_k} ..."
			)
			try:
				domain_hint = None
				if assessment is not None:
					domain_hint = getattr(assessment, "task_type", None)
				docs = ctx.rag.retrieve(
					query=execution_prompt,
					k=planner_rag_k,
					domain_hint=domain_hint,
				)
				rag_context = _serialize_rag_docs_for_planner(docs)
			except Exception as exc:
				logger.warning(
					f"planner_or_default(): RAG retrieval failed, continuing without RAG: {exc}"
				)
				rag_context = []

		planner_prompt = _build_planner_prompt(
			execution_prompt=execution_prompt,
			assessment=assessment,
			prepared_assets=prepared_assets,
			rag_context=rag_context,
		)

		logger.info("--> planner_or_default(): invoking planner_agent ...")

		response = ctx.agents.planner_agent.invoke({
			"messages": [HumanMessage(content=planner_prompt)]
		})
		plan: TaskPlan = response["structured_response"]

	else:
		#worker = "general"
		worker = primary_worker
		
		#if assessment is not None and assessment.suggested_worker:
		#	if assessment.suggested_worker in {"general", "image", "catalog", "literature"}:
		#		worker = assessment.suggested_worker
		
		step = _default_step_for_worker(worker)

		plan = TaskPlan(
			objective=execution_prompt,
			requires_rag_context=(worker == "literature"),
			rationale=(
				"Prompt assessment indicates this can be handled as a direct "
				f"single-step task. The selected worker is '{worker}'."
			),
			steps=[step],
		)
		
		#plan = TaskPlan(
		#	objective=execution_prompt,
		#	requires_rag_context=False,
		#	rationale=(
		#		"Prompt assessment indicates this can be handled as a direct "
		#		"single-step task without explicit multi-step planning."
		#	),
		#	steps=[
		#		PlanStep(
		#			id="step-1",
		#			title="Direct execution",
		#			description="Answer the execution prompt directly.",
		#			worker=worker,
		#			inputs=["execution_prompt"],
		#			expected_output="A complete response to the approved execution prompt.",
		#		)
		#	],
		#)

	plan = plan.model_copy(
		update={
			"requires_rag_context": bool(rag_context) or any(
				step.worker == "literature"
				for step in plan.steps
			)
		}
	)

	logger.info(
		f"--> planner_or_default(): created plan with {len(plan.steps)} step(s)"
	)

	logger.info(_format_task_plan(plan))

	return {
		"task_plan": plan,
		"planner_rag_context": rag_context,
		"status": "planned",
	}


def execute_plan(state: GraphState, ctx: NodeContext) -> dict[str, Any]:
	"""Execute each step in the task plan using the assigned worker agent."""
	plan: TaskPlan | None = state.get("task_plan")

	if plan is None:
		return {
			"execution_results": [],
			"status": "execution_failed",
			"route_reason": "Cannot execute plan because task_plan is missing.",
		}

	worker_map = {
		"catalog": ctx.agents.catalog_agent,
		"image": ctx.agents.image_agent,
		"literature": ctx.agents.literature_agent,
		"general": ctx.agents.general_agent,
	}

	results: list[StepResult] = []

	for step in plan.steps:
		agent = worker_map.get(step.worker)

		if agent is None:
			results.append(
				StepResult(
					step_id=step.id,
					worker=step.worker,
					status="failed",
					output="",
					evidence=[],
					error=f"No worker agent is available for worker={step.worker!r}.",
				)
			)
			continue

		# - Retrieve info from RAG
		worker_rag_context: list[dict[str, Any]] = []
		
		if step.worker == "literature" and ctx.rag is not None:
			literature_rag_k = int(state.get("literature_rag_k", 8) or 8)

			query = "\n".join([
				plan.objective,
				step.title,
				step.description,
				state.get("execution_prompt", ""),
			]).strip()

			logger.info(
				f"--> execute_plan(): retrieving literature RAG context for step {step.id}, k={literature_rag_k} ..."
			)

			try:
				docs = ctx.rag.retrieve(
					query=query,
					k=literature_rag_k,
					domain_hint="literature",
				)
				worker_rag_context = _serialize_rag_docs_for_worker(docs)

			except Exception as exc:
				logger.warning(
					f"execute_plan(): literature RAG retrieval failed for step {step.id}: {exc}"
				)
				worker_rag_context = []

		#elif step.worker == "general" and state.get("planner_rag_context"):
		#	worker_rag_context = state.get("planner_rag_context", [])

		elif step.worker == "general" and state.get("planner_rag_context"):
			has_prior_literature_result = any(
				item.worker == "literature" and item.status == "ok"
				for item in results
			)
			if not has_prior_literature_result:
				worker_rag_context = state.get("planner_rag_context", [])

		# - Build worker prompt
		worker_prompt = _build_worker_prompt(
			plan=plan,
			step=step,
			state=state,
			previous_results=results,
			worker_rag_context=worker_rag_context,
		)

		logger.info(
			f"--> execute_plan(): invoking {step.worker}_agent for step {step.id} ..."
		)

		try:
			response = agent.invoke({
				"messages": [HumanMessage(content=worker_prompt)]
			})

			output = _extract_agent_text(response)
			
			logger.info(
				f"--> execute_plan(): completed {step.id} [{step.worker}], "
				f"output_len={len(output or '')}, evidence_count={len(worker_rag_context)}"
			)

			results.append(
				StepResult(
					step_id=step.id,
					worker=step.worker,
					status="ok",
					output=output,
					evidence=worker_rag_context,
					error=None,
				)
			)

		except Exception as exc:
			logger.exception(
				f"execute_plan(): step {step.id} failed for worker={step.worker}"
			)
			results.append(
				StepResult(
					step_id=step.id,
					worker=step.worker,
					status="failed",
					output="",
					evidence=worker_rag_context,
					error=str(exc),
				)
			)

	return {
		"execution_results": results,
		"status": "executed",
	}


def aggregate(state: GraphState, ctx: NodeContext) -> dict[str, Any]:
	"""Aggregate worker results into the final MAASAI answer."""
	plan: TaskPlan | None = state.get("task_plan")
	results: list[StepResult] = state.get("execution_results", [])

	logger.info(f"--> aggregate(): received {len(results)} execution result(s)")
	for item in results:
		logger.info(
			f"--> aggregate(): result {item.step_id} [{item.worker}] "
			f"status={item.status}, output_len={len(item.output or '')}"
		)

	if plan is None:
		final = FinalAnswer(
			status="error",
			message="Cannot aggregate results because task_plan is missing.",
			answer=None,
			citations=[],
			artifacts=[],
			debug={
				"execution_results": [
					item.model_dump() if hasattr(item, "model_dump") else item
					for item in results
				],
			},
		)
		return {"final_answer": final, "status": "done"}

	caveats = [
		f"Step {item.step_id} [{item.worker}] failed: {item.error}"
		for item in results
		if item.status != "ok"
	]

	# - Fill citations
	citations = _collect_citations_from_results(results)
	
	# - Create aggregation prompt
	aggregation_prompt = _build_aggregation_prompt(
		plan=plan,
		results=results,
		state=state,
		citations=citations,
	)

	logger.info("--> aggregate(): invoking aggregator_agent ...")

	try:
		response = ctx.agents.aggregator_agent.invoke({
			"messages": [HumanMessage(content=aggregation_prompt)]
		})
		final_text = _extract_agent_text(response)
		final_text = _strip_references_section(final_text)
		final_text = _normalize_inline_citations(final_text)

	except Exception as exc:
		logger.exception("aggregate(): aggregator_agent failed")
		final_text = (
			"The task was executed, but the aggregation agent failed. "
			"Raw step results are available in debug output."
		)
		caveats.append(f"Aggregation failed: {exc}")

	
	# - Create final answer
	final = FinalAnswer(
		status="ok" if not caveats else "error",
		message=(
			"Execution completed."
			if not caveats
			else "Execution completed with one or more caveats."
		),
		answer=final_text,
		citations=citations,
		artifacts=[],
		debug={
			"task_plan": plan.model_dump(),
			"execution_results": [
				item.model_dump() if hasattr(item, "model_dump") else item
				for item in results
			],
			"caveats": caveats,
			"planner_rag_context": state.get("planner_rag_context", []),
		},
	)

	return {
		"final_answer": final,
		"status": "done",
	}


def final_guardrail(state: GraphState, ctx: NodeContext) -> dict[str, Any]:
	_ = ctx

	status = state.get("status")

	# - FINAL ANSWER
	if status == "done" and state.get("final_answer") is not None:
		return {
			"final_answer": state.get("final_answer"),
			"status": "done",
		}

	# - ERROR: Invalid attachment
	if status == "invalid_attachments":
		final = FinalAnswer(
			status="error",
			message="One or more attachments could not be opened or processed.",
			answer=None,
			citations=[],
			artifacts=[],
			debug={
				"route_reason": state.get("route_reason"),
				"intake_reason": state.get("intake_reason"),
				"attachment_errors": state.get("attachment_errors", []),
			},
		)
		return {"final_answer": final, "status": "done"}

	# - ERROR: Input user request cannot be validated by intake agent
	if status == "blocked_intake" and not state.get("intake_decision"):
		final = FinalAnswer(
			status="error",
			message="The intake model is currently unavailable.",
			answer=None,
			citations=[],
			artifacts=[],
			debug={
				"route_reason": state.get("route_reason"),
				"intake_reason": state.get("intake_reason"),
			},
		)
		return {"final_answer": final, "status": "done"}

	# - Input user request blocked by intake agent
	if status in {"blocked_language", "blocked_pii", "blocked_domain", "blocked_intake", "rejected_after_iterations"}:
		language_precheck_ok = state.get("language_precheck_ok", True)
		language_model_ok = state.get("language_model_ok", None)
		language_ok= state.get("language_ok", None)
		
		pii_precheck_detected = state.get("pii_precheck_detected", False)
		pii_model_detected = state.get("pii_model_detected", None)
		pii_detected= state.get("pii_detected", None)
		
		domain_ok = state.get("domain_ok", True)
		intake_reason = state.get("intake_reason")
		
		reasons = []
		if not language_ok:
			reasons.append("it is not in English")
		if pii_detected:
			reasons.append("it appears to contain personally identifiable information")
		if not domain_ok:
			reasons.append("it is outside the supported astronomy and astrophysics domain")

		if reasons:
			if len(reasons) == 1:
				user_message = f"The request was rejected because {reasons[0]}."
			elif len(reasons) == 2:
				user_message = f"The request was rejected because {reasons[0]} and {reasons[1]}."
			else:
				user_message = (
					f"The request was rejected because {reasons[0]}, {reasons[1]}, "
					f"and {reasons[2]}."
				)
		else:
			user_message = state.get("route_reason", "The request was rejected.")

		final = FinalAnswer(
			status="rejected",
			message=user_message,
			answer=None,
			citations=[],
			artifacts=[],
			debug={
				"route_reason": state.get("route_reason"),
				"language_precheck_ok": language_precheck_ok,
				"language_model_ok": language_model_ok,
				"language_ok": language_ok,
				"pii_precheck_detected": pii_precheck_detected,
				"pii_model_detected": pii_model_detected,
				"pii_detected": pii_detected,
				"domain_ok": domain_ok,
				"intake_reason": intake_reason,
			},
		)
		
		print("--> guardrail final")
		print(final)
		return {"final_answer": final, "status": "done"}

	# - Request marked as to be re-written by prompt assessment agent
	if status == "needs_rewrite":
		assessment = state.get("prompt_assessment")
		final = FinalAnswer(
			status="ok",
			message="Request passed intake triage but requires rewrite before execution.",
			answer=(
				"The request is in scope, but it is not yet ready for downstream execution.\n\n"
				f"Rewrite goal: {getattr(assessment, 'rewrite_goal', None)}\n"
				f"Missing details: {getattr(assessment, 'missing_details', None)}\n"
				f"Ambiguities: {getattr(assessment, 'ambiguities', None)}\n"
				f"Reasoning summary: {getattr(assessment, 'reasoning_summary', None)}"
			),
			citations=[],
			artifacts=[],
			debug={
				"prompt_assessment": assessment.model_dump() if assessment else None,
			},
		)
		return {"final_answer": final, "status": "done"}


	# - Prepared execution prompt
	if status == "prepared":
		final = FinalAnswer(
			status="ok",
			message="Prompt approved and prepared for downstream execution.",
			answer=(
				"The prompt has been approved and prepared for downstream execution.\n\n"
				f"Execution prompt:\n{state.get('execution_prompt')}"
			),
			citations=[],
			artifacts=[],
			debug={
				"execution_prompt": state.get("execution_prompt"),
				"approved_prompt": state.get("approved_prompt"),
				"approval_decision": (
					state.get("approval_decision").model_dump()
					if state.get("approval_decision") else None
				),
				"optimized_prompt": (
					state.get("optimized_prompt").model_dump()
					if state.get("optimized_prompt") else None
				),
				"prompt_assessment": (
					state.get("prompt_assessment").model_dump()
					if state.get("prompt_assessment") else None
				),
			},
		)
		return {"final_answer": final, "status": "done"}


	# - Request re-written
	if status == "running" and state.get("optimized_prompt") is not None:
		optimized = state.get("optimized_prompt")

		final = FinalAnswer(
			status="ok",
			message="Request passed intake triage, was assessed, and has been rewritten for execution.",
			answer=(
				"The request was accepted, assessed as needing rewrite, and rewritten into a more execution-ready form.\n\n"
				f"Rewritten prompt:\n{getattr(optimized, 'rewritten_prompt', None)}\n\n"
				f"Assumptions: {getattr(optimized, 'assumptions', None)}\n"
				f"Open questions: {getattr(optimized, 'open_questions', None)}\n"
				f"Rationale: {getattr(optimized, 'rationale', None)}"
			),
			citations=[],
			artifacts=[],
			debug={
				"optimized_prompt": optimized.model_dump() if optimized else None,
				"prompt_assessment": (
					state.get("prompt_assessment").model_dump()
					if state.get("prompt_assessment") else None
				),
			},
		)
		return {"final_answer": final, "status": "done"}


	# - Approved prompt re-write
	if status == "approved":
		final = FinalAnswer(
			status="ok",
			message="Prompt approved for downstream execution.",
			answer=(
				"The prompt has been approved and is ready for downstream execution.\n\n"
				f"Approved prompt:\n{state.get('approved_prompt')}"
			),
			citations=[],
			artifacts=[],
			debug={
				"approved_prompt": state.get("approved_prompt"),
				"approval_decision": (
					state.get("approval_decision").model_dump()
					if state.get("approval_decision") else None
				),
				"optimized_prompt": (
					state.get("optimized_prompt").model_dump()
					if state.get("optimized_prompt") else None
				),
				"prompt_assessment": (
					state.get("prompt_assessment").model_dump()
					if state.get("prompt_assessment") else None
				),
			},
		)
		return {"final_answer": final, "status": "done"}

	# - Rejected prompt re-write
	if status == "rejected_by_user":
		final = FinalAnswer(
			status="rejected",
			message="Prompt approval was rejected by the user.",
			answer=None,
			citations=[],
			artifacts=[],
			debug={
				"approval_decision": (
					state.get("approval_decision").model_dump()
					if state.get("approval_decision") else None
				),
			},
		)
		return {"final_answer": final, "status": "done"}
		
	# - Task planned message
	if status == "planned":
		plan = state.get("task_plan")
		
		rag_citations = []
		for idx, item in enumerate(state.get("planner_rag_context", []) or [], start=1):
			if not _has_meaningful_rag_source(item):
				continue

			metadata = item.get("metadata", {}) or {}

			rag_citations.append({
				"index": idx,
				"doc_id": item.get("doc_id"),
				"title": item.get("title"),
				"collection": item.get("collection") or metadata.get("collection"),
				"score": item.get("score"),
				"doctype": metadata.get("kind") or metadata.get("doctype"),
				"file_name": metadata.get("file_name"),
				"file_path": (
					metadata.get("file_path") or metadata.get("filepath")
				),
				"page_label": metadata.get("page_label"),
				"year": metadata.get("year") or metadata.get("pub_year"),
				"authors": metadata.get("authors") or metadata.get("author"),
				"doi": metadata.get("doi"),
				"arxiv_id": metadata.get("arxiv_id"),
				"arxiv_abs_url": metadata.get("arxiv_abs_url"),
				"url": metadata.get("url") or metadata.get("download_url"),
				"metadata": metadata,
			})
		
		final = FinalAnswer(
			status="ok",
			message="Execution prompt has been planned.",
			answer=(
				"The approved execution prompt has been converted into a task plan.\n\n"
				f"Objective:\n{getattr(plan, 'objective', None)}\n\n"
				f"Rationale:\n{getattr(plan, 'rationale', None)}\n\n"
				"Steps:\n"
				+ "\n".join(
					f"- {step.id} [{step.worker}] {step.title}: {step.description}"
					for step in getattr(plan, "steps", [])
				)
			),
			citations=rag_citations,
			artifacts=[],
			debug={
				"task_plan": plan.model_dump() if plan else None,
				"planner_rag_enabled": state.get("planner_rag_enabled"),
				"planner_rag_k": state.get("planner_rag_k"),
				"planner_rag_context": state.get("planner_rag_context", []),
				"execution_prompt": state.get("execution_prompt"),
			},
		)
		return {"final_answer": final, "status": "done"}
		
	#######################################
	##      SUCCESS
	#######################################
	#final = FinalAnswer(
	#	status="ok",
	#	message="Request passed intake triage.",
	#	answer="The request passed intake triage, but no downstream execution node is enabled in this test graph yet.",
	#	citations=[],
	#	artifacts=[],
	#	debug={},
	#)
	#return {"final_answer": final, "status": "done"}
	

	assessment = state.get("prompt_assessment")

	final = FinalAnswer(
		status="ok",
		message="Request passed intake triage and prompt assessment.",
		answer=(
			"The request passed intake triage and prompt assessment.\n\n"
			f"Assessment summary:\n"
			f"- needs_rewrite: {getattr(assessment, 'needs_rewrite', None)}\n"
			f"- rewrite_would_help: {getattr(assessment, 'rewrite_would_help', None)}\n"
			f"- executable_as_is: {getattr(assessment, 'executable_as_is', None)}\n"
			f"- complexity: {getattr(assessment, 'complexity', None)}\n"
			f"- requires_planning: {getattr(assessment, 'requires_planning', None)}\n"
			f"- task_type: {getattr(assessment, 'task_type', None)}\n"
			f"- suggested_worker: {getattr(assessment, 'suggested_worker', None)}\n"
			f"- required_workers: {getattr(assessment, 'required_workers', None)}\n"
			f"- missing_details: {getattr(assessment, 'missing_details', None)}\n"
			f"- ambiguities: {getattr(assessment, 'ambiguities', None)}\n"
			f"- rewrite_goal: {getattr(assessment, 'rewrite_goal', None)}\n"
			f"- reasoning_summary: {getattr(assessment, 'reasoning_summary', None)}"
		),
		citations=[],
		artifacts=[],
		debug={
			"prompt_assessment": assessment.model_dump() if assessment else None,
		},
	)
	return {"final_answer": final, "status": "done"}
	
	
