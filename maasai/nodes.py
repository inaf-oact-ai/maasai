from __future__ import print_function
from __future__ import annotations

##################################################
###          MODULE IMPORT
##################################################
# - STANDARD MODULES
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
#from .schemas import StepResult
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
		f"suggested_worker={assessment.suggested_worker}"
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

	requires_planner = False
	if assessment is not None:
		requires_planner = (
			assessment.requires_planning
			or assessment.complexity == "complex"
		)

	planner_rag_enabled = bool(state.get("planner_rag_enabled", False))
	planner_rag_k = int(state.get("planner_rag_k", 5) or 5)

	rag_context: list[dict[str, str]] = []

	if requires_planner:
		if planner_rag_enabled and ctx.rag is not None:
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
		worker = "general"
		if assessment is not None and assessment.suggested_worker:
			if assessment.suggested_worker in {"general", "image", "catalog", "literature"}:
				worker = assessment.suggested_worker

		plan = TaskPlan(
			objective=execution_prompt,
			requires_rag_context=False,
			rationale=(
				"Prompt assessment indicates this can be handled as a direct "
				"single-step task without explicit multi-step planning."
			),
			steps=[
				PlanStep(
					id="step-1",
					title="Direct execution",
					description="Answer the execution prompt directly.",
					worker=worker,
					inputs=["execution_prompt"],
					expected_output="A complete response to the approved execution prompt.",
				)
			],
		)

	logger.info(
		f"--> planner_or_default(): created plan with {len(plan.steps)} step(s)"
	)

	return {
		"task_plan": plan,
		"planner_rag_context": rag_context,
		"status": "planned",
	}


#def execute_plan(state: GraphState, ctx: NodeContext) -> dict[str, Any]:
#	results: list[StepResult] = []
#	plan = state["task_plan"]
#	worker_map = {
#		"catalog": ctx.agents.catalog_agent,
#		"image": ctx.agents.image_agent,
#		"literature": ctx.agents.literature_agent,
#		"general": ctx.agents.general_agent,
#	}
#	for step in plan.steps:
#		agent = worker_map[step.worker]
#		try:
#			response = agent.invoke({
#				"messages": [HumanMessage(content=f"Task objective: {plan.objective}\n\nStep: {step.model_dump_json(indent=2)}")]
#			})
#			last_message = response["messages"][-1]
#			results.append(
#				StepResult(
#					step_id=step.id,
#					worker=step.worker,
#					status="ok",
#					output=getattr(last_message, "content", str(last_message)),
#					evidence=[],
#				)
#			)
#		except Exception as exc:  # noqa: BLE001
#			results.append(
#				StepResult(
#					step_id=step.id,
#					worker=step.worker,
#					status="failed",
#					output="",
#					evidence=[],
#					error=str(exc),
#				)
#			)
#	return {"execution_results": results}


#def aggregate(state: GraphState, ctx: NodeContext) -> dict[str, Any]:
#	plan = state["task_plan"]
#	results = state.get("execution_results", [])
#	summary_lines = [f"Objective: {plan.objective}", "", "Step results:"]
#	caveats: list[str] = []
#	for item in results:
#		summary_lines.append(f"- {item.step_id} [{item.worker}] {item.status}: {item.output or item.error}")
#		if item.status != "ok":
#			caveats.append(f"Step {item.step_id} did not complete successfully.")
#	request = "\n".join(summary_lines)
#	response = ctx.agents.aggregator_agent.invoke({
#		"messages": [HumanMessage(content=request)]
#	})
#	final_text = getattr(response["messages"][-1], "content", str(response["messages"][-1]))
#	final = FinalAnswer(answer=final_text, citations=[], caveats=caveats, confidence="medium")
#	return {"final_answer": final}


def final_guardrail(state: GraphState, ctx: NodeContext) -> dict[str, Any]:
	_ = ctx

	status = state.get("status")

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
	
	
