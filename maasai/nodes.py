from __future__ import print_function
from __future__ import annotations

##################################################
###          MODULE IMPORT
##################################################
# - STANDARD MODULES
from typing import Any

# - LLM/LANGCHAIN MODULES
from langchain.messages import HumanMessage
from langgraph.types import interrupt

# - MAASAI MODULES
from .agents import AgentFactory
from .config import Settings
from .guardrails import detect_pii, is_probably_english, is_scientific_or_astronomy_related, wrap_guardrail_response
from .rag import PromptRAG
#from .schemas import ApprovalDecision, DomainDecision, FinalAnswer, OptimizedPrompt, PlanStep, PromptAssessment, StepResult, TaskPlan
from .state import GraphState

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
			path = getattr(asset, "path", None) or asset.get("path", "")
			kind = getattr(asset, "kind", None) or asset.get("kind", "unknown")
			notes = getattr(asset, "notes", None) or asset.get("notes", [])

			lines.append(f"- Asset {idx}:")
			lines.append(f"  kind: {kind}")
			lines.append(f"  path: {path}")
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

##################################################
###          GRAPH NODES
##################################################
class NodeContext:
	def __init__(
		self, *, 
		settings: Settings, 
		rag: PromptRAG, 
		agents: AgentFactory
	) -> None:
		""" Node context """
		self.settings = settings
		self.rag = rag
		self.agents = agents



def intake_triage(state: GraphState, ctx: NodeContext) -> dict[str, Any]:
	""" User input triage operations """
	messages = state.get("messages", [])
	raw_user_text = _extract_text(messages)

	# 1. detect files / image paths from state
	# state may include e.g. input_files, image_paths, attachments
	attachments = state.get("attachments", [])

	# 2. preprocess assets
	prepared_assets = []
	for item in attachments:
		prepared_assets.append(
			_prepare_asset(item, ctx)
		)

	# 3. cheap deterministic checks first
	language_ok = True if not ctx.settings.workflow.strict_english_only else is_probably_english(raw_user_text)
	pii_reasons = detect_pii(raw_user_text)
	pii_detected = bool(pii_reasons)

	# 4. domain/scope decision from text + prepared image summaries
	intake_prompt = _build_intake_prompt(
		text=raw_user_text,
		prepared_assets=prepared_assets
	)

	decision = ctx.agents.intake_agent.invoke({
		"messages": [HumanMessage(content=intake_prompt)]
	})["structured_response"]

	accepted = (
		language_ok
		and (not pii_detected or ctx.settings.workflow.pii_policy != "block")
		and decision.accepted
	)

	if not accepted:
		return {
			"raw_user_text": raw_user_text,
			"multimodal": len(prepared_assets) > 0,
			"prepared_assets": prepared_assets,
			"intake_decision": decision,
			"status": "blocked_intake",
			"route_reason": wrap_guardrail_response(decision.reason),
		}

	return {
		"raw_user_text": raw_user_text,
		"multimodal": len(prepared_assets) > 0,
		"prepared_assets": prepared_assets,
		"intake_decision": decision,
		"approval_iterations": state.get("approval_iterations", 0),
		"max_approval_iterations": state.get("max_approval_iterations", ctx.settings.workflow.max_approval_iterations),
		"status": "running",
	}


def assess_prompt(state: GraphState, ctx: NodeContext) -> dict[str, Any]:
	assessment = ctx.agents.assessment_agent.invoke({
		"messages": [HumanMessage(content=state["raw_user_text"])]
	})["structured_response"]
	return {"prompt_assessment": assessment}



#def normalize_input(state: GraphState, ctx: NodeContext) -> dict[str, Any]:
#	_ = ctx
#	messages = state.get("messages", [])
#	raw_user_text = _extract_text(messages)
#	multimodal = any(isinstance(getattr(m, "content", None), list) for m in messages)
#	return {
#		"raw_user_text": raw_user_text,
#		"multimodal": multimodal,
#		"approval_iterations": state.get("approval_iterations", 0),
#		"max_approval_iterations": state.get("max_approval_iterations", ctx.settings.workflow.max_approval_iterations),
#		"status": "running",
#	}


#def language_gate(state: GraphState, ctx: NodeContext) -> dict[str, Any]:
#	text = state["raw_user_text"]
#	ok = True if not ctx.settings.workflow.strict_english_only else is_probably_english(text)
#	update: dict[str, Any] = {"language_ok": ok}
#	if not ok:
#		update["status"] = "blocked_language"
#		update["route_reason"] = wrap_guardrail_response("The system currently accepts only English requests.")
#	return update


#def pii_gate(state: GraphState, ctx: NodeContext) -> dict[str, Any]:
#	text = state["raw_user_text"]
#	reasons = detect_pii(text)
#	pii_detected = bool(reasons)
#	update: dict[str, Any] = {"pii_detected": pii_detected, "pii_reasons": reasons}
#	if pii_detected and ctx.settings.workflow.pii_policy == "block":
#		update["status"] = "blocked_pii"
#		update["route_reason"] = wrap_guardrail_response(
#			"The request was blocked because it appears to contain personally identifiable information."
#		)
#	return update


#def domain_affinity(state: GraphState, ctx: NodeContext) -> dict[str, Any]:
#	text = state["raw_user_text"]
#	if is_scientific_or_astronomy_related(text):
#		decision = ctx.agents.domain_agent.invoke({"messages": [HumanMessage(content=text)]})["structured_response"]
#	else:
#		decision = DomainDecision(allowed=False, confidence=0.99, reason="Not astronomy/science related.", topic=None)
#	update: dict[str, Any] = {"domain_decision": decision}
#	if not decision.allowed:
#		update["status"] = "blocked_domain"
#		update["route_reason"] = wrap_guardrail_response(decision.reason)
#	return update



#def rewrite_prompt(state: GraphState, ctx: NodeContext) -> dict[str, Any]:
#	assessment: PromptAssessment = state["prompt_assessment"]
#	docs = ctx.rag.retrieve(state["raw_user_text"], k=5)
#	context_blob = "\n\n".join(f"[{doc.doc_id}] {doc.title}\n{doc.text}" for doc in docs)
#	request = (
#		f"Original request:\n{state['raw_user_text']}\n\n"
#		f"Missing details: {assessment.missing_details}\n\n"
#		f"Retrieved optimization context:\n{context_blob or 'No context retrieved.'}"
#	)
#	optimized = ctx.agents.optimizer_agent.invoke({
#		"messages": [HumanMessage(content=request)]
#	})["structured_response"]
#	return {"optimized_prompt": optimized}


#def approval_node(state: GraphState, ctx: NodeContext) -> dict[str, Any]:
#	_ = ctx
#	candidate = state.get("optimized_prompt") or OptimizedPrompt(
#		rewritten_prompt=state["raw_user_text"],
#		assumptions=[],
#		rationale="Prompt was already specific enough.",
#		retrieved_context_ids=[],
#	)
#	decision_payload = interrupt({
#		"type": "prompt_approval",
#		"candidate_prompt": candidate.rewritten_prompt,
#		"assumptions": candidate.assumptions,
#		"rationale": candidate.rationale,
#		"iteration": state.get("approval_iterations", 0),
#		"max_iterations": state.get("max_approval_iterations", 3),
#	})
#	decision = ApprovalDecision.model_validate(decision_payload)
#	return {
#		"approval_decision": decision,
#		"approval_iterations": state.get("approval_iterations", 0) + 1,
#		"status": "awaiting_approval",
#	}


#def refine_from_feedback(state: GraphState, ctx: NodeContext) -> dict[str, Any]:
#	decision = state["approval_decision"]
#	candidate = state.get("optimized_prompt")
#	base_text = candidate.rewritten_prompt if candidate else state["raw_user_text"]
#	feedback_text = decision.feedback or "Please improve clarity and specificity."
#	request = (
#		f"Current rewritten prompt:\n{base_text}\n\n"
#		f"User feedback:\n{feedback_text}\n\n"
#		"Produce a revised version."
#	)
#	optimized = ctx.agents.optimizer_agent.invoke({
#		"messages": [HumanMessage(content=request)]
#	})["structured_response"]
#	return {"optimized_prompt": optimized, "status": "running"}


#def give_up(state: GraphState, ctx: NodeContext) -> dict[str, Any]:
#	_ = ctx
#	return {
#		"status": "rejected_after_iterations",
#		"route_reason": wrap_guardrail_response(
#			"The system could not converge on an approved prompt within the allowed number of iterations."
#		),
#	}


#def prepare_prompt(state: GraphState, ctx: NodeContext) -> dict[str, Any]:
#	_ = ctx
#	optimized = state.get("optimized_prompt")
#	approved_prompt = optimized.rewritten_prompt if optimized else state["raw_user_text"]
#	return {"approved_prompt": approved_prompt, "status": "running"}


#def planner_or_default(state: GraphState, ctx: NodeContext) -> dict[str, Any]:
#	assessment: PromptAssessment = state["prompt_assessment"]
#	prompt = state["approved_prompt"]
#	if assessment.complexity == "complex":
#		plan = ctx.agents.planner_agent.invoke({"messages": [HumanMessage(content=prompt)]})["structured_response"]
#	else:
#		plan = TaskPlan(
#			objective=prompt,
#			steps=[
#				PlanStep(
#					id="step-1",
#					title="Direct execution",
#					description="Answer directly with the general scientific worker.",
#					worker="general",
#				)
#			],
#		)
#	return {"task_plan": plan}


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
	if status in {"blocked_language", "blocked_pii", "blocked_domain", "rejected_after_iterations"}:
		final = FinalAnswer(answer=state.get("route_reason", "Request rejected."), citations=[], caveats=[], confidence="low")
		return {"final_answer": final, "status": "done"}
	return {"status": "done"}
