from __future__ import print_function
from __future__ import annotations

##################################################
###          MODULE IMPORT
##################################################
# - STANDARD MODULES
from typing import Any, Literal
from typing_extensions import TypedDict

# - MAASAI MODULES
from .schemas import FinalAnswer
from .schemas import PreparedAsset
from .schemas import IntakeDecision
from .schemas import PromptAssessment
from .schemas import OptimizedPrompt
from .schemas import ApprovalDecision
from .schemas import TaskPlan
	#from .schemas import StepResult

##################################################
###          GRAPH STATE
##################################################
class GraphState(TypedDict, total=False):
	messages: list[Any]
	raw_user_text: str
	multimodal: bool
	attachments: list[dict[str, Any]]
	prepared_assets: list[PreparedAsset]
	intake_decision: IntakeDecision | None

	language_ok: bool | None
	pii_detected: bool | None
	language_model_ok: bool | None
	pii_model_detected: bool | None
	language_precheck_ok: bool | None
	pii_precheck_detected: bool | None

	domain_ok: bool | None
	intake_reason: str | None
	pii_reasons: list[str]
	
	attachment_errors: list[dict[str, Any]]
	
	prompt_assessment: PromptAssessment | None
	optimized_prompt: OptimizedPrompt | None
	approval_decision: ApprovalDecision | None
	approval_iterations: int
	max_approval_iterations: int
	approved_prompt: str | None
	execution_prompt: str | None
	
	task_plan: TaskPlan | None
	planner_rag_enabled: bool
	planner_rag_context: list[dict[str, str]]
	planner_rag_k: int
	
	#execution_results: list[StepResult]
	final_answer: FinalAnswer | None

	route_reason: str
	status: Literal[
		"running",
		"invalid_attachments",
		"blocked_language",
		"blocked_pii",
		"blocked_domain",
		"blocked_intake",
		"needs_rewrite",
		"awaiting_approval",
		"approved",
		"prepared",
		"planned",
		"rejected_after_iterations",
		"rejected_by_user",
		"done",
	]
	
