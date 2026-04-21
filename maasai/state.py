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
#from .schemas import ApprovalDecision
#from .schemas import DomainDecision
#from .schemas import OptimizedPrompt
#from .schemas import PromptAssessment
#from .schemas import StepResult
#from .schemas import TaskPlan

##################################################
###          GRAPH STATE
##################################################
class GraphState(TypedDict, total=False):
	messages: list[Any]
	raw_user_text: str
	multimodal: bool
	attachments: list[dict[str, Any]]

	language_ok: bool | None
	pii_detected: bool | None
	language_model_ok: bool | None
	pii_model_detected: bool | None
	language_precheck_ok: bool | None
	pii_precheck_detected: bool | None

	domain_ok: bool | None
	intake_reason: str | None
	pii_reasons: list[str]
	
	#domain_decision: DomainDecision

	#prompt_assessment: PromptAssessment
	#optimized_prompt: OptimizedPrompt
	#approval_iterations: int
	#max_approval_iterations: int
	#approval_decision: ApprovalDecision
	#approved_prompt: str

	#task_plan: TaskPlan
	#execution_results: list[StepResult]
	final_answer: FinalAnswer | None

	route_reason: str
	status: Literal[
		"running",
		"blocked_language",
		"blocked_pii",
		"blocked_domain",
		"blocked_intake",
		"awaiting_approval",
		"rejected_after_iterations",
		"done",
	]
