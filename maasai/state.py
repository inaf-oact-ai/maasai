from __future__ import print_function
from __future__ import annotations

##################################################
###          MODULE IMPORT
##################################################
# - STANDARD MODULES
from typing import Any, Literal
from typing_extensions import TypedDict

# - MAASAI MODULES
#from .schemas import ApprovalDecision
#from .schemas DomainDecision
#from .schemas FinalAnswer
#from .schemas OptimizedPrompt
#from .schemas PromptAssessment
#from .schemas StepResult
#from .schemas TaskPlan

##################################################
###          GRAPH STATE
##################################################
class GraphState(TypedDict, total=False):
	messages: list[Any]
	raw_user_text: str
	multimodal: bool
	attachments: list[dict[str, Any]]

	language_ok: bool
	pii_detected: bool
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
	#final_answer: FinalAnswer

	route_reason: str
	status: Literal[
		"running",
		"blocked_language",
		"blocked_pii",
		"blocked_domain",
		"awaiting_approval",
		"rejected_after_iterations",
		"done",
	]
