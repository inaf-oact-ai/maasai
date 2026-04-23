from __future__ import print_function
from __future__ import annotations

##################################################
###          MODULE IMPORT
##################################################
# - STANDARD MODULES

# - LLM/LANGCHAIN MODULES
from langchain.agents import create_agent

# - MAASAI MODULES
from .model_router import ModelRouter
from .schemas import IntakeDecision
from .schemas import PromptAssessment
#from .schemas import DomainDecision
#from .schemas import OptimizedPrompt
#from .schemas import TaskPlan
from .tools import AstronomyToolRegistry

##################################################
###          AGENTS
##################################################
class AgentFactory:
	def __init__(
		self, 
		router: ModelRouter, 
		tools: AstronomyToolRegistry
	) -> None:
		""" Create all agents """
	
		# - Intake agent (accepting/rejecting user inputs)
		self.intake_agent = create_agent(
			model=router.get_llm(
				stage="intake",
				tool_required=False,
				structured_output_required=True,
				temperature=0.0,
			),
			tools=[],
			response_format=IntakeDecision,
			system_prompt=(
				"You are the MAASAI system intake gate agent. "
				"Your role is to decide whether the user request is acceptable according to these requirements: \n"
				"- Accept only English requests about astronomy, astrophysics, scientific analysis, or astronomical data analysis. \n"
				"- Reject if personal identifiable information (PII) is present. \n"
				"- Reject if the provided input data (if any) do not belong to the astronomy domain, e.g. an image or a table with non-astronomical content. \n"
				"Return a precise reason for acceptance/rejection."
			),
		)
	
		# - Prompt assessment agent (eval/analyze user request)
		self.assessment_agent = create_agent(
			model=router.get_llm(
				stage="prompt_assessment",
				tool_required=False,
				structured_output_required=True,
				temperature=0.0,
			),
			tools=[],
			response_format=PromptAssessment,
			system_prompt=(
				"You are the MAASAI prompt assessment agent.\n"
				"Your role is to evaluate an already accepted astronomy/astrophysics user request "
				"and decide how ready it is for downstream execution.\n\n"

				"Important distinctions:\n"
				"- needs_rewrite = True only if the request is too underspecified, ambiguous, "
				"or poorly formed to proceed reliably.\n"
				"- rewrite_would_help = True if the request is understandable and executable, "
				"but could benefit from normalization into a clearer or more operational task specification.\n"
				"- executable_as_is = True if a useful answer or downstream action can already be produced.\n\n"

				"Assess:\n"
				"- whether rewrite is required\n"
				"- whether rewrite would help\n"
				"- whether the task is executable as-is\n"
				"- complexity\n"
				"- whether explicit planning is required\n"
				"- task type\n"
				"- suggested downstream worker\n"
				"- missing details\n"
				"- ambiguities\n"
				"- rewrite goal, if any\n"
				"- a concise reasoning summary\n\n"

				"Do not reject requests for safety, language, or domain reasons; intake triage already handled that.\n"
				"Return only the structured assessment."
			)
		)

		#self.domain_agent = create_agent(
		#	model=router.pick(stage="domain"),
		#	tools=[],
		#	response_format=DomainDecision,
		#	system_prompt=(
		#		"Decide whether the request belongs to astronomy or scientific analysis. "
		#		"Return allowed=false when it is clearly out of scope."
		#	),
		#)

		#self.optimizer_agent = create_agent(
		#	model=router.pick(stage="optimizer"),
		#	tools=[],
		#	response_format=OptimizedPrompt,
		#	system_prompt=(
		#		"Rewrite the task into a clearer, executable astronomy/science prompt. "
		#		"Never invent facts. Keep assumptions explicit."
		#	),
		#)

		#self.planner_agent = create_agent(
		#	model=router.pick(stage="planner", complexity="complex"),
		#	tools=[],
		#	response_format=TaskPlan,
		#	system_prompt=(
		#		"Break the approved prompt into executable astronomy workflow steps. "
		#		"Keep the plan concise and practical."
		#	),
		#)

		#worker_tools = [tools.query_caesar_rest, tools.call_mcp_tool]

		#self.catalog_agent = create_agent(
		#	model=router.pick(stage="worker"),
		#	tools=worker_tools,
		#	system_prompt="You are the astronomy catalog/database specialist.",
		#)

		#self.image_agent = create_agent(
		#	model=router.pick(stage="worker", multimodal=True),
		#	tools=worker_tools,
		#	system_prompt="You are the astronomy image-analysis specialist.",
		#)

		#self.literature_agent = create_agent(
		#	model=router.pick(stage="worker"),
		#	tools=worker_tools,
		#	system_prompt="You are the astronomy literature/evidence specialist.",
		#)

		#self.general_agent = create_agent(
		#	model=router.pick(stage="worker"),
		#	tools=worker_tools,
		#	system_prompt="You are the general scientific workflow specialist.",
		#)

		#self.aggregator_agent = create_agent(
		#	model=router.pick(stage="aggregator", complexity="complex"),
		#	tools=[],
		#	system_prompt=(
		#		"Aggregate the worker outputs into a coherent final response. "
		#		"Mention caveats where evidence is weak or incomplete."
		#	),
		#)
		
		
