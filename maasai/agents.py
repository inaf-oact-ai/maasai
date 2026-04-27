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
from .schemas import OptimizedPrompt
from .schemas import TaskPlan
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

		# - Prompt optimizer agent
		self.optimizer_agent = create_agent(
			model=router.get_llm(
				stage="prompt_optimization",
				tool_required=False,
				structured_output_required=True,
				temperature=0.0,
			),
			tools=[],
			response_format=OptimizedPrompt,
			system_prompt=(
				"You are the MAASAI prompt rewrite agent. "
				"Your role is to rewrite an accepted astronomy/astrophysics request "
				"into a clearer and more operational task specification for downstream execution.\n\n"

				"Rules:\n"
				"- Do not invent facts, observations, datasets, or results.\n"
				"- Preserve the user's scientific intent.\n"
				"- Use the assessment information to address missing details and ambiguities.\n"
				"- Keep assumptions explicit, but place them only in the assumptions field.\n"
				"- Place unresolved issues only in the open_questions field.\n"
				"- Do not include sections titled 'Assumptions' or 'Open Questions' inside rewritten_prompt.\n"
				"- rewritten_prompt should contain only the improved execution-ready task request.\n"
				"- The rewritten prompt should be concise, scientifically meaningful, and suitable for downstream planning or direct execution.\n\n"

				"Return only the structured rewritten prompt."
			),
		)
		
		# - Task planner agent
		self.planner_agent = create_agent(
			model=router.get_llm(
				stage="planner",
				tool_required=False,
				structured_output_required=True,
				temperature=0.0,
			),
			tools=[],
			response_format=TaskPlan,
			system_prompt=(
				"You are the MAASAI task planner. "
				"Your role is to convert an approved astronomy/astrophysics execution prompt "
				"into a concrete ordered execution plan.\n\n"

				"Available workers:\n"
				"- general: conceptual explanations, scientific reasoning, coding guidance, workflow design.\n"
				"- image: astronomical image/FITS analysis, morphology inspection, image-derived measurements.\n"
				"- catalog: catalog queries, cross-matching, source metadata, survey table analysis.\n"
				"- literature: literature search, paper summaries, reference discovery.\n\n"

				"Planning rules:\n"
				"- Use the fewest steps needed for reliable execution.\n"
				"- Assign exactly one worker to each step.\n"
				"- Do not invent unavailable observations, data, or tools.\n"
				"- Use retrieved planning context only to improve the plan structure, not as final scientific evidence.\n"
				"- If no data are attached, avoid image/catalog execution steps unless the prompt asks for a workflow involving such data.\n"
				"- For workflow-design or explanatory tasks, the general worker is often sufficient.\n\n"

				"Return only the structured task plan."
			),
		)


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
		
		
