from __future__ import print_function
from __future__ import annotations

##################################################
###          MODULE IMPORT
##################################################
# - STANDARD MODULES

# - LLM/LANGCHAIN MODULES
from langchain.agents import create_agent

# - MAASAI MODULES
from .config import Settings
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
		tools: AstronomyToolRegistry,
		settings: Settings | None = None,
	) -> None:
		""" Create all agents """
	
		settings = settings or Settings()
	
		# - Intake agent (accepting/rejecting user inputs)
		self.intake_agent = create_agent(
			model=router.get_llm(
				stage="intake",
				tool_required=False,
				structured_output_required=True,
				temperature=settings.llm.intake_temperature,
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
				temperature=settings.llm.assessment_temperature,
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
				"- suggested downstream worker, or step-dependent for multi-worker plans\n"
				"- required workers for multi-step specialist plans\n"
				"- missing details\n"
				"- ambiguities\n"
				"- rewrite goal, if any\n"
				"- a concise reasoning summary\n\n"

				"Worker routing policy:\n"
				"- Route literature-grounded tasks to suggested_worker='literature'. This includes requests for papers, references, citations, inline references, bibliography, related work, literature reviews, state-of-the-art summaries, and introduction/background sections with references.\n"
				"- Route image/FITS analysis tasks to suggested_worker='image'. This includes object/source classification, morphology analysis, segmentation, source detection, photometry, and visual/FITS inspection.\n"
				"- Route catalog/database tasks to suggested_worker='catalog'. This includes catalog lookup, cross-matching, counterpart searches, coordinate/cone searches, source metadata retrieval, and survey-table analysis.\n"
				"- Route only non-specialist explanation, coding, workflow-design, and general reasoning tasks to suggested_worker='general'.\n"
				"- If requires_planning=True and the task needs multiple specialist capabilities, set suggested_worker='step-dependent' and populate required_workers with the executable workers likely needed.\n"
				"- required_workers must contain only executable workers: general, image, catalog, literature. Do not include 'step-dependent' in required_workers.\n\n"

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
				temperature=settings.llm.optimizer_temperature,
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
				temperature=settings.llm.planner_temperature,
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

		# - Define tools
		worker_tools = tools.get_tools() if hasattr(tools, "get_tools") else []
		
		# - Worker agents
		self.general_agent = create_agent(
			model=router.get_llm(
				stage="worker",
				tool_required=False,
				structured_output_required=False,
				temperature=settings.llm.general_temperature,
			),
			tools=[],
			system_prompt=(
				"You are the MAASAI general scientific worker. "
				"Handle conceptual explanations, scientific reasoning, coding guidance, "
				"workflow design, and general astronomy/astrophysics tasks. "
				"Do not invent unavailable observations, data products, citations, or tool results. "
				"If required information is missing, state what is missing and proceed only with justified assumptions. "
				"Clearly distinguish hard requirements, recommended defaults, and illustrative parameter values. "
				"Do not present mission-specific thresholds as universal constants. "
			),
		)
		
		self.catalog_agent = create_agent(
			model=router.get_llm(
				stage="worker",
				tool_required=bool(worker_tools),
				structured_output_required=False,
				temperature=settings.llm.catalog_temperature,
			),
			tools=worker_tools,
			system_prompt=(
				"You are the MAASAI catalog/database worker. "
				"Handle catalog queries, cross-matching, source metadata retrieval, "
				"coordinate-based searches, survey-table analysis, and CAESAR/MCP-backed data access. "
				"Use tools when needed and available. "
				"Do not analyze image pixels directly. "
				"Do not fabricate catalog rows, coordinates, measurements, or source metadata."
			),
		)

		self.image_agent = create_agent(
			model=router.get_llm(
				stage="worker",
				tool_required=bool(worker_tools),
				structured_output_required=False,
				temperature=settings.llm.image_temperature,
			),
			tools=worker_tools,
			system_prompt=(
				"You are the MAASAI astronomical image-analysis worker. "
				"Handle FITS/image inspection, morphology assessment, source detection guidance, "
				"image-derived measurements, cutouts, segmentation products, and multimodal image tasks. "
				"Use tools when needed and available. "
				"Do not fabricate catalog metadata; request catalog-worker support when source metadata or cross-matches are needed."
			),
		)

		self.literature_agent = create_agent(
			model=router.get_llm(
				stage="worker",
				tool_required=bool(worker_tools),
				structured_output_required=False,
				temperature=settings.llm.literature_temperature,
			),
			tools=worker_tools,
			system_prompt=(
				"You are the MAASAI literature/evidence worker. "
				"Handle literature search, paper summaries, reference discovery, method comparison, "
				"related-work synthesis, introduction/background drafting with citations, "
				"and scientific evidence synthesis. "
				"When retrieved literature context is provided, use only that context for factual claims, inline citations, and bibliography entries. "
				"Do not fabricate papers, citations, DOIs, arXiv identifiers, years, journals, or claims. "
				"Do not add standalone bibliography entries for works that are merely mentioned inside a retrieved review article unless those works are themselves retrieved as evidence. "
				"Do not include a separate References or Bibliography section in the answer body unless the user explicitly asks for one; "
				"the system will attach structured citations separately. "
				"If retrieved context is insufficient for the requested references, say so clearly."
			),
		)

		# - Aggregator agent
		self.aggregator_agent = create_agent(
			model=router.get_llm(
				stage="aggregator",
				tool_required=False,
				structured_output_required=False,
				temperature=settings.llm.aggregator_temperature,
			),
			tools=[],
			system_prompt=(
				"You are the MAASAI final-response aggregator. "
				"Aggregate worker outputs into a coherent astronomy/astrophysics answer. "
				"Preserve caveats, mention failed or incomplete steps, and do not invent evidence. "
				"If the workers did not retrieve enough information, say so clearly."
			),
		)
		
		
