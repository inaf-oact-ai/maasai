from __future__ import print_function
from __future__ import annotations

##################################################
###          MODULE IMPORT
##################################################
# - STANDARD MODULES
from functools import partial

# - LLM/LANGCHAIN MODULES
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, StateGraph

# - MAASAI MODULES
from .agents import AgentFactory
from .config import Settings
from .nodes import NodeContext
from .nodes import intake_triage
from .nodes import assess_prompt
from .nodes import final_guardrail
#from .nodes import (
#	NodeContext,
#	aggregate,
#	approval_node,
#	assess_prompt,
#	domain_affinity,
#	execute_plan,
#	final_guardrail,
#	give_up,
#	language_gate,
#	normalize_input,
#	pii_gate,
#	planner_or_default,
#	prepare_prompt,
#	refine_from_feedback,
#	rewrite_prompt,
#)
from .rag import PromptRAG
from .state import GraphState
from .tools import AstronomyToolRegistry
from .model_router import ModelRouter

from maasai import logger

##################################################
###          HELPER METHODS
##################################################
def _after_intake(state: GraphState) -> str:
	decision = state.get("intake_decision")
	if decision and decision.accepted:
		return "assess_prompt"
	return "final_guardrail"

def _after_language(state: GraphState) -> str:
	return "pii_gate" if state.get("language_ok", False) else "final_guardrail"


def _after_pii(state: GraphState) -> str:
	return "domain_affinity" if not state.get("pii_detected", False) else "final_guardrail"


def _after_domain(state: GraphState) -> str:
	decision = state.get("domain_decision")
	return "assess_prompt" if decision and decision.allowed else "final_guardrail"


def _after_assessment(state: GraphState) -> str:
	assessment = state.get("prompt_assessment")
	if assessment and assessment.needs_rewrite:
		return "rewrite_prompt"
	return "approval"


def _after_approval(state: GraphState) -> str:
	decision = state.get("approval_decision")
	if decision is None:
		return "give_up"
	if decision.decision == "approve":
		return "prepare_prompt"
	if decision.decision == "revise":
		if state.get("approval_iterations", 0) >= state.get("max_approval_iterations", 3):
			return "give_up"
		return "refine_from_feedback"
	return "give_up"

##################################################
###          GRAPH
##################################################
#def build_graph(settings: Settings | None = None):
#	settings = settings or Settings()
#	ctx = NodeContext(
#		settings=settings,
#		rag=PromptRAG(),
#		agents=AgentFactory(ModelRouter(settings.litellm), AstronomyToolRegistry()),
#	)

def build_graph(
	agents: AgentFactory,
	prompt_rag: PromptRAG | None = None,
	settings: Settings | None = None,
	ckp_saver: BaseCheckpointSaver = InMemorySaver(),
):
	settings = settings or Settings()
	ctx = NodeContext(
		settings=settings,
		rag=prompt_rag,
		agents=agents
	)
	""" Build agent graph """
	
	# - Create builder
	logger.info("Creating builder ...")
	builder = StateGraph(GraphState)

	# - Create nodes
	logger.info("Creating nodes ...")
	builder.add_node("intake_triage", partial(intake_triage, ctx=ctx))
	##builder.add_node("normalize_input", partial(normalize_input, ctx=ctx))
	##builder.add_node("language_gate", partial(language_gate, ctx=ctx))
	##builder.add_node("pii_gate", partial(pii_gate, ctx=ctx))
	##builder.add_node("domain_affinity", partial(domain_affinity, ctx=ctx))
	
	#builder.add_node("assess_prompt", partial(assess_prompt, ctx=ctx))
	#builder.add_node("rewrite_prompt", partial(rewrite_prompt, ctx=ctx))
	
	#builder.add_node("approval", partial(approval_node, ctx=ctx))
	#builder.add_node("refine_from_feedback", partial(refine_from_feedback, ctx=ctx))
	#builder.add_node("give_up", partial(give_up, ctx=ctx))
	#builder.add_node("prepare_prompt", partial(prepare_prompt, ctx=ctx))
	#builder.add_node("planner_or_default", partial(planner_or_default, ctx=ctx))
	#builder.add_node("execute_plan", partial(execute_plan, ctx=ctx))
	#builder.add_node("aggregate", partial(aggregate, ctx=ctx))
	builder.add_node("final_guardrail", partial(final_guardrail, ctx=ctx))

	# - Connect nodes
	#builder.add_edge(START, "intake_triage")
	#builder.add_conditional_edges("intake_triage", _after_intake, {
	#	"assess_prompt": "assess_prompt",
	#	"final_guardrail": "final_guardrail",
	#})
	
	############# TEST ################
	builder.add_edge(START, "intake_triage")
	builder.add_edge("intake_triage", "final_guardrail")
	builder.add_edge("final_guardrail", END)
	####################################

	####  TO BE REMOVED ###
	#builder.add_edge(START, "normalize_input")
	#builder.add_edge("normalize_input", "language_gate")
	#builder.add_conditional_edges("language_gate", _after_language, {
	#	"pii_gate": "pii_gate",
	#	"final_guardrail": "final_guardrail",
	#})
	#builder.add_conditional_edges("pii_gate", _after_pii, {
	#	"domain_affinity": "domain_affinity",
	#	"final_guardrail": "final_guardrail",
	#})
	#builder.add_conditional_edges("domain_affinity", _after_domain, {
	#	"assess_prompt": "assess_prompt",
	#	"final_guardrail": "final_guardrail",
	#})
	########################
	
	#builder.add_conditional_edges("assess_prompt", _after_assessment, {
	#	"rewrite_prompt": "rewrite_prompt",
	#	"approval": "approval",
	#})
	
	#### TO BE REVISED ######
	#builder.add_edge("rewrite_prompt", "approval")
	#builder.add_conditional_edges("approval", _after_approval, {
	#	"prepare_prompt": "prepare_prompt",
	#	"refine_from_feedback": "refine_from_feedback",
	#	"give_up": "give_up",
	#})
	#builder.add_edge("refine_from_feedback", "approval")
	#builder.add_edge("give_up", "final_guardrail")
	#builder.add_edge("prepare_prompt", "planner_or_default")
	#builder.add_edge("planner_or_default", "execute_plan")
	#builder.add_edge("execute_plan", "aggregate")
	#builder.add_edge("aggregate", "final_guardrail")
	##########################
	
	#builder.add_edge("final_guardrail", END)

	# - Compile and return graph
	logger.info("Compiling graph ...")
	return builder.compile(checkpointer=ckp_saver)
