#!/usr/bin/env python

from __future__ import print_function
from __future__ import annotations

##################################################
###          MODULE IMPORT
##################################################
# - STANDARD MODULES
import sys
import os
import random
import numpy as np
import argparse
from pathlib import Path
import json
from datetime import datetime, timedelta
from IPython.display import Image, display
from typing import Literal
from typing import TypedDict
import yaml

# - LLANGCHAIN/LLM MODULES
from pydantic import BaseModel, Field
import mlflow
from litellm import Router
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from langchain_core.tools import tool
from langgraph.graph import StateGraph, START, END
from langchain.agents import create_agent
from langgraph.checkpoint.memory import MemorySaver
from langgraph.types import Command, interrupt
from langgraph.checkpoint.memory import InMemorySaver, MemorySaver

# - MAASAI MODULES
from maasai.config import Settings
from maasai.graph import build_graph
from maasai.model_router import ModelRouter
from maasai.tools import AstronomyToolRegistry
from maasai.agents import AgentFactory
from maasai.rag import PromptRAG

# - LOGGER
from maasai import logger

# Set mlflow autologging and experiment name
os.environ["MLFLOW_EXPERIMENT_NAME"] = "multiagent_traces"

mlflow.set_tracking_uri(os.environ.get("MLFLOW_TRACKING_URI"))
mlflow.set_experiment(os.environ.get("MLFLOW_EXPERIMENT_NAME"))
mlflow.langchain.autolog()

###########################
##     ARGS
###########################
def get_args():
	"""This function parses and return arguments passed in"""
	parser = argparse.ArgumentParser(description="Parse args.")

	# - Input options
	parser.add_argument('-query','--query', dest='query', required=False, default=None, type=str, help='Input user query') 
	parser.add_argument("--input_imgs", default="")
	
	# - LLM options
	parser.add_argument('-temperature','--temperature', dest='temperature', required=False, type=float, default=0.0, help='Default temperature value') 
	parser.add_argument("--llm-timeout", type=float, default=None, help="Timeout in seconds for LLM calls",)
	parser.add_argument("--litellm-timeout", type=float, default=None, help="Timeout in seconds for LiteLLM Router calls.")
	parser.add_argument("--litellm-num-retries", type=int, default=None, help="Number of LiteLLM Router retries.")
	parser.add_argument("--litellm-retry-after", type=float, default=None, help="Delay in seconds between LiteLLM Router retries.")
	
	# - RUN options
	parser.add_argument('-config_litellm','--config_litellm', dest='config_litellm', required=True, type=str, help='Input YAML config file for LiteLLM') 
	parser.add_argument("--mode", choices=["cli", "api"], default="cli")
	parser.add_argument("--thread-id", type=str, default="maasai-thread")
	parser.add_argument("--print-graph", dest="print_graph", action="store_true", default=None, help="Print compiled LangGraph ASCII graph.")
	parser.add_argument("--no-print-graph", dest="print_graph", action="store_false", help="Do not print compiled LangGraph ASCII graph.")
	
	# - Workflow options
	parser.add_argument("--max-approval-iterations", type=int, default=None, help="Maximum number of prompt approval/revision iterations.")
	parser.add_argument("--strict-english-only", dest="strict_english_only", action="store_true", default=None, help="Require English input.")
	parser.add_argument("--no-strict-english-only", dest="strict_english_only", action="store_false", help="Disable English-only input requirement.")
	parser.add_argument("--pii-policy", choices=["block", "allow"], default=None, help="PII policy used by intake guardrails.")

	# - Planner RAG options
	parser.add_argument("--enable-planner-rag", dest="enable_planner_rag", action="store_true", default=None, help="Enable RAG retrieval for the planner node.")
	parser.add_argument("--disable-planner-rag", dest="enable_planner_rag", action="store_false", help="Disable RAG retrieval for the planner node.")
	parser.add_argument("--planner-rag-k", type=int, default=None, help="Number of final RAG documents to keep for planner context.")
	parser.add_argument("--rag-qdrant-url", type=str, default=None, help="Remote Qdrant endpoint URL.")
	parser.add_argument("--rag-embedding-model", type=str, default=None, help="Embedding model used by the planner RAG retriever.")
	parser.add_argument("--rag-top-k-per-collection", type=int, default=None, help="Number of chunks retrieved from each selected Qdrant collection.")
	parser.add_argument("--rag-final-top-k", type=int, default=None, help="Maximum number of merged RAG chunks retained after collection merge.")
	parser.add_argument("--rag-score-threshold", type=float, default=None, help="Optional score threshold for retrieved chunks.")
	parser.add_argument("--rag-base-collections", type=str, default=None, help="Comma-separated collections always included, e.g. annreviews.")
	parser.add_argument("--rag-default-collections", type=str, default=None, help="Comma-separated fallback collections used when no domain-specific collection is selected.")
	parser.add_argument("--rag-content-payload-key", type=str, default=None, help="Qdrant payload key containing chunk text, e.g. text, _text, page_content.")
	parser.add_argument("--rag-metadata-payload-key", type=str, default=None, help="Optional Qdrant payload key containing metadata dictionary.")
	parser.add_argument("--rag-debug-payload", action="store_true", default=False, help="Print raw Qdrant payload samples and exit.")
	parser.add_argument("--rag-debug-collections", type=str, default=None, help="Comma-separated collections to inspect with --rag-debug-payload.")
	parser.add_argument("--rag-backend", choices=["llama-index-service", "langchain-qdrant", "raw-qdrant"], default=None, help="RAG retrieval backend. Use llama-index as default, langchain-qdrant for QdrantVectorStore + post-processing, raw-qdrant for direct QdrantClient retrieval.")
	parser.add_argument("--rag-fallback-to-local", dest="rag_fallback_to_local", action="store_true", default=None, help="If the external RAG backend fails, fall back to the local RAG backend.")
	parser.add_argument("--no-rag-fallback-to-local", dest="rag_fallback_to_local", action="store_false", help="Disable fallback from external RAG backend to local RAG backend.")
	parser.add_argument("--llama-index-rag-url", type=str, default=None, help="Base URL of the llama-index-rag service, e.g. http://127.0.0.1:8010.")
	parser.add_argument("--llama-index-num-queries", type=int, default=None, help="Number of query-fusion generated queries used by llama-index-rag.")
	parser.add_argument("--rag-request-timeout", type=float, default=None, help="Timeout in seconds for external RAG service calls.")

	# - API options
	parser.add_argument("--host", type=str, default="127.0.0.1")
	parser.add_argument("--port", type=int, default=8000)
	

	args, _unknown = parser.parse_known_args()
	
	return args		
	
#################
##   HELPERS   ##
#################
def _parse_csv_list(value: str | None) -> list[str] | None:
	""" Parse CSV list"""
	if value is None:
		return None

	return [
		item.strip()
		for item in value.split(",")
		if item.strip()
	]
	
def print_interrupt(payload: dict) -> None:
	print("\n=== INTERRUPT ===")
	print(json.dumps(payload, indent=2))
	print("=================\n")

def print_approval_interrupt(payload: dict) -> None:
	print("\n=== PROMPT APPROVAL REQUIRED ===")

	candidate_prompt = payload.get("candidate_prompt")
	assumptions = payload.get("assumptions", [])
	open_questions = payload.get("open_questions", [])
	rationale = payload.get("rationale")
	iteration = payload.get("iteration")
	max_iterations = payload.get("max_iterations")
	instructions = payload.get("instructions")

	print(f"\nIteration: {iteration}/{max_iterations}")

	if instructions:
		print("\nInstructions:")
		print(instructions)

	print("\n--- Candidate Prompt ---")
	print(candidate_prompt or "<empty>")

	print("\n--- Assumptions ---")
	if assumptions:
		for item in assumptions:
			print(f"- {item}")
	else:
		print("- none")

	print("\n--- Open Questions ---")
	if open_questions:
		for item in open_questions:
			print(f"- {item}")
	else:
		print("- none")

	if rationale:
		print("\n--- Rationale ---")
		print(rationale)

	assessment = payload.get("assessment")
	if assessment:
		print("\n--- Assessment Summary ---")
		print(f"needs_rewrite: {assessment.get('needs_rewrite')}")
		print(f"rewrite_would_help: {assessment.get('rewrite_would_help')}")
		print(f"executable_as_is: {assessment.get('executable_as_is')}")
		print(f"complexity: {assessment.get('complexity')}")
		print(f"requires_planning: {assessment.get('requires_planning')}")
		print(f"task_type: {assessment.get('task_type')}")
		print(f"suggested_worker: {assessment.get('suggested_worker')}")

def ask_approval_decision() -> dict:
	while True:
		print("\nChoose an action:")
		print("  [a] approve")
		print("  [r] revise")
		print("  [x] reject")

		choice = input("> ").strip().lower()

		if choice in {"a", "approve"}:
			return {
				"decision": "approve",
				"feedback": None,
			}

		if choice in {"r", "revise"}:
			feedback = input("Revision feedback: ").strip()
			if not feedback:
				feedback = "Please improve clarity and specificity."
			return {
				"decision": "revise",
				"feedback": feedback,
			}

		if choice in {"x", "reject"}:
			feedback = input("Optional rejection reason: ").strip()
			return {
				"decision": "reject",
				"feedback": feedback or None,
			}

		print("Invalid choice. Please enter a, r, or x.")

###########################
##   AGENT GRAPH METHODS
###########################
def build_runtime(args):
	""" Create agent graph """
	
	#===========================
	#==   CREATE SETTINGS
	#===========================
	# - Create settings
	logger.info("Create settings ...")
	settings= Settings()
	
	# - Override settings
	logger.info("Overriding default settings with provided input options ...")
	settings = apply_cli_overrides(settings, args)
	#if args.llm_timeout is not None:
	#	settings.llm.timeout_seconds = args.llm_timeout
	#	settings.litellm.timeout_seconds = args.llm_timeout
	
	#===========================
	#==   CREATE LLM ROUTER
	#===========================
	# - Load the YAML config
	logger.info(f"Loading LiteLLM config file {args.config_litellm} ...")
	try:
		with open(args.config_litellm, "r") as f:
			config = yaml.safe_load(f)
	except Exception as e:
		logger.error(f"Failed to load LiteLLM config file {args.config_litellm} (err={str(e)}, exit!")
		return None

	# - Set router settings
	router_settings = dict(config.get("router_settings", {}))
	router_settings["timeout"] = settings.litellm.timeout_seconds
	router_settings["num_retries"] = settings.litellm.num_retries
	router_settings["retry_after"] = settings.litellm.retry_after

	# - Create a LiteLLM Router with the model list from the config
	logger.info(f"Creating a LiteLLM Router with model list parsed from config file {args.config_litellm} ...")
	litellm_router = Router(
		model_list=config['model_list'],
		#**config.get("router_settings", {})
		**router_settings
	)
	
	# - Create a model router using LiteLLM router
	logger.info("Creating a model router using LiteLLM router ...")
	model_router = ModelRouter(
		litellm_router=litellm_router,
		config=config,
		default_temperature=args.temperature,
	)

	#===========================
	#==   BUILD TOOLS
	#===========================
	# - Create tools
	logger.info("Creating tool inventory ...")
	tool_inventory= AstronomyToolRegistry()
	
	# - Create RAG
	logger.info("Creating prompt RAG ...")
	prompt_rag = PromptRAG(settings=settings.rag)
	
	logger.info(
		"Planner RAG settings: "
		f"enabled={settings.rag.enabled}, "
		f"backend={settings.rag.backend}, "
		f"fallback_to_local={settings.rag.fallback_to_local}, "
		f"llamaindex_service_url={settings.rag.llamaindex_service_url}, "
		f"llamaindex_num_queries={settings.rag.llamaindex_num_queries}, "
		f"qdrant_url={settings.rag.qdrant_url}, "
		f"embedding_model={settings.rag.embedding_model}, "
		f"top_k_per_collection={settings.rag.top_k_per_collection}, "
		f"final_top_k={settings.rag.final_top_k}, "
		f"content_payload_key={settings.rag.content_payload_key}, "
		f"metadata_payload_key={settings.rag.metadata_payload_key}, "
		f"default_collections={settings.rag.default_collections}, "
		f"base_collections={settings.rag.always_include_collections}"
	)
	
	if getattr(args, "rag_debug_payload", False):
		collections = (
			_parse_csv_list(args.rag_debug_collections)
			or settings.rag.default_collections
		)

		for collection_name in collections:
			print(f"\n=== QDRANT PAYLOAD SAMPLE: {collection_name} ===")
			samples = prompt_rag.debug_sample_payload(collection_name, limit=1)
			print(json.dumps(samples, indent=2, default=str))

		return "debug_done", settings
		
	
	
	#===========================
	#==   BUILD AGENTS
	#===========================
	logger.info("Creating agents ...")
	agents= AgentFactory(model_router, tool_inventory)

	#===========================
	#==   BUILD GRAPH
	#===========================
	# - Create memory
	logger.info("Creating graph memory ...")
	ckp_saver = MemorySaver()
	#ckp_saver = InMemorySaver()
	
	# - Build graph
	logger.info("Creating agent graph ...")
	graph= build_graph(
		agents=agents,
		prompt_rag=prompt_rag,
		settings=settings,
		ckp_saver=ckp_saver
	)
	
	# - Visualize compiled graph
	logger.info("Visualize graph ...")
	if settings.runtime.print_graph:
		print(graph.get_graph().draw_ascii())
	
	return graph, settings


def apply_cli_overrides(settings: Settings, args) -> Settings:
	"""Apply command-line overrides to Settings.

	Environment variables and dataclass defaults are loaded first.
	CLI arguments have highest priority.
	"""

	# - Generic LLM settings
	if args.llm_timeout is not None:
		settings.llm.timeout_seconds = args.llm_timeout

	# - LiteLLM router settings
	if args.litellm_timeout is not None:
		settings.litellm.timeout_seconds = args.litellm_timeout
	elif args.llm_timeout is not None:
		# Backward-compatible behavior: --llm-timeout also controls LiteLLM
		# unless a dedicated --litellm-timeout is supplied.
		settings.litellm.timeout_seconds = args.llm_timeout

	if args.litellm_num_retries is not None:
		settings.litellm.num_retries = args.litellm_num_retries

	if args.litellm_retry_after is not None:
		settings.litellm.retry_after = args.litellm_retry_after

	# - Workflow settings
	if args.max_approval_iterations is not None:
		settings.workflow.max_approval_iterations = args.max_approval_iterations

	if args.strict_english_only is not None:
		settings.workflow.strict_english_only = args.strict_english_only

	if args.pii_policy is not None:
		settings.workflow.pii_policy = args.pii_policy

	# - Runtime settings
	if args.print_graph is not None:
		settings.runtime.print_graph = args.print_graph

	# - Planner RAG settings
	if args.enable_planner_rag is not None:
		settings.rag.enabled = args.enable_planner_rag
		
	if args.rag_backend is not None:
		settings.rag.backend = args.rag_backend
		
	if args.rag_fallback_to_local is not None:
		settings.rag.fallback_to_local = args.rag_fallback_to_local

	if args.llama_index_rag_url is not None:
		settings.rag.llamaindex_service_url = args.llama_index_rag_url

	if args.llama_index_num_queries is not None:
		settings.rag.llamaindex_num_queries = args.llama_index_num_queries

	if args.rag_request_timeout is not None:
		settings.rag.request_timeout = args.rag_request_timeout

	if args.rag_qdrant_url is not None:
		settings.rag.qdrant_url = args.rag_qdrant_url

	if args.rag_embedding_model is not None:
		settings.rag.embedding_model = args.rag_embedding_model

	if args.rag_top_k_per_collection is not None:
		settings.rag.top_k_per_collection = args.rag_top_k_per_collection

	if args.rag_final_top_k is not None:
		settings.rag.final_top_k = args.rag_final_top_k

	if args.rag_score_threshold is not None:
		settings.rag.score_threshold = args.rag_score_threshold

	base_collections = _parse_csv_list(args.rag_base_collections)
	if base_collections is not None:
		settings.rag.always_include_collections = base_collections

	default_collections = _parse_csv_list(args.rag_default_collections)
	if default_collections is not None:
		settings.rag.default_collections = default_collections
		
	if args.rag_content_payload_key is not None:
		settings.rag.content_payload_key = args.rag_content_payload_key

	if args.rag_metadata_payload_key is not None:
		settings.rag.metadata_payload_key = args.rag_metadata_payload_key

	return settings

def invoke_with_cli_approval(graph, initial_input: dict, config: dict):
	result = graph.invoke(initial_input, config=config)

	while "__interrupt__" in result:
		interrupts = result["__interrupt__"]
		if not interrupts:
			break

		payload = interrupts[0].value
	
		if payload.get("type") != "prompt_approval":
			print("\n=== UNKNOWN INTERRUPT ===")
			print(payload)
			return result

		print_approval_interrupt(payload)
		resume_payload = ask_approval_decision()

		result = graph.invoke(
			Command(resume=resume_payload),
			config=config,
		)

	return result
	
def run_graph(graph, args, settings: Settings) -> None:
	""" Helper to run the agentic graph and return the final answer. """

	# - Set input arguments
	attachments = []
	if args.input_imgs.strip():
		for raw_path in args.input_imgs.split(","):
			path = raw_path.strip()
			if not path:
				continue
			attachments.append({"path": os.path.abspath(path)})

	planner_rag_k = (
		args.planner_rag_k
		if args.planner_rag_k is not None
		else settings.rag.final_top_k
	)

	# - Define config    
	config = {"configurable": {"thread_id": args.thread_id}}

	# - Define input message    
	attachments = attachments or []
	initial_state = {
		"messages": [
			HumanMessage(content=args.query)
		],
		"attachments": attachments,
		"planner_rag_enabled": settings.rag.enabled,
		"planner_rag_k": planner_rag_k,
	}
	
	print("initial_state")
	print(initial_state)
	
	# - Invoke graph with approval
	logger.info(f"Run user query: {args.query} ...")
	#result = graph.invoke(initial_state, config=config)
	result = invoke_with_cli_approval(
		graph=graph,
		initial_input=initial_state,
		config=config,
	)
	
	# - Parse response
	final = result.get("final_answer")
	print("final")
	print(final)
	if final is None:
		logger.warning("No final answer produced.")
		return

	print("\n=== FINAL ANSWER ===")
	print(f"Status: {final.status}")
	print(f"Message: {final.message}")

	if final.answer:
		print("\nAnswer:")
		print(final.answer)

	if final.citations:
		print("\nCitations:")
		for item in final.citations:
			print(f"- {item}")

	if final.artifacts:
		print("\nArtifacts:")
		for item in final.artifacts:
			print(f"- {item}")

	if final.debug:
		print("\nDebug:")
		print(final.debug)
		
	# - Print other debug state fields
	#print("\n=== GRAPH RESULT KEYS ===")
	#print(result.keys())

	#print("\n=== PREPARED ASSETS IN STATE ===")
	#print(result.get("prepared_assets"))
			
	
def run_cli(graph, args, settings: Settings):
	""" Run graph in CLI mode for test purposes """

	# - Check for user prompt
	if not args.query or args.query is None or args.query=="":
		print("Missing --query or empty query in cli mode")
		return 1
		
	# - Run graph
	logger.info(f"Run graph with user query: {args.query}")
	return run_graph(
		graph=graph,
		args=args,
		settings=settings,
	)
	
###########################
##   AGENT GRAPH APP
###########################
def run_api(graph, args):
	import uvicorn
	from maasai.app import create_fastapi_app
	
	# - Create app
	logger.info("Creating app for agent graph ...")
	app = create_fastapi_app(graph)
	
	# - Launch app
	logger.info(f"Launching app on {args.host}:{args.port} ...")
	uvicorn.run(app, host=args.host, port=args.port)
	return 0
	

##############
##   MAIN   ##
##############
def main():
	"""Main function"""

	#===========================
	#==   PARSE ARGS
	#===========================
	logger.info("Parse and retrieve input script args ...")
	try:
		args= get_args()
	except Exception as ex:
		logger.error("Failed to get and parse options (err=%s)", str(ex))
		return 1

	
	#===========================
	#==   CREATE GRAPH
	#===========================
	logger.info("Creating agent graph ...")
	runtime = build_runtime(args)
	if runtime is None:
		return 1

	graph, settings = runtime
	
	if graph == "debug_done":
		return 0
	
	#===========================
	#==   RUN
	#===========================
	if args.mode == "cli":
		return run_cli(graph, args, settings)
	elif args.mode == "api":
		return run_api(graph, args)
	
	return 0
		
###################
##   MAIN EXEC   ##
###################
if __name__ == "__main__":
	sys.exit(main())	
