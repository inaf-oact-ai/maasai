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

	# - RUN options
	parser.add_argument('-config_litellm','--config_litellm', dest='config_litellm', required=True, type=str, help='Input yaml config file for LiteLLM') 
	parser.add_argument("--mode", choices=["cli", "api"], default="cli")
	parser.add_argument("--thread-id", type=str, default="maasai-thread")
	parser.add_argument("--enable-planner-rag", action="store_true", help="Enable RAG retrieval for the planner node.")
	parser.add_argument("--planner-rag-k", type=int, default=5, help="Number of RAG documents to retrieve for planner context.")

	# - API options
	parser.add_argument("--host", type=str, default="127.0.0.1")
	parser.add_argument("--port", type=int, default=8000)
	

	args, _unknown = parser.parse_known_args()
	
	return args		
	
#################
##   HELPERS   ##
#################
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
	if args.llm_timeout is not None:
		settings.llm.timeout_seconds = args.llm_timeout
		settings.litellm.timeout_seconds = args.llm_timeout
	
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
	router_settings.setdefault("timeout", settings.litellm.timeout_seconds)
	router_settings.setdefault("num_retries", 0)
	router_settings.setdefault("retry_after", 0)

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
	prompt_rag= PromptRAG()

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
	print(graph.get_graph().draw_ascii())
	
	return graph

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
	
def run_graph(graph, args) -> None:
	""" Helper to run the agentic graph and return the final answer. """

	# - Set input arguments
	attachments = []
	if args.input_imgs.strip():
		for raw_path in args.input_imgs.split(","):
			path = raw_path.strip()
			if not path:
				continue
			attachments.append({"path": os.path.abspath(path)})

	# - Define config    
	config = {"configurable": {"thread_id": args.thread_id}}

	# - Define input message    
	attachments = attachments or []
	initial_state = {
		"messages": [
			HumanMessage(content=args.query)
		],
		"attachments": attachments,
		"planner_rag_enabled": getattr(args, "enable_planner_rag", False),
		"planner_rag_k": getattr(args, "planner_rag_k", 5),
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
			
	
def run_cli(graph, args):
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
	graph= build_runtime(args)
	
	#===========================
	#==   RUN
	#===========================
	if args.mode == "cli":
		return run_cli(graph, args)
	elif args.mode == "api":
		return run_api(graph, args)
	
	
	
	return 0
		
###################
##   MAIN EXEC   ##
###################
if __name__ == "__main__":
	sys.exit(main())	
