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

	##### MANDATORY ######
	# - Input options
	parser.add_argument('-config_litellm','--config_litellm', dest='config_litellm', required=True, type=str, help='Input yaml config file for LiteLLM') 
	
	# - Task options (MANDATORY ONLY IN CLI MODE)
	parser.add_argument('-query','--query', dest='query', required=False, default=None, type=str, help='Input user query') 
	
	##### OPTIONAL #######
	# - LLM options
	parser.add_argument('-temperature','--temperature', dest='temperature', required=False, type=float, default=0.0, help='Default temperature value') 
	
	# - RUN options
	parser.add_argument("--mode", choices=["cli", "api"], default="cli")
	parser.add_argument("--thread-id", type=str, default="maasai-demo-thread")
	
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


###########################
##   AGENT GRAPH METHODS
###########################
def build_runtime(args):
	""" Create agent graph """
	
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

	# - Create a LiteLLM Router with the model list from the config
	logger.info(f"Creating a LiteLLM Router with model list parsed from config file {args.config_litellm} ...")
	litellm_router = Router(
		model_list=config['model_list'],
		**config.get("router_settings", {})
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
	# - Create settings
	logger.info("Create settings ...")
	settings= Settings()
	
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


def run_graph(
	graph,
	query: str, 
	thread_id: str
) -> str:
	"""Helper to run the agentic graph and return the final answer."""

	# - Define config    
	config = {"configurable": {"thread_id": thread_id}}

	# - Define input message    
	initial_state = {
		"messages": [
			HumanMessage(
				content=query
			)
		]
	}

	# - Invoke graph
	logger.info(f"Run user query: {query} ...")
	result = graph.invoke(initial_state, config=config)
	
	# - Get response
	if "__interrupt__" in result:
		payload = result["__interrupt__"][0].value
		print_interrupt(payload)
		resume = {
			"decision": "approve",
			"feedback": None,
		}
		result = graph.invoke(Command(resume=resume), config=config)

	final = result.get("final_answer")
	if final is None:
		logger.warn("No final answer produced.")
		return

	print("\n=== FINAL ANSWER ===")
	print(final.answer)
	if final.caveats:
		print("\nCaveats:")
		for item in final.caveats:
			print(f"- {item}") 
			
	return final.answer


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
		query=args.query,
		thread_id="maasai-thread"
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
