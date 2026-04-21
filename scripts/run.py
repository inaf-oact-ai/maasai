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
	parser.add_argument('-config_litellm','--config_litellm', dest='config_litellm', required=True, type=str, help='Input yaml config file for LiteLLM') 
	
	# - LLM settings
	parser.add_argument('-temperature','--temperature', dest='temperature', required=False, type=float, default=0.0, help='Default temperature value') 
	
	
	args, _unknown = parser.parse_known_args()
	
	return args		
	
#################
##   HELPERS   ##
#################
def print_interrupt(payload: dict) -> None:
	print("\n=== INTERRUPT ===")
	print(json.dumps(payload, indent=2))
	print("=================\n")


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
	#==   CREATE LLM ROUTER
	#===========================
	# - Load the YAML config
	logger.info(f"Loading LiteLLM config file {args.config_litellm} ...")
	try:
		with open(args.config_litellm, "r") as f:
			config = yaml.safe_load(f)
	except Exception as e:
		logger.error(f"Failed to load LiteLLM config file {args.config_litellm} (err={str(e)}, exit!")
		return 1

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
	logger.info("Creating agent graph ...")
	settings= Settings()
	#graph= build_graph(settings)
	graph= build_graph(
		agents=agents,
		prompt_rag=prompt_rag,
		settings=settings
	)
	
	
	#config = {"configurable": {"thread_id": "maasai-demo-thread"}}

	#initial_state = {
	#	"messages": [
	#		HumanMessage(
	#			content="Analyze a radio astronomy image of a candidate supernova remnant and suggest a minimal workflow."
	#		)
	#	]
	#}

	#result = graph.invoke(initial_state, config=config)
	#if "__interrupt__" in result:
	#	payload = result["__interrupt__"][0].value
	#	print_interrupt(payload)
	#	resume = {
	#		"decision": "approve",
	#		"feedback": None,
	#	}
	#	result = graph.invoke(Command(resume=resume), config=config)

	#final = result.get("final_answer")
	#if final is None:
	#	print("No final answer produced.")
	#	return

	#print("\n=== FINAL ANSWER ===")
	#print(final.answer)
	#if final.caveats:
	#	print("\nCaveats:")
	#	for item in final.caveats:
	#		print(f"- {item}")


	return 0
		
###################
##   MAIN EXEC   ##
###################
if __name__ == "__main__":
	sys.exit(main())	
