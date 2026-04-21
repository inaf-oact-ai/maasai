from __future__ import print_function
from __future__ import annotations

##################################################
###          MODULE IMPORT
##################################################
# - STANDARD MODULES
from dataclasses import dataclass
from typing import Any

# - LLANGCHAIN MODULES
from fastapi import FastAPI
from pydantic import BaseModel
from langgraph.types import Command
from langchain_core.messages import HumanMessage

# - MAASAI MODULES
from maasai import logger

###########################
##   AGENT GRAPH APP
###########################
def create_fastapi_app(graph):
	""" Create web app """
	
	class InvokeRequest(BaseModel):
		thread_id: str
		message: str

	class ResumeRequest(BaseModel):
		thread_id: str
		decision: str
		feedback: str | None = None

	app = FastAPI()

	@app.post("/invoke")
	def invoke(req: InvokeRequest):
		config = {"configurable": {"thread_id": req.thread_id}}
		initial_state = {
			"messages": [HumanMessage(content=req.message)]
		}

		result = graph.invoke(initial_state, config=config)

		if "__interrupt__" in result:
			payload = result["__interrupt__"][0].value
			return {"status": "interrupt", "payload": payload}

		final = result.get("final_answer")
		return {
			"status": "completed",
			"final_answer": None if final is None else final.answer,
			"caveats": [] if final is None else final.caveats,
		}

	@app.post("/resume")
	def resume(req: ResumeRequest):
		config = {"configurable": {"thread_id": req.thread_id}}
		resume_payload = {
			"decision": req.decision,
			"feedback": req.feedback,
		}

		result = graph.invoke(Command(resume=resume_payload), config=config)

		if "__interrupt__" in result:
			payload = result["__interrupt__"][0].value
			return {"status": "interrupt", "payload": payload}

		final = result.get("final_answer")
		return {
			"status": "completed",
			"final_answer": None if final is None else final.answer,
			"caveats": [] if final is None else final.caveats,
		}

	return app
