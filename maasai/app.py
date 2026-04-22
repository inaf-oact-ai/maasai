from __future__ import print_function
from __future__ import annotations

##################################################
###          MODULE IMPORT
##################################################
# - STANDARD MODULES
from typing import Any

# - FASTAPI / PYDANTIC
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

# - LANGCHAIN / LANGGRAPH
from langgraph.types import Command
from langchain_core.messages import HumanMessage

# - MAASAI MODULES
from maasai import logger

###########################
##   PYDANTIC MODELS
###########################
class Attachment(BaseModel):
	path: str


class InvokeRequest(BaseModel):
	thread_id: str = Field(..., description="Conversation/session identifier")
	message: str = Field(..., description="User message")
	attachments: list[Attachment] = Field(default_factory=list)


class ResumeRequest(BaseModel):
	thread_id: str = Field(..., description="Conversation/session identifier")
	decision: str = Field(..., description="Human approval decision")
	feedback: str | None = Field(default=None, description="Optional human feedback")


class ChatRequest(BaseModel):
	thread_id: str = Field(..., description="Conversation/session identifier")
	message: str = Field(..., description="User message")
	attachments: list[Attachment] = Field(default_factory=list)


###########################
##   SERIALIZATION HELPERS
###########################
def _serialize_final(final: Any) -> dict[str, Any] | None:
	if final is None:
		return None

	return {
		"status": getattr(final, "status", None),
		"message": getattr(final, "message", None),
		"answer": getattr(final, "answer", None),
		"citations": getattr(final, "citations", []) or [],
		"artifacts": getattr(final, "artifacts", []) or [],
		"debug": getattr(final, "debug", None),
	}


def _completed_response(thread_id: str, final: Any) -> dict[str, Any]:
	result = _serialize_final(final)
	answer_text = None if result is None else result.get("answer")

	return {
		"status": "completed",
		"thread_id": thread_id,
		"result": result,
		"output": answer_text,   # handy alias for thin UI clients
	}


def _interrupt_response(thread_id: str, payload: Any) -> dict[str, Any]:
	return {
		"status": "interrupt",
		"thread_id": thread_id,
		"payload": payload,
	}


def _error_response(thread_id: str | None, err: Exception) -> dict[str, Any]:
	return {
		"status": "error",
		"thread_id": thread_id,
		"error": str(err),
	}


###########################
##   AGENT GRAPH APP
###########################
def create_fastapi_app(graph):
	"""Create FastAPI app exposing the MAASAI LangGraph workflow."""

	app = FastAPI(
		title="MAASAI Agent API",
		version="0.1.0",
		description="FastAPI wrapper for the MAASAI LangGraph assistant workflow.",
	)

	@app.get("/health")
	def health():
		return {
			"ok": True,
			"service": "maasai-agent-api",
		}

	@app.get("/info")
	def info():
		return {
			"name": "MAASAI Agent API",
			"interrupts_supported": True,
			"attachments_supported": True,
			"endpoints": [
				"/health",
				"/info",
				"/invoke",
				"/resume",
				"/chat",
			],
		}

	@app.post("/invoke")
	def invoke(req: InvokeRequest):
		try:
			config = {"configurable": {"thread_id": req.thread_id}}
			initial_state = {
				"messages": [HumanMessage(content=req.message)],
				"attachments": [a.model_dump() for a in req.attachments],
			}

			logger.info("Invoke request received for thread_id=%s", req.thread_id)
			result = graph.invoke(initial_state, config=config)

			if "__interrupt__" in result:
				payload = result["__interrupt__"][0].value
				logger.info("Interrupt returned for thread_id=%s", req.thread_id)
				return _interrupt_response(req.thread_id, payload)

			final = result.get("final_answer")
			return _completed_response(req.thread_id, final)

		except Exception as e:
			logger.exception("Invoke failed for thread_id=%s", req.thread_id)
			raise HTTPException(status_code=500, detail=_error_response(req.thread_id, e))

	@app.post("/resume")
	def resume(req: ResumeRequest):
		try:
			config = {"configurable": {"thread_id": req.thread_id}}
			resume_payload = {
				"decision": req.decision,
				"feedback": req.feedback,
			}

			logger.info("Resume request received for thread_id=%s", req.thread_id)
			result = graph.invoke(Command(resume=resume_payload), config=config)

			if "__interrupt__" in result:
				payload = result["__interrupt__"][0].value
				logger.info("Interrupt returned again for thread_id=%s", req.thread_id)
				return _interrupt_response(req.thread_id, payload)

			final = result.get("final_answer")
			return _completed_response(req.thread_id, final)

		except Exception as e:
			logger.exception("Resume failed for thread_id=%s", req.thread_id)
			raise HTTPException(status_code=500, detail=_error_response(req.thread_id, e))

	@app.post("/chat")
	def chat(req: ChatRequest):
		"""
		OpenWebUI-friendly alias for /invoke.
		Returns both the structured result and a top-level assistant message field.
		"""
		try:
			config = {"configurable": {"thread_id": req.thread_id}}
			initial_state = {
				"messages": [HumanMessage(content=req.message)],
				"attachments": [a.model_dump() for a in req.attachments],
			}

			logger.info("Chat request received for thread_id=%s", req.thread_id)
			result = graph.invoke(initial_state, config=config)

			if "__interrupt__" in result:
				payload = result["__interrupt__"][0].value
				return {
					"status": "interrupt",
					"thread_id": req.thread_id,
					"payload": payload,
					"assistant": None,
				}

			final = result.get("final_answer")
			serialized = _serialize_final(final)
			assistant_text = None if serialized is None else serialized.get("answer")

			return {
				"status": "completed",
				"thread_id": req.thread_id,
				"assistant": assistant_text,
				"result": serialized,
			}

		except Exception as e:
			logger.exception("Chat failed for thread_id=%s", req.thread_id)
			raise HTTPException(status_code=500, detail=_error_response(req.thread_id, e))

	return app
