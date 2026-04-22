from __future__ import print_function
from __future__ import annotations

##################################################
###          MODULE IMPORT
##################################################
# - STANDARD MODULES
from typing import Any, Literal
from pydantic import BaseModel, Field

##################################################
###          SCHEMAS
##################################################
class PreparedAsset(BaseModel):
	""" A data structure to hold user provided data assets """
	path: str
	kind: Literal["fits", "image", "other"]
	original_mime_type: str | None = None
	preview_mime_type: str | None = None
	preview_path: str | None = None
	base64_data: str | None = None
	notes: list[str] = Field(default_factory=list)
	
	
class IntakeDecision(BaseModel):
	""" A data structure to hold results of user prompt/asset triage analysis """
	accepted: bool
	language_ok: bool
	pii_detected: bool
	domain_ok: bool
	images_astronomy_ok: bool = True
	reason: str
	normalized_text: str
	#prepared_assets: list[PreparedAsset] = Field(default_factory=list)
		
#class RAGDocument(BaseModel):
#	doctype: str | None = None
#	title: str | None = None
#	text: str
#	score: float | None = None
#	source_id: str | None = None
#	metadata: dict[str, Any] = Field(default_factory=dict)

#class OptimizedPrompt(BaseModel):
#	rewritten_prompt: str
#	assumptions: list[str] = Field(default_factory=list)
#	rationale: str = ""
#	retrieved_context_ids: list[str] = Field(default_factory=list)


#class ApprovalDecision(BaseModel):
#	decision: Literal["approve", "revise", "reject"]
#	feedback: str | None = None


#class PlanStep(BaseModel):
#	step_id: str
#	objective: str
#	worker: Literal["literature", "caesar", "general"] = "general"
#	tool_hint: str | None = None
#	inputs: dict[str, Any] = Field(default_factory=dict)
#	depends_on: list[str] = Field(default_factory=list)


#class Plan(BaseModel):
#	steps: list[PlanStep] = Field(default_factory=list)


#class CaesarApp(BaseModel):
#	name: str
#	description: str | None = None
#	raw: dict[str, Any] = Field(default_factory=dict)


#class CaesarJobSubmission(BaseModel):
#	app_name: str
#	job_id: str | None = None
#	payload: dict[str, Any] = Field(default_factory=dict)
#	raw: dict[str, Any] = Field(default_factory=dict)


#class CaesarJobStatus(BaseModel):
#	job_id: str
#	status: str
#	terminal: bool = False
#	raw: dict[str, Any] = Field(default_factory=dict)


#class CaesarJobOutput(BaseModel):
#	job_id: str
#	content_type: str | None = None
#	output_path: str | None = None
#	data: Any = None
#	metadata: dict[str, Any] = Field(default_factory=dict)


class FinalAnswer(BaseModel):
	status: Literal["ok", "rejected", "aborted", "error"]
	message: str
	answer: str | None = None
	citations: list[dict[str, Any]] = Field(default_factory=list)
	artifacts: list[dict[str, Any]] = Field(default_factory=list)
	debug: dict[str, Any] = Field(default_factory=dict)

