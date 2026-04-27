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
	error: str | None = None
	is_valid: bool | None = None
	
class IntakeDecision(BaseModel):
	""" A data structure to hold results of user prompt/asset triage analysis """
	accepted: bool
	language_ok: bool
	pii_detected: bool
	domain_ok: bool
	images_astronomy_ok: bool = True
	reason: str
	normalized_text: str


class PromptAssessment(BaseModel):
	"""Assessment of whether a user request is ready for downstream execution."""

	needs_rewrite: bool = Field(
		...,
		description=(
			"True only when the request is not sufficiently actionable or specific "
			"to proceed reliably to downstream execution."
		),
	)

	rewrite_would_help: bool = Field(
		...,
		description=(
			"True when the request is understandable and executable, but a rewrite "
			"would improve clarity, scientific grounding, or execution-readiness."
		),
	)

	executable_as_is: bool = Field(
		...,
		description=(
			"Whether the request can be handled directly in its current form."
		),
	)

	complexity: Literal["simple", "moderate", "complex"] = Field(
		...,
		description="Estimated execution complexity of the request.",
	)

	requires_planning: bool = Field(
		...,
		description=(
			"Whether the task likely needs explicit multi-step planning rather than direct execution."
		),
	)

	task_type: str = Field(
		...,
		description=(
			"High-level category of task, e.g. explanation, coding, image-analysis, "
			"catalog-query, literature-review, workflow-design."
		),
	)

	suggested_worker: str | None = Field(
		default=None,
		description=(
			"Suggested downstream worker, e.g. general, image, catalog, literature."
		),
	)

	missing_details: list[str] = Field(
		default_factory=list,
		description=(
			"Important missing details that should ideally be provided before or during execution."
		),
	)

	ambiguities: list[str] = Field(
		default_factory=list,
		description=(
			"Non-blocking ambiguities or underspecified aspects of the request."
		),
	)

	rewrite_goal: str | None = Field(
		default=None,
		description=(
			"If a rewrite is useful or required, describe what the rewrite should achieve."
		),
	)

	reasoning_summary: str = Field(
		...,
		description="Short explanation of the assessment."
	)
	
class OptimizedPrompt(BaseModel):
	"""Rewritten prompt ready for downstream execution."""

	rewritten_prompt: str = Field(
		...,
		description="A clearer, more operational rewrite of the original request.",
	)

	assumptions: list[str] = Field(
		default_factory=list,
		description=(
			"Explicit assumptions introduced to make the request executable "
			"without inventing facts."
		),
	)

	open_questions: list[str] = Field(
		default_factory=list,
		description=(
			"Questions that remain unresolved and may still need user clarification."
		),
	)

	rationale: str = Field(
		...,
		description="Short explanation of why this rewrite is better for execution.",
	)
	
class ApprovalDecision(BaseModel):
	"""Human approval decision for an optimized prompt."""

	decision: Literal["approve", "revise", "reject"] = Field(
		...,
		description=(
			"User decision on the rewritten prompt. "
			"approve means continue execution; revise means rewrite again using feedback; "
			"reject means stop the workflow."
		),
	)

	feedback: str | None = Field(
		default=None,
		description=(
			"Optional user feedback. Required or strongly recommended when decision is revise."
		),
	)
			

class PlanStep(BaseModel):
	"""Single execution step in a MAASAI task plan."""

	id: str = Field(
		...,
		description="Stable step identifier, e.g. step-1.",
	)

	title: str = Field(
		...,
		description="Short human-readable step title.",
	)

	description: str = Field(
		...,
		description="Detailed description of what this step should do.",
	)

	worker: Literal["general", "image", "catalog", "literature"] = Field(
		...,
		description="Worker that should execute this step.",
	)

	inputs: list[str] = Field(
		default_factory=list,
		description="Inputs required by this step.",
	)

	expected_output: str = Field(
		...,
		description="Expected output of this step.",
	)


class TaskPlan(BaseModel):
	"""Execution plan for an accepted and prepared user request."""

	objective: str = Field(
		...,
		description="Overall objective to be executed.",
	)

	requires_rag_context: bool = Field(
		default=False,
		description="Whether retrieved planning context was used or is considered useful.",
	)

	rationale: str = Field(
		...,
		description="Short explanation of why this plan structure was chosen.",
	)

	steps: list[PlanStep] = Field(
		...,
		description="Ordered list of execution steps.",
	)			
			
#class RAGDocument(BaseModel):
#	doctype: str | None = None
#	title: str | None = None
#	text: str
#	score: float | None = None
#	source_id: str | None = None
#	metadata: dict[str, Any] = Field(default_factory=dict)


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

