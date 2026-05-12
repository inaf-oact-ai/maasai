from __future__ import print_function
from __future__ import annotations

##################################################
###          MODULE IMPORT
##################################################
# - STANDARD MODULES
import json
import time
import re
from pathlib import Path
from typing import Any, Callable

# - THIRD PARTY MODULES
import requests
from pydantic import BaseModel, Field

# - LANGCHAIN MODULES
from langchain.tools import tool
from langchain_core.tools import StructuredTool

# - LOGGER
from maasai import logger

##################################################
###          CAESAR REST SCHEMAS
##################################################
class CaesarRunAppArgs(BaseModel):
	data_inputs: dict[str, Any] = Field(
		...,
		description=(
			"CAESAR REST data_inputs object. This should contain input files, "
			"dataset ids, file ids, or other app-specific input data."
		),
	)
	job_options: dict[str, Any] = Field(
		default_factory=dict,
		description=(
			"CAESAR REST job_options object. Valid app-specific options are listed "
			"in this tool description under 'Parameters discovered from caesar-rest describe endpoint'."
		),
	)
	tag: str = Field(
		default="",
		description="Optional user-defined tag for the submitted CAESAR job.",
	)
	wait: bool = Field(
		default=False,
		description="If true, wait until the job reaches a terminal status.",
	)
	poll_seconds: float = Field(
		default=5.0,
		description="Polling interval used when wait=true.",
	)
	timeout_seconds: float = Field(
		default=600.0,
		description="Maximum waiting time used when wait=true.",
	)


class CaesarSubmitJobArgs(BaseModel):
	app: str = Field(..., description="Name of the CAESAR REST application to run.")
	data_inputs: dict[str, Any] = Field(..., description="CAESAR REST data_inputs object.")
	job_options: dict[str, Any] = Field(default_factory=dict, description="CAESAR REST job_options object.")
	tag: str = Field(default="", description="Optional job tag.")
	wait: bool = Field(default=False, description="If true, wait for job completion.")
	poll_seconds: float = Field(default=5.0, description="Polling interval when wait=true.")
	timeout_seconds: float = Field(default=600.0, description="Timeout when wait=true.")


##################################################
###          CAESAR REST CLIENT
##################################################
class CaesarRestClient:
	def __init__(
		self,
		base_url: str,
		timeout_seconds: float = 30.0,
		api_prefix: str = "/caesar/api/v1.0",
		auth_token: str | None = None,
	) -> None:
		self.base_url = base_url.rstrip("/")
		self.timeout_seconds = timeout_seconds
		self.api_prefix = api_prefix.strip("/")
		self.auth_token = auth_token

	def _url(self, path: str) -> str:
		path = path.strip("/")
		return f"{self.base_url}/{self.api_prefix}/{path}"

	def _headers(self) -> dict[str, str]:
		headers = {
			"Accept": "application/json",
		}
		if self.auth_token:
			headers["Authorization"] = f"Bearer {self.auth_token}"
		return headers

	def _request(
		self,
		method: str,
		path: str,
		json_payload: dict[str, Any] | None = None,
		files: dict[str, Any] | None = None,
	) -> Any:
		url = self._url(path)

		logger.info(
			f"--> CAESAR HTTP REQUEST: method={method.upper()}, url={url}, "
			f"has_json={json_payload is not None}, has_files={files is not None}"
		)

		try:
			response = requests.request(
				method=method.upper(),
				url=url,
				headers=self._headers(),
				json=json_payload,
				files=files,
				timeout=self.timeout_seconds,
			)
			
			logger.info(
				f"--> CAESAR HTTP RESPONSE: method={method.upper()}, url={url}, "
				f"status_code={response.status_code}, content_type={response.headers.get('content-type', '')}"
			)
			
			response.raise_for_status()

			content_type = response.headers.get("content-type", "")
			if "application/json" in content_type:
				return response.json()

			return response.text

		except requests.RequestException as exc:
			logger.exception("CAESAR REST request failed")
			raise RuntimeError(f"CAESAR REST request failed: {method} {url}: {exc}") from exc

	def list_apps(self) -> list[str]:
		result = self._request("GET", "apps")

		if isinstance(result, list):
			return [str(item) for item in result]

		if isinstance(result, dict):
			for key in ["apps", "data", "results"]:
				if isinstance(result.get(key), list):
					return [str(item) for item in result[key]]

		raise RuntimeError(f"Unexpected CAESAR /apps response: {result!r}")

	def describe_app(self, app: str) -> dict[str, Any]:
		result = self._request("GET", f"app/{app}/describe")

		if not isinstance(result, dict):
			raise RuntimeError(f"Unexpected CAESAR describe response for app={app!r}: {result!r}")

		return result

	def discover_apps(self) -> dict[str, dict[str, Any]]:
		apps = self.list_apps()
		specs: dict[str, dict[str, Any]] = {}

		for app in apps:
			try:
				specs[app] = self.describe_app(app)
			except Exception as exc:
				logger.warning(f"Could not describe CAESAR app {app!r}: {exc}")
				specs[app] = {
					"name": app,
					"description": f"CAESAR REST app {app}. Description could not be retrieved.",
					"error": str(exc),
				}

		return specs

	def submit_job(
		self,
		app: str,
		data_inputs: dict[str, Any],
		job_options: dict[str, Any] | None = None,
		tag: str = "",
	) -> dict[str, Any]:
		payload = {
			"app": app,
			"data_inputs": data_inputs,
			"job_options": job_options or {},
		}

		if tag:
			payload["tag"] = tag

		result = self._request("POST", "job", json_payload=payload)

		if isinstance(result, dict):
			return result

		return {"raw_response": result}

	def get_job_status(self, job_id: str) -> dict[str, Any]:
		result = self._request("GET", f"job/{job_id}/status")

		if isinstance(result, dict):
			return result

		return {"raw_response": result}

	def wait_for_job(
		self,
		job_id: str,
		poll_seconds: float = 5.0,
		timeout_seconds: float = 600.0,
	) -> dict[str, Any]:
		start = time.monotonic()
		terminal_statuses = {
			"SUCCESS",
			"FAILED",
			"CANCELLED",
			"CANCELED",
			"ERROR",
			"DONE",
		}

		last_status: dict[str, Any] = {}

		while True:
			last_status = self.get_job_status(job_id)
			status_value = self._extract_status_value(last_status)

			if status_value and status_value.upper() in terminal_statuses:
				return last_status

			if time.monotonic() - start > timeout_seconds:
				return {
					"status": "TIMEOUT",
					"job_id": job_id,
					"last_status": last_status,
					"message": f"Timed out after {timeout_seconds} seconds.",
				}

			time.sleep(max(float(poll_seconds), 0.5))

	def get_job_output(self, job_id: str) -> Any:
		return self._request("GET", f"job/{job_id}/output")

	def cancel_job(self, job_id: str) -> dict[str, Any]:
		result = self._request("POST", f"job/{job_id}/cancel")

		if isinstance(result, dict):
			return result

		return {"raw_response": result}

	def list_jobs(self) -> Any:
		return self._request("GET", "jobs")

	def list_file_ids(self) -> Any:
		return self._request("GET", "fileids")

	def list_datasets(self) -> Any:
		return self._request("GET", "datasets")

	def upload_file(self, path: str) -> Any:
		file_path = Path(path)

		if not file_path.exists():
			raise FileNotFoundError(f"File does not exist: {path}")

		with file_path.open("rb") as handle:
			files = {
				"file": (file_path.name, handle),
			}
			return self._request("POST", "upload", files=files)

	@staticmethod
	def _extract_file_uid(result: Any) -> str | None:
		uid_keys = {
			"uuid",
			"uid",
			"file_uid",
			"fileid",
			"file_id",
			"id",
		}

		def walk(value: Any) -> str | None:
			if isinstance(value, dict):
				for key in uid_keys:
					item = value.get(key)
					if item:
						return str(item)

				for item in value.values():
					found = walk(item)
					if found:
						return found

			if isinstance(value, list):
				for item in value:
					found = walk(item)
					if found:
						return found

			return None

		return walk(result)

	@staticmethod
	def _extract_job_id(result: dict[str, Any]) -> str | None:
		for key in ["job_id", "id", "jobId"]:
			value = result.get(key)
			if value:
				return str(value)

		job = result.get("job")
		if isinstance(job, dict):
			for key in ["job_id", "id", "jobId"]:
				value = job.get(key)
				if value:
					return str(value)

		return None

	@staticmethod
	def _extract_status_value(result: dict[str, Any]) -> str | None:
		for key in ["status", "state", "job_status"]:
			value = result.get(key)
			if value:
				return str(value)

		job = result.get("job")
		if isinstance(job, dict):
			for key in ["status", "state", "job_status"]:
				value = job.get(key)
				if value:
					return str(value)

		return None


##################################################
###          CAESAR TOOL CACHE
##################################################
class CaesarRestToolCache:
	_memory_cache: dict[str, Any] | None = None
	_memory_cache_timestamp: float | None = None

	def __init__(
		self,
		cache_path: str | Path | None = None,
		ttl_seconds: float = 86400.0,
	) -> None:
		self.cache_path = Path(cache_path) if cache_path else None
		self.ttl_seconds = ttl_seconds

	def load(self) -> dict[str, Any] | None:
		if self._memory_cache is not None and not self._is_expired(self._memory_cache_timestamp):
			return self._memory_cache

		if self.cache_path is None or not self.cache_path.exists():
			return None

		try:
			with self.cache_path.open("r", encoding="utf-8") as handle:
				payload = json.load(handle)

			timestamp = payload.get("timestamp")
			if self._is_expired(timestamp):
				return None

			app_specs = payload.get("app_specs")
			if not isinstance(app_specs, dict):
				return None

			self.__class__._memory_cache = app_specs
			self.__class__._memory_cache_timestamp = float(timestamp or time.time())

			return app_specs

		except Exception as exc:
			logger.warning(f"Could not load CAESAR tool cache: {exc}")
			return None

	def save(self, app_specs: dict[str, Any]) -> None:
		now = time.time()

		self.__class__._memory_cache = app_specs
		self.__class__._memory_cache_timestamp = now

		if self.cache_path is None:
			return

		try:
			self.cache_path.parent.mkdir(parents=True, exist_ok=True)
			with self.cache_path.open("w", encoding="utf-8") as handle:
				json.dump(
					{
						"timestamp": now,
						"app_specs": app_specs,
					},
					handle,
					indent=2,
					sort_keys=True,
				)
		except Exception as exc:
			logger.warning(f"Could not save CAESAR tool cache: {exc}")

	def _is_expired(self, timestamp: float | int | str | None) -> bool:
		if timestamp is None:
			return True

		try:
			age = time.time() - float(timestamp)
		except Exception:
			return True

		return age > self.ttl_seconds


##################################################
###          CAESAR REST TOOLKIT
##################################################
class CaesarRestToolkit:
	def __init__(
		self,
		client: CaesarRestClient,
		cache: CaesarRestToolCache | None = None,
		refresh_tools: bool = False,
		enable_dynamic_app_tools: bool = True,
	) -> None:
		self.client = client
		self.cache = cache or CaesarRestToolCache()
		self.enable_dynamic_app_tools = enable_dynamic_app_tools

		self.app_specs = self._load_or_discover(refresh_tools=refresh_tools)
		self._tools = self._build_tools()

	def _load_or_discover(self, refresh_tools: bool = False) -> dict[str, dict[str, Any]]:
		if not refresh_tools:
			cached = self.cache.load()
			if cached is not None:
				logger.info(f"Loaded {len(cached)} CAESAR REST app specs from cache.")
				return cached

		logger.info("Discovering CAESAR REST apps from API.")
		app_specs = self.client.discover_apps()
		logger.info(f"Discovered {len(app_specs)} CAESAR REST app specs from API.")
		self.cache.save(app_specs)
		return app_specs

	def refresh(self) -> None:
		self.app_specs = self.client.discover_apps()
		self.cache.save(self.app_specs)
		self._tools = self._build_tools()

	def get_tools(self) -> list[Any]:
		return list(self._tools)

	def _build_tools(self) -> list[Any]:
		#tools: list[Any] = [
		#	StructuredTool.from_function(
		#		func=self._caesar_list_apps,
		#		name="caesar_list_apps",
		#		description="List CAESAR REST applications discovered from the cached app registry.",
		#	),
		#	StructuredTool.from_function(
		#		func=self._caesar_describe_app,
		#		name="caesar_describe_app",
		#		description="Return the CAESAR REST describe metadata for one application.",
		#	),
		#	StructuredTool.from_function(
		#		func=self._caesar_submit_job,
		#		name="caesar_submit_job",
		#		description="Submit a generic CAESAR REST job by app name.",
		#		args_schema=CaesarSubmitJobArgs,
		#	),
		#	StructuredTool.from_function(
		#		func=self._caesar_get_job_status,
		#		name="caesar_get_job_status",
		#		description="Get the status of a CAESAR REST job.",
		#	),
		#	StructuredTool.from_function(
		#		func=self._caesar_wait_for_job,
		#		name="caesar_wait_for_job",
		#		description="Poll a CAESAR REST job until it reaches a terminal state or timeout.",
		#	),
		#	StructuredTool.from_function(
		#		func=self._caesar_get_job_output,
		#		name="caesar_get_job_output",
		#		description="Retrieve the output metadata or output payload for a completed CAESAR REST job.",
		#	),
		#	StructuredTool.from_function(
		#		func=self._caesar_cancel_job,
		#		name="caesar_cancel_job",
		#		description="Cancel a CAESAR REST job.",
		#	),
		#	StructuredTool.from_function(
		#		func=self._caesar_list_jobs,
		#		name="caesar_list_jobs",
		#		description="List CAESAR REST jobs visible to the service.",
		#	),
		#	StructuredTool.from_function(
		#		func=self._caesar_list_file_ids,
		#		name="caesar_list_file_ids",
		#		description="List file ids known to CAESAR REST.",
		#	),
		#	StructuredTool.from_function(
		#		func=self._caesar_list_datasets,
		#		name="caesar_list_datasets",
		#		description="List datasets known to CAESAR REST.",
		#	),
		#	StructuredTool.from_function(
		#		func=self._caesar_upload_file,
		#		name="caesar_upload_file",
		#		description="Upload a local file to CAESAR REST and return the upload response.",
		#	),
		#]

		tools: list[Any] = []
		
		if self.enable_dynamic_app_tools:
			for app_name, app_spec in sorted(self.app_specs.items()):
				tools.append(self._build_dynamic_app_tool(app_name, app_spec))

		logger.info(
			f"Built {len(tools)} CAESAR REST app tools "
			f"({len(self.app_specs)} dynamic app specs, "
			f"dynamic_enabled={self.enable_dynamic_app_tools})."
		)

		logger.debug(
			"CAESAR REST app tool names: "
			+ ", ".join(getattr(item, "name", str(item)) for item in tools)
		)
		
		return tools

	def _build_dynamic_app_tool(
		self,
		app_name: str,
		app_spec: dict[str, Any],
	) -> StructuredTool:
		safe_app_name = self._safe_tool_name(app_name)
		tool_name = f"caesar_run_{safe_app_name}"
		description = self._build_dynamic_tool_description(app_name, app_spec)

		def _normalize_data_inputs_for_caesar(
			data_inputs: dict[str, Any],
		) -> tuple[dict[str, Any], dict[str, Any] | None]:
			"""Upload local absolute paths before submitting CAESAR jobs.

			If the LLM passes {'data': '/local/file.fits', 'format': 'abspath'},
			convert it to {'data': '<uploaded_uid>', 'format': 'uid'}.
			"""
			if not isinstance(data_inputs, dict):
				return data_inputs, None

			data = data_inputs.get("data")
			fmt = data_inputs.get("format")

			if isinstance(data, str) and fmt == "abspath":
				path = Path(data)

				if path.exists():
					logger.info(
						f"--> CAESAR PREUPLOAD: uploading local file before app submit: {path}"
					)

					upload_result = self.client.upload_file(str(path))
			
					logger.info(
						f"--> CAESAR PREUPLOAD RESPONSE: {json.dumps(upload_result, indent=2, default=str)}"
					)
					
					file_uid = self.client._extract_file_uid(upload_result)

					if not file_uid:
						raise RuntimeError(
							"CAESAR upload succeeded but no file uid could be extracted "
							f"from response: {upload_result!r}"
						)

					normalized = dict(data_inputs)
					normalized["data"] = file_uid
					normalized["format"] = "uid"

					logger.info(
						f"--> CAESAR PREUPLOAD DONE: path={path}, uid={file_uid}"
					)

					return normalized, upload_result

			return data_inputs, None

		def _run_caesar_app(
			data_inputs: dict[str, Any],
			job_options: dict[str, Any] | None = None,
			tag: str = "",
			wait: bool = False,
			poll_seconds: float = 5.0,
			timeout_seconds: float = 600.0,
		) -> str:
			
			# - Normalize data inputs
			data_inputs, upload_result = _normalize_data_inputs_for_caesar(data_inputs)
			
			# IMPORTANT:
			# Agent-called CAESAR tools must not block the worker thread for long jobs.
			# Submit the job and return the job_id. Polling/output retrieval should be
			# handled by explicit follow-up workflow steps.		
			if wait:
				logger.warning(
					f"CAESAR tool {app_name} was called with wait=True. "
					"Overriding to wait=False to avoid blocking the worker graph."
				)
				wait = False
		
			logger.info(
				f"--> CAESAR TOOL CALLED: app={app_name}, "
				f"data_inputs={data_inputs}, "
				f"job_options={job_options or {}}, "
				f"tag={tag!r}, wait={wait}, "
				f"poll_seconds={poll_seconds}, timeout_seconds={timeout_seconds}"
			)
			
			result = self.client.submit_job(
				app=app_name,
				data_inputs=data_inputs,
				job_options=job_options or {},
				tag=tag,
			)
			
			logger.info(
				f"--> CAESAR TOOL SUBMIT RETURNED: app={app_name}, result={result}"
			)

			
			if wait:
				job_id = self.client._extract_job_id(result)
				if not job_id:
					return json.dumps(
						{
							"upload_result": upload_result,
							"submit_result": result,
							"wait_error": "Could not extract job_id from submit response.",
						},
						indent=2,
					)

				logger.info(
					f"--> CAESAR TOOL WAITING: app={app_name}, job_id={job_id}, "
					f"poll_seconds={poll_seconds}, timeout_seconds={timeout_seconds}"
				)

				status = self.client.wait_for_job(
					job_id=job_id,
					poll_seconds=poll_seconds,
					timeout_seconds=timeout_seconds,
				)

				return json.dumps(
					{
						"upload_result": upload_result,
						"submit_result": result,
						"final_status": status,
						"message": (
							"CAESAR job submitted asynchronously. "
							"Use the returned job_id to check status and retrieve outputs."
						),
					},
					indent=2,
				)

			return json.dumps(
				{
					"upload_result": upload_result,
					"submit_result": result,
					"message": (
						"CAESAR job submitted asynchronously. "
						"Use the returned job_id to check status and retrieve outputs."
					),
				},
				indent=2,
			)
		

		return StructuredTool.from_function(
			func=_run_caesar_app,
			name=tool_name,
			description=description,
			args_schema=CaesarRunAppArgs,
		)

	def _build_dynamic_tool_description(
		self,
		app_name: str,
		app_spec: dict[str, Any],
	) -> str:
		
		# - Top description
		lines = [
			f"Run the CAESAR REST application '{app_name}'.",
			"",
			"This tool submits a CAESAR REST job. The app name is implicit in this tool.",
		]
		
		# - Add detailed description
		summary = self._extract_description_text(app_spec)
		if summary:
			lines.extend([
				"",
				"Application description:",
				summary,
			])

		
		# - Add data inputs requirements
		input_text = self._format_input_requirements(app_spec.get("input_requirements", {}))
		if input_text:
			lines.extend([
				"",
				"Input requirements:",
				input_text,
			])

		# - Add tool call
		lines.extend([
			"",
			"Call arguments:",
			#"- data_inputs: CAESAR input data descriptor. It must usually contain:",
			#"  {'data': <input>, 'format': <format>}",
			#"  where format is one of:",
			#"  - 'uid': data is a file UID already registered in CAESAR, e.g. from upload API",
			#"  - 'abspath': data is an absolute path on the CAESAR storage filesystem",
			#"  - 'dataset': data is a dataset label supported by the CAESAR service",
			#"  For multiple inputs, use lists, e.g.:",
			#"  {'data': ['dataset_name', 'file_uid'], 'format': ['dataset', 'uid']}",
			"- data_inputs: CAESAR input data descriptor.",
			"  Preferred format:",
			"  {'data': '<file_uid>', 'format': 'uid'}",
			"  If a local path is provided as {'data': '<local_path>', 'format': 'abspath'},",
			"  this tool will upload the file to CAESAR REST first and replace the path",
			"  with the returned file uid before submitting the job.",
			"- job_options: app-specific execution parameters.",
			"  Place the listed CAESAR parameters inside job_options using their exact names.",
			"  Boolean/flag parameters listed as type=none should be passed as true when enabled.",
			"- tag: optional job label",
			"- wait: keep this false for normal agent workflows. Submit asynchronously and return the job_id. Do not set wait=true unless a caller explicitly requests blocking execution.",
		])
		
		# - Add tool call example
		lines.extend([
			"",
			"Example tool call arguments:",
			"{",
			"  'data_inputs': {'data': '<file_uid_or_path_or_dataset>', 'format': 'uid'},",
			"  'job_options': {'model': 'yolov11l_imgsize640', 'imgsize': 640, 'preprocessing': True, 'normalize': True, 'normmin': 0.0, 'normmax': 255.0, 'zscale': True, 'zscale-contrasts': '0.25:0.25:0.25', 'score-thr': 0.5, 'iou-thr': 0.5, 'merge-overlap-iou-thr-soft': 0.3, 'merge-overlap-iou-thr-hard': 0.8, 'save-plots': True, 'draw-class-label-in-caption': True},",
			"  'wait': False",
			"}",
		])
		
		# - Add job options
		job_options_spec = self._extract_job_options_spec(app_spec)
		param_text = self._format_app_parameters(job_options_spec)
		
		if param_text:
			lines.extend([
				"",
				"Parameters discovered from caesar-rest describe endpoint:",
				param_text,
			])

		lines.extend([
			"",
			#"The tool returns the job submission response. If wait=true, it also polls the job status.",
			"The tool returns the job submission response. In normal agent workflows it submits asynchronously and returns a job_id; wait=true is ignored to avoid blocking execution."
		])
		
		# - Add job output description
		output_text = self._format_job_outputs(app_spec.get("job_outputs", {}))
		if output_text:
			lines.extend([
			"",
			"Expected output products:",
			output_text,
		])
		
		# - Add app limitations
		limitations_text = self._format_limitations(app_spec.get("limitations", []))
		if limitations_text:
			lines.extend([
				"",
				"Limitations:",
				limitations_text,
			])

		return "\n".join(lines)

	@staticmethod
	def _extract_description_text(app_spec: dict[str, Any]) -> str:
		for key in ["description", "help", "summary", "doc", "documentation"]:
			value = app_spec.get(key)
			if isinstance(value, str) and value.strip():
				return value.strip()

		return ""

	
	@staticmethod
	def _looks_like_flat_parameter_spec(value: Any) -> bool:
		if not isinstance(value, dict) or not value:
			return False

		n_param_like = 0

		for item in value.values():
			if isinstance(item, dict) and (
				"type" in item
				or "description" in item
				or "mandatory" in item
				or "default" in item
				or "category" in item
			):
				n_param_like += 1

		return n_param_like > 0

	@staticmethod
	def _format_parameter_dict(
		params: dict[str, Any],
		indent: str = "",
		include_advanced: bool = False,
	) -> str:
		lines: list[str] = []

		grouped: dict[str, list[tuple[str, dict[str, Any]]]] = {}

		for name, spec in params.items():
			if not isinstance(spec, dict):
				continue

			if not include_advanced and int(spec.get("advanced", 0) or 0) != 0:
				continue

			category = str(spec.get("category") or "GENERAL")
			grouped.setdefault(category, []).append((name, spec))

		for category in sorted(grouped):
			lines.append(f"{indent}{category}:")

			for name, spec in sorted(grouped[category], key=lambda item: item[0]):
				type_name = spec.get("type", "unknown")
				description = spec.get("description", "")
				default = spec.get("default", None)
				mandatory = spec.get("mandatory", False)
				min_value = spec.get("min", None)
				max_value = spec.get("max", None)
				allowed_values = spec.get("allowed_values", None)

				bits = [f"type={type_name}"]

				if mandatory:
					bits.append("mandatory=true")

				if default not in [None, ""]:
					bits.append(f"default={default}")

				if min_value not in [None, ""]:
					bits.append(f"min={min_value}")

				if max_value not in [None, ""]:
					bits.append(f"max={max_value}")

				if allowed_values:
					bits.append(f"allowed_values={allowed_values}")

				if description:
					bits.append(str(description))

				lines.append(f"{indent}\t- {name}: " + " | ".join(bits))

		return "\n".join(lines).strip()
	
	@staticmethod
	def _format_app_parameters(app_spec: dict[str, Any]) -> str:
		if CaesarRestToolkit._looks_like_flat_parameter_spec(app_spec):
			return CaesarRestToolkit._format_parameter_dict(app_spec)

		candidate_keys = [
			"parameters",
			"params",
			"options",
			"job_options",
			"data_inputs",
			"inputs",
		]

		parts: list[str] = []

		for key in candidate_keys:
			value = app_spec.get(key)
			if not value:
				continue

			parts.append(f"{key}:")
			if isinstance(value, dict) and CaesarRestToolkit._looks_like_flat_parameter_spec(value):
				parts.append(CaesarRestToolkit._format_parameter_dict(value, indent="\t"))
			else:
				parts.append(CaesarRestToolkit._format_schema_fragment(value, indent="\t"))

		return "\n".join(parts).strip()


	@staticmethod
	def _extract_job_options_spec(app_spec: dict[str, Any]) -> dict[str, Any]:
		if isinstance(app_spec.get("job_options"), dict):
			return app_spec["job_options"]

		return app_spec

	@staticmethod
	def _format_job_outputs(job_outputs: Any) -> str:
		if not isinstance(job_outputs, dict) or not job_outputs:
			return ""

		lines: list[str] = []

		for name, spec in sorted(job_outputs.items()):
			if not isinstance(spec, dict):
				continue

			role = spec.get("role", "")
			parser = spec.get("parser", "")
			type_name = spec.get("type", "")
			path = spec.get("path") or spec.get("glob") or ""
			required = spec.get("required", False)
			description = spec.get("description", "")
			notes = spec.get("notes", "")

			bits = []
			if role:
				bits.append(f"role={role}")
			if parser:
				bits.append(f"parser={parser}")
			if type_name:
				bits.append(f"type={type_name}")
			if path:
				bits.append(f"path/glob={path}")
			if required:
				bits.append("required=true")

			lines.append(f"- {name}: " + " | ".join(bits))

			if description:
				lines.append(f"  description: {description}")

			if notes:
				lines.append(f"  notes: {notes}")

		return "\n".join(lines)


	@staticmethod
	def _format_input_requirements(input_requirements: Any) -> str:
		if not isinstance(input_requirements, dict) or not input_requirements:
			return ""

		lines: list[str] = []

		expected_data = input_requirements.get("expected_data")
		if expected_data:
			lines.append(f"- expected_data: {expected_data}")

		supported_formats = input_requirements.get("supported_formats")
		if supported_formats:
			lines.append(f"- supported_formats: {supported_formats}")

		notes = input_requirements.get("notes")
		if isinstance(notes, list) and notes:
			lines.append("- notes:")
			for item in notes:
				lines.append(f"  - {item}")
		elif isinstance(notes, str) and notes:
			lines.append(f"- notes: {notes}")

		return "\n".join(lines)

	@staticmethod
	def _format_limitations(limitations: Any) -> str:
		if not limitations:
			return ""

		if isinstance(limitations, list):
			return "\n".join(f"- {item}" for item in limitations)

		if isinstance(limitations, str):
			return f"- {limitations}"

		return json.dumps(limitations, indent=2, default=str)

	@staticmethod
	def _format_schema_fragment(value: Any, indent: str = "") -> str:
		if isinstance(value, dict):
			lines = []
			for key, item in value.items():
				if isinstance(item, dict):
					desc = item.get("description") or item.get("help") or item.get("doc") or ""
					default = item.get("default", None)
					required = item.get("required", None)
					type_name = item.get("type", None)

					bits = []
					if type_name is not None:
						bits.append(f"type={type_name}")
					if default is not None:
						bits.append(f"default={default}")
					if required is not None:
						bits.append(f"required={required}")
					if desc:
						bits.append(str(desc))

					suffix = " | ".join(bits)
					lines.append(f"{indent}- {key}: {suffix}".rstrip())
				else:
					lines.append(f"{indent}- {key}: {item}")
			return "\n".join(lines)

		if isinstance(value, list):
			lines = []
			for item in value:
				lines.append(f"{indent}- {item}")
			return "\n".join(lines)

		return f"{indent}{value}"

	@staticmethod
	def _safe_tool_name(name: str) -> str:
		name = name.strip().lower()
		name = re.sub(r"[^a-z0-9_]+", "_", name)
		name = re.sub(r"_+", "_", name).strip("_")
		return name or "app"

	def _caesar_list_apps(self) -> str:
		"""List CAESAR REST applications discovered from the cached app registry."""
		return json.dumps(sorted(self.app_specs.keys()), indent=2)

	def _caesar_describe_app(self, app: str) -> str:
		"""Return the CAESAR REST describe metadata for one application."""
		if app in self.app_specs:
			return json.dumps(self.app_specs[app], indent=2)

		spec = self.client.describe_app(app)
		return json.dumps(spec, indent=2)

	def _caesar_submit_job(
		self,
		app: str,
		data_inputs: dict[str, Any],
		job_options: dict[str, Any] | None = None,
		tag: str = "",
		wait: bool = False,
		poll_seconds: float = 5.0,
		timeout_seconds: float = 600.0,
	) -> str:
		"""Submit a generic CAESAR REST job by app name."""
		result = self.client.submit_job(
			app=app,
			data_inputs=data_inputs,
			job_options=job_options or {},
			tag=tag,
		)

		if wait:
			job_id = self.client._extract_job_id(result)
			if not job_id:
				return json.dumps(
					{
						"submit_result": result,
						"wait_error": "Could not extract job_id from submit response.",
					},
					indent=2,
				)

			status = self.client.wait_for_job(
				job_id=job_id,
				poll_seconds=poll_seconds,
				timeout_seconds=timeout_seconds,
			)

			return json.dumps(
				{
					"submit_result": result,
					"final_status": status,
				},
				indent=2,
			)

		return json.dumps(result, indent=2)

	def _caesar_get_job_status(self, job_id: str) -> str:
		"""Get the status of a CAESAR REST job."""
		return json.dumps(self.client.get_job_status(job_id), indent=2)

	def _caesar_wait_for_job(
		self,
		job_id: str,
		poll_seconds: float = 5.0,
		timeout_seconds: float = 600.0,
	) -> str:
		"""Poll a CAESAR REST job until it reaches a terminal state or timeout."""
		return json.dumps(
			self.client.wait_for_job(
				job_id=job_id,
				poll_seconds=poll_seconds,
				timeout_seconds=timeout_seconds,
			),
			indent=2,
		)

	def _caesar_get_job_output(self, job_id: str) -> str:
		"""Retrieve the output metadata or output payload for a completed CAESAR REST job."""
		result = self.client.get_job_output(job_id)
		if isinstance(result, str):
			return result
		return json.dumps(result, indent=2)

	def _caesar_cancel_job(self, job_id: str) -> str:
		"""Cancel a CAESAR REST job."""
		return json.dumps(self.client.cancel_job(job_id), indent=2)

	def _caesar_list_jobs(self) -> str:
		"""List CAESAR REST jobs visible to the service."""
		result = self.client.list_jobs()
		if isinstance(result, str):
			return result
		return json.dumps(result, indent=2)

	def _caesar_list_file_ids(self) -> str:
		"""List file ids known to CAESAR REST."""
		result = self.client.list_file_ids()
		if isinstance(result, str):
			return result
		return json.dumps(result, indent=2)

	def _caesar_list_datasets(self) -> str:
		"""List datasets known to CAESAR REST."""
		result = self.client.list_datasets()
		if isinstance(result, str):
			return result
		return json.dumps(result, indent=2)

	def _caesar_upload_file(self, path: str) -> str:
		"""Upload a local file to CAESAR REST and return the upload response."""
		result = self.client.upload_file(path)
		if isinstance(result, str):
			return result
		return json.dumps(result, indent=2)

