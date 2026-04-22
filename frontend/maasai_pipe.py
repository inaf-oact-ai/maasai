from __future__ import print_function
from __future__ import annotations

##################################################
###          MODULE IMPORT
##################################################
# - STANDARD MODULES
from typing import Any, Dict, Optional
import uuid
import os
import httpx

# - LLANGCHAIN MODULES
from pydantic import BaseModel, Field

##################################################
###          PIPE CLASS
##################################################
class Pipe:
	class Valves(BaseModel):
		FASTAPI_BASE_URL: str = Field(
			default="http://127.0.0.1:8000",
			description="Base URL for the MAASAI FastAPI server.",
		)
		API_PATH: str = Field(
			default="/chat",
			description="Path for the MAASAI chat endpoint.",
		)

		CONNECT_TIMEOUT: float = Field(default=5.0, description="HTTP connect timeout (s).")
		READ_TIMEOUT: Optional[float] = Field(default=120.0, description="HTTP read timeout (s).")
		WRITE_TIMEOUT: float = Field(default=30.0, description="HTTP write timeout (s).")
		POOL_TIMEOUT: float = Field(default=5.0, description="HTTP connection pool timeout (s).")

		DEBUG: bool = Field(default=False, description="Enable debug prints in OpenWebUI logs.")

	def __init__(self):
		self.valves = self.Valves()

	def pipes(self):
		return [
			{
				"id": "maasai.pipe",
				"name": "MAASAI",
			}
		]

	# ------------------------------------------------------------------
	# Helpers
	# ------------------------------------------------------------------
	def _debug(self, *args):
		if self.valves.DEBUG:
			print("[MAASAI PIPE]", *args)

	def _extract_last_user_message(self, body: Dict[str, Any]) -> str:
		msgs = body.get("messages") or []
		for msg in reversed(msgs):
			if msg.get("role") == "user":
				content = msg.get("content")
				if isinstance(content, str):
					return content.strip()

				if isinstance(content, list):
					parts = []
					for item in content:
						if (
							isinstance(item, dict)
							and item.get("type") == "text"
							and isinstance(item.get("text"), str)
						):
							parts.append(item["text"])
					if parts:
						return "\n".join(parts).strip()

		if isinstance(body.get("prompt"), str):
			return body["prompt"].strip()

		return ""

	def _extract_thread_id(self, body: Dict[str, Any], __metadata__: Dict[str, Any]) -> str:
		# Prefer OpenWebUI metadata if available
		for key in ("chat_id", "conversation_id", "session_id"):
			value = __metadata__.get(key)
			if value:
				return str(value)

		for key in ("chat_id", "conversation_id", "thread_id", "session_id"):
			value = body.get(key)
			if value:
				return str(value)

		return f"maasai-{uuid.uuid4()}"

	def _maybe_add_path(self, attachments: list[dict[str, str]], candidate: Any):
		if not isinstance(candidate, str):
			return

		path = candidate.strip()
		if not path:
			return

		# Accept absolute paths directly
		if os.path.isabs(path):
			attachments.append({"path": path})
			return

		# Also accept file:// URLs
		if path.startswith("file://"):
			local_path = path[len("file://"):].strip()
			if local_path:
				attachments.append({"path": local_path})
			return

	def _extract_attachments_from_content_list(self, content: list[Any]) -> list[dict[str, str]]:
		attachments: list[dict[str, str]] = []

		for item in content:
			if not isinstance(item, dict):
				continue

			# Text chunks are ignored here
			item_type = item.get("type")

			# Common custom path-style fields one might encounter
			for key in ("path", "file_path", "local_path"):
				self._maybe_add_path(attachments, item.get(key))

			# image_url may occasionally hold file:// paths
			image_url = item.get("image_url")
			if isinstance(image_url, dict):
				self._maybe_add_path(attachments, image_url.get("url"))
			elif isinstance(image_url, str):
				self._maybe_add_path(attachments, image_url)

			# Some UIs attach file metadata objects
			file_obj = item.get("file")
			if isinstance(file_obj, dict):
				for key in ("path", "file_path", "local_path"):
					self._maybe_add_path(attachments, file_obj.get(key))

			# Generic nested sources
			source = item.get("source")
			if isinstance(source, dict):
				for key in ("path", "file_path", "local_path", "url"):
					self._maybe_add_path(attachments, source.get(key))

		return attachments

	def _extract_attachments(self, body: Dict[str, Any]) -> list[dict[str, str]]:
		"""
		Best-effort attachment extraction for common OpenWebUI payload shapes.

		Important:
		- MAASAI currently expects local filesystem paths.
		- This works only if the OpenWebUI runtime can access the same filesystem paths
		  as the MAASAI server, or if file:// paths resolve correctly.
		"""
		attachments: list[dict[str, str]] = []

		# 1) Look in the latest user message content list
		msgs = body.get("messages") or []
		for msg in reversed(msgs):
			if msg.get("role") != "user":
				continue

			content = msg.get("content")
			if isinstance(content, list):
				attachments.extend(self._extract_attachments_from_content_list(content))

			# Also inspect top-level per-message file metadata if present
			if isinstance(msg.get("files"), list):
				for f in msg["files"]:
					if isinstance(f, dict):
						for key in ("path", "file_path", "local_path", "url"):
							self._maybe_add_path(attachments, f.get(key))

			break

		# 2) Inspect top-level files on request body if present
		if isinstance(body.get("files"), list):
			for f in body["files"]:
				if isinstance(f, dict):
					for key in ("path", "file_path", "local_path", "url"):
						self._maybe_add_path(attachments, f.get(key))

		# 3) Deduplicate
		seen = set()
		deduped: list[dict[str, str]] = []
		for item in attachments:
			path = item.get("path")
			if path and path not in seen:
				seen.add(path)
				deduped.append(item)

		return deduped

	def _format_maasai_response(self, data: Dict[str, Any]) -> str:
		status = data.get("status")
		result = data.get("result") or {}

		if status == "completed":
			assistant = data.get("assistant")
			if assistant:
				return assistant

			answer = result.get("answer")
			if answer:
				return answer

			message = result.get("message")
			if message:
				return message

			return "MAASAI completed the request but returned no text."

		if status == "interrupt":
			payload = data.get("payload", {})
			return f"MAASAI requires approval before continuing.\n\nPayload:\n{payload}"

		if status == "error":
			err = data.get("error")
			if err:
				return f"MAASAI error: {err}"

		if result:
			message = result.get("message")
			if message:
				return message

		return str(data)

	# ------------------------------------------------------------------
	# Main entrypoint
	# ------------------------------------------------------------------
	def pipe(self, body: Dict[str, Any], __metadata__: dict):
		base = (self.valves.FASTAPI_BASE_URL or "").rstrip("/")
		path = (self.valves.API_PATH or "").lstrip("/")
		url = f"{base}/{path}"

		user_msg = self._extract_last_user_message(body)
		if not user_msg:
			return "No prompt provided. Please type your question."

		thread_id = self._extract_thread_id(body, __metadata__)
		attachments = self._extract_attachments(body)

		payload = {
			"thread_id": thread_id,
			"message": user_msg,
			"attachments": attachments,
		}

		self._debug("URL:", url)
		self._debug("thread_id:", thread_id)
		self._debug("message:", user_msg)
		self._debug("attachments:", attachments)

		timeout = httpx.Timeout(
			connect=self.valves.CONNECT_TIMEOUT,
			read=self.valves.READ_TIMEOUT,
			write=self.valves.WRITE_TIMEOUT,
			pool=self.valves.POOL_TIMEOUT,
		)

		try:
			with httpx.Client(timeout=timeout) as client:
				resp = client.post(url, json=payload)
				resp.raise_for_status()
				data = resp.json()
				self._debug("response:", data)
		except httpx.HTTPStatusError as e:
			try:
				err_text = e.response.text
			except Exception:
				err_text = str(e)
			return f"MAASAI API error {e.response.status_code}: {err_text}"
		except Exception as e:
			return f"Error contacting MAASAI API: {e}"

		return self._format_maasai_response(data)
