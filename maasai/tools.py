from __future__ import print_function
from __future__ import annotations

##################################################
###          MODULE IMPORT
##################################################
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

# - MAASAI MODULES
from .caesar_tools import CaesarRestToolkit, CaesarRestClient, CaesarRestToolCache
from maasai import logger
		
##################################################
###          TOOL REGISTRY
##################################################
class AstronomyToolRegistry:
	def __init__(
		self,
		caesar_base_url: str | None = "http://localhost:8080",
		caesar_api_prefix: str = "/caesar/api/v1.0",
		caesar_cache_path: str | Path | None = ".maasai/cache/caesar_apps.json",
		caesar_refresh_tools: bool = False,
		caesar_enable_dynamic_app_tools: bool = True,
		caesar_timeout_seconds: float = 30.0,
		caesar_cache_ttl_seconds: float = 86400.0,
		caesar_auth_token: str | None = None,
		caesar_enabled_apps: list[str] | None = None,
	) -> None:
		self.caesar: CaesarRestToolkit | None = None

		if caesar_base_url:
			client = CaesarRestClient(
				base_url=caesar_base_url,
				timeout_seconds=caesar_timeout_seconds,
				api_prefix=caesar_api_prefix,
				auth_token=caesar_auth_token,
			)

			cache = CaesarRestToolCache(
				cache_path=caesar_cache_path,
				ttl_seconds=caesar_cache_ttl_seconds,
			)

			self.caesar = CaesarRestToolkit(
				client=client,
				cache=cache,
				refresh_tools=caesar_refresh_tools,
				enable_dynamic_app_tools=caesar_enable_dynamic_app_tools,
				enabled_apps=caesar_enabled_apps,
			)
			
		if self.caesar is not None:
			logger.info(
				f"CAESAR REST toolkit enabled: "
				f"base_url={caesar_base_url}, "
				f"api_prefix={caesar_api_prefix}, "
				f"cache_path={caesar_cache_path}, "
				f"dynamic_tools={caesar_enable_dynamic_app_tools}"
			)
		else:
			logger.info("CAESAR REST toolkit disabled.")	


	
	def get_tools(self) -> list[Any]:
		tools: list[Any] = []

		if self.caesar is not None:
			tools.extend(self.caesar.get_tools())

		tools.append(
			StructuredTool.from_function(
				func=self._call_mcp_tool,
				name="call_mcp_tool",
				description="Call an MCP tool exposed by a remote astronomy service.",
			)
		)

		logger.info(
			"AstronomyToolRegistry loaded tools: "
			+ ", ".join(getattr(item, "name", str(item)) for item in tools)
		)
		
		print("\n=== TOOL INVENTORY ===")
		print(f"Loaded tools: {len(tools)}")
		for item in tools:
			print(f"- {getattr(item, 'name', str(item))}")
		print("======================\n")
		
		debug_tool_name = "caesar_run_caesar_yolo"

		for item in tools:
			if getattr(item, "name", None) == debug_tool_name:
				print(f"\n=== TOOL DESCRIPTION: {debug_tool_name} ===")
				print(item.description)
				print("\n=== TOOL ARGS SCHEMA ===")
				print(item.args_schema.model_json_schema() if item.args_schema else "<none>")
				print("========================\n")
				break
				
		#debug_app_name = "caesar-yolo"
		#if self.caesar is not None:
		#	app_spec = self.caesar.app_specs.get(debug_app_name)
		#	print(f"\n=== RAW CAESAR APP SPEC: {debug_app_name} ===")
		#	print(json.dumps(app_spec, indent=2, default=str))
		#	print("============================================\n")

		return tools

	def _call_mcp_tool(self, server: str, tool_name: str, arguments: dict) -> str:
		"""Call an MCP tool exposed by a remote astronomy service."""
		return f"MCP call to {server}.{tool_name} with arguments={arguments}"
