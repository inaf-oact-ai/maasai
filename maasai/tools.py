from __future__ import print_function
from __future__ import annotations

##################################################
###          MODULE IMPORT
##################################################
# - STANDARD MODULES

# - LLANGCHAIN MODULES
from langchain.tools import tool

# - LOGGER
from maasai import logger

##################################################
###          TOOLS
##################################################
class AstronomyToolRegistry:
	@tool
	def query_caesar_rest(self, endpoint: str, payload: dict) -> str:
		"""Call an astronomy REST endpoint such as Caesar REST."""
		return f"REST call to {endpoint} with payload keys={list(payload.keys())}"

	@tool
	def call_mcp_tool(self, server: str, tool_name: str, arguments: dict) -> str:
		"""Call an MCP tool exposed by a remote astronomy service."""
		return f"MCP call to {server}.{tool_name} with arguments={arguments}"
