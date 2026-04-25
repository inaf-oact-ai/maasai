from __future__ import annotations

from .agents import AgentFactory
from .config import Settings
from .rag import PromptRAG


class NodeContext:
	def __init__(
		self, *, 
		settings: Settings, 
		rag: PromptRAG, 
		agents: AgentFactory
	) -> None:
		""" Node context """
		self.settings = settings
		self.rag = rag
		self.agents = agents
