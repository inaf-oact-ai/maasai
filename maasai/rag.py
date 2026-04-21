from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(slots=True)
class RAGDocument:
	doc_id: str
	title: str
	text: str
	metadata: dict[str, Any] = field(default_factory=dict)


class PromptRAG:
	"""Stub for prompt-optimization retrieval.

	Replace this with your real ensemble retriever over prompt examples,
	tool docs, ontology notes, or astronomy-specific task templates.
	"""

	def retrieve(self, query: str, k: int = 5) -> list[RAGDocument]:
		_ = (query, k)
		return []
