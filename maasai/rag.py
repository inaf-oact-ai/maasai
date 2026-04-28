from __future__ import print_function
from __future__ import annotations

##################################################
###          MODULE IMPORT
##################################################
# - STANDARD MODULES
from dataclasses import dataclass, field
from typing import Any
import os
import re
import requests

# - LANGCHAIN MODULES
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient

# - MAASAI MODULES
from .keywords import KEYWORDS_DEFAULT_RADIO
from .keywords import KEYWORDS_DEFAULT_SOLAR
from .keywords import KEYWORDS_DEFAULT_EXOPLANETS


###################################################
###         RAG CLASSES
###################################################
@dataclass(slots=True)
class RAGDocument:
	"""Retrieved document chunk with citation-ready metadata."""

	doc_id: str
	title: str
	text: str
	metadata: dict[str, Any] = field(default_factory=dict)
	score: float | None = None
	collection: str | None = None


@dataclass(slots=True)
class PlannerRAGDomain:
	"""Mapping between an astronomy domain and Qdrant collections."""

	name: str
	keywords: list[str] = field(default_factory=list)
	collections: list[str] = field(default_factory=list)


@dataclass(slots=True)
class PlannerRAGSettings:
	"""Runtime settings for MAASAI planner RAG."""

	enabled: bool = field(
		default_factory=lambda: os.getenv("MAASAI_PLANNER_RAG_ENABLED", "false").lower() == "true"
	)
	backend: str = field(
		default_factory=lambda: os.getenv("MAASAI_RAG_BACKEND", "llama-index-service")
	)
	llamaindex_service_url: str = field(
		default_factory=lambda: os.getenv(
			"MAASAI_LLAMA_INDEX_RAG_URL",
			"http://127.0.0.1:8010",
		)
	)
	llamaindex_num_queries: int = field(
		default_factory=lambda: int(os.getenv("MAASAI_LLAMA_INDEX_NUM_QUERIES", "4"))
	)
	request_timeout: float = field(
		default_factory=lambda: float(os.getenv("MAASAI_RAG_REQUEST_TIMEOUT", "60"))
	)
	fallback_to_local: bool = field(
		default_factory=lambda: os.getenv("MAASAI_RAG_FALLBACK_TO_LOCAL", "false").lower() == "true"
	)
	qdrant_url: str = field(
		default_factory=lambda: os.getenv("MAASAI_QDRANT_URL", "http://localhost:6333")
	)
	embedding_model: str = field(
		#default_factory=lambda: os.getenv("MAASAI_RAG_EMBEDDING_MODEL", "mixedbread-ai/mxbai-embed-large-v1")
		default_factory=lambda: os.getenv("MAASAI_RAG_EMBEDDING_MODEL", "Qwen/Qwen3-Embedding-0.6B")
	)
	top_k_per_collection: int = field(
		default_factory=lambda: int(os.getenv("MAASAI_RAG_TOP_K_PER_COLLECTION", "4"))
	)
	final_top_k: int = field(
		default_factory=lambda: int(os.getenv("MAASAI_RAG_FINAL_TOP_K", "8"))
	)
	always_include_collections: list[str] = field(
		default_factory=lambda: [
			item.strip()
			for item in os.getenv("MAASAI_RAG_BASE_COLLECTIONS", "annreviews").split(",")
			if item.strip()
		]
	)
	default_collections: list[str] = field(
		default_factory=lambda: [
			item.strip()
			for item in os.getenv(
				"MAASAI_RAG_DEFAULT_COLLECTIONS",
				"radiopapers,radiobooks,annreviews",
			).split(",")
			if item.strip()
		]
	)
	score_threshold: float | None = field(
		default_factory=lambda: (
			float(os.getenv("MAASAI_RAG_SCORE_THRESHOLD"))
			if os.getenv("MAASAI_RAG_SCORE_THRESHOLD")
			else None
		)
	)
	content_payload_key: str = field(
		default_factory=lambda: os.getenv("MAASAI_RAG_CONTENT_PAYLOAD_KEY", "_node_content")
	)
	metadata_payload_key: str | None = field(
		default_factory=lambda: os.getenv("MAASAI_RAG_METADATA_PAYLOAD_KEY") or None
	)
	


DEFAULT_DOMAINS: list[PlannerRAGDomain] = [
	PlannerRAGDomain(
		name="radio",
		keywords=list(set(KEYWORDS_DEFAULT_RADIO)),
		collections=["radiopapers", "radiobooks", "annreviews"],
	),
	PlannerRAGDomain(
		name="solar",
		keywords=list(set(KEYWORDS_DEFAULT_SOLAR)),
		collections=["solar-papers", "annreviews"],
	),
	PlannerRAGDomain(
		name="exoplanet",
		keywords=list(set(KEYWORDS_DEFAULT_EXOPLANETS)),
		collections=["exoplanet-papers", "annreviews"],
	),
]


class PromptRAG:
	"""Planner-oriented LangChain/Qdrant RAG.

	This class intentionally retrieves chunks and metadata only. It does not
	synthesize an answer. The planner agent receives the retrieved context and
	uses it only to improve task decomposition.
	"""

	def __init__(
		self,
		settings: PlannerRAGSettings | None = None,
		domains: list[PlannerRAGDomain] | None = None,
	) -> None:
		self.settings = settings or PlannerRAGSettings()
		self.domains = domains or DEFAULT_DOMAINS

		self._client: QdrantClient | None = None
		self._embeddings: HuggingFaceEmbeddings | None = None
		self._vectorstores: dict[str, QdrantVectorStore] = {}

	def retrieve(
		self,
		query: str,
		k: int = 5,
		domain_hint: str | None = None,
		collections: list[str] | None = None,
	) -> list[RAGDocument]:
		"""Retrieve planner context from dynamically selected collections."""

		query = (query or "").strip()
		if not query:
			return []

		selected_collections = self.select_collections(
			query=query,
			domain_hint=domain_hint,
			collections=collections,
		)
		
		if not selected_collections:
			selected_collections = list(self.settings.default_collections)

		top_k_per_collection = max(1, self.settings.top_k_per_collection)
		final_k = max(1, k or self.settings.final_top_k)
		backend = (self.settings.backend or "langchain-qdrant").strip().lower()

		# - Use llama-index service RAG
		if backend == "llama-index-service":
			try:
				return self._retrieve_from_llama_index_service(...)
			except Exception as exc:
				if self.settings.fallback_to_local:
					logger.warning(f"LlamaIndex RAG service failed, falling back to local RAG: {exc}")
				else:
					raise

		# - Use langchain RAG (retrieval not working fully with llama-index populated Qdrant collections)
		results: list[RAGDocument] = []

		for collection_name in selected_collections:
			try:
				results.extend(
					self._retrieve_from_collection_by_backend(
						query=query,
						collection_name=collection_name,
						k=top_k_per_collection,
					)
				)
			except Exception:
				# Do not fail the planner if one optional collection is missing
				# or temporarily unavailable.
				continue

		results = self._deduplicate(results)
		results = self._sort_results(results)

		if self.settings.score_threshold is not None:
			results = [
				item
				for item in results
				if item.score is None or item.score >= self.settings.score_threshold
			]

		return results[:final_k]

	def select_collections(
		self,
		query: str,
		domain_hint: str | None = None,
		collections: list[str] | None = None,
	) -> list[str]:
		"""Select Qdrant collections from query/domain hints."""

		if collections:
			return self._unique_preserve_order(collections)

		selected: list[str] = []

		if domain_hint:
			for domain in self.domains:
				if domain.name.lower() == domain_hint.lower():
					selected.extend(domain.collections)

		query_norm = self._normalize(query)

		for domain in self.domains:
			if self._domain_matches(query_norm, domain):
				selected.extend(domain.collections)

		selected.extend(self.settings.always_include_collections)

		if not selected:
			selected.extend(self.settings.default_collections)

		return self._unique_preserve_order(selected)

	def _retrieve_from_collection_by_backend(
		self,
		query: str,
		collection_name: str,
		k: int,
	) -> list[RAGDocument]:
		backend = (self.settings.backend or "langchain-qdrant").strip().lower()

		if backend == "langchain-qdrant":
			return self._retrieve_from_collection(
				query=query,
				collection_name=collection_name,
				k=k,
			)

		if backend == "raw-qdrant":
			return self._retrieve_from_collection_raw(
				query=query,
				collection_name=collection_name,
				k=k,
			)

		raise ValueError(
			f"Unsupported RAG backend={self.settings.backend!r}. "
			"Supported values are: langchain-qdrant, raw-qdrant."
		)

	def _retrieve_from_collection(
		self,
		query: str,
		collection_name: str,
		k: int,
	) -> list[RAGDocument]:
		"""Retrieve using LangChain QdrantVectorStore.

		This is Option A. For LlamaIndex-created Qdrant collections,
		content_payload_key should usually be "_node_content". The returned
		page_content is then a serialized LlamaIndex TextNode, which we parse
		to recover clean text and metadata.
		"""
		vectorstore = self._get_vectorstore(collection_name)

		pairs: list[tuple[Document, float]] = vectorstore.similarity_search_with_score(
			query,
			k=k,
		)

		out: list[RAGDocument] = []

		for doc, score in pairs:
			metadata = dict(doc.metadata or {})
			raw_text = doc.page_content or ""

			text = raw_text

			parsed = self._parse_serialized_payload(raw_text)
			if isinstance(parsed, dict):
				parsed_text = self._extract_text_from_serialized_payload(raw_text)
				if parsed_text:
					text = parsed_text

				parsed_metadata = parsed.get("metadata")
				if isinstance(parsed_metadata, dict):
					metadata.update(parsed_metadata)

				for key in [
					"id_",
					"mimetype",
					"start_char_idx",
					"end_char_idx",
					"class_name",
				]:
					value = parsed.get(key)
					if value is not None:
						metadata.setdefault(key, value)

				if parsed.get("id_"):
					metadata.setdefault("node_id", parsed.get("id_"))

			metadata["collection"] = collection_name
			metadata["score"] = score

			doc_id = self._make_doc_id(
				doc=doc,
				metadata=metadata,
				collection_name=collection_name,
			)
			title = self._extract_title(metadata=metadata, fallback=doc_id)

			out.append(
				RAGDocument(
					doc_id=doc_id,
					title=title,
					text=text,
					metadata=metadata,
					score=score,
					collection=collection_name,
				)
			)

		return out


	def _retrieve_from_collection_raw(
		self,
		query: str,
		collection_name: str,
		k: int,
	) -> list[RAGDocument]:
		query_vector = self._embed_query(query)

		points = self._get_client().query_points(
			collection_name=collection_name,
			query=query_vector,
			limit=k,
			with_payload=True,
			with_vectors=False,
		)

		out: list[RAGDocument] = []

		for point in points.points:
			payload = dict(point.payload or {})
			score = getattr(point, "score", None)
	
			text = self._extract_text_from_payload(payload)
			metadata = self._extract_metadata_from_payload(payload)
	
			metadata["_id"] = str(point.id)
			metadata["_collection_name"] = collection_name
			metadata["collection"] = collection_name
			metadata["score"] = score

			doc_id = self._make_doc_id_from_payload(
				point_id=str(point.id),
				payload=payload,
				metadata=metadata,
				collection_name=collection_name,
			)
			title = self._extract_title(metadata=metadata, fallback=doc_id)

			out.append(
				RAGDocument(
					doc_id=doc_id,
					title=title,
					text=text,
					metadata=metadata,
					score=score,
					collection=collection_name,
				)
			)

		return out


	def _retrieve_from_llama_index_service(
		self,
		query: str,
		collections: list[str],
		k: int,
	) -> list[RAGDocument]:
		
		url = self.settings.llamaindex_service_url.rstrip("/") + "/api/retrieve"

		payload = {
			"query": query,
			"collections": collections,
			"similarity_top_k": k,
			"num_queries": self.settings.llamaindex_num_queries,
			"include_text": True,
		}

		response = requests.post(
			url,
			json=payload,
			timeout=self.settings.request_timeout,
		)
		response.raise_for_status()

		data = response.json()

		if data.get("status") != 0:
			raise RuntimeError(
				f"llama-index-rag retrieval failed: {data.get('message')}"
			)

		docs: list[RAGDocument] = []

		for item in data.get("documents", []):
			metadata = dict(item.get("metadata") or {})

			collection = (
				item.get("collection")
				or metadata.get("collection")
				or metadata.get("_collection_name")
			)

			doc_id = str(
				item.get("doc_id")
				or metadata.get("node_id")
				or metadata.get("doc_id")
				or metadata.get("source_id")
				or metadata.get("doi")
				or metadata.get("arxiv_id")
				or metadata.get("file_name")
				or ""
			)

			title = str(
				item.get("title")
				or metadata.get("title")
				or metadata.get("paper_title")
				or metadata.get("document_title")
				or metadata.get("file_name")
				or doc_id
			)

			docs.append(
				RAGDocument(
					doc_id=doc_id,
					title=title,
					text=str(item.get("text") or ""),
					score=item.get("score"),
					collection=collection,
					metadata=metadata,
				)
			)

		return docs

	def _get_vectorstore(self, collection_name: str) -> QdrantVectorStore:
		if collection_name in self._vectorstores:
			return self._vectorstores[collection_name]

		kwargs = {
			"client": self._get_client(),
			"collection_name": collection_name,
			"embedding": self._get_embeddings(),
			"content_payload_key": self.settings.content_payload_key,
		}

		if self.settings.metadata_payload_key:
			kwargs["metadata_payload_key"] = self.settings.metadata_payload_key

		vectorstore = QdrantVectorStore(**kwargs)

		#vectorstore = QdrantVectorStore(
		#	client=self._get_client(),
		#	collection_name=collection_name,
		#	embedding=self._get_embeddings(),
		#)
		self._vectorstores[collection_name] = vectorstore
		return vectorstore

	def _get_client(self) -> QdrantClient:
		if self._client is None:
			self._client = QdrantClient(url=self.settings.qdrant_url)
		return self._client

	def _get_embeddings(self) -> HuggingFaceEmbeddings:
		if self._embeddings is None:
			self._embeddings = HuggingFaceEmbeddings(
				model_name=self.settings.embedding_model
			)
		return self._embeddings

	def _domain_matches(self, query_norm: str, domain: PlannerRAGDomain) -> bool:
		for keyword in domain.keywords:
			keyword_norm = self._normalize(keyword)
			if keyword_norm and keyword_norm in query_norm:
				return True
		return False

	def _normalize(self, text: str) -> str:
		return re.sub(r"\s+", " ", text.lower()).strip()

	def _unique_preserve_order(self, values: list[str]) -> list[str]:
		seen: set[str] = set()
		out: list[str] = []

		for value in values:
			value = value.strip()
			if not value or value in seen:
				continue
			seen.add(value)
			out.append(value)

		return out

	def _make_doc_id(
		self,
		doc: Document,
		metadata: dict[str, Any],
		collection_name: str,
	) -> str:
		for key in [
			"node_id",
			"id",
			"id_",
			"doc_id",
			"document_id",
			"source_id",
			"file_path",
			"filepath",
			"file_name",
			"arxiv_id",
			"doi",
			"url",
		]:
			value = metadata.get(key)
			if value:
				page = metadata.get("page_label") or metadata.get("page") or metadata.get("page_number")
				if page is not None:
					return f"{collection_name}:{value}:p{page}"
				return f"{collection_name}:{value}"

		content_hash = abs(hash(doc.page_content or ""))
		return f"{collection_name}:chunk:{content_hash}"

	def _extract_title(self, metadata: dict[str, Any], fallback: str) -> str:
		for key in [
			"title",
			"paper_title",
			"document_title",
			"book_title",
			"container_title",
			"file_name",
		]:
			value = metadata.get(key)
			if value:
				return str(value)

		return fallback

	def _deduplicate(self, docs: list[RAGDocument]) -> list[RAGDocument]:
		seen: set[str] = set()
		out: list[RAGDocument] = []

		for doc in docs:
			key = doc.doc_id
			if key in seen:
				continue
			seen.add(key)
			out.append(doc)

		return out

	def _sort_results(self, docs: list[RAGDocument]) -> list[RAGDocument]:
		# LangChain/Qdrant can expose distance-like scores depending on the
		# vector configuration. In many setups lower means closer. However,
		# your previous LlamaIndex service treated scores as similarity-like.
		# Keep ordering stable and conservative here: prefer non-None scores,
		# then sort descending. If your Qdrant collection returns distances,
		# flip this ordering after checking observed values.
		return sorted(
			docs,
			key=lambda item: item.score if item.score is not None else float("-inf"),
			reverse=True,
		)
		
		
	def _embed_query(self, query: str) -> list[float]:
		embeddings = self._get_embeddings()

		if hasattr(embeddings, "embed_query"):
			return embeddings.embed_query(query)

		# Defensive fallback for embedding wrappers exposing only embed_documents.
		return embeddings.embed_documents([query])[0]


	def _extract_text_from_payload(self, payload: dict[str, Any]) -> str:
		"""Extract chunk text from raw Qdrant payload.

		Supports LangChain-style, LlamaIndex-style, and custom layouts.
		"""

		candidate_keys = [
			self.settings.content_payload_key,
			"text",
			"_text",
			"page_content",
			"content",
			"chunk",
			"document",
		]

		for key in candidate_keys:
			if not key:
				continue

			value = payload.get(key)
			if isinstance(value, str) and value.strip():
				return value.strip()

		# LlamaIndex often stores useful values under nested dictionaries.
		for nested_key in [
			"_node_content",
			"node_content",
			"_data",
			"data",
			"metadata",
		]:
			value = payload.get(nested_key)

			if isinstance(value, str):
				text = self._extract_text_from_serialized_payload(value)
				if text:
					return text

			if isinstance(value, dict):
				text = self._extract_text_from_payload(value)
				if text:
					return text

		return ""


	def _extract_metadata_from_payload(self, payload: dict[str, Any]) -> dict[str, Any]:
		"""Extract and flatten useful metadata from raw Qdrant payload."""

		metadata: dict[str, Any] = {}

		if self.settings.metadata_payload_key:
			value = payload.get(self.settings.metadata_payload_key)
			if isinstance(value, dict):
				metadata.update(value)

		# Common LlamaIndex metadata keys.
		for key in [
			"metadata",
			"extra_info",
			"_node_metadata",
			"node_metadata",
			"doc_metadata",
		]:
			value = payload.get(key)
			if isinstance(value, dict):
				metadata.update(value)

		# Preserve all simple top-level payload fields too.
		for key, value in payload.items():
			if key in {
				"_node_content",
				"node_content",
				"_data",
				"data",
				"vector",
			}:
				continue

			if isinstance(value, (str, int, float, bool)) or value is None:
				metadata.setdefault(key, value)
			elif isinstance(value, list):
				metadata.setdefault(key, value)

		# Try to parse LlamaIndex serialized node content for metadata.
		for key in ["_node_content", "node_content"]:
			value = payload.get(key)
			parsed = self._parse_serialized_payload(value)
			if isinstance(parsed, dict):
				for meta_key in ["metadata", "extra_info"]:
					meta_value = parsed.get(meta_key)
					if isinstance(meta_value, dict):
						metadata.update(meta_value)

		return metadata


	def _extract_text_from_serialized_payload(self, value: str) -> str:
		parsed = self._parse_serialized_payload(value)
		if not isinstance(parsed, dict):
			return ""

		for key in [
			"text",
			"_text",
			"text_resource",
			"content",
			"page_content",
		]:
			candidate = parsed.get(key)

			if isinstance(candidate, str) and candidate.strip():
				return candidate.strip()

			if isinstance(candidate, dict):
				for subkey in ["text", "content"]:
					subvalue = candidate.get(subkey)
					if isinstance(subvalue, str) and subvalue.strip():
						return subvalue.strip()

		return ""


	def _parse_serialized_payload(self, value: Any) -> Any:
		if not isinstance(value, str) or not value.strip():
			return None

		try:
			import json
			return json.loads(value)
		except Exception:
			return None


	def _make_doc_id_from_payload(
		self,
		point_id: str,
		payload: dict[str, Any],
		metadata: dict[str, Any],
		collection_name: str,
	) -> str:
		for key in [
			"node_id",
			"id",
			"id_",
			"doc_id",
			"document_id",
			"source_id",
			"file_path",
			"filepath",
			"file_name",
			"arxiv_id",
			"doi",
			"url",
		]:
			value = metadata.get(key) or payload.get(key)
			if value:
				page = (
					metadata.get("page_label")
					or metadata.get("page")
					or metadata.get("page_number")
					or payload.get("page_label")
					or payload.get("page")
					or payload.get("page_number")
				)
				if page is not None:
					return f"{collection_name}:{value}:p{page}"
				return f"{collection_name}:{value}"

		return f"{collection_name}:{point_id}"	
		
	def debug_sample_payload(
		self,
		collection_name: str,
		limit: int = 1,
	) -> list[dict[str, Any]]:
		points, _ = self._get_client().scroll(
			collection_name=collection_name,
			limit=limit,
			with_payload=True,
			with_vectors=False,
		)

		return [
			{
				"id": str(point.id),
				"payload": point.payload,
			}
			for point in points
		]	
