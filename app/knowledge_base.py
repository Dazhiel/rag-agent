"""
Knowledge base built on Milvus hybrid retrieval with BGE-M3 embeddings.
"""
import hashlib
import os
import uuid
from dataclasses import dataclass
from datetime import datetime
from typing import Dict, List, Optional

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from app.config import RAGConfig
from app.loaders import DocumentLoader


def _project_root() -> str:
    return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _project_path(path: str) -> str:
    if os.path.isabs(path):
        return path
    return os.path.join(_project_root(), path)


class DedupService:
    """MD5 based content deduplication persisted in a local record file."""

    def __init__(self, record_path: str):
        self.record_path = record_path
        self._records: set = self._load_records()

    def _load_records(self) -> set:
        os.makedirs(os.path.dirname(self.record_path), exist_ok=True)
        if not os.path.exists(self.record_path):
            open(self.record_path, "w", encoding="utf-8").close()
            return set()
        with open(self.record_path, "r", encoding="utf-8") as f:
            return {line.strip() for line in f if line.strip()}

    @staticmethod
    def _md5(text: str) -> str:
        return hashlib.md5(text.encode("utf-8")).hexdigest()

    def exists(self, content: str) -> bool:
        return self._md5(content) in self._records

    def mark(self, content: str) -> None:
        md5_hex = self._md5(content)
        self._records.add(md5_hex)
        with open(self.record_path, "a", encoding="utf-8") as f:
            f.write(md5_hex + "\n")


class TextSplitter:
    """Text splitter wrapper."""

    def __init__(self, config: RAGConfig):
        self.config = config
        self._splitter = RecursiveCharacterTextSplitter(
            chunk_size=config.chunk_size,
            chunk_overlap=config.chunk_overlap,
            separators=config.separators,
            length_function=len,
        )

    def split(self, text: str) -> List[str]:
        if len(text) <= self.config.max_split_threshold:
            return [text]
        return self._splitter.split_text(text)


class BgeM3EmbeddingService:
    """BGE-M3 dense + sparse embedding adapter."""

    def __init__(self, model_name: str):
        try:
            from FlagEmbedding import BGEM3FlagModel
        except ImportError as exc:
            raise RuntimeError(
                "Missing dependency FlagEmbedding. Install it with: pip install FlagEmbedding"
            ) from exc

        model_path = self._resolve_model_path(
            model_name,
            allow_patterns=[
                "config.json",
                "modules.json",
                "sentence_bert_config.json",
                "sentencepiece.bpe.model",
                "special_tokens_map.json",
                "tokenizer.json",
                "tokenizer_config.json",
                "pytorch_model.bin",
                "model.safetensors",
                "1_Pooling/**",
            ],
        )
        self.model = BGEM3FlagModel(model_path, use_fp16=True)

    @staticmethod
    def _resolve_model_path(
        model_name: str,
        allow_patterns: Optional[List[str]] = None,
    ) -> str:
        if os.path.exists(model_name):
            return model_name

        project_model_path = _project_path(model_name)
        if os.path.exists(project_model_path):
            return project_model_path

        try:
            from huggingface_hub import snapshot_download
        except ImportError:
            return model_name

        return snapshot_download(
            repo_id=model_name,
            allow_patterns=allow_patterns,
            ignore_patterns=[".DS_Store", "**/.DS_Store", "imgs/**"],
        )

    @staticmethod
    def _normalize_sparse(weights: Dict) -> Dict[int, float]:
        sparse = {}
        for key, value in weights.items():
            sparse[int(key)] = float(value)
        return sparse

    def encode(self, texts: List[str]) -> tuple[List[List[float]], List[Dict[int, float]]]:
        output = self.model.encode(
            texts,
            return_dense=True,
            return_sparse=True,
            return_colbert_vecs=False,
        )
        dense_vectors = [list(map(float, vector)) for vector in output["dense_vecs"]]
        sparse_vectors = [
            self._normalize_sparse(weights)
            for weights in output["lexical_weights"]
        ]
        return dense_vectors, sparse_vectors


class BgeReranker:
    """BGE reranker adapter. If unavailable, keeps original retrieval order."""

    def __init__(self, model_name: str):
        try:
            from FlagEmbedding import FlagReranker
        except ImportError:
            self.reranker = None
            return

        try:
            model_path = BgeM3EmbeddingService._resolve_model_path(
                model_name,
                allow_patterns=[
                    "config.json",
                    "sentencepiece.bpe.model",
                    "special_tokens_map.json",
                    "tokenizer.json",
                    "tokenizer_config.json",
                    "model.safetensors",
                    "pytorch_model.bin",
                ],
            )
            self.reranker = FlagReranker(model_path, use_fp16=True)
        except Exception as exc:
            print(f"Warning: reranker unavailable, hybrid retrieval will continue without rerank: {exc}")
            self.reranker = None

    def rerank(self, query: str, docs: List[Document], top_k: int) -> List[Document]:
        if not docs or self.reranker is None:
            return docs[:top_k]

        pairs = [[query, doc.page_content] for doc in docs]
        scores = self.reranker.compute_score(pairs, normalize=True)
        if not isinstance(scores, list):
            scores = [scores]

        ranked = sorted(zip(docs, scores), key=lambda item: item[1], reverse=True)
        results = []
        for doc, score in ranked[:top_k]:
            doc.metadata["rerank_score"] = float(score)
            results.append(doc)
        return results


@dataclass
class HybridSearchResult:
    text: str
    source: str
    create_time: str
    score: float


class MilvusHybridRetriever:
    """LangChain-like retriever facade for Milvus hybrid search."""

    def __init__(self, kb: "KnowledgeBaseBuilder"):
        self.kb = kb

    def invoke(self, query: str) -> List[Document]:
        return self.kb.search(query)


class KnowledgeBaseBuilder:
    """Load, split, embed, index, and retrieve documents with Milvus."""

    def __init__(self, config: Optional[RAGConfig] = None):
        self.config = config or RAGConfig()
        self.config.data_dir = _project_path(self.config.data_dir)
        self.config.md5_record_path = _project_path(self.config.md5_record_path)
        self.embedding = BgeM3EmbeddingService(self.config.embedding_model)
        self.reranker = BgeReranker(self.config.reranker_model)
        self.splitter = TextSplitter(self.config)
        self.dedup = DedupService(self.config.md5_record_path)
        self.collection = self._connect_collection()

    def _connect_collection(self):
        try:
            from pymilvus import (
                AnnSearchRequest,
                Collection,
                CollectionSchema,
                DataType,
                FieldSchema,
                connections,
                utility,
            )
        except ImportError as exc:
            raise RuntimeError(
                "Missing dependency pymilvus. Install it with: pip install pymilvus"
            ) from exc

        self._AnnSearchRequest = AnnSearchRequest

        connection_args = {"uri": self.config.milvus_uri}
        if self.config.milvus_token:
            connection_args["token"] = self.config.milvus_token
        if self.config.milvus_db_name:
            connection_args["db_name"] = self.config.milvus_db_name

        connections.connect(alias="default", **connection_args)

        if not utility.has_collection(self.config.milvus_collection):
            fields = [
                FieldSchema(name="id", dtype=DataType.VARCHAR, is_primary=True, max_length=64),
                FieldSchema(name="text", dtype=DataType.VARCHAR, max_length=65535),
                FieldSchema(name="source", dtype=DataType.VARCHAR, max_length=512),
                FieldSchema(name="create_time", dtype=DataType.VARCHAR, max_length=32),
                FieldSchema(name="dense_vector", dtype=DataType.FLOAT_VECTOR, dim=self.config.dense_vector_dim),
                FieldSchema(name="sparse_vector", dtype=DataType.SPARSE_FLOAT_VECTOR),
            ]
            schema = CollectionSchema(fields=fields, description="RAG knowledge base")
            collection = Collection(self.config.milvus_collection, schema=schema)
            collection.create_index(
                field_name="dense_vector",
                index_params={
                    "index_type": "HNSW",
                    "metric_type": "COSINE",
                    "params": {"M": 16, "efConstruction": 200},
                },
            )
            collection.create_index(
                field_name="sparse_vector",
                index_params={
                    "index_type": "SPARSE_INVERTED_INDEX",
                    "metric_type": "IP",
                    "params": {"drop_ratio_build": 0.2},
                },
            )
        else:
            collection = Collection(self.config.milvus_collection)

        collection.load()
        return collection

    def build_from_directory(self, directory: Optional[str] = None) -> int:
        directory = directory or self.config.data_dir
        if not os.path.isdir(directory):
            raise FileNotFoundError(f"Directory does not exist: {directory}")

        documents = DocumentLoader.load_directory(directory)
        total_chunks = 0

        for doc in documents:
            source = doc.metadata.get("source", "unknown")
            if self.dedup.exists(doc.page_content):
                print(f"[skip] {source} already indexed")
                continue

            total_chunks += self.add_text(
                content=doc.page_content,
                source=source,
                mark_dedup=False,
            )
            self.dedup.mark(doc.page_content)

        print(f"[done] indexed {total_chunks} chunks")
        return total_chunks

    def add_text(self, content: str, source: str = "manual", mark_dedup: bool = True):
        if mark_dedup and self.dedup.exists(content):
            return "[skip] content already exists"

        chunks = self.splitter.split(content)
        dense_vectors, sparse_vectors = self.embedding.encode(chunks)
        now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        rows = [
            {
                "id": uuid.uuid4().hex,
                "text": chunk,
                "source": source,
                "create_time": now,
                "dense_vector": dense_vectors[index],
                "sparse_vector": sparse_vectors[index],
            }
            for index, chunk in enumerate(chunks)
        ]
        self.collection.insert(rows)
        self.collection.flush()

        if mark_dedup:
            self.dedup.mark(content)
            return f"[ok] {source}: indexed {len(chunks)} chunks"
        return len(chunks)

    def search(self, query: str) -> List[Document]:
        from pymilvus import WeightedRanker

        dense_vectors, sparse_vectors = self.embedding.encode([query])
        dense_request = self._AnnSearchRequest(
            data=[dense_vectors[0]],
            anns_field="dense_vector",
            param={"metric_type": "COSINE", "params": {"ef": 64}},
            limit=self.config.retrieval_candidates,
        )
        sparse_request = self._AnnSearchRequest(
            data=[sparse_vectors[0]],
            anns_field="sparse_vector",
            param={"metric_type": "IP", "params": {"drop_ratio_search": 0.2}},
            limit=self.config.retrieval_candidates,
        )

        results = self.collection.hybrid_search(
            reqs=[dense_request, sparse_request],
            rerank=WeightedRanker(self.config.dense_weight, self.config.sparse_weight),
            limit=self.config.retrieval_candidates,
            output_fields=["text", "source", "create_time"],
        )

        docs = []
        for hit in results[0]:
            entity = hit.entity
            docs.append(
                Document(
                    page_content=entity.get("text"),
                    metadata={
                        "source": entity.get("source"),
                        "create_time": entity.get("create_time"),
                        "retrieval_score": float(hit.score),
                    },
                )
            )

        return self.reranker.rerank(query, docs, self.config.retrieval_top_k)

    def get_retriever(self):
        return MilvusHybridRetriever(self)
