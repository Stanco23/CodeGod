"""
Memory System with Vector DB and RAG Workflow
Handles context storage, retrieval, and management for the autonomous AI system
"""

from typing import List, Dict, Optional, Any
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
import hashlib
import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer


class ContentType(Enum):
    """Types of content stored in memory"""
    API_DOC = "api_doc"
    CODE = "code"
    TASK = "task"
    TEST = "test"
    ERROR = "error"
    DEPENDENCY = "dependency"
    CONFIG = "config"
    DOCUMENTATION = "documentation"


class Platform(Enum):
    """Target platforms"""
    WEB = "web"
    IOS = "ios"
    ANDROID = "android"
    BACKEND = "backend"
    SHARED = "shared"


@dataclass
class MemoryMetadata:
    """Metadata for stored content"""
    type: ContentType
    source: str
    task_id: Optional[str] = None
    timestamp: str = field(default_factory=lambda: datetime.utcnow().isoformat())
    version: Optional[str] = None
    platform: Platform = Platform.SHARED
    tags: List[str] = field(default_factory=list)
    language: Optional[str] = None
    file_path: Optional[str] = None
    dependencies: List[str] = field(default_factory=list)
    validated: bool = False


@dataclass
class MemoryChunk:
    """A chunk of content with metadata"""
    id: str
    content: str
    metadata: MemoryMetadata
    embedding: Optional[List[float]] = None


class MemorySystem:
    """
    Vector database memory system for RAG workflows

    Features:
    - Efficient chunking of large documents
    - Semantic search with embeddings
    - Metadata filtering
    - Context retrieval for worker agents
    - Version tracking
    """

    def __init__(
        self,
        persist_directory: str = "./chroma_db",
        embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2",
        chunk_size: int = 1024,
        chunk_overlap: int = 128
    ):
        """Initialize memory system"""
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

        # Initialize embedding model
        self.embedding_model = SentenceTransformer(embedding_model)

        # Initialize ChromaDB
        self.client = chromadb.Client(Settings(
            persist_directory=persist_directory,
            anonymized_telemetry=False
        ))

        # Create or get collection
        self.collection = self.client.get_or_create_collection(
            name="autonomous_ai_memory",
            metadata={"description": "Memory for autonomous multi-agent AI system"}
        )

    def chunk_content(self, content: str) -> List[str]:
        """
        Chunk large content into manageable pieces

        Strategy:
        - Split by paragraphs/sections first
        - Ensure chunks don't exceed chunk_size
        - Maintain overlap for context continuity
        """
        chunks = []

        # Try to split by double newlines (paragraphs)
        paragraphs = content.split('\n\n')

        current_chunk = ""
        for para in paragraphs:
            # If adding this paragraph exceeds chunk_size
            if len(current_chunk) + len(para) > self.chunk_size:
                if current_chunk:
                    chunks.append(current_chunk.strip())
                    # Keep overlap from previous chunk
                    overlap_start = max(0, len(current_chunk) - self.chunk_overlap)
                    current_chunk = current_chunk[overlap_start:] + "\n\n" + para
                else:
                    # Single paragraph too large, force split
                    words = para.split()
                    temp_chunk = ""
                    for word in words:
                        if len(temp_chunk) + len(word) > self.chunk_size:
                            chunks.append(temp_chunk.strip())
                            temp_chunk = word + " "
                        else:
                            temp_chunk += word + " "
                    current_chunk = temp_chunk
            else:
                current_chunk += para + "\n\n"

        if current_chunk:
            chunks.append(current_chunk.strip())

        return chunks

    def _generate_id(self, content: str, metadata: MemoryMetadata) -> str:
        """Generate unique ID for content"""
        unique_string = f"{content[:100]}{metadata.source}{metadata.timestamp}"
        return hashlib.sha256(unique_string.encode()).hexdigest()[:16]

    def store(
        self,
        content: str,
        metadata: MemoryMetadata,
        auto_chunk: bool = True
    ) -> List[str]:
        """
        Store content in memory with metadata

        Args:
            content: Text content to store
            metadata: Associated metadata
            auto_chunk: Whether to automatically chunk large content

        Returns:
            List of stored chunk IDs
        """
        chunks = self.chunk_content(content) if auto_chunk else [content]
        chunk_ids = []

        for idx, chunk_content in enumerate(chunks):
            chunk_id = self._generate_id(chunk_content, metadata)

            # Generate embedding
            embedding = self.embedding_model.encode(chunk_content).tolist()

            # Prepare metadata dict
            meta_dict = {
                "type": metadata.type.value,
                "source": metadata.source,
                "timestamp": metadata.timestamp,
                "platform": metadata.platform.value,
                "tags": ",".join(metadata.tags),
                "chunk_index": idx,
                "total_chunks": len(chunks),
                "validated": str(metadata.validated)
            }

            # Add optional fields
            if metadata.task_id:
                meta_dict["task_id"] = metadata.task_id
            if metadata.version:
                meta_dict["version"] = metadata.version
            if metadata.language:
                meta_dict["language"] = metadata.language
            if metadata.file_path:
                meta_dict["file_path"] = metadata.file_path
            if metadata.dependencies:
                meta_dict["dependencies"] = ",".join(metadata.dependencies)

            # Store in ChromaDB
            self.collection.add(
                ids=[chunk_id],
                embeddings=[embedding],
                documents=[chunk_content],
                metadatas=[meta_dict]
            )

            chunk_ids.append(chunk_id)

        return chunk_ids

    def retrieve(
        self,
        query: str,
        filters: Optional[Dict[str, Any]] = None,
        top_k: int = 5,
        include_embeddings: bool = False
    ) -> List[Dict[str, Any]]:
        """
        Retrieve relevant content using RAG

        Args:
            query: Search query
            filters: Metadata filters (e.g., {"type": "api_doc", "platform": "web"})
            top_k: Number of results to return
            include_embeddings: Whether to include embeddings in results

        Returns:
            List of retrieved chunks with content and metadata
        """
        # Generate query embedding
        query_embedding = self.embedding_model.encode(query).tolist()

        # Build where clause from filters
        where_clause = None
        if filters:
            where_clause = {}
            for key, value in filters.items():
                if isinstance(value, list):
                    where_clause[key] = {"$in": value}
                else:
                    where_clause[key] = value

        # Query ChromaDB
        results = self.collection.query(
            query_embeddings=[query_embedding],
            where=where_clause,
            n_results=top_k,
            include=["documents", "metadatas", "distances"] +
                    (["embeddings"] if include_embeddings else [])
        )

        # Format results
        retrieved = []
        for idx in range(len(results["ids"][0])):
            result = {
                "id": results["ids"][0][idx],
                "content": results["documents"][0][idx],
                "metadata": results["metadatas"][0][idx],
                "distance": results["distances"][0][idx],
                "relevance_score": 1 - results["distances"][0][idx]  # Convert distance to score
            }
            if include_embeddings:
                result["embedding"] = results["embeddings"][0][idx]
            retrieved.append(result)

        return retrieved

    def update(self, chunk_id: str, content: str, metadata: Optional[MemoryMetadata] = None):
        """Update existing content"""
        # Generate new embedding
        embedding = self.embedding_model.encode(content).tolist()

        update_data = {
            "documents": [content],
            "embeddings": [embedding]
        }

        if metadata:
            meta_dict = {
                "type": metadata.type.value,
                "source": metadata.source,
                "timestamp": metadata.timestamp,
                "platform": metadata.platform.value,
                "tags": ",".join(metadata.tags),
                "validated": str(metadata.validated)
            }
            if metadata.task_id:
                meta_dict["task_id"] = metadata.task_id
            if metadata.version:
                meta_dict["version"] = metadata.version

            update_data["metadatas"] = [meta_dict]

        self.collection.update(
            ids=[chunk_id],
            **update_data
        )

    def delete(self, chunk_ids: List[str]):
        """Delete content by IDs"""
        self.collection.delete(ids=chunk_ids)

    def search_by_metadata(
        self,
        filters: Dict[str, Any],
        limit: Optional[int] = None
    ) -> List[Dict[str, Any]]:
        """Search purely by metadata without semantic similarity"""
        where_clause = {}
        for key, value in filters.items():
            if isinstance(value, list):
                where_clause[key] = {"$in": value}
            else:
                where_clause[key] = value

        results = self.collection.get(
            where=where_clause,
            limit=limit,
            include=["documents", "metadatas"]
        )

        # Format results
        items = []
        for idx in range(len(results["ids"])):
            items.append({
                "id": results["ids"][idx],
                "content": results["documents"][idx],
                "metadata": results["metadatas"][idx]
            })

        return items

    def get_task_context(
        self,
        task_description: str,
        task_type: str,
        platform: str,
        max_tokens: int = 4000
    ) -> str:
        """
        Retrieve relevant context for a task using RAG

        This is the key method for providing context to worker agents
        """
        # Search for relevant content
        filters = {
            "platform": {"$in": [platform, Platform.SHARED.value]}
        }

        # Retrieve relevant chunks
        results = self.retrieve(
            query=task_description,
            filters=filters,
            top_k=10
        )

        # Build context string, respecting token limit
        context_parts = []
        current_length = 0

        for result in results:
            content = result["content"]
            content_length = len(content.split())  # Rough token estimate

            if current_length + content_length > max_tokens:
                break

            metadata = result["metadata"]
            source = metadata.get("source", "unknown")

            context_parts.append(
                f"--- Source: {source} (Relevance: {result['relevance_score']:.2f}) ---\n"
                f"{content}\n"
            )
            current_length += content_length

        return "\n".join(context_parts)

    def store_api_documentation(self, api_spec: str, source_url: str) -> List[str]:
        """
        Store API documentation with optimized chunking

        Special handling for API docs to preserve structure
        """
        metadata = MemoryMetadata(
            type=ContentType.API_DOC,
            source=source_url,
            tags=["api", "documentation", "reference"]
        )

        return self.store(api_spec, metadata, auto_chunk=True)

    def store_code_module(
        self,
        code: str,
        file_path: str,
        language: str,
        task_id: str,
        platform: Platform,
        dependencies: List[str] = None
    ) -> List[str]:
        """Store generated code module"""
        metadata = MemoryMetadata(
            type=ContentType.CODE,
            source=file_path,
            task_id=task_id,
            platform=platform,
            language=language,
            file_path=file_path,
            dependencies=dependencies or [],
            tags=["code", language, platform.value]
        )

        return self.store(code, metadata, auto_chunk=False)

    def store_task_result(
        self,
        task_id: str,
        task_description: str,
        output: str,
        success: bool
    ) -> List[str]:
        """Store task execution result"""
        metadata = MemoryMetadata(
            type=ContentType.TASK,
            source=f"task_{task_id}",
            task_id=task_id,
            tags=["task_result", "success" if success else "failure"]
        )

        content = f"Task: {task_description}\n\nOutput:\n{output}"
        return self.store(content, metadata, auto_chunk=False)

    def get_project_state(self) -> Dict[str, Any]:
        """Get current project state from memory"""
        # Get all code modules
        code_modules = self.search_by_metadata({"type": ContentType.CODE.value})

        # Get all tasks
        tasks = self.search_by_metadata({"type": ContentType.TASK.value})

        # Get all errors
        errors = self.search_by_metadata({"type": ContentType.ERROR.value})

        return {
            "code_modules": len(code_modules),
            "completed_tasks": len([t for t in tasks if "success" in t["metadata"].get("tags", "")]),
            "failed_tasks": len([t for t in tasks if "failure" in t["metadata"].get("tags", "")]),
            "errors": len(errors),
            "modules": code_modules
        }


# RAG Workflow Functions

def prepare_worker_context(
    memory: MemorySystem,
    task_description: str,
    task_type: str,
    platform: str,
    dependencies: List[str] = None
) -> Dict[str, Any]:
    """
    Prepare comprehensive context for a worker agent

    Returns:
        Dict with relevant context, examples, and documentation
    """
    context = {
        "task_description": task_description,
        "task_type": task_type,
        "platform": platform,
        "dependencies": dependencies or []
    }

    # Get relevant API documentation
    api_context = memory.retrieve(
        query=task_description,
        filters={"type": ContentType.API_DOC.value},
        top_k=3
    )
    context["api_docs"] = [r["content"] for r in api_context]

    # Get relevant existing code for examples
    code_context = memory.retrieve(
        query=task_description,
        filters={
            "type": ContentType.CODE.value,
            "platform": {"$in": [platform, Platform.SHARED.value]}
        },
        top_k=2
    )
    context["code_examples"] = [r["content"] for r in code_context]

    # Get dependency information
    if dependencies:
        dep_context = []
        for dep in dependencies:
            dep_results = memory.search_by_metadata({
                "task_id": dep
            })
            if dep_results:
                dep_context.append(dep_results[0]["content"])
        context["dependency_outputs"] = dep_context

    return context


def store_project_initialization(
    memory: MemorySystem,
    project_name: str,
    user_prompt: str,
    api_specs: List[str],
    tech_stack: Dict[str, str]
) -> None:
    """
    Store initial project information in memory
    """
    # Store user prompt
    metadata = MemoryMetadata(
        type=ContentType.DOCUMENTATION,
        source="user_input",
        tags=["project_init", "requirements"]
    )
    memory.store(user_prompt, metadata)

    # Store each API spec
    for spec_url in api_specs:
        # In real implementation, would fetch the spec
        # For now, just store the URL
        metadata = MemoryMetadata(
            type=ContentType.API_DOC,
            source=spec_url,
            tags=["api", "spec", "external"]
        )
        memory.store(f"API Specification from: {spec_url}", metadata)

    # Store tech stack info
    tech_content = "\n".join([f"{k}: {v}" for k, v in tech_stack.items()])
    metadata = MemoryMetadata(
        type=ContentType.CONFIG,
        source="tech_stack",
        tags=["config", "stack", "technology"]
    )
    memory.store(tech_content, metadata)
