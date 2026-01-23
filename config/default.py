"""Entity-Event RAG Configuration.

This module provides default configuration for the Entity-Event RAG system.
Configuration can be overridden by creating a custom config and passing it
to the EntityEventRAG constructor.
"""

from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any


@dataclass
class ModelConfig:
    """LLM and embedding model configuration."""
    llm_model: str = "gpt-4o-mini"
    embedding_model: str = "text-embedding-3-small"
    embedding_dim: int = 1536
    max_token_size: int = 32768
    max_async_calls: int = 16


@dataclass
class ExtractionConfig:
    """Entity and event extraction configuration."""
    entity_types: List[str] = field(default_factory=lambda: [
        "person", "organization", "location", "event", "concept", 
        "object", "document", "time_period"
    ])
    event_types: List[str] = field(default_factory=lambda: [
        "battle", "discovery", "meeting", "journey", "transformation", 
        "conflict", "ceremony", "communication", "transaction", "creation"
    ])
    max_gleaning: int = 1
    summary_max_tokens: int = 500


@dataclass
class GraphConfig:
    """Knowledge graph configuration."""
    event_hop_depth: int = 2
    max_nodes: int = 1000
    storage_type: str = "NetworkXStorage"  # Options: NetworkXStorage, Neo4JStorage, OracleGraphStorage


@dataclass
class RetrievalConfig:
    """Retrieval configuration."""
    mode: str = "entity_event"  # Options: naive, local, global, hybrid, entity_event
    top_k: int = 60
    max_token_for_text_unit: int = 4000
    max_token_for_global_context: int = 4000
    max_token_for_local_context: int = 4000
    enable_hypothetical_document: bool = False
    use_direct_embedding: bool = False
    use_combined_embedding: bool = False


@dataclass  
class ChunkingConfig:
    """Text chunking configuration."""
    chunk_token_size: int = 1200
    overlap_token_size: int = 100
    tiktoken_model: str = "gpt-4o-mini"


@dataclass
class EntityEventRAGConfig:
    """Main configuration class for Entity-Event RAG.
    
    Example usage:
        from config.default import EntityEventRAGConfig
        
        config = EntityEventRAGConfig()
        config.model.llm_model = "gpt-4o"
        config.retrieval.event_hop_depth = 4
    """
    model: ModelConfig = field(default_factory=ModelConfig)
    extraction: ExtractionConfig = field(default_factory=ExtractionConfig)
    graph: GraphConfig = field(default_factory=GraphConfig)
    retrieval: RetrievalConfig = field(default_factory=RetrievalConfig)
    chunking: ChunkingConfig = field(default_factory=ChunkingConfig)
    
    # General settings
    working_dir: str = "./entity_event_rag_cache"
    enable_llm_cache: bool = True
    log_level: str = "INFO"
    
    def to_rag_params(self) -> Dict[str, Any]:
        """Convert configuration to parameters for EntityEventRAG initialization."""
        return {
            "working_dir": self.working_dir,
            "chunk_token_size": self.chunking.chunk_token_size,
            "chunk_overlap_token_size": self.chunking.overlap_token_size,
            "tiktoken_model_name": self.chunking.tiktoken_model,
            "entity_extract_max_gleaning": self.extraction.max_gleaning,
            "entity_summary_to_max_tokens": self.extraction.summary_max_tokens,
            "llm_model_max_token_size": self.model.max_token_size,
            "llm_model_max_async": self.model.max_async_calls,
            "embedding_func_max_async": self.model.max_async_calls,
            "graph_storage": self.graph.storage_type,
            "enable_llm_cache": self.enable_llm_cache,
            "log_level": self.log_level,
        }


# Pre-configured profiles for common use cases
class ConfigProfiles:
    """Pre-configured settings for common use cases."""
    
    @staticmethod
    def fast_prototyping() -> EntityEventRAGConfig:
        """Optimized for quick iteration and development."""
        config = EntityEventRAGConfig()
        config.model.llm_model = "gpt-4o-mini"
        config.extraction.max_gleaning = 0
        config.retrieval.top_k = 30
        return config
    
    @staticmethod
    def high_quality() -> EntityEventRAGConfig:
        """Optimized for best retrieval quality."""
        config = EntityEventRAGConfig()
        config.model.llm_model = "gpt-4o"
        config.extraction.max_gleaning = 2
        config.retrieval.top_k = 100
        config.graph.event_hop_depth = 4
        return config
    
    @staticmethod
    def narrative_analysis() -> EntityEventRAGConfig:
        """Optimized for analyzing narrative content (novels, stories)."""
        config = EntityEventRAGConfig()
        config.extraction.event_types = [
            "plot_event", "character_action", "dialogue", "scene_change",
            "conflict", "resolution", "revelation", "relationship_change"
        ]
        config.graph.event_hop_depth = 4
        config.retrieval.enable_hypothetical_document = True
        return config


# Default configuration instance
DEFAULT_CONFIG = EntityEventRAGConfig()
