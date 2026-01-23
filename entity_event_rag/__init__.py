"""Entity-Event RAG: Bipartite Graph-based Retrieval-Augmented Generation.

This package implements a RAG system that builds a bipartite knowledge graph
connecting entities and events, enabling more comprehensive retrieval for
complex queries involving temporal and narrative understanding.
"""

from .core import EntityEventRAG, QueryParam

__version__ = "1.0.0"
__author__ = "Entity-Event RAG Team"
__url__ = "https://github.com/zyzhang1130/EntityEventRAG"

# Exported classes
__all__ = ["EntityEventRAG", "QueryParam"]
