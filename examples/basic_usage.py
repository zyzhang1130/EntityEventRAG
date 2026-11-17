#!/usr/bin/env python3
"""Basic usage example for Entity-Event RAG.

This example demonstrates the core functionality of the Entity-Event RAG system:
1. Loading a document (Lady_Susan.txt)
2. Building the entity-event knowledge graph
3. Querying the graph with different modes
"""

import os
import shutil
import sys
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Add parent directory to path for development
sys.path.insert(0, str(Path(__file__).parent.parent))

from entity_event_rag import EntityEventRAG
from entity_event_rag.base import QueryParam
from examples.interactive_query import construct_bipartite_graph, create_combined_kg


def main():
    # Configuration
    WORKING_DIR = "./example_cache"
    DATA_FILE = Path(__file__).parent.parent / "RAG_documents" / "Lady_Susan.txt"
    entity_graph_path = Path(WORKING_DIR) / "entity_kg" / "graph_chunk_entity_relation.graphml"
    event_graph_path = Path(WORKING_DIR) / "event_kg" / "graph_chunk_entity_relation.graphml"
    
    # Ensure working directory exists
    os.makedirs(WORKING_DIR, exist_ok=True)
    
    # Check if data file exists
    if not DATA_FILE.exists():
        print(f"Sample data not found at {DATA_FILE}")
        print("Please provide a text file to index.")
        return

    has_cached_graphs = entity_graph_path.exists() and event_graph_path.exists()

    # Check for API Key (only strictly required when building the graphs)
    if not os.getenv("OPENAI_API_KEY"):
        if not has_cached_graphs:
            print("Error: OPENAI_API_KEY not found in environment variables.")
            print("Please set it in your .env file or export it in your shell.")
            return
        print("Warning: OPENAI_API_KEY not found; using existing cache only.")
    
    print("=" * 60)
    print("Entity-Event RAG - Basic Usage Example")
    print("=" * 60)
    
    # Initialize Entity-Event RAG
    print("\n[1/4] Initializing EntityEventRAG...")
    
    # Create an instance for entity extraction
    entity_rag = EntityEventRAG(
        working_dir=os.path.join(WORKING_DIR, "entity_kg"),
        extraction_type="entity",  # Extract entities
    )
    
    # Create an instance for event extraction
    event_rag = EntityEventRAG(
        working_dir=os.path.join(WORKING_DIR, "event_kg"),
        extraction_type="event",  # Extract events
    )
    
    print("  ✓ EntityEventRAG instances created")
    
    # Load and index document (skipped if cached graphs already exist)
    print("\n[2/4] Loading and indexing document...")
    if has_cached_graphs:
        print(f"  Found existing cache under: {WORKING_DIR}")
        print("  Skipping indexing step.")
    else:
        print(f"  Source: {DATA_FILE}")

        with open(DATA_FILE, "r", encoding="utf-8") as f:
            # For demo, just use first 50k chars
            content = f.read()[:50000]

        print(f"  Content length: {len(content):,} characters")
        print("  Indexing with entity extraction...")
        entity_rag.insert(content)
        print("  ✓ Entity knowledge graph built")

        print("  Indexing with event extraction...")
        event_rag.insert(content)
        print("  ✓ Event knowledge graph built")
    
    # Run sample queries
    print("\n[3/4] Running sample queries...")
    
    queries = [
        "What are Lady Susan's real motives for visiting Churchhill?",
        "How does Mrs. Vernon feel about Lady Susan's arrival?",
        "What is the relationship between Lady Susan and Mainwaring?"
    ]
    
    for i, query in enumerate(queries, 1):
        print(f"\n  Query {i}: {query}")
        print("-" * 50)
        
        # Query using hybrid mode (combines entity and relationship context)
        response = entity_rag.query(
            query,
            param=QueryParam(
                mode="hybrid",
                top_k=10,
            )
        )
        
        # Print truncated response
        answer = str(response)[:500] + "..." if len(str(response)) > 500 else str(response)
        print(f"  Answer: {answer}")
    
    # Query using entity-event mode (if available)
    print("\n[4/4] Demonstrating Entity-Event Query Mode...")
    
    combined_query = "How does Lady Susan manipulate Reginald De Courcy?"
    print(f"  Query: {combined_query}")
    print("-" * 50)
    
    try:
        # Event vector hits must resolve in a graph containing both subgraphs.
        # Reuse the interactive example's assembly, including EVENT normalization.
        combined_dir = os.path.join(WORKING_DIR, "combined_kg")
        bipartite_graph = construct_bipartite_graph(entity_graph_path, event_graph_path)
        combined_graph = create_combined_kg(
            entity_graph_path, event_graph_path, bipartite_graph, combined_dir
        )
        if combined_graph is None:
            raise RuntimeError("Failed to construct the combined entity-event graph")
        for name in ("kv_store_text_chunks.json", "kv_store_full_docs.json"):
            shutil.copy2(
                os.path.join(WORKING_DIR, "entity_kg", name),
                os.path.join(combined_dir, name),
            )

        # Create a combined RAG instance for entity-event queries
        combined_rag = EntityEventRAG(
            working_dir=combined_dir,
            extraction_type="entity",
            entity_storage_dir=os.path.join(WORKING_DIR, "entity_kg"),
            event_storage_dir=os.path.join(WORKING_DIR, "event_kg"),
        )
        
        response = combined_rag.query(
            combined_query,
            param=QueryParam(
                mode="entity_event",
                event_hop_depth=2,
            )
        )
        
        answer = str(response)[:500] + "..." if len(str(response)) > 500 else str(response)
        print(f"  Answer: {answer}")
    except Exception as e:
        print(f"  Note: Entity-event mode requires both graphs. Error: {e}")
    
    print("\n" + "=" * 60)
    print("Example complete!")
    print(f"Knowledge graphs saved to: {WORKING_DIR}")
    print("=" * 60)


if __name__ == "__main__":
    main()
