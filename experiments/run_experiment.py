#!/usr/bin/env python3
"""Entity-Event RAG Experiment Runner.

This script provides a unified interface for running batch experiments
with the Entity-Event RAG system across different configurations.
"""

import os
import sys
import json
import argparse
import asyncio
from datetime import datetime
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables
load_dotenv()
from typing import List, Dict, Any, Optional

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from entity_event_rag import EntityEventRAG
from entity_event_rag.base import QueryParam
from entity_event_rag.llm import gpt_4o_mini_complete, gpt_4o_complete
from config import EntityEventRAGConfig, ConfigProfiles


class ExperimentRunner:
    """Runs batch experiments with Entity-Event RAG."""
    
    def __init__(
        self,
        working_dir: str,
        config: Optional[EntityEventRAGConfig] = None,
        output_dir: Optional[str] = None
    ):
        """Initialize the experiment runner.
        
        Args:
            working_dir: Base working directory for RAG instances
            config: Configuration to use (defaults to EntityEventRAGConfig)
            output_dir: Directory for output files (defaults to working_dir/results)
        """
        self.working_dir = Path(working_dir)
        self.config = config or EntityEventRAGConfig()
        self.output_dir = Path(output_dir) if output_dir else self.working_dir / "results"
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.rag_instances: Dict[str, EntityEventRAG] = {}
        self.results: List[Dict[str, Any]] = []
    
    def get_or_create_rag(
        self,
        extraction_type: str = "entity",
        storage_subdir: str = ""
    ) -> EntityEventRAG:
        """Get or create a RAG instance for the given configuration.
        
        Args:
            extraction_type: "entity" or "event"
            storage_subdir: Subdirectory for storage (e.g., story/document ID)
        
        Returns:
            EntityEventRAG instance
        """
        cache_key = f"{extraction_type}:{storage_subdir}"
        
        if cache_key not in self.rag_instances:
            if storage_subdir:
                working_dir = self.working_dir / storage_subdir / f"{extraction_type}_kg"
            else:
                working_dir = self.working_dir / f"{extraction_type}_kg"
            
            working_dir.mkdir(parents=True, exist_ok=True)
            
            rag = EntityEventRAG(
                working_dir=str(working_dir),
                extraction_type=extraction_type,
                llm_model_func=gpt_4o_mini_complete,
                **self.config.to_rag_params()
            )
            self.rag_instances[cache_key] = rag
        
        return self.rag_instances[cache_key]
    
    async def run_query(
        self,
        query: str,
        mode: str = "entity_event",
        extraction_type: str = "entity",
        storage_subdir: str = "",
        event_hop_depth: int = 2,
        enable_hyde: bool = False,
        **additional_params
    ) -> Dict[str, Any]:
        """Run a single query and return results.
        
        Args:
            query: Query string
            mode: Query mode (naive, local, global, hybrid, entity_event)
            extraction_type: "entity" or "event"
            storage_subdir: Subdirectory for storage
            event_hop_depth: Hops for entity-event graph traversal
            enable_hyde: Enable hypothetical document embedding
            **additional_params: Additional QueryParam parameters
        
        Returns:
            Dictionary with query results and metadata
        """
        rag = self.get_or_create_rag(extraction_type, storage_subdir)
        
        query_param = QueryParam(
            mode=mode,
            event_hop_depth=event_hop_depth,
            enable_hypothetical_document=enable_hyde,
            **additional_params
        )
        
        start_time = datetime.now()
        
        try:
            response = await rag.aquery(query, query_param)
            
            # Handle different response types
            if isinstance(response, tuple):
                answer = str(response[0])
                metadata = response[1] if len(response) > 1 else {}
            else:
                answer = str(response)
                metadata = {}
            
            result = {
                "query": query,
                "answer": answer,
                "mode": mode,
                "extraction_type": extraction_type,
                "event_hop_depth": event_hop_depth,
                "enable_hyde": enable_hyde,
                "metadata": metadata,
                "timestamp": datetime.now().isoformat(),
                "duration_seconds": (datetime.now() - start_time).total_seconds(),
                "success": True,
                "error": None
            }
        except Exception as e:
            result = {
                "query": query,
                "answer": None,
                "mode": mode,
                "extraction_type": extraction_type,
                "event_hop_depth": event_hop_depth,
                "enable_hyde": enable_hyde,
                "metadata": {},
                "timestamp": datetime.now().isoformat(),
                "duration_seconds": (datetime.now() - start_time).total_seconds(),
                "success": False,
                "error": str(e)
            }
        
        self.results.append(result)
        return result
    
    def run_queries_sync(
        self,
        queries: List[str],
        **query_params
    ) -> List[Dict[str, Any]]:
        """Run multiple queries synchronously.
        
        Args:
            queries: List of query strings
            **query_params: Parameters passed to run_query
        
        Returns:
            List of result dictionaries
        """
        results = []
        for i, query in enumerate(queries, 1):
            print(f"Processing query {i}/{len(queries)}: {query[:50]}...")
            result = asyncio.get_event_loop().run_until_complete(
                self.run_query(query, **query_params)
            )
            results.append(result)
            
            if not result["success"]:
                print(f"  Error: {result['error']}")
            else:
                print(f"  Done ({result['duration_seconds']:.2f}s)")
        
        return results
    
    def save_results(self, filename: str = "experiment_results.json"):
        """Save all results to a JSON file.
        
        Args:
            filename: Output filename
        """
        output_path = self.output_dir / filename
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(self.results, f, indent=2, ensure_ascii=False)
        print(f"Results saved to: {output_path}")
    
    def load_queries_from_file(self, filepath: str) -> List[Dict[str, Any]]:
        """Load queries from a JSON file.
        
        Expected format: List of dicts with at least a "query" field.
        
        Args:
            filepath: Path to JSON file
        
        Returns:
            List of query dictionaries
        """
        with open(filepath, 'r', encoding='utf-8') as f:
            return json.load(f)


def main():
    parser = argparse.ArgumentParser(
        description="Run Entity-Event RAG experiments"
    )
    parser.add_argument(
        "--working-dir", "-w",
        default="./experiments_cache",
        help="Working directory for RAG cache"
    )
    parser.add_argument(
        "--queries-file", "-q",
        help="JSON file containing queries"
    )
    parser.add_argument(
        "--query", 
        action="append",
        help="Single query to run (can be used multiple times)"
    )
    parser.add_argument(
        "--mode", "-m",
        choices=["naive", "local", "global", "hybrid", "entity_event"],
        default="entity_event",
        help="Query mode (default: entity_event)"
    )
    parser.add_argument(
        "--extraction-type",
        choices=["entity", "event"],
        default="entity",
        help="Extraction type used during indexing"
    )
    parser.add_argument(
        "--event-hop-depth",
        type=int,
        default=2,
        help="Event hop depth for entity-event mode (default: 2)"
    )
    parser.add_argument(
        "--enable-hyde",
        action="store_true",
        help="Enable hypothetical document embedding"
    )
    parser.add_argument(
        "--output-dir", "-o",
        help="Output directory for results"
    )
    parser.add_argument(
        "--config-profile",
        choices=["default", "fast", "high_quality", "narrative"],
        default="default",
        help="Configuration profile to use"
    )
    
    args = parser.parse_args()
    
    # Load configuration
    if args.config_profile == "fast":
        config = ConfigProfiles.fast_prototyping()
    elif args.config_profile == "high_quality":
        config = ConfigProfiles.high_quality()
    elif args.config_profile == "narrative":
        config = ConfigProfiles.narrative_analysis()
    else:
        config = EntityEventRAGConfig()
    
    # Initialize runner
    runner = ExperimentRunner(
        working_dir=args.working_dir,
        config=config,
        output_dir=args.output_dir
    )
    
    # Collect queries
    queries = []
    if args.queries_file:
        query_data = runner.load_queries_from_file(args.queries_file)
        queries = [q["query"] for q in query_data]
    if args.query:
        queries.extend(args.query)
    
    if not queries:
        print("No queries provided. Use --query or --queries-file.")
        sys.exit(1)
    
    print(f"Running {len(queries)} queries with mode={args.mode}, " 
          f"extraction_type={args.extraction_type}, event_hop_depth={args.event_hop_depth}")
    
    # Run queries
    runner.run_queries_sync(
        queries,
        mode=args.mode,
        extraction_type=args.extraction_type,
        event_hop_depth=args.event_hop_depth,
        enable_hyde=args.enable_hyde
    )
    
    # Save results
    runner.save_results()
    
    # Print summary
    successful = sum(1 for r in runner.results if r["success"])
    failed = len(runner.results) - successful
    avg_time = sum(r["duration_seconds"] for r in runner.results) / len(runner.results)
    
    print(f"\n=== Experiment Summary ===")
    print(f"Total queries: {len(runner.results)}")
    print(f"Successful: {successful}")
    print(f"Failed: {failed}")
    print(f"Average time: {avg_time:.2f}s")


if __name__ == "__main__":
    main()
