"""Module for query expansion using hypothetical document generation."""

import asyncio
import time
import json
import os
from typing import Dict, Optional, Tuple

class HypotheticalDocumentTransformer:
    """Transform queries by generating hypothetical answers to expand the original query."""
    
    def __init__(self, llm_model_func, enable_cache=True):
        """Initialize the hypothetical document transformer.
        
        Args:
            llm_model_func: Function to call the language model
            enable_cache: Whether to cache generated hypothetical documents
        """
        self.llm_model_func = llm_model_func
        self.cache = {} if enable_cache else None
        
    async def transform_query(self, original_query: str) -> Tuple[str, float]:
        """Generate a hypothetical document that answers the original query.
        
        Args:
            original_query: Original user query
            
        Returns:
            A tuple of (generated hypothetical document, generation time in seconds)
        """
        # Check cache first if enabled
        if self.cache is not None and original_query in self.cache:
            # Return cached document with zero generation time (since it's cached)
            return self.cache[original_query], 0.0
        
        # Start timing the generation
        start_time = time.time()
            
        # Prompt for hypothetical document generation
        prompt = f"""Please write a passage to answer the question.
Question: {original_query}
Passage:"""
        
        # Generate the hypothetical document
        hypothetical_doc = await self.llm_model_func(prompt, max_tokens=512, temperature=0.7)
        
        # Calculate generation time
        end_time = time.time()
        generation_time = end_time - start_time
        
        # Cache the result if caching is enabled
        if self.cache is not None:
            self.cache[original_query] = hypothetical_doc
            
        # Return both the document and generation time
        return hypothetical_doc, generation_time
    
    def save_to_file(self, original_query: str, hypothetical_doc: str, model_name: str = "unknown", output_dir: str = "hypothetical_docs"):
        """Save the hypothetical document to a JSON file.
        
        Args:
            original_query: The original query that generated this document
            hypothetical_doc: The generated hypothetical document
            model_name: Name of the model used for generation
            output_dir: Directory to save the output files
        
        Returns:
            Path to the saved file or None if saving failed
        """
        try:
            # Create output directory if it doesn't exist
            os.makedirs(output_dir, exist_ok=True)
            
            # Create a filename based on a timestamp
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            filename = f"hyp_doc_{timestamp}.json"
            filepath = os.path.join(output_dir, filename)
            
            # Prepare the data
            data = {
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                "query": original_query,
                "model": model_name,
                "hypothetical_document": hypothetical_doc
            }
            
            # Write to file
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
                
            print(f"Hypothetical document saved to {filepath}")
            return filepath
            
        except Exception as e:
            print(f"Error saving hypothetical document to file: {e}")
            return None
