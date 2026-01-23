"""Module for query expansion using hypothetical document generation with combined embedding."""

import asyncio
import time
import json
import os
from typing import Dict, Optional, Tuple, List
import numpy as np

class HypotheticalDocumentDirectEmbeddingTransformer:
    """Transform queries by generating hypothetical answers and using combined embeddings."""
    
    def __init__(self, llm_model_func, embedding_func, enable_cache=True):
        """Initialize the hypothetical document transformer with combined embedding support.
        
        Args:
            llm_model_func: Function to call the language model
            embedding_func: Function to generate embeddings from text
            enable_cache: Whether to cache generated hypothetical documents and embeddings
        """
        self.llm_model_func = llm_model_func
        self.embedding_func = embedding_func
        self.cache = {} if enable_cache else None
        
    async def transform_query(self, original_query: str) -> Tuple[str, List[float], List[float], List[float]]:
        """Generate a hypothetical document that answers the query and combine embeddings.
        
        Args:
            original_query: Original user query
            
        Returns:
            Tuple of (generated hypothetical document, 
                     query embedding, 
                     hypothetical document embedding,
                     combined hyde embedding)
        """
        # Check cache first if enabled
        if self.cache is not None and original_query in self.cache:
            return self.cache[original_query]
            
        # Prompt for hypothetical document generation
        prompt = f"""Please write a passage to answer the question.
Question: {original_query}
Passage:"""
        
        # Generate the hypothetical document
        hypothetical_doc = await self.llm_model_func(prompt, max_tokens=512, temperature=0.7)
        
        # Generate embeddings using the same pattern as NanoVectorDBStorage
        # Wrap input in list and extract first element to ensure consistency
        query_embedding_list = await self.embedding_func([original_query])
        query_embedding = query_embedding_list[0]
        
        doc_embedding_list = await self.embedding_func([hypothetical_doc])
        doc_embedding = doc_embedding_list[0]
        
        # Combine embeddings by averaging them
        if isinstance(query_embedding, list) and isinstance(doc_embedding, list):
            combined_embedding = [(a + b) / 2 for a, b in zip(query_embedding, doc_embedding)]
        else:
            # Handle numpy arrays or other formats
            combined_embedding = (np.array(query_embedding) + np.array(doc_embedding)) / 2
            combined_embedding = combined_embedding.tolist()
        
        result = (hypothetical_doc, query_embedding, doc_embedding, combined_embedding)
        
        # Cache the result if caching is enabled
        if self.cache is not None:
            self.cache[original_query] = result
            
        return result
    
    async def find_similar_graph_elements(self, combined_embedding, graph_vectors, top_k=10):
        """Find graph elements most similar to the combined HyDE embedding.
        
        Args:
            combined_embedding: The averaged embedding vector combining query and hypothetical document
            graph_vectors: Dictionary mapping graph element IDs to their embedding vectors
            top_k: Number of top matches to return
            
        Returns:
            List of (graph_element_id, similarity_score) tuples sorted by similarity
        """
        results = []
        
        # Calculate similarity between the combined embedding and all graph vectors
        for element_id, element_embedding in graph_vectors.items():
            # Compute cosine similarity
            similarity = self._compute_cosine_similarity(combined_embedding, element_embedding)
            results.append((element_id, similarity))
        
        # Sort by similarity score in descending order and take top_k
        results.sort(key=lambda x: x[1], reverse=True)
        return results[:top_k]
    
    def _compute_cosine_similarity(self, vec1, vec2):
        """Compute cosine similarity between two vectors."""
        # Convert to numpy arrays if they aren't already
        vec1_array = np.array(vec1)
        vec2_array = np.array(vec2)
        
        # Calculate dot product
        dot_product = np.dot(vec1_array, vec2_array)
        
        # Calculate magnitudes
        magnitude1 = np.linalg.norm(vec1_array)
        magnitude2 = np.linalg.norm(vec2_array)
        
        # Prevent division by zero
        if magnitude1 == 0 or magnitude2 == 0:
            return 0
            
        # Calculate cosine similarity
        return dot_product / (magnitude1 * magnitude2)
    
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
            filename = f"hyp_doc_direct_{timestamp}.json"
            filepath = os.path.join(output_dir, filename)
            
            # Prepare the data
            data = {
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                "query": original_query,
                "model": model_name,
                "hypothetical_document": hypothetical_doc,
                "approach": "direct_embedding"
            }
            
            # Write to file
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
                
            print(f"Hypothetical document (direct embedding) saved to {filepath}")
            return filepath
            
        except Exception as e:
            print(f"Error saving hypothetical document to file: {e}")
            return None
