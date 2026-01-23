"""Module for synergistic querying that combines entity and event knowledge graphs."""

import asyncio
from typing import Tuple
from .base import BaseGraphStorage, BaseKVStorage, BaseVectorStorage, QueryParam
from .utils import logger, truncate_list_by_token_size, list_of_list_to_csv
from .prompt import PROMPTS, GRAPH_FIELD_SEP
import json
from .operate import (
    _handle_entity_relation_summary,
    split_string_by_multi_markers,
    locate_json_string_body_from_string
)
from .hypothetical_document import HypotheticalDocumentTransformer

async def synergistic_query(
    query: str,
    # Entity KG components
    entity_graph: BaseGraphStorage,
    entity_vdb: BaseVectorStorage,
    # Event KG components
    event_graph: BaseGraphStorage,
    event_vdb: BaseVectorStorage,
    # Shared components
    text_chunks_db: BaseKVStorage,
    query_param: QueryParam,
    global_config: dict,
) -> str:
    """
    Query both entity and event KGs synergistically to provide a comprehensive answer.
    
    This method:
    1. Extracts keywords relevant to both entities and events
    2. Queries both KGs in parallel
    3. Finds connections between entities and events
    4. Combines the results in a meaningful way
    5. Generates a final, integrated response
    """
    use_model_func = global_config["llm_model_func"]
    
    # Apply hypothetical document transformation if enabled
    original_query = query
    if query_param.enable_hypothetical_document:
        # Initialize the hypothetical document transformer
        hypothetical_transformer = HypotheticalDocumentTransformer(
            llm_model_func=use_model_func,
            enable_cache=global_config.get("enable_llm_cache", True)
        )
        
        # Transform the query to a hypothetical document
        hypothetical_doc = await hypothetical_transformer.transform_query(query)
        
        # Log original and transformed queries if debug mode is enabled
        if query_param.hypothetical_document_debug:
            logger.info(f"Original query: {query}")
            logger.info(f"Hypothetical document: {hypothetical_doc}")
            
        # Replace original query with hypothetical document
        query = hypothetical_doc

    # Extract keywords for both entities and events
    kw_prompt = PROMPTS["keywords_extraction"].format(query=query)
    result = await use_model_func(kw_prompt)
    json_text = locate_json_string_body_from_string(result)
    
    try:
        keywords_data = json.loads(json_text)
        entity_keywords = keywords_data.get("low_level_keywords", [])  # For entities
        event_keywords = keywords_data.get("high_level_keywords", [])  # For events
        entity_keywords = ", ".join(entity_keywords)
        event_keywords = ", ".join(event_keywords)
    except json.JSONDecodeError as e:
        logger.error(f"Failed to parse keywords: {e}")
        return PROMPTS["fail_response"]

    # Query both KGs in parallel
    entity_context, event_context = await asyncio.gather(
        _build_entity_context(
            entity_keywords,
            entity_graph,
            entity_vdb,
            text_chunks_db,
            query_param,
        ),
        _build_event_context(
            event_keywords,
            event_graph,
            event_vdb,
            text_chunks_db,
            query_param,
        )
    )

    if query_param.only_need_context:
        return _combine_entity_event_contexts(entity_context, event_context)

    if entity_context is None and event_context is None:
        return PROMPTS["fail_response"]

    # Find connections between entities and events
    combined_context = await _enrich_contexts_with_connections(
        entity_context,
        event_context,
        entity_graph,
        event_graph,
        text_chunks_db,
        query_param,
    )

    # Generate final response
    sys_prompt = PROMPTS["synergistic_rag_response"].format(
        context_data=combined_context,
        response_type=query_param.response_type
    )
    response = await use_model_func(query, system_prompt=sys_prompt)

    # Clean up response
    if len(response) > len(sys_prompt):
        response = (
            response.replace(sys_prompt, "")
            .replace("user", "")
            .replace("model", "")
            .replace(query, "")
            .replace("<system>", "")
            .replace("</system>", "")
            .strip()
        )

    return response

async def _build_entity_context(
    keywords: str,
    knowledge_graph: BaseGraphStorage,
    entities_vdb: BaseVectorStorage,
    text_chunks_db: BaseKVStorage,
    query_param: QueryParam,
) -> str:
    """Build context from entity KG."""
    if not keywords:
        return None

    # Query entity vector DB
    results = await entities_vdb.query(keywords, top_k=query_param.top_k)
    if not results:
        return None

    # Get node data and degrees
    node_data = await asyncio.gather(
        *[knowledge_graph.get_node(r["entity_name"]) for r in results]
    )
    node_degrees = await asyncio.gather(
        *[knowledge_graph.node_degree(r["entity_name"]) for r in results]
    )

    # Combine data
    nodes = [
        {**n, "entity_name": k["entity_name"], "rank": d}
        for k, n, d in zip(results, node_data, node_degrees)
        if n is not None
    ]

    # Format as CSV
    section_list = [["id", "entity", "type", "description", "rank"]]
    for i, n in enumerate(nodes):
        section_list.append([
            i,
            n["entity_name"],
            n.get("entity_type", "UNKNOWN"),
            n.get("description", "UNKNOWN"),
            n["rank"]
        ])

    return list_of_list_to_csv(section_list)

async def _build_event_context(
    keywords: str,
    knowledge_graph: BaseGraphStorage,
    events_vdb: BaseVectorStorage,
    text_chunks_db: BaseKVStorage,
    query_param: QueryParam,
) -> str:
    """Build context from event KG."""
    if not keywords:
        return None

    # Query event vector DB
    results = await events_vdb.query(keywords, top_k=query_param.top_k)
    if not results:
        return None

    # Get node data and degrees
    node_data = await asyncio.gather(
        *[knowledge_graph.get_node(r["entity_name"]) for r in results]
    )
    node_degrees = await asyncio.gather(
        *[knowledge_graph.node_degree(r["entity_name"]) for r in results]
    )

    # Combine data
    events = [
        {**n, "event_name": k["entity_name"], "rank": d}
        for k, n, d in zip(results, node_data, node_degrees)
        if n is not None
    ]

    # Format as CSV
    section_list = [["id", "event", "type", "description", "rank"]]
    for i, e in enumerate(events):
        section_list.append([
            i,
            e["event_name"],
            e.get("entity_type", "EVENT"),
            e.get("description", "UNKNOWN"),
            e["rank"]
        ])

    return list_of_list_to_csv(section_list)

async def _enrich_contexts_with_connections(
    entity_context: str,
    event_context: str,
    entity_graph: BaseGraphStorage,
    event_graph: BaseGraphStorage,
    text_chunks_db: BaseKVStorage,
    query_param: QueryParam,
) -> str:
    """Find and add connections between entities and events."""
    # Extract entity and event names
    entity_names = set()
    event_names = set()

    if entity_context:
        for line in entity_context.split('\n')[1:]:  # Skip header
            if line.strip():
                parts = line.split(',')
                if len(parts) > 1:
                    entity_names.add(parts[1].strip())

    if event_context:
        for line in event_context.split('\n')[1:]:  # Skip header
            if line.strip():
                parts = line.split(',')
                if len(parts) > 1:
                    event_names.add(parts[1].strip())

    # Find connections
    connections = []
    for entity in entity_names:
        entity_node = await entity_graph.get_node(entity)
        if entity_node:
            entity_desc = entity_node.get('description', '')
            for event in event_names:
                event_node = await event_graph.get_node(event)
                if event_node:
                    event_desc = event_node.get('description', '')
                    # Check if entity appears in event description or vice versa
                    if (entity.lower() in event_desc.lower() or 
                        event.lower() in entity_desc.lower()):
                        connections.append({
                            'entity': entity,
                            'event': event,
                            'description': f"Entity '{entity}' is involved in event '{event}'"
                        })

    # Format connections as CSV
    connections_list = [["id", "entity", "event", "description"]]
    for i, conn in enumerate(connections):
        connections_list.append([
            i,
            conn['entity'],
            conn['event'],
            conn['description']
        ])
    connections_context = list_of_list_to_csv(connections_list)

    # Combine all contexts
    return f"""
-----Entities-----
```csv
{entity_context if entity_context else 'No relevant entities found.'}
```
-----Events-----
```csv
{event_context if event_context else 'No relevant events found.'}
```
-----Connections-----
```csv
{connections_context if connections else 'No connections found between entities and events.'}
```
"""

def _combine_entity_event_contexts(entity_context: str, event_context: str) -> str:
    """Combine entity and event contexts."""
    return f"""
-----Entities-----
```csv
{entity_context if entity_context else 'No relevant entities found.'}
```
-----Events-----
```csv
{event_context if event_context else 'No relevant events found.'}
```
"""
