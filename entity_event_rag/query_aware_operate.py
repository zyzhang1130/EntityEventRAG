import asyncio
from typing import Union
from collections import defaultdict
import re

from .utils import (
    logger,
    clean_str,
    compute_mdhash_id,
    pack_user_ass_to_openai_messages,
    split_string_by_multi_markers,
)
from .base import (
    BaseGraphStorage,
    BaseKVStorage,
    BaseVectorStorage,
    TextChunkSchema,
    QueryParam,
)
from .prompt import PROMPTS
from .operate import (
    _handle_single_entity_extraction,
    _handle_single_relationship_extraction,
    _merge_nodes_then_upsert,
    _merge_edges_then_upsert,
    local_query,
    global_query,
    hybrid_query,
)

async def query_aware_extract_entities(
    query: str,
    chunks: dict[str, TextChunkSchema],
    knowledge_graph_inst: BaseGraphStorage,
    entity_vdb: BaseVectorStorage,
    relationships_vdb: BaseVectorStorage,
    global_config: dict,
) -> Union[BaseGraphStorage, None]:
    """Extract entities and relationships with awareness of the user's query"""
    # [Previous implementation remains unchanged]

async def query_aware_local_query(
    query: str,
    knowledge_graph_inst: BaseGraphStorage,
    entities_vdb: BaseVectorStorage,
    relationships_vdb: BaseVectorStorage,
    text_chunks_db: BaseKVStorage[TextChunkSchema],
    query_param: QueryParam,
    global_config: dict,
) -> str:
    """Try local query first, if fails then try query-aware extraction"""
    response = await local_query(
        query,
        knowledge_graph_inst,
        entities_vdb,
        relationships_vdb,
        text_chunks_db,
        query_param,
        global_config,
    )
    
    if response == PROMPTS["fail_response"]:
        # Try query-aware entity extraction
        chunks = await text_chunks_db.get_all()
        maybe_new_kg = await query_aware_extract_entities(
            query,
            chunks,
            knowledge_graph_inst,
            entities_vdb,
            relationships_vdb,
            global_config,
        )
        if maybe_new_kg is not None:
            # Try query again with updated KG
            response = await local_query(
                query,
                knowledge_graph_inst,
                entities_vdb,
                relationships_vdb,
                text_chunks_db,
                query_param,
                global_config,
            )
    
    return response

async def query_aware_global_query(
    query: str,
    knowledge_graph_inst: BaseGraphStorage,
    entities_vdb: BaseVectorStorage,
    relationships_vdb: BaseVectorStorage,
    text_chunks_db: BaseKVStorage[TextChunkSchema],
    query_param: QueryParam,
    global_config: dict,
) -> str:
    """Try global query first, if fails then try query-aware extraction"""
    response = await global_query(
        query,
        knowledge_graph_inst,
        entities_vdb,
        relationships_vdb,
        text_chunks_db,
        query_param,
        global_config,
    )
    
    if response == PROMPTS["fail_response"]:
        # Try query-aware entity extraction
        chunks = await text_chunks_db.get_all()
        maybe_new_kg = await query_aware_extract_entities(
            query,
            chunks,
            knowledge_graph_inst,
            entities_vdb,
            relationships_vdb,
            global_config,
        )
        if maybe_new_kg is not None:
            # Try query again with updated KG
            response = await global_query(
                query,
                knowledge_graph_inst,
                entities_vdb,
                relationships_vdb,
                text_chunks_db,
                query_param,
                global_config,
            )
    
    return response

async def query_aware_hybrid_query(
    query: str,
    knowledge_graph_inst: BaseGraphStorage,
    entities_vdb: BaseVectorStorage,
    relationships_vdb: BaseVectorStorage,
    text_chunks_db: BaseKVStorage[TextChunkSchema],
    query_param: QueryParam,
    global_config: dict,
) -> str:
    """Try hybrid query first, if fails then try query-aware extraction"""
    response = await hybrid_query(
        query,
        knowledge_graph_inst,
        entities_vdb,
        relationships_vdb,
        text_chunks_db,
        query_param,
        global_config,
    )
    
    if response == PROMPTS["fail_response"]:
        # Try query-aware entity extraction
        chunks = await text_chunks_db.get_all()
        maybe_new_kg = await query_aware_extract_entities(
            query,
            chunks,
            knowledge_graph_inst,
            entities_vdb,
            relationships_vdb,
            global_config,
        )
        if maybe_new_kg is not None:
            # Try query again with updated KG
            response = await hybrid_query(
                query,
                knowledge_graph_inst,
                entities_vdb,
                relationships_vdb,
                text_chunks_db,
                query_param,
                global_config,
            )
    
    return response
