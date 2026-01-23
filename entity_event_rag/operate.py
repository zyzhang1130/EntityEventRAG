import asyncio
import json
import re
import time
from typing import Union
from collections import Counter, defaultdict, deque
from dataclasses import asdict
import warnings
from .utils import (
    logger,
    clean_str,
    compute_mdhash_id,
    decode_tokens_by_tiktoken,
    encode_string_by_tiktoken,
    is_float_regex,
    list_of_list_to_csv,
    pack_user_ass_to_openai_messages,
    split_string_by_multi_markers,
    truncate_list_by_token_size,
    process_combine_contexts,
    locate_json_string_body_from_string,
)
from .base import (
    BaseGraphStorage,
    BaseKVStorage,
    BaseVectorStorage,
    TextChunkSchema,
    QueryParam,
)
from .prompt import GRAPH_FIELD_SEP, PROMPTS


def _normalize_similarity(distance) -> float:
    """Convert vector distance to a normalized similarity in [0, 1]."""
    if distance is None:
        return 1.0
    try:
        value = float(distance)
    except (TypeError, ValueError):
        return 0.0
    # Distances closer to 0 indicate higher similarity; clamp to avoid negatives or >1 values
    if value < 0:
        value = 0.0
    if value > 1:
        value = 1.0
    return max(0.0, 1.0 - value)


def _clamp_score(score: float) -> float:
    """Ensure propagated similarity scores stay within [0, 1]."""
    try:
        return max(0.0, min(1.0, float(score)))
    except (TypeError, ValueError):
        return 0.0


def _safe_weight(weight) -> float:
    """Safely parse edge weights; default to 1.0 when unavailable."""
    try:
        parsed = float(weight)
    except (TypeError, ValueError):
        return 1.0
    return max(0.0, parsed)


def chunking_by_token_size(
    content: str, overlap_token_size=128, max_token_size=1024, tiktoken_model="gpt-4o"
):
    tokens = encode_string_by_tiktoken(content, model_name=tiktoken_model)
    results = []
    for index, start in enumerate(
        range(0, len(tokens), max_token_size - overlap_token_size)
    ):
        chunk_content = decode_tokens_by_tiktoken(
            tokens[start : start + max_token_size], model_name=tiktoken_model
        )
        results.append(
            {
                "tokens": min(max_token_size, len(tokens) - start),
                "content": chunk_content.strip(),
                "chunk_order_index": index,
            }
        )
    return results


async def _handle_entity_relation_summary(
    entity_or_relation_name: str,
    description: str,
    global_config: dict,
) -> str:
    use_llm_func: callable = global_config["llm_model_func"]
    llm_max_tokens = global_config["llm_model_max_token_size"]
    tiktoken_model_name = global_config["tiktoken_model_name"]
    summary_max_tokens = global_config["entity_summary_to_max_tokens"]

    tokens = encode_string_by_tiktoken(description, model_name=tiktoken_model_name)
    if len(tokens) < summary_max_tokens:  # No need for summary
        return description
    prompt_template = PROMPTS["summarize_entity_descriptions"]
    use_description = decode_tokens_by_tiktoken(
        tokens[:llm_max_tokens], model_name=tiktoken_model_name
    )
    context_base = dict(
        entity_name=entity_or_relation_name,
        description_list=use_description.split(GRAPH_FIELD_SEP),
    )
    use_prompt = prompt_template.format(**context_base)
    logger.debug(f"Trigger summary: {entity_or_relation_name}")
    summary = await use_llm_func(use_prompt, max_tokens=summary_max_tokens)
    return summary


async def _handle_single_entity_extraction(
    record_attributes: list[str],
    chunk_key: str,
    extraction_type: str = "entity",  # Add extraction_type parameter
):
    if len(record_attributes) < 4 or record_attributes[0] != '"entity"':
        return None
    # Handle entity name based on extraction type
    base_entity_name = clean_str(record_attributes[1].upper())
    # For entity extraction, append chunk_key; for event extraction, use name directly
    entity_name = f"{base_entity_name}_{chunk_key}" if extraction_type == "entity" else base_entity_name
    if not entity_name.strip():
        return None
    entity_type = clean_str(record_attributes[2].upper())
    entity_description = clean_str(record_attributes[3])
    entity_source_id = chunk_key
    return dict(
        entity_name=entity_name,
        entity_type=entity_type,
        description=entity_description,
        source_id=entity_source_id,
    )

async def _handle_single_relationship_extraction(
    record_attributes: list[str],
    chunk_key: str,
    extraction_type: str = "entity",  # Add extraction_type parameter
):
    if len(record_attributes) < 5 or record_attributes[0] != '"relationship"':
        return None
    # Handle source/target names based on extraction type
    base_source = clean_str(record_attributes[1].upper())
    base_target = clean_str(record_attributes[2].upper())
    # For entity extraction, append chunk_key; for event extraction, use names directly
    source = f"{base_source}_{chunk_key}" if extraction_type == "entity" else base_source
    target = f"{base_target}_{chunk_key}" if extraction_type == "entity" else base_target
    edge_description = clean_str(record_attributes[3])
    edge_keywords = clean_str(record_attributes[4])
    edge_source_id = chunk_key
    weight = (
        float(record_attributes[-1]) if is_float_regex(record_attributes[-1]) else 1.0
    )
    return dict(
        src_id=source,
        tgt_id=target,
        weight=weight,
        description=edge_description,
        keywords=edge_keywords,
        source_id=edge_source_id,
    )


async def _merge_nodes_then_upsert(
    entity_name: str,
    nodes_data: list[dict],
    knowledge_graph_inst: BaseGraphStorage,
    global_config: dict,
):
    already_entitiy_types = []
    already_source_ids = []
    already_description = []

    already_node = await knowledge_graph_inst.get_node(entity_name)
    if already_node is not None:
        already_entitiy_types.append(already_node["entity_type"])
        already_source_ids.extend(
            split_string_by_multi_markers(already_node["source_id"], [GRAPH_FIELD_SEP])
        )
        already_description.append(already_node["description"])

    entity_type = sorted(
        Counter(
            [dp["entity_type"] for dp in nodes_data] + already_entitiy_types
        ).items(),
        key=lambda x: x[1],
        reverse=True,
    )[0][0]
    description = GRAPH_FIELD_SEP.join(
        sorted(set([dp["description"] for dp in nodes_data] + already_description))
    )
    source_id = GRAPH_FIELD_SEP.join(
        set([dp["source_id"] for dp in nodes_data] + already_source_ids)
    )
    description = await _handle_entity_relation_summary(
        entity_name, description, global_config
    )
    node_data = dict(
        entity_type=entity_type,
        description=description,
        source_id=source_id,
    )
    await knowledge_graph_inst.upsert_node(
        entity_name,
        node_data=node_data,
    )
    node_data["entity_name"] = entity_name
    return node_data


async def _merge_edges_then_upsert(
    src_id: str,
    tgt_id: str,
    edges_data: list[dict],
    knowledge_graph_inst: BaseGraphStorage,
    global_config: dict,
):
    already_weights = []
    already_source_ids = []
    already_description = []
    already_keywords = []

    if await knowledge_graph_inst.has_edge(src_id, tgt_id):
        already_edge = await knowledge_graph_inst.get_edge(src_id, tgt_id)
        already_weights.append(already_edge["weight"])
        already_source_ids.extend(
            split_string_by_multi_markers(already_edge["source_id"], [GRAPH_FIELD_SEP])
        )
        already_description.append(already_edge["description"])
        already_keywords.extend(
            split_string_by_multi_markers(already_edge["keywords"], [GRAPH_FIELD_SEP])
        )

    weight = sum([dp["weight"] for dp in edges_data] + already_weights)
    description = GRAPH_FIELD_SEP.join(
        sorted(set([dp["description"] for dp in edges_data] + already_description))
    )
    keywords = GRAPH_FIELD_SEP.join(
        sorted(set([dp["keywords"] for dp in edges_data] + already_keywords))
    )
    source_id = GRAPH_FIELD_SEP.join(
        set([dp["source_id"] for dp in edges_data] + already_source_ids)
    )
    for need_insert_id in [src_id, tgt_id]:
        if not (await knowledge_graph_inst.has_node(need_insert_id)):
            await knowledge_graph_inst.upsert_node(
                need_insert_id,
                node_data={
                    "source_id": source_id,
                    "description": description,
                    "entity_type": '"UNKNOWN"',
                },
            )
    description = await _handle_entity_relation_summary(
        (src_id, tgt_id), description, global_config
    )
    await knowledge_graph_inst.upsert_edge(
        src_id,
        tgt_id,
        edge_data=dict(
            weight=weight,
            description=description,
            keywords=keywords,
            source_id=source_id,
        ),
    )

    edge_data = dict(
        src_id=src_id,
        tgt_id=tgt_id,
        description=description,
        keywords=keywords,
    )

    return edge_data


async def extract_entities(
    chunks: dict[str, TextChunkSchema],
    knowledge_graph_inst: BaseGraphStorage,
    entity_vdb: BaseVectorStorage,
    relationships_vdb: BaseVectorStorage,
    global_config: dict,
) -> Union[BaseGraphStorage, None]:
    use_llm_func: callable = global_config["llm_model_func"]
    entity_extract_max_gleaning = global_config["entity_extract_max_gleaning"]
    extraction_type = global_config.get("extraction_type", "entity")  # Get extraction type from config

    ordered_chunks = list(chunks.items())

    # Use the appropriate prompt based on extraction_type
    extraction_prompt = global_config.get("extraction_prompt", "entity_extraction")
    entity_extract_prompt = PROMPTS[extraction_prompt]
    
    
    context_base = dict(
        tuple_delimiter=PROMPTS["DEFAULT_TUPLE_DELIMITER"],
        record_delimiter=PROMPTS["DEFAULT_RECORD_DELIMITER"],
        completion_delimiter=PROMPTS["DEFAULT_COMPLETION_DELIMITER"],
        entity_types=",".join(PROMPTS["DEFAULT_ENTITY_TYPES"]),
        event_types=",".join(PROMPTS.get("DEFAULT_EVENT_TYPES", PROMPTS["DEFAULT_ENTITY_TYPES"])),
    )
    continue_prompt = PROMPTS["entiti_continue_extraction"]
    if_loop_prompt = PROMPTS["entiti_if_loop_extraction"]

    already_processed = 0
    already_entities = 0
    already_relations = 0

    async def _process_single_content(chunk_key_dp: tuple[str, TextChunkSchema]):
        nonlocal already_processed, already_entities, already_relations
        chunk_key = chunk_key_dp[0]
        chunk_dp = chunk_key_dp[1]
        content = chunk_dp["content"]
        hint_prompt = entity_extract_prompt.format(**context_base, input_text=content)
        final_result = await use_llm_func(hint_prompt)

        history = pack_user_ass_to_openai_messages(hint_prompt, final_result)
        for now_glean_index in range(entity_extract_max_gleaning):
            glean_result = await use_llm_func(continue_prompt, history_messages=history)

            history += pack_user_ass_to_openai_messages(continue_prompt, glean_result)
            final_result += glean_result
            if now_glean_index == entity_extract_max_gleaning - 1:
                break

            if_loop_result: str = await use_llm_func(
                if_loop_prompt, history_messages=history
            )
            if_loop_result = if_loop_result.strip().strip('"').strip("'").lower()
            if if_loop_result != "yes":
                break

        records = split_string_by_multi_markers(
            final_result,
            [context_base["record_delimiter"], context_base["completion_delimiter"]],
        )

        maybe_nodes = defaultdict(list)
        maybe_edges = defaultdict(list)
        for record in records:
            record = re.search(r"\((.*)\)", record)
            if record is None:
                continue
            record = record.group(1)
            record_attributes = split_string_by_multi_markers(
                record, [context_base["tuple_delimiter"]]
            )
            if_entities = await _handle_single_entity_extraction(
                record_attributes, chunk_key, extraction_type  # Pass extraction_type
            )
            if if_entities is not None:
                maybe_nodes[if_entities["entity_name"]].append(if_entities)
                continue

            if_relation = await _handle_single_relationship_extraction(
                record_attributes, chunk_key, extraction_type  # Pass extraction_type
            )
            if if_relation is not None:
                maybe_edges[(if_relation["src_id"], if_relation["tgt_id"])].append(
                    if_relation
                )
        already_processed += 1
        already_entities += len(maybe_nodes)
        already_relations += len(maybe_edges)
        now_ticks = PROMPTS["process_tickers"][
            already_processed % len(PROMPTS["process_tickers"])
        ]
        print(
            f"{now_ticks} Processed {already_processed} chunks, {already_entities} entities(duplicated), {already_relations} relations(duplicated)\r",
            end="",
            flush=True,
        )
        return dict(maybe_nodes), dict(maybe_edges)

    # use_llm_func is wrapped in ascynio.Semaphore, limiting max_async callings
    results = await asyncio.gather(
        *[_process_single_content(c) for c in ordered_chunks]
    )
    print()  # clear the progress bar
    maybe_nodes = defaultdict(list)
    maybe_edges = defaultdict(list)
    for m_nodes, m_edges in results:
        for k, v in m_nodes.items():
            maybe_nodes[k].extend(v)
        for k, v in m_edges.items():
            maybe_edges[tuple(sorted(k))].extend(v)
    all_entities_data = await asyncio.gather(
        *[
            _merge_nodes_then_upsert(k, v, knowledge_graph_inst, global_config)
            for k, v in maybe_nodes.items()
        ]
    )
    all_relationships_data = await asyncio.gather(
        *[
            _merge_edges_then_upsert(k[0], k[1], v, knowledge_graph_inst, global_config)
            for k, v in maybe_edges.items()
        ]
    )
    if not len(all_entities_data):
        logger.warning("Didn't extract any entities, maybe your LLM is not working")
        return None
    if not len(all_relationships_data):
        logger.warning(
            "Didn't extract any relationships, maybe your LLM is not working"
        )
        return None

    if entity_vdb is not None:
        data_for_vdb = {
            compute_mdhash_id(dp["entity_name"], prefix="ent-"): {
                "content": dp["entity_name"] + dp["description"],
                "entity_name": dp["entity_name"],
            }
            for dp in all_entities_data
        }
        await entity_vdb.upsert(data_for_vdb)

    if relationships_vdb is not None:
        data_for_vdb = {
            compute_mdhash_id(dp["src_id"] + dp["tgt_id"], prefix="rel-"): {
                "src_id": dp["src_id"],
                "tgt_id": dp["tgt_id"],
                "content": dp["keywords"]
                + dp["src_id"]
                + dp["tgt_id"]
                + dp["description"],
            }
            for dp in all_relationships_data
        }
        await relationships_vdb.upsert(data_for_vdb)

    return knowledge_graph_inst


async def local_query(
    query,
    knowledge_graph_inst: BaseGraphStorage,
    entities_vdb: BaseVectorStorage,
    relationships_vdb: BaseVectorStorage,
    text_chunks_db: BaseKVStorage[TextChunkSchema],
    query_param: QueryParam,
    global_config: dict,
) -> str:
    context = None
    use_model_func = global_config["llm_model_func"]

    kw_prompt_temp = PROMPTS["keywords_extraction"]
    kw_prompt = kw_prompt_temp.format(query=query, language="English", examples="")
    result = await use_model_func(kw_prompt)
    json_text = locate_json_string_body_from_string(result)

    try:
        keywords_data = json.loads(json_text)
        keywords = keywords_data.get("low_level_keywords", [])
        keywords = ", ".join(keywords)
    except json.JSONDecodeError:
        try:
            result = (
                result.replace(kw_prompt[:-1], "")
                .replace("user", "")
                .replace("model", "")
                .strip()
            )
            result = "{" + result.split("{")[-1].split("}")[0] + "}"

            keywords_data = json.loads(result)
            keywords = keywords_data.get("low_level_keywords", [])
            keywords = ", ".join(keywords)
        # Handle parsing error
        except json.JSONDecodeError as e:
            print(f"JSON parsing error: {e}")
            return PROMPTS["fail_response"]
    if keywords:
        context = await _build_local_query_context(
            keywords,
            knowledge_graph_inst,
            entities_vdb,
            text_chunks_db,
            query_param,
        )
    if query_param.only_need_context:
        return context
    if context is None:
        return PROMPTS["fail_response"]
    sys_prompt_temp = PROMPTS["rag_response"]
    sys_prompt = sys_prompt_temp.format(
        context_data=context, response_type=query_param.response_type, user_prompt=""
    )
    response = await use_model_func(
        query,
        system_prompt=sys_prompt,
    )
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


async def _build_local_query_context(
    query,
    knowledge_graph_inst: BaseGraphStorage,
    entities_vdb: BaseVectorStorage,
    text_chunks_db: BaseKVStorage[TextChunkSchema],
    query_param: QueryParam,
):
    results = await entities_vdb.query(query, top_k=query_param.top_k)

    if not len(results):
        return None
    node_datas = await asyncio.gather(
        *[knowledge_graph_inst.get_node(r["entity_name"]) for r in results]
    )
    if not all([n is not None for n in node_datas]):
        logger.warning("Some nodes are missing, maybe the storage is damaged")
    node_degrees = await asyncio.gather(
        *[knowledge_graph_inst.node_degree(r["entity_name"]) for r in results]
    )
    node_datas = [
        {**n, "entity_name": k["entity_name"], "rank": d}
        for k, n, d in zip(results, node_datas, node_degrees)
        if n is not None
    ]  # what is this text_chunks_db doing.  dont remember it in airvx.  check the diagram.
    use_text_units = await _find_most_related_text_unit_from_entities(
        node_datas, query_param, text_chunks_db, knowledge_graph_inst
    )
    use_relations = await _find_most_related_edges_from_entities(
        node_datas, query_param, knowledge_graph_inst
    )
    logger.info(
        f"Local query uses {len(node_datas)} entites, {len(use_relations)} relations, {len(use_text_units)} text units"
    )
    entites_section_list = [["id", "entity", "type", "description", "rank"]]
    for i, n in enumerate(node_datas):
        entites_section_list.append(
            [
                i,
                n["entity_name"],
                n.get("entity_type", "UNKNOWN"),
                n.get("description", "UNKNOWN"),
                n["rank"],
            ]
        )
    entities_context = list_of_list_to_csv(entites_section_list)

    relations_section_list = [
        ["id", "source", "target", "description", "keywords", "weight", "rank"]
    ]
    for i, e in enumerate(use_relations):
        relations_section_list.append(
            [
                i,
                e["src_tgt"][0],
                e["src_tgt"][1],
                e["description"],
                e["keywords"],
                e["weight"],
                e["rank"],
            ]
        )
    relations_context = list_of_list_to_csv(relations_section_list)

    text_units_section_list = [["id", "content"]]
    for i, t in enumerate(use_text_units):
        text_units_section_list.append([i, t["content"]])
    text_units_context = list_of_list_to_csv(text_units_section_list)
    return f"""
-----Entities-----
```csv
{entities_context}
```
-----Relationships-----
```csv
{relations_context}
```
-----Sources-----
```csv
{text_units_context}
```
"""


async def _find_most_related_text_unit_from_entities(
    node_datas: list[dict],
    query_param: QueryParam,
    text_chunks_db: BaseKVStorage[TextChunkSchema],
    knowledge_graph_inst: BaseGraphStorage,
):
    text_units = [
        split_string_by_multi_markers(dp["source_id"], [GRAPH_FIELD_SEP])
        for dp in node_datas
    ]
    chunk_score_lookup = {}
    for dp, chunks_for_node in zip(node_datas, text_units):
        node_score = dp.get("score", 0.0)
        for chunk_id in chunks_for_node:
            if chunk_id:
                chunk_score_lookup[chunk_id] = max(
                    chunk_score_lookup.get(chunk_id, 0.0), node_score
                )
    edges = await asyncio.gather(
        *[knowledge_graph_inst.get_node_edges(dp["entity_name"]) for dp in node_datas]
    )
    all_one_hop_nodes = set()
    for this_edges in edges:
        if not this_edges:
            continue
        all_one_hop_nodes.update([e[1] for e in this_edges])

    all_one_hop_nodes = list(all_one_hop_nodes)
    all_one_hop_nodes_data = await asyncio.gather(
        *[knowledge_graph_inst.get_node(e) for e in all_one_hop_nodes]
    )

    # Add null check for node data
    all_one_hop_text_units_lookup = {
        k: set(split_string_by_multi_markers(v["source_id"], [GRAPH_FIELD_SEP]))
        for k, v in zip(all_one_hop_nodes, all_one_hop_nodes_data)
        if v is not None and "source_id" in v  # Add source_id check
    }

    all_text_units_lookup = {}
    for index, (this_text_units, this_edges) in enumerate(zip(text_units, edges)):
        for c_id in this_text_units:
            if c_id in all_text_units_lookup:
                continue
            relation_counts = 0
            if this_edges:  # Add check for None edges
                for e in this_edges:
                    if (
                        e[1] in all_one_hop_text_units_lookup
                        and c_id in all_one_hop_text_units_lookup[e[1]]
                    ):
                        relation_counts += 1

            chunk_data = await text_chunks_db.get_by_id(c_id)
            if chunk_data is not None and "content" in chunk_data:  # Add content check
                all_text_units_lookup[c_id] = {
                    "data": chunk_data,
                    "order": index,
                    "relation_counts": relation_counts,
                    "score": chunk_score_lookup.get(c_id, 0.0),
                }

    # Filter out None values and ensure data has content
    all_text_units = [
        {"id": k, **v}
        for k, v in all_text_units_lookup.items()
        if v is not None and v.get("data") is not None and "content" in v["data"]
    ]

    if not all_text_units:
        logger.warning("No valid text units found")
        return []

    all_text_units = sorted(
        all_text_units,
        key=lambda x: (x.get("score", 0.0), x["relation_counts"], -x["order"]),
        reverse=True,
    )

    all_text_units = truncate_list_by_token_size(
        all_text_units,
        key=lambda x: x["data"]["content"],
        max_token_size=query_param.max_token_for_text_unit,
    )

    all_text_units = [t["data"] for t in all_text_units]
    return all_text_units


async def _find_most_related_edges_from_entities(
    node_datas: list[dict],
    query_param: QueryParam,
    knowledge_graph_inst: BaseGraphStorage,
):
    node_lookup = {dp.get("entity_name"): dp for dp in node_datas if dp.get("entity_name")}
    all_related_edges = await asyncio.gather(
        *[knowledge_graph_inst.get_node_edges(dp["entity_name"]) for dp in node_datas]
    )
    all_edges = []
    seen = set()

    for this_edges in all_related_edges:
        for e in this_edges:
            sorted_edge = tuple(sorted(e))
            if sorted_edge not in seen:
                seen.add(sorted_edge)
                all_edges.append(sorted_edge)

    edge_detail_cache = {}

    async def _get_edge(src: str, tgt: str):
        key = tuple(sorted((src, tgt)))
        if key not in edge_detail_cache:
            edge_detail_cache[key] = await knowledge_graph_inst.get_edge(src, tgt)
        return edge_detail_cache[key]

    all_edges_pack = await asyncio.gather(
        *[_get_edge(e[0], e[1]) for e in all_edges]
    )
    all_edges_degree = await asyncio.gather(
        *[knowledge_graph_inst.edge_degree(e[0], e[1]) for e in all_edges]
    )
    all_edges_data = [
        {
            "src_tgt": k,
            "rank": d,
            **v,
        }
        for k, v, d in zip(all_edges, all_edges_pack, all_edges_degree)
        if v is not None
    ]
    for edge in all_edges_data:
        src, tgt = edge["src_tgt"]
        edge["weight"] = _safe_weight(edge.get("weight", 1.0))
        src_score = node_lookup.get(src, {}).get("score", 0.0)
        tgt_score = node_lookup.get(tgt, {}).get("score", 0.0)
        edge["score"] = max(src_score, tgt_score)
    all_edges_data = sorted(
        all_edges_data,
        key=lambda x: (
            x.get("score", 0.0),
            x["rank"],
            x.get("weight", 0.0),
        ),
        reverse=True,
    )
    all_edges_data = truncate_list_by_token_size(
        all_edges_data,
        key=lambda x: x["description"],
        max_token_size=query_param.max_token_for_global_context,
    )
    return all_edges_data


async def _expand_entity_event_nodes(
    knowledge_graph_inst: BaseGraphStorage,
    seed_entities: list[dict],
    seed_events: list[dict],
    query_param: QueryParam,
) -> dict[str, dict]:
    hop_depth = max(0, int(getattr(query_param, "event_hop_depth", 2)))
    decay = _clamp_score(getattr(query_param, "event_neighbor_decay", 0.6))

    node_attr_cache: dict[str, dict] = {}
    degree_cache: dict[str, int] = {}
    edge_cache: dict[str, list[tuple]] = {}
    edge_detail_cache = {}

    async def get_node_attr(node_id: str):
        if node_id not in node_attr_cache:
            node_attr_cache[node_id] = await knowledge_graph_inst.get_node(node_id)
        return node_attr_cache[node_id]

    async def get_degree(node_id: str) -> int:
        if node_id not in degree_cache:
            degree_cache[node_id] = await knowledge_graph_inst.node_degree(node_id)
        return degree_cache[node_id]

    async def get_edges(node_id: str):
        if node_id not in edge_cache:
            edge_cache[node_id] = await knowledge_graph_inst.get_node_edges(node_id) or []
        return edge_cache[node_id]

    async def get_edge_detail(src: str, tgt: str):
        key = tuple(sorted((src, tgt)))
        if key not in edge_detail_cache:
            edge_detail_cache[key] = await knowledge_graph_inst.get_edge(src, tgt)
        return edge_detail_cache[key]

    node_lookup: dict[str, dict] = {}
    for base in seed_entities + seed_events:
        node_id = base.get("entity_name")
        if not node_id:
            continue
        entry = dict(base)
        entry["entity_name"] = node_id
        entry["score"] = _clamp_score(entry.get("score", 1.0))
        entry["depth"] = int(entry.get("depth", 0))
        entry["node_type"] = entry.get("node_type", "").strip('"')
        entry.setdefault("rank", base.get("rank", 0))
        node_lookup[node_id] = entry

    queue = deque()
    for seed in seed_events:
        node_id = seed.get("entity_name")
        if node_id:
            queue.append((node_id, _clamp_score(seed.get("score", 1.0)), 0))

    best_scores = {node_id: data.get("score", 0.0) for node_id, data in node_lookup.items()}

    while queue:
        current_id, current_score, depth = queue.popleft()
        if depth >= hop_depth:
            continue
        edges = await get_edges(current_id)
        if not edges:
            continue
        for edge in edges:
            if len(edge) < 2:
                continue
            neighbor_id = edge[1] if edge[0] == current_id else edge[0]
            if neighbor_id == current_id:
                continue
            neighbor_attr = await get_node_attr(neighbor_id)
            if neighbor_attr is None:
                continue
            node_type = neighbor_attr.get("node_type", "").strip('"').lower()
            if node_type not in {"entity", "event"}:
                continue
            edge_detail = await get_edge_detail(current_id, neighbor_id)
            weight = _safe_weight(edge_detail.get("weight", 1.0)) if edge_detail else 1.0
            propagated = _clamp_score(current_score * decay * min(weight, 1.0))
            existing = node_lookup.get(neighbor_id)
            if existing is None:
                degree = await get_degree(neighbor_id)
                enriched = dict(neighbor_attr)
                enriched["entity_name"] = neighbor_id
                enriched["node_type"] = node_type
                enriched["rank"] = degree
                enriched["score"] = propagated
                enriched["depth"] = depth + 1
                node_lookup[neighbor_id] = enriched
            else:
                if propagated > existing.get("score", 0.0):
                    existing["score"] = propagated
                existing["depth"] = min(existing.get("depth", depth + 1), depth + 1)
            if depth + 1 < hop_depth and propagated > best_scores.get(neighbor_id, -1.0):
                best_scores[neighbor_id] = propagated
                queue.append((neighbor_id, propagated, depth + 1))

    for node_id, data in node_lookup.items():
        node_attr = await get_node_attr(node_id)
        if node_attr:
            data.setdefault("description", node_attr.get("description"))
            data.setdefault("source_id", node_attr.get("source_id"))
            data.setdefault("node_type", node_attr.get("node_type", "").strip('"'))
        data["score"] = _clamp_score(data.get("score", 0.0))
        data["depth"] = int(max(0, data.get("depth", 0)))
        if "rank" not in data or data["rank"] is None:
            data["rank"] = await get_degree(node_id)

    return node_lookup


async def _assemble_entity_event_context(
    nodes_dict: dict[str, dict],
    query_param: QueryParam,
    text_chunks_db: BaseKVStorage[TextChunkSchema],
    knowledge_graph_inst: BaseGraphStorage,
):
    if not nodes_dict:
        return None, [], []

    sorted_nodes = sorted(
        nodes_dict.values(),
        key=lambda x: (
            x.get("score", 0.0),
            -x.get("depth", 0),
            x.get("rank", 0),
        ),
        reverse=True,
    )

    use_text_units = await _find_most_related_text_unit_from_entities(
        sorted_nodes, query_param, text_chunks_db, knowledge_graph_inst
    )

    use_relations = await _find_most_related_edges_from_entities(
        sorted_nodes, query_param, knowledge_graph_inst
    )

    entities_section_list = [
        ["id", "node", "type", "description", "rank", "score", "hop_depth"]
    ]
    for idx, node in enumerate(sorted_nodes):
        entities_section_list.append(
            [
                idx,
                node.get("entity_name"),
                node.get("node_type", "UNKNOWN"),
                node.get("description", "UNKNOWN"),
                node.get("rank", 0),
                f"{node.get('score', 0.0):.4f}",
                node.get("depth", 0),
            ]
        )
    entities_context = list_of_list_to_csv(entities_section_list)

    relations_section_list = [
        [
            "id",
            "source",
            "target",
            "description",
            "keywords",
            "weight",
            "rank",
            "score",
        ]
    ]
    for idx, rel in enumerate(use_relations):
        relations_section_list.append(
            [
                idx,
                rel["src_tgt"][0],
                rel["src_tgt"][1],
                rel.get("description", ""),
                rel.get("keywords", ""),
                rel.get("weight", 0.0),
                rel.get("rank", 0),
                f"{rel.get('score', 0.0):.4f}",
            ]
        )
    relations_context = list_of_list_to_csv(relations_section_list)

    text_units_section_list = [["id", "content"]]
    for idx, unit in enumerate(use_text_units):
        text_units_section_list.append([idx, unit.get("content", "")])
    text_units_context = list_of_list_to_csv(text_units_section_list)

    context_data = f"""
-----Nodes-----
```csv
{entities_context}
```
-----Relationships-----
```csv
{relations_context}
```
-----Sources-----
```csv
{text_units_context}
```
"""

    return context_data, sorted_nodes, use_relations


async def global_query(
    query,
    knowledge_graph_inst: BaseGraphStorage,
    entities_vdb: BaseVectorStorage,
    relationships_vdb: BaseVectorStorage,
    text_chunks_db: BaseKVStorage[TextChunkSchema],
    query_param: QueryParam,
    global_config: dict,
) -> str:
    context = None
    use_model_func = global_config["llm_model_func"]

    kw_prompt_temp = PROMPTS["keywords_extraction"]
    kw_prompt = kw_prompt_temp.format(query=query, language="English", examples="")
    result = await use_model_func(kw_prompt)
    json_text = locate_json_string_body_from_string(result)

    try:
        keywords_data = json.loads(json_text)
        keywords = keywords_data.get("high_level_keywords", [])
        keywords = ", ".join(keywords)
    except json.JSONDecodeError:
        try:
            result = (
                result.replace(kw_prompt[:-1], "")
                .replace("user", "")
                .replace("model", "")
                .strip()
            )
            result = "{" + result.split("{")[-1].split("}")[0] + "}"

            keywords_data = json.loads(result)
            keywords = keywords_data.get("high_level_keywords", [])
            keywords = ", ".join(keywords)

        except json.JSONDecodeError as e:
            # Handle parsing error
            print(f"JSON parsing error: {e}")
            return PROMPTS["fail_response"]
    if keywords:
        context = await _build_global_query_context(
            keywords,
            knowledge_graph_inst,
            entities_vdb,
            relationships_vdb,
            text_chunks_db,
            query_param,
        )

    if query_param.only_need_context:
        return context
    if context is None:
        return PROMPTS["fail_response"]

    sys_prompt_temp = PROMPTS["rag_response"]
    sys_prompt = sys_prompt_temp.format(
        context_data=context, response_type=query_param.response_type, user_prompt=""
    )
    response = await use_model_func(
        query,
        system_prompt=sys_prompt,
    )
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


async def _build_global_query_context(
    keywords,
    knowledge_graph_inst: BaseGraphStorage,
    entities_vdb: BaseVectorStorage,
    relationships_vdb: BaseVectorStorage,
    text_chunks_db: BaseKVStorage[TextChunkSchema],
    query_param: QueryParam,
):
    results = await relationships_vdb.query(keywords, top_k=query_param.top_k)

    if not len(results):
        return None

    edge_datas = await asyncio.gather(
        *[knowledge_graph_inst.get_edge(r["src_id"], r["tgt_id"]) for r in results]
    )

    if not all([n is not None for n in edge_datas]):
        logger.warning("Some edges are missing, maybe the storage is damaged")
    edge_degree = await asyncio.gather(
        *[knowledge_graph_inst.edge_degree(r["src_id"], r["tgt_id"]) for r in results]
    )
    edge_datas = [
        {"src_id": k["src_id"], "tgt_id": k["tgt_id"], "rank": d, **v}
        for k, v, d in zip(results, edge_datas, edge_degree)
        if v is not None
    ]
    edge_datas = sorted(
        edge_datas, key=lambda x: (x["rank"], x["weight"]), reverse=True
    )
    edge_datas = truncate_list_by_token_size(
        edge_datas,
        key=lambda x: x["description"],
        max_token_size=query_param.max_token_for_global_context,
    )

    use_entities = await _find_most_related_entities_from_relationships(
        edge_datas, query_param, knowledge_graph_inst
    )
    use_text_units = await _find_related_text_unit_from_relationships(
        edge_datas, query_param, text_chunks_db, knowledge_graph_inst
    )
    logger.info(
        f"Global query uses {len(use_entities)} entites, {len(edge_datas)} relations, {len(use_text_units)} text units"
    )
    relations_section_list = [
        ["id", "source", "target", "description", "keywords", "weight", "rank"]
    ]
    for i, e in enumerate(edge_datas):
        relations_section_list.append(
            [
                i,
                e["src_id"],
                e["tgt_id"],
                e["description"],
                e["keywords"],
                e["weight"],
                e["rank"],
            ]
        )
    relations_context = list_of_list_to_csv(relations_section_list)

    entites_section_list = [["id", "entity", "type", "description", "rank"]]
    for i, n in enumerate(use_entities):
        entites_section_list.append(
            [
                i,
                n["entity_name"],
                n.get("entity_type", "UNKNOWN"),
                n.get("description", "UNKNOWN"),
                n["rank"],
            ]
        )
    entities_context = list_of_list_to_csv(entites_section_list)

    text_units_section_list = [["id", "content"]]
    for i, t in enumerate(use_text_units):
        text_units_section_list.append([i, t["content"]])
    text_units_context = list_of_list_to_csv(text_units_section_list)

    return f"""
-----Entities-----
```csv
{entities_context}
```
-----Relationships-----
```csv
{relations_context}
```
-----Sources-----
```csv
{text_units_context}
```
"""


async def _find_most_related_entities_from_relationships(
    edge_datas: list[dict],
    query_param: QueryParam,
    knowledge_graph_inst: BaseGraphStorage,
):
    entity_names = []
    seen = set()

    for e in edge_datas:
        if e["src_id"] not in seen:
            entity_names.append(e["src_id"])
            seen.add(e["src_id"])
        if e["tgt_id"] not in seen:
            entity_names.append(e["tgt_id"])
            seen.add(e["tgt_id"])

    node_datas = await asyncio.gather(
        *[knowledge_graph_inst.get_node(entity_name) for entity_name in entity_names]
    )

    node_degrees = await asyncio.gather(
        *[knowledge_graph_inst.node_degree(entity_name) for entity_name in entity_names]
    )
    node_datas = [
        {**n, "entity_name": k, "rank": d}
        for k, n, d in zip(entity_names, node_datas, node_degrees)
    ]

    node_datas = truncate_list_by_token_size(
        node_datas,
        key=lambda x: x["description"],
        max_token_size=query_param.max_token_for_local_context,
    )

    return node_datas


async def _find_related_text_unit_from_relationships(
    edge_datas: list[dict],
    query_param: QueryParam,
    text_chunks_db: BaseKVStorage[TextChunkSchema],
    knowledge_graph_inst: BaseGraphStorage,
):
    text_units = [
        split_string_by_multi_markers(dp["source_id"], [GRAPH_FIELD_SEP])
        for dp in edge_datas
    ]

    all_text_units_lookup = {}

    for index, unit_list in enumerate(text_units):
        for c_id in unit_list:
            if c_id not in all_text_units_lookup:
                all_text_units_lookup[c_id] = {
                    "data": await text_chunks_db.get_by_id(c_id),
                    "order": index,
                }

    if any([v is None for v in all_text_units_lookup.values()]):
        logger.warning("Text chunks are missing, maybe the storage is damaged")
    all_text_units = [
        {"id": k, **v} for k, v in all_text_units_lookup.items() if v is not None
    ]
    all_text_units = sorted(all_text_units, key=lambda x: x["order"])
    all_text_units = truncate_list_by_token_size(
        all_text_units,
        key=lambda x: x["data"]["content"],
        max_token_size=query_param.max_token_for_text_unit,
    )
    all_text_units: list[TextChunkSchema] = [t["data"] for t in all_text_units]

    return all_text_units


async def hybrid_hyde_embedding_query(
    query,
    knowledge_graph_inst: BaseGraphStorage,
    entities_vdb: BaseVectorStorage,
    relationships_vdb: BaseVectorStorage,
    text_chunks_db: BaseKVStorage[TextChunkSchema],
    query_param: QueryParam,
    global_config: dict,
    embedding=None,  # Add optional embedding parameter for averaged embedding
) -> str:
    """
    Hybrid HyDE embedding-based retrieval: supports direct, combined, or averaged embedding for hybrid retrieval.
    """
    use_model_func = global_config["llm_model_func"]
    use_direct_embedding = getattr(query_param, 'use_direct_embedding', False)
    use_combined_embedding = getattr(query_param, 'use_combined_embedding', False)
    use_averaged_embedding = getattr(query_param, 'use_averaged_embedding', False)

    # If an embedding is provided (averaged embedding mode), use it for vector search
    if embedding is not None:
        print(f"DEBUG: [hybrid_hyde_embedding_query] Using pre-computed averaged embedding for hybrid retrieval")
        # Use the averaged embedding for both local and global context retrieval
        local_results = await entities_vdb.query_with_embedding(embedding, top_k=query_param.top_k)
        global_results = await relationships_vdb.query_with_embedding(embedding, top_k=query_param.top_k)
        # For demonstration, just combine the results (real implementation may need more logic)
        all_results = local_results + global_results
        # Build a simple context from the results
        context = "\n".join([str(r) for r in all_results])
        return context

    # Otherwise, fall back to text-based hybrid retrieval (direct or combined)
    # This is a simplified version; real implementation may need to mirror hybrid_query logic
    if use_direct_embedding:
        print(f"DEBUG: [hybrid_hyde_embedding_query] Using hypothetical document directly for hybrid retrieval")
        query_text = query  # query is the hypothetical document
    elif use_combined_embedding:
        print(f"DEBUG: [hybrid_hyde_embedding_query] Using combined (query + hypothetical) for hybrid retrieval")
        query_text = query  # in practice, should be combined text
    else:
        query_text = query

    # Use the existing hybrid_query for text-based retrieval
    return await hybrid_query(
        query_text,
        knowledge_graph_inst,
        entities_vdb,
        relationships_vdb,
        text_chunks_db,
        query_param,
        global_config,
    )

async def hybrid_query(
    query,
    knowledge_graph_inst: BaseGraphStorage,
    entities_vdb: BaseVectorStorage,
    relationships_vdb: BaseVectorStorage,
    text_chunks_db: BaseKVStorage[TextChunkSchema],
    query_param: QueryParam,
    global_config: dict,
) -> str:
    low_level_context = None
    high_level_context = None
    use_model_func = global_config["llm_model_func"]

    # Create a modified query param with half the token sizes for hybrid mode
    hybrid_query_param = QueryParam(
        **{**asdict(query_param), 
           'max_token_for_text_unit': query_param.max_token_for_text_unit // 2,
           'max_token_for_global_context': query_param.max_token_for_global_context // 2,
           'max_token_for_local_context': query_param.max_token_for_local_context // 2}
    )

    kw_prompt_temp = PROMPTS["keywords_extraction"]
    kw_prompt = kw_prompt_temp.format(query=query, language="English", examples="")

    result = await use_model_func(kw_prompt)
    json_text = locate_json_string_body_from_string(result)
    try:
        keywords_data = json.loads(json_text)
        hl_keywords = keywords_data.get("high_level_keywords", [])
        ll_keywords = keywords_data.get("low_level_keywords", [])
        hl_keywords = ", ".join(hl_keywords)
        ll_keywords = ", ".join(ll_keywords)
    except json.JSONDecodeError:
        try:
            result = (
                result.replace(kw_prompt[:-1], "")
                .replace("user", "")
                .replace("model", "")
                .strip()
            )
            result = "{" + result.split("{")[-1].split("}")[0] + "}"
            keywords_data = json.loads(result)
            hl_keywords = keywords_data.get("high_level_keywords", [])
            ll_keywords = keywords_data.get("low_level_keywords", [])
            hl_keywords = ", ".join(hl_keywords)
            ll_keywords = ", ".join(ll_keywords)
        # Handle parsing error
        except json.JSONDecodeError as e:
            print(f"JSON parsing error: {e}")
            return PROMPTS["fail_response"]

    if ll_keywords:
        low_level_context = await _build_local_query_context(
            ll_keywords,
            knowledge_graph_inst,
            entities_vdb,
            text_chunks_db,
            hybrid_query_param,  # Use modified param with half token size
        )

    if hl_keywords:
        high_level_context = await _build_global_query_context(
            hl_keywords,
            knowledge_graph_inst,
            entities_vdb,
            relationships_vdb,
            text_chunks_db,
            hybrid_query_param,  # Use modified param with half token size
        )

    context = combine_contexts(high_level_context, low_level_context)

    if query_param.only_need_context:
        return context
    if context is None:
        return PROMPTS["fail_response"]

    sys_prompt_temp = PROMPTS["rag_response"]
    sys_prompt = sys_prompt_temp.format(
        context_data=context, response_type=query_param.response_type, user_prompt=""
    )
    response = await use_model_func(
        query,
        system_prompt=sys_prompt,
    )
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


def combine_contexts(high_level_context, low_level_context):
    # Function to extract entities, relationships, and sources from context strings

    def extract_sections(context):
        entities_match = re.search(
            r"-----Entities-----\s*```csv\s*(.*?)\s*```", context, re.DOTALL
        )
        relationships_match = re.search(
            r"-----Relationships-----\s*```csv\s*(.*?)\s*```", context, re.DOTALL
        )
        sources_match = re.search(
            r"-----Sources-----\s*```csv\s*(.*?)\s*```", context, re.DOTALL
        )

        entities = entities_match.group(1) if entities_match else ""
        relationships = relationships_match.group(1) if relationships_match else ""
        sources = sources_match.group(1) if sources_match else ""

        return entities, relationships, sources

    # Extract sections from both contexts

    if high_level_context is None:
        warnings.warn(
            "High Level context is None. Return empty High entity/relationship/source"
        )
        hl_entities, hl_relationships, hl_sources = "", "", ""
    else:
        hl_entities, hl_relationships, hl_sources = extract_sections(high_level_context)

    if low_level_context is None:
        warnings.warn(
            "Low Level context is None. Return empty Low entity/relationship/source"
        )
        ll_entities, ll_relationships, ll_sources = "", "", ""
    else:
        ll_entities, ll_relationships, ll_sources = extract_sections(low_level_context)

    # Combine and deduplicate the entities
    combined_entities = process_combine_contexts(hl_entities, ll_entities)

    # Combine and deduplicate the relationships
    combined_relationships = process_combine_contexts(
        hl_relationships, ll_relationships
    )

    # Combine and deduplicate the sources
    combined_sources = process_combine_contexts(hl_sources, ll_sources)

    # Format the combined context
    return f"""
-----Entities-----
```csv
{combined_entities}
```
-----Relationships-----
```csv
{combined_relationships}
```
-----Sources-----
```csv
{combined_sources}
```
"""


async def entity_event_hyde_embedding_query(
    query,
    knowledge_graph_inst: BaseGraphStorage,
    entity_vdb: BaseVectorStorage,  # Vector DB for entities
    event_vdb: BaseVectorStorage,   # Vector DB for events
    relationships_vdb: BaseVectorStorage,
    text_chunks_db: BaseKVStorage[TextChunkSchema],
    query_param: QueryParam,
    global_config: dict,
    hypothetical_doc: str = None,
    # Add parameter to receive timing info from aquery
    calculated_hypothetical_doc_time: float = None,
    # Add parameter for pre-computed embedding (for averaged embedding mode)
    embedding = None,
) -> tuple[str, list[dict], list[dict]]:  # Return (response, nodes, edges)
    """Query using embedding-based similarity for entity-event HyDE.
    
    This variant uses vector similarity against entity and event embeddings and then
    expands the retrieved nodes to nearby context before generating the final answer.
    """
    original_query = query
    use_model_func = global_config["llm_model_func"]

    use_direct_embedding = getattr(query_param, 'use_direct_embedding', False)
    use_combined_embedding = getattr(query_param, 'use_combined_embedding', False)

    entity_query_text = original_query
    event_query_text = original_query
    if use_direct_embedding and hypothetical_doc:
        print("DEBUG: Using hypothetical document directly for embedding queries")
        entity_query_text = hypothetical_doc
        event_query_text = hypothetical_doc
    elif use_combined_embedding and hypothetical_doc:
        print("DEBUG: Using combined (query + hypothetical) text for embedding queries")
        combined_text = f"{original_query} {hypothetical_doc}"
        entity_query_text = combined_text
        event_query_text = combined_text

    if embedding is not None:
        print("DEBUG: Using provided embedding vector for entity/event retrieval")
        entity_results = await entity_vdb.query_with_embedding(embedding, top_k=query_param.top_k)
        event_results = await event_vdb.query_with_embedding(embedding, top_k=query_param.top_k)
    else:
        if entity_query_text is None:
            entity_query_text = original_query
        if event_query_text is None:
            event_query_text = original_query
        print("DEBUG: Querying entity/event VDBs with text input")
        entity_results = await entity_vdb.query(entity_query_text, top_k=query_param.top_k)
        event_results = await event_vdb.query(event_query_text, top_k=query_param.top_k)

    if not entity_results and not event_results:
        print("WARNING: Embedding search returned no entities or events")
        return PROMPTS["fail_response"], [], []

    async def build_seeds(results, expected_type: str) -> list[dict]:
        seeds: list[dict] = []
        seen = set()
        for res in results:
            node_id = res.get("entity_name") or res.get("id")
            if not node_id or node_id in seen:
                continue
            node_attr = await knowledge_graph_inst.get_node(node_id)
            if node_attr is None:
                continue
            node_type = node_attr.get("node_type", "").strip('"').lower()
            if node_type != expected_type:
                continue
            degree = await knowledge_graph_inst.node_degree(node_id)
            seeds.append(
                {
                    **node_attr,
                    "entity_name": node_id,
                    "node_type": node_attr.get("node_type", "").strip('"'),
                    "rank": degree,
                    "score": _normalize_similarity(res.get("distance")),
                    "depth": 0,
                }
            )
            seen.add(node_id)
        return seeds

    seed_entities = await build_seeds(entity_results, "entity")
    seed_events = await build_seeds(event_results, "event")

    node_lookup = await _expand_entity_event_nodes(
        knowledge_graph_inst, seed_entities, seed_events, query_param
    )

    print(
        "DEBUG: Embedding search seeds -> entities: %d, events: %d, expanded nodes: %d",
        len(seed_entities),
        len(seed_events),
        len(node_lookup),
    )

    context_data, nodes_data, use_relations = await _assemble_entity_event_context(
        node_lookup, query_param, text_chunks_db, knowledge_graph_inst
    )

    if context_data is None:
        return PROMPTS["fail_response"], [], []

    sys_prompt_temp = PROMPTS["rag_response"]
    sys_prompt = sys_prompt_temp.format(
        context_data=context_data, response_type=query_param.response_type, user_prompt=""
    )
    response = await use_model_func(
        original_query,
        system_prompt=sys_prompt,
    )

    if len(response) > len(sys_prompt):
        response = (
            response.replace(sys_prompt, "")
            .replace("user", "")
            .replace("model", "")
            .replace(original_query, "")
            .replace("<system>", "")
            .replace("</system>", "")
            .strip()
        )

    response_str = str(response)

    if calculated_hypothetical_doc_time is not None:
        class ResponseWithTiming(str):
            pass

        response_obj = ResponseWithTiming(response_str)
        response_obj.hypothetical_doc_time_seconds = calculated_hypothetical_doc_time
        return response_obj, nodes_data, use_relations

    return response_str, nodes_data, use_relations


async def naive_query(
    query,
    chunks_vdb: BaseVectorStorage,
    text_chunks_db: BaseKVStorage[TextChunkSchema],
    query_param: QueryParam,
    global_config: dict,
    embedding=None,  # Add optional embedding parameter
):
    # Prefer custom_prompt from query_param if set
    custom_prompt = getattr(query_param, "custom_prompt", None)

    use_model_func = global_config["llm_model_func"]
    
    # Use the provided embedding if it exists, otherwise use regular text query
    if embedding is not None:
        print(f"Using pre-computed embedding for naive_query")
        results = await chunks_vdb.query_with_embedding(embedding, top_k=query_param.top_k)
    else:
        print(f"Using text query for naive_query: {query[:50]}...")
        results = await chunks_vdb.query(query, top_k=query_param.top_k)
        
    if not len(results):
        return PROMPTS["fail_response"]
    chunks_ids = [r["id"] for r in results]
    chunks = await text_chunks_db.get_by_ids(chunks_ids)

    maybe_trun_chunks = truncate_list_by_token_size(
        chunks,
        key=lambda x: x["content"],
        max_token_size=query_param.max_token_for_text_unit,
    )
    logger.info(f"Truncate {len(chunks)} to {len(maybe_trun_chunks)} chunks")
    section = "--New Chunk--\n".join([c["content"] for c in maybe_trun_chunks])
    if query_param.only_need_context:
        return section

    if custom_prompt:
        # Use the custom prompt, substituting query, and context
        context = section
        sys_prompt = custom_prompt.format(
            query=query,
            context=context
        )
    else:
        sys_prompt_temp = PROMPTS["naive_rag_response"]
        sys_prompt = sys_prompt_temp.format(
            content_data=section, response_type=query_param.response_type
        )

    response = await use_model_func(
        query,
        system_prompt=sys_prompt,
        max_tokens=1024,
        temperature=0.7
    )

    if len(response) > len(sys_prompt):
        response = (
            response[len(sys_prompt) :]
            .replace(sys_prompt, "")
            .replace("user", "")
            .replace("model", "")
            .replace(query, "")
            .replace("<system>", "")
            .replace("</system>", "")
            .strip()
        )

    return response

async def entity_event_query(
    query,
    knowledge_graph_inst: BaseGraphStorage,
    entity_vdb: BaseVectorStorage,  # Vector DB for entities
    event_vdb: BaseVectorStorage,   # Vector DB for events
    relationships_vdb: BaseVectorStorage,
    text_chunks_db: BaseKVStorage[TextChunkSchema],
    query_param: QueryParam,
    global_config: dict,
    # Add parameter to receive timing info from aquery
    calculated_hypothetical_doc_time: float = None,
    # Add parameter for hypothetical document
    hypothetical_doc: str = None, 
) -> tuple[str, list[dict], list[dict]]:  # Return (response, nodes, edges)
    context = None
    use_model_func = global_config["llm_model_func"]
    
    # The original query is used for extraction and final response generation
    original_query = query
    
    # Check if we're using embedding-based methods (they don't use entity extraction)
    use_direct_embedding = getattr(query_param, 'use_direct_embedding', False)
    use_combined_embedding = getattr(query_param, 'use_combined_embedding', False)
    
    # If we're using embedding-based methods, don't do any entity extraction
    if use_direct_embedding or use_combined_embedding:
        print(f"DEBUG: Using embedding-based method (direct={use_direct_embedding}, combined={use_combined_embedding})")
        # Skip entity extraction for embedding-based methods
        entity_keywords = ""
        event_keywords = ""
    else:
        # Only do entity extraction for non-embedding methods
        # Extract from original query if enabled (combined extraction mode)
        if getattr(query_param, 'extract_from_original_query', True):
            # Extract entities and events from BOTH the original query AND hypothetical document 
            # and combine the results
            print(f"DEBUG: Combined extraction mode - extracting from both original query and hypothetical document")
            
            # First, extract from original query
            kw_prompt_temp = PROMPTS["entity_event_extraction"]
            kw_prompt = kw_prompt_temp.format(query=original_query)
            result = await use_model_func(kw_prompt)
            json_text = locate_json_string_body_from_string(result)

            try:
                # Extract from original query
                keywords_data = json.loads(json_text)
                entity_keywords_from_query = keywords_data.get("entities", [])
                event_keywords_from_query = keywords_data.get("events", [])
            except json.JSONDecodeError:
                try:
                    result = (
                        result.replace(kw_prompt[:-1], "")
                        .replace("user", "")
                        .replace("model", "")
                        .strip()
                    )
                    result = "{" + result.split("{")[-1].split("}")[0] + "}"
                    keywords_data = json.loads(result)
                    entity_keywords_from_query = keywords_data.get("entities", [])
                    event_keywords_from_query = keywords_data.get("events", [])
                except json.JSONDecodeError as e:
                    print(f"JSON parsing error for original query: {e}")
                    entity_keywords_from_query = []
                    event_keywords_from_query = []
            
            print(f"DEBUG: Extracted from original query - entities: {entity_keywords_from_query}, events: {event_keywords_from_query}")
            
            # Now extract from hypothetical document if it's different from the original query
            entity_keywords_from_hypothetical = []
            event_keywords_from_hypothetical = []
            
            # hypothetical_doc is now passed directly as a parameter
            # rather than trying to extract it from the query
            
            if hypothetical_doc is not None:
                print(f"DEBUG: Also extracting from hypothetical document")
                hypo_kw_prompt = PROMPTS["entity_event_extraction"]
                hypo_kw_prompt = hypo_kw_prompt.format(query=hypothetical_doc)  # query contains the hypothetical doc
                hypo_result = await use_model_func(hypo_kw_prompt)
                hypo_json_text = locate_json_string_body_from_string(hypo_result)
                
                try:
                    hypo_keywords_data = json.loads(hypo_json_text)
                    entity_keywords_from_hypothetical = hypo_keywords_data.get("entities", [])
                    event_keywords_from_hypothetical = hypo_keywords_data.get("events", [])
                except json.JSONDecodeError:
                    try:
                        hypo_result = (
                            hypo_result.replace(hypo_kw_prompt[:-1], "")
                            .replace("user", "")
                            .replace("model", "")
                            .strip()
                        )
                        hypo_result = "{" + hypo_result.split("{")[-1].split("}")[0] + "}"
                        hypo_keywords_data = json.loads(hypo_result)
                        entity_keywords_from_hypothetical = hypo_keywords_data.get("entities", [])
                        event_keywords_from_hypothetical = hypo_keywords_data.get("events", [])
                    except json.JSONDecodeError as e:
                        print(f"JSON parsing error for hypothetical document: {e}")
                
                print(f"DEBUG: Extracted from hypothetical document - entities: {entity_keywords_from_hypothetical}, events: {event_keywords_from_hypothetical}")
            
            # Combine both sets of entities and events, removing duplicates
            combined_entity_keywords = list(set(entity_keywords_from_query + entity_keywords_from_hypothetical))
            combined_event_keywords = list(set(event_keywords_from_query + event_keywords_from_hypothetical))
            
            # Convert lists to comma-separated strings
            entity_keywords = ", ".join(combined_entity_keywords)
            event_keywords = ", ".join(combined_event_keywords)
            
            print(f"DEBUG: Combined keywords - entities: {entity_keywords}, events: {event_keywords}")
            
            # If we couldn't extract anything, fail gracefully
            if not entity_keywords and not event_keywords:
                print(f"WARNING: Failed to extract any entities or events")
                return PROMPTS["fail_response"], [], []
        else:
            # Extract from hypothetical document (which is now in the query variable)
            print(f"DEBUG: Extracting entities/events from hypothetical document")
            # Use the same extraction logic but with the current query (hypothetical document)
            kw_prompt_temp = PROMPTS["entity_event_extraction"]
            kw_prompt = kw_prompt_temp.format(query=hypothetical_doc)  # Use 'query' which now contains the hypothetical doc
            result = await use_model_func(kw_prompt)
            json_text = locate_json_string_body_from_string(result)

            try:
                keywords_data = json.loads(json_text)
                entity_keywords = keywords_data.get("entities", [])
                event_keywords = keywords_data.get("events", [])
                entity_keywords = ", ".join(entity_keywords)
                event_keywords = ", ".join(event_keywords)
                print(f"DEBUG: Extracted entities: {entity_keywords}")
                print(f"DEBUG: Extracted events: {event_keywords}")
            except json.JSONDecodeError:
                try:
                    result = (
                        result.replace(kw_prompt[:-1], "")
                        .replace("user", "")
                        .replace("model", "")
                        .strip()
                    )
                    result = "{" + result.split("{")[-1].split("}")[0] + "}"
                    keywords_data = json.loads(result)
                    entity_keywords = keywords_data.get("entities", [])
                    event_keywords = keywords_data.get("events", [])
                    entity_keywords = ", ".join(entity_keywords)
                    event_keywords = ", ".join(event_keywords)
                    print(f"DEBUG: Extracted entities (after format): {entity_keywords}")
                    print(f"DEBUG: Extracted events (after format): {event_keywords}")
                except json.JSONDecodeError as e:
                    print(f"DEBUG: JSON parsing error when extracting from hypothetical document: {e}")
                    entity_keywords = ""
                    event_keywords = ""
    
    # Remove the incorrect timing calculation here
    # mid_time = time.time()
    # if getattr(query_param, 'extract_from_original_query', True) and not original_query == query:
    #     hypothetical_doc_time = mid_time - start_time

    context_data = None
    nodes_data = None
    edges_data = None
    if entity_keywords or event_keywords:
        context_data, nodes_data, edges_data = await _build_entity_event_query_context(
            entity_keywords,
            event_keywords,
            knowledge_graph_inst,
            entity_vdb,
            event_vdb,
            text_chunks_db,
            query_param,
        )
    else:
        print("WARNING: No entity or event keywords available to drive retrieval")

    if query_param.only_need_context:
        return context_data, nodes_data, edges_data
    if context_data is None:
        return PROMPTS["fail_response"], [], []

    sys_prompt_temp = PROMPTS["rag_response"]
    sys_prompt = sys_prompt_temp.format(
        context_data=context_data, response_type=query_param.response_type, user_prompt=""
    )
    # Use original_query for the final response generation
    response = await use_model_func(
        original_query, 
        system_prompt=sys_prompt,
    )
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

    # Use the passed-in timing value (calculated_hypothetical_doc_time)
    if query_param.enable_hypothetical_document and calculated_hypothetical_doc_time is not None:
        # Create a custom string class to hold the attribute
        class ResponseWithTiming(str):
            pass
        
        response_obj = ResponseWithTiming(response)
        # Attach the correct time passed from aquery
        response_obj.hypothetical_doc_time_seconds = calculated_hypothetical_doc_time 
        
        # Log the timing for debugging purposes
        print(f"DEBUG: Attached hypothetical_doc_time_seconds = {calculated_hypothetical_doc_time}")
        
        return response_obj, nodes_data, edges_data
    else:
        # Log that no timing info is attached
        if query_param.enable_hypothetical_document:
            # Check if time was calculated but None, or just not calculated
            if calculated_hypothetical_doc_time is None:
                 print(f"DEBUG: Hypothetical doc time was not calculated or passed correctly.")
            # else: # This case should not happen if enable_hypothetical_document is True
            #    print(f"DEBUG: Hypothetical doc time is None.") 
        else:
            print(f"DEBUG: Not using HyDE, returning plain string response")
        
        # Return a plain string
        return response, nodes_data, edges_data

async def _build_entity_event_query_context(
    entity_keywords,
    event_keywords,
    knowledge_graph_inst: BaseGraphStorage,
    entity_vdb: BaseVectorStorage,
    event_vdb: BaseVectorStorage,
    text_chunks_db: BaseKVStorage[TextChunkSchema],
    query_param: QueryParam,
) -> tuple[str, list[dict], list[dict]]:  # Return (context, nodes, edges)
    entity_results = []
    if entity_keywords:
        entity_results = await entity_vdb.query(entity_keywords, top_k=query_param.top_k)

    event_results = []
    if event_keywords:
        event_results = await event_vdb.query(event_keywords, top_k=query_param.top_k)

    if not entity_results and not event_results:
        print(
            "DEBUG: No entity or event results found for keywords: "
            f"entity_keywords={entity_keywords}, event_keywords={event_keywords}"
        )
        return None, [], []

    seed_entities: list[dict] = []
    seen_entity_names = set()
    for result in entity_results:
        node_id = result.get("entity_name")
        if not node_id:
            continue
        # Use the full node_id as stored in the graph (includes chunk reference)
        if node_id in seen_entity_names:
            continue
        seen_entity_names.add(node_id)
        
        node_attr = await knowledge_graph_inst.get_node(node_id)
        if node_attr is None:
            continue
        # The graph uses 'entity_type' with values like "PERSON", "GEO", "EVENT", "CONCEPT"
        entity_type = node_attr.get("entity_type", "").strip('"').upper()
        # Filter out EVENT types - we only want entity types (PERSON, GEO, ORGANIZATION, CONCEPT, etc)
        if entity_type == "EVENT":
            continue
        degree = await knowledge_graph_inst.node_degree(node_id)
        seed_entities.append(
            {
                **node_attr,
                "entity_name": node_id,
                "entity_type": entity_type,
                "rank": degree,
                "score": _normalize_similarity(result.get("distance")),
                "depth": 0,
            }
        )

    seed_events: list[dict] = []
    seen_event_names = set()
    for result in event_results:
        node_id = result.get("entity_name")
        if not node_id:
            continue
        # Use the full node_id as stored in the graph
        if node_id in seen_event_names:
            continue
        seen_event_names.add(node_id)
        
        node_attr = await knowledge_graph_inst.get_node(node_id)
        if node_attr is None:
            continue
        # The graph uses 'entity_type' - for events we want type == "EVENT"
        entity_type = node_attr.get("entity_type", "").strip('"').upper()
        if entity_type != "EVENT":
            continue
        degree = await knowledge_graph_inst.node_degree(node_id)
        seed_events.append(
            {
                **node_attr,
                "entity_name": node_id,
                "entity_type": entity_type,
                "rank": degree,
                "score": _normalize_similarity(result.get("distance")),
                "depth": 0,
            }
        )

    node_lookup = await _expand_entity_event_nodes(
        knowledge_graph_inst, seed_entities, seed_events, query_param
    )

    logger.info(
        "Entity-Event query uses %d seed entities, %d seed events, expanded to %d nodes",
        len(seed_entities),
        len(seed_events),
        len(node_lookup),
    )

    return await _assemble_entity_event_context(
        node_lookup, query_param, text_chunks_db, knowledge_graph_inst
    )
