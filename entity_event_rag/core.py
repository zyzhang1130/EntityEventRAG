import asyncio
import os
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime
from functools import partial
from typing import Optional, Type, cast

from .llm import (
    gpt_4o_mini_complete,
    openai_embedding,
)
from .operate import (
    chunking_by_token_size,
    extract_entities,
    local_query,
    global_query,
    hybrid_query,
    naive_query,
    entity_event_query,
    entity_event_hyde_embedding_query,
)

from .utils import (
    EmbeddingFunc,
    compute_mdhash_id,
    limit_async_func_call,
    convert_response_to_json,
    logger,
    set_logger,
)
from .base import (
    BaseGraphStorage,
    BaseKVStorage,
    BaseVectorStorage,
    StorageNameSpace,
    QueryParam,
)

from .storage import (
    JsonKVStorage,
    NanoVectorDBStorage,
    NetworkXStorage,
)

from .kg.neo4j_impl import Neo4JStorage

from .kg.oracle_impl import OracleKVStorage, OracleGraphStorage, OracleVectorDBStorage

# future KG integrations

# from .kg.ArangoDB_impl import (
#     GraphStorage as ArangoDBStorage
# )


def always_get_an_event_loop() -> asyncio.AbstractEventLoop:
    try:
        return asyncio.get_event_loop()

    except RuntimeError:
        logger.info("Creating a new event loop in main thread.")
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)

        return loop


@dataclass
class EntityEventRAG:
    """Entity-Event RAG: Bipartite Graph-based Retrieval-Augmented Generation.
    
    This system builds a bipartite knowledge graph connecting entities and events,
    enabling more comprehensive retrieval for complex queries involving temporal
    and narrative understanding.
    """
    working_dir: str = field(
        default_factory=lambda: f"./entity_event_rag_cache_{datetime.now().strftime('%Y-%m-%d-%H:%M:%S')}"
    )

    # Optional override locations for the dedicated entity/event vector stores used
    # by entity-event querying. Defaults to the main working_dir when not provided.
    entity_storage_dir: Optional[str] = None
    event_storage_dir: Optional[str] = None

    kv_storage: str = field(default="JsonKVStorage")
    vector_storage: str = field(default="NanoVectorDBStorage")
    graph_storage: str = field(default="NetworkXStorage")

    # Add extraction type parameter
    extraction_type: str = field(default="entity")  # Can be "entity" or "event"

    current_log_level = logger.level
    log_level: str = field(default=current_log_level)

    # text chunking
    chunk_token_size: int = 1200
    chunk_overlap_token_size: int = 100
    tiktoken_model_name: str = "gpt-4o-mini"

    # entity extraction
    entity_extract_max_gleaning: int = 1
    entity_summary_to_max_tokens: int = 500

    # node embedding
    node_embedding_algorithm: str = "node2vec"
    node2vec_params: dict = field(
        default_factory=lambda: {
            "dimensions": 1536,
            "num_walks": 10,
            "walk_length": 40,
            "window_size": 2,
            "iterations": 3,
            "random_seed": 3,
        }
    )

    # embedding_func: EmbeddingFunc = field(default_factory=lambda:hf_embedding)
    embedding_func: EmbeddingFunc = field(default_factory=lambda: openai_embedding)
    embedding_batch_num: int = 32
    embedding_func_max_async: int = 16

    # LLM
    llm_model_func: callable = gpt_4o_mini_complete  # hf_model_complete#
    llm_model_name: str = "meta-llama/Llama-3.2-1B-Instruct"  #'meta-llama/Llama-3.2-1B'#'google/gemma-2-2b-it'
    llm_model_max_token_size: int = 32768
    llm_model_max_async: int = 16
    llm_model_kwargs: dict = field(default_factory=dict)

    # storage
    vector_db_storage_cls_kwargs: dict = field(default_factory=dict)

    enable_llm_cache: bool = True

    # extension
    addon_params: dict = field(default_factory=dict)
    convert_response_to_json_func: callable = convert_response_to_json

    def __post_init__(self):
        # [Previous post_init code remains unchanged until extract_entities call]

        # Validate extraction type
        if self.extraction_type not in ["entity", "event"]:
            raise ValueError("extraction_type must be either 'entity' or 'event'")

        # Ensure entity/event storage directories default to working_dir when not set
        self.entity_storage_dir = self.entity_storage_dir or self.working_dir
        self.event_storage_dir = self.event_storage_dir or self.working_dir
        os.makedirs(self.entity_storage_dir, exist_ok=True)
        os.makedirs(self.event_storage_dir, exist_ok=True)

        log_file = os.path.join(self.working_dir, "entity_event_rag.log")
        set_logger(log_file)
        logger.setLevel(self.log_level)

        logger.info(f"Logger initialized for working directory: {self.working_dir}")

        _print_config = ",\n  ".join([f"{k} = {v}" for k, v in asdict(self).items()])
        logger.debug(f"EntityEventRAG init with param:\n  {_print_config}\n")

        # @TODO: should move all storage setup here to leverage initial start params attached to self.

        self.key_string_value_json_storage_cls: Type[BaseKVStorage] = (
            self._get_storage_class()[self.kv_storage]
        )
        self.vector_db_storage_cls: Type[BaseVectorStorage] = self._get_storage_class()[
            self.vector_storage
        ]
        self.graph_storage_cls: Type[BaseGraphStorage] = self._get_storage_class()[
            self.graph_storage
        ]

        if not os.path.exists(self.working_dir):
            logger.info(f"Creating working directory {self.working_dir}")
            os.makedirs(self.working_dir)

        self.llm_response_cache = (
            self.key_string_value_json_storage_cls(
                namespace="llm_response_cache",
                global_config=asdict(self),
                embedding_func=None,
            )
            if self.enable_llm_cache
            else None
        )

        self.embedding_func = limit_async_func_call(self.embedding_func_max_async)(
            self.embedding_func
        )

        ####
        # add embedding func by walter
        ####
        self.full_docs = self.key_string_value_json_storage_cls(
            namespace="full_docs",
            global_config=asdict(self),
            embedding_func=self.embedding_func,
        )
        self.text_chunks = self.key_string_value_json_storage_cls(
            namespace="text_chunks",
            global_config=asdict(self),
            embedding_func=self.embedding_func,
        )
        self.chunk_entity_relation_graph = self.graph_storage_cls(
            namespace="chunk_entity_relation", global_config=asdict(self)
        )
        ####
        # add embedding func by walter over
        ####

        # Main entities VDB used by all operations except entity_event_query
        self.entities_vdb = self.vector_db_storage_cls(
            namespace="entities",
            global_config=asdict(self),
            embedding_func=self.embedding_func,
            meta_fields={"entity_name"},
        )
        
        # Separate VDBs specifically for entity_event_query
        entity_config = asdict(self)
        entity_config["working_dir"] = self.entity_storage_dir
        self.entity_vdb = self.vector_db_storage_cls(
            namespace="entities",
            global_config=entity_config,
            embedding_func=self.embedding_func,
            meta_fields={"entity_name"},
        )
        
        event_config = asdict(self)
        event_config["working_dir"] = self.event_storage_dir
        self.event_vdb = self.vector_db_storage_cls(
            namespace="entities",
            global_config=event_config,
            embedding_func=self.embedding_func,
            meta_fields={"entity_name"},
        )
        self.relationships_vdb = self.vector_db_storage_cls(
            namespace="relationships",
            global_config=asdict(self),
            embedding_func=self.embedding_func,
            meta_fields={"src_id", "tgt_id"},
        )
        self.chunks_vdb = self.vector_db_storage_cls(
            namespace="chunks",
            global_config=asdict(self),
            embedding_func=self.embedding_func,
        )

        self.llm_model_func = limit_async_func_call(self.llm_model_max_async)(
            partial(
                self.llm_model_func,
                hashing_kv=self.llm_response_cache,
                **self.llm_model_kwargs,
            )
        )

    def _get_storage_class(self) -> Type[BaseGraphStorage]:
        return {
            # kv storage
            "JsonKVStorage": JsonKVStorage,
            "OracleKVStorage": OracleKVStorage,
            # vector storage
            "NanoVectorDBStorage": NanoVectorDBStorage,
            "OracleVectorDBStorage": OracleVectorDBStorage,
            # graph storage
            "NetworkXStorage": NetworkXStorage,
            "Neo4JStorage": Neo4JStorage,
            "OracleGraphStorage": OracleGraphStorage,
            # "ArangoDBStorage": ArangoDBStorage
        }

    def insert(self, string_or_strings):
        loop = always_get_an_event_loop()
        return loop.run_until_complete(self.ainsert(string_or_strings))

    async def ainsert(self, string_or_strings):
        update_storage = False
        try:
            if isinstance(string_or_strings, str):
                string_or_strings = [string_or_strings]

            new_docs = {
                compute_mdhash_id(c.strip(), prefix="doc-"): {"content": c.strip()}
                for c in string_or_strings
            }
            _add_doc_keys = await self.full_docs.filter_keys(list(new_docs.keys()))
            new_docs = {k: v for k, v in new_docs.items() if k in _add_doc_keys}
            if not len(new_docs):
                logger.warning("All docs are already in the storage")
                return
            update_storage = True
            logger.info(f"[New Docs] inserting {len(new_docs)} docs")

            inserting_chunks = {}
            for doc_key, doc in new_docs.items():
                chunks = {
                    compute_mdhash_id(dp["content"], prefix="chunk-"): {
                        **dp,
                        "full_doc_id": doc_key,
                    }
                    for dp in chunking_by_token_size(
                        doc["content"],
                        overlap_token_size=self.chunk_overlap_token_size,
                        max_token_size=self.chunk_token_size,
                        tiktoken_model=self.tiktoken_model_name,
                    )
                }
                inserting_chunks.update(chunks)
            _add_chunk_keys = await self.text_chunks.filter_keys(
                list(inserting_chunks.keys())
            )
            inserting_chunks = {
                k: v for k, v in inserting_chunks.items() if k in _add_chunk_keys
            }
            if not len(inserting_chunks):
                logger.warning("All chunks are already in the storage")
                return
            logger.info(f"[New Chunks] inserting {len(inserting_chunks)} chunks")

            await self.chunks_vdb.upsert(inserting_chunks)

            # Use appropriate prompt based on extraction type
            extraction_type_str = "Event" if self.extraction_type == "event" else "Entity"
            logger.info(f"[{extraction_type_str} Extraction]...")

            # Create config with appropriate prompt
            config = asdict(self)
            config["extraction_prompt"] = "event_extraction" if self.extraction_type == "event" else "entity_extraction"

            maybe_new_kg = await extract_entities(
                inserting_chunks,
                knowledge_graph_inst=self.chunk_entity_relation_graph,
                entity_vdb=self.entities_vdb,
                relationships_vdb=self.relationships_vdb,
                global_config=config,
            )
            if maybe_new_kg is None:
                logger.warning(f"No new {extraction_type_str.lower()}s and relationships found")
                return
            self.chunk_entity_relation_graph = maybe_new_kg

            await self.full_docs.upsert(new_docs)
            await self.text_chunks.upsert(inserting_chunks)
        finally:
            if update_storage:
                await self._insert_done()

    async def _insert_done(self):
        tasks = []
        for storage_inst in [
            self.full_docs,
            self.text_chunks,
            self.llm_response_cache,
            self.entities_vdb,
            self.relationships_vdb,
            self.chunks_vdb,
            self.chunk_entity_relation_graph,
        ]:
            if storage_inst is None:
                continue
            tasks.append(cast(StorageNameSpace, storage_inst).index_done_callback())
        await asyncio.gather(*tasks)

    def query(self, query: str, param: QueryParam = QueryParam()):
        loop = always_get_an_event_loop()
        return loop.run_until_complete(self.aquery(query, param))

    async def aquery(self, query: str, param: QueryParam = QueryParam()):
        # Apply hypothetical document transformation if enabled
        original_query = query
        hypothetical_doc = None
        doc_embedding = None
        query_embedding = None
        combined_embedding = None
        # Variable to store the calculated hypothetical doc time
        calculated_hypothetical_doc_time = None 
        
        if param.enable_hypothetical_document:
            # Check if we need to use direct embedding or combined embedding
            if param.use_direct_embedding or param.use_combined_embedding or getattr(param, 'use_averaged_embedding', False):
                from .hypothetical_document_direct import HypotheticalDocumentDirectEmbeddingTransformer
                
                # Initialize the direct embedding transformer
                hypothetical_transformer = HypotheticalDocumentDirectEmbeddingTransformer(
                    llm_model_func=self.llm_model_func,
                    embedding_func=self.embedding_func,
                    enable_cache=self.enable_llm_cache
                )
                
                # Track time for hypothetical document generation
                start_time = time.time()
                
                # Transform the query and get embeddings
                hypothetical_doc, query_embedding, doc_embedding, combined_embedding = await hypothetical_transformer.transform_query(query)
                
                # Calculate time spent generating hypothetical document
                end_time = time.time()
                calculated_hypothetical_doc_time = end_time - start_time
                logger.info(f"Embedding-based hypothetical document generation time: {calculated_hypothetical_doc_time:.2f} seconds")
                
                # Log info if debug mode is enabled
                if param.hypothetical_document_debug:
                    logger.info(f"Original query: {query}")
                    logger.info(f"Hypothetical document: {hypothetical_doc}")
                    logger.info(f"Using {'direct' if param.use_direct_embedding else 'combined'} embedding mode")
                
            else:
                # Standard HyDE without special embedding handling
                from .hypothetical_document import HypotheticalDocumentTransformer
                
                # Initialize the hypothetical document transformer
                hypothetical_transformer = HypotheticalDocumentTransformer(
                    llm_model_func=self.llm_model_func,
                    enable_cache=self.enable_llm_cache
                )
                
                # Transform the query to a hypothetical document and get timing
                hypothetical_doc, generation_time = await hypothetical_transformer.transform_query(query)
                
                # Log original and transformed queries if debug mode is enabled
                if param.hypothetical_document_debug:
                    logger.info(f"Original query: {query}")
                    logger.info(f"Hypothetical document: {hypothetical_doc}")
                    logger.info(f"Generation time: {generation_time:.2f} seconds")
                
                # Store the generation time for later use
                # hypothetical_doc_time = generation_time # This variable is local and not used later
                calculated_hypothetical_doc_time = generation_time # Store in the accessible variable
                
                # Keep original query intact and pass hypothetical_doc separately
                # (The hypothetical_doc will be kept as a separate variable)

        if param.mode == "local":
            response = await local_query(
                query,
                self.chunk_entity_relation_graph,
                self.entities_vdb,
                self.relationships_vdb,
                self.text_chunks,
                param,
                asdict(self),
            )
        elif param.mode == "global":
            response = await global_query(
                query,
                self.chunk_entity_relation_graph,
                self.entities_vdb,
                self.relationships_vdb,
                self.text_chunks,
                param,
                asdict(self),
            )
        elif param.mode == "hybrid":
            query_to_hybrid = hypothetical_doc if hypothetical_doc is not None else original_query
            response = await hybrid_query(
                query_to_hybrid,
                self.chunk_entity_relation_graph,
                self.entities_vdb,
                self.relationships_vdb,
                self.text_chunks,
                param,
                asdict(self),
            )
        elif param.mode == "naive":
            # Check which approach we should use
            if param.enable_hypothetical_document and hypothetical_doc:
                if getattr(param, 'use_averaged_embedding', False) and combined_embedding is not None:
                    # Use the pre-computed averaged embedding directly
                    logger.info("Using averaged embedding directly for naive mode")
                    response = await naive_query(
                        query,  # Use original query for final answer
                        self.chunks_vdb,
                        self.text_chunks,
                        param,
                        asdict(self),
                        embedding=combined_embedding  # Pass averaged embedding
                    )
                elif getattr(param, 'use_combined_embedding', False) and hypothetical_doc:
                    # Traditional approach - concatenate text
                    combined_query_text = f"{query} {hypothetical_doc}"
                    logger.info(f"Using combined query text for naive mode: {combined_query_text[:100]}...")
                    response = await naive_query(
                        combined_query_text,  # Use combined text here
                        self.chunks_vdb,
                        self.text_chunks,
                        param,
                        asdict(self),
                    )
                else:
                    # Use original query
                    response = await naive_query(
                        query,
                        self.chunks_vdb,
                        self.text_chunks,
                        param,
                        asdict(self),
                    )
            else:
                # Original behavior without hypothetical doc
                response = await naive_query(
                    query,
                    self.chunks_vdb,
                    self.text_chunks,
                    param,
                    asdict(self),
                )
        elif param.mode == "entity_event":
            # Determine if we should use embedding-based approach or extraction-based approach
            use_direct_embedding = getattr(param, 'use_direct_embedding', False)
            use_combined_embedding = getattr(param, 'use_combined_embedding', False)
            use_averaged_embedding = getattr(param, 'use_averaged_embedding', False)
            
            # Use different query functions based on whether we're using embeddings
            if use_direct_embedding or use_combined_embedding or use_averaged_embedding:
                # For embedding-based methods, use the dedicated embedding query function
                if use_direct_embedding:
                    embedding_type = "direct"
                elif use_combined_embedding:
                    embedding_type = "combined"
                elif use_averaged_embedding:
                    embedding_type = "averaged"
                else:
                    embedding_type = "unknown"
                    
                logger.info(f"Using entity_event_hyde_embedding_query with {embedding_type} embedding")
                
                # If we're using averaged embedding, pass the combined_embedding
                if use_averaged_embedding and combined_embedding is not None:
                    response = await entity_event_hyde_embedding_query(
                        query,
                        self.chunk_entity_relation_graph,
                        self.entity_vdb,
                        self.event_vdb,
                        self.relationships_vdb,
                        self.text_chunks,
                        param,
                        asdict(self),
                        hypothetical_doc=hypothetical_doc,
                        calculated_hypothetical_doc_time=calculated_hypothetical_doc_time,
                        embedding=combined_embedding  # Pass the pre-computed averaged embedding
                    )
                else:
                    response = await entity_event_hyde_embedding_query(
                        query,
                        self.chunk_entity_relation_graph,
                        self.entity_vdb,
                        self.event_vdb,
                        self.relationships_vdb,
                        self.text_chunks,
                        param,
                        asdict(self),
                        hypothetical_doc=hypothetical_doc,
                        calculated_hypothetical_doc_time=calculated_hypothetical_doc_time,
                    )
            else:
                # For extraction-based methods, use the original entity_event_query function
                logger.info(f"Using entity_event_query with extraction_from_original_query={getattr(param, 'extract_from_original_query', True)}")
                response = await entity_event_query(
                    query,
                    self.chunk_entity_relation_graph,
                    self.entity_vdb,
                    self.event_vdb,
                    self.relationships_vdb,
                    self.text_chunks,
                    param,
                    asdict(self),
                    # Pass the calculated time down
                    calculated_hypothetical_doc_time=calculated_hypothetical_doc_time,
                    # Pass hypothetical_doc as a separate parameter
                    hypothetical_doc=hypothetical_doc
                )
            
            # Timing is now attached within entity_event_query/entity_event_hyde_embedding_query
            # No need to check getattr here anymore.
            # if param.enable_hypothetical_document:
            #     hypothetical_doc_time = getattr(response[0], 'hypothetical_doc_time_seconds', None)
            #     if hypothetical_doc_time is not None:
            #         # Make sure the attribute is preserved
            #         logger.info(f"Hypothetical document generation time: {hypothetical_doc_time:.2f} seconds")
            #     else:
            #         logger.warning(f"No hypothetical_doc_time_seconds attribute found on response")
        else:
            raise ValueError(f"Unknown mode {param.mode}")
        
        # If hypothetical document was generated, attach timing information to response
        if param.enable_hypothetical_document and calculated_hypothetical_doc_time is not None:
            try:
                # Don't overwrite existing timing if it's already there (e.g., from entity_event methods)
                if not hasattr(response, 'hypothetical_doc_time_seconds'):
                    # Create a custom string class to hold timing info and hypothetical document
                    class ResponseWithMetadata(str):
                        pass
                    
                    # For string responses (most modes), attach the timing attribute and hypothetical document
                    if isinstance(response, str):
                        response_obj = ResponseWithMetadata(response)
                        response_obj.hypothetical_doc_time_seconds = calculated_hypothetical_doc_time
                        # Also attach the hypothetical document text itself
                        response_obj.hypothetical_document = hypothetical_doc
                        response = response_obj
                        logger.info(f"Added metadata to {param.mode} response: time={calculated_hypothetical_doc_time:.2f}s, doc={len(hypothetical_doc) if hypothetical_doc else 0} chars")
                    elif isinstance(response, tuple) and len(response) > 0 and isinstance(response[0], str):
                        # For tuple responses (entity_event mode), attach to first element
                        response_obj = ResponseWithMetadata(response[0])
                        response_obj.hypothetical_doc_time_seconds = calculated_hypothetical_doc_time
                        # Also attach the hypothetical document text itself
                        response_obj.hypothetical_document = hypothetical_doc
                        response = (response_obj,) + response[1:]
                        logger.info(f"Added metadata to {param.mode} tuple response: time={calculated_hypothetical_doc_time:.2f}s, doc={len(hypothetical_doc) if hypothetical_doc else 0} chars")
            except Exception as e:
                # Ensure any errors here don't prevent the response from being returned
                logger.warning(f"Error attaching hypothetical_doc_time_seconds: {e}")
        
        await self._query_done()
        return response

    async def _query_done(self):
        tasks = []
        for storage_inst in [self.llm_response_cache]:
            if storage_inst is None:
                continue
            tasks.append(cast(StorageNameSpace, storage_inst).index_done_callback())
        await asyncio.gather(*tasks)

    def delete_by_entity(self, entity_name: str):
        loop = always_get_an_event_loop()
        return loop.run_until_complete(self.adelete_by_entity(entity_name))

    async def adelete_by_entity(self, entity_name: str):
        entity_name = f'"{entity_name.upper()}"'

        try:
            await self.entities_vdb.delete_entity(entity_name)
            await self.relationships_vdb.delete_relation(entity_name)
            await self.chunk_entity_relation_graph.delete_node(entity_name)

            logger.info(
                f"Entity '{entity_name}' and its relationships have been deleted."
            )
            await self._delete_by_entity_done()
        except Exception as e:
            logger.error(f"Error while deleting entity '{entity_name}': {e}")

    async def _delete_by_entity_done(self):
        tasks = []
        for storage_inst in [
            self.entities_vdb,
            self.relationships_vdb,
            self.chunk_entity_relation_graph,
        ]:
            if storage_inst is None:
                continue
            tasks.append(cast(StorageNameSpace, storage_inst).index_done_callback())
        await asyncio.gather(*tasks)
