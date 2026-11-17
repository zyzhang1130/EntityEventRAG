"""Regression for the basic example's missing combined-graph assembly.

Real ingestion, GraphML, NanoVectorDB, retrieval, and context rendering are used.
Only model/tokenizer boundaries and unused remote providers are deterministic.
Run with: python -m pytest -q tests/test_basic_entity_event.py
"""

import asyncio
import json
import os
from pathlib import Path
import sys
from types import ModuleType
from unittest.mock import patch

import networkx as nx
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


async def unexpected_provider(*args, **kwargs):
    raise AssertionError("The test must supply a deterministic model")


unexpected_provider.embedding_dim = 3
unexpected_provider.max_token_size = 8192
llm = ModuleType("entity_event_rag.llm")
llm.gpt_4o_mini_complete = unexpected_provider
llm.openai_embedding = unexpected_provider
providers = {llm.__name__: llm}
for name, classes in {
    "entity_event_rag.kg.neo4j_impl": ("Neo4JStorage",),
    "entity_event_rag.kg.oracle_impl": (
        "OracleKVStorage", "OracleGraphStorage", "OracleVectorDBStorage"
    ),
}.items():
    module = ModuleType(name)
    for cls in classes:
        setattr(module, cls, object)
    providers[name] = module

with patch.dict(sys.modules, providers):
    from entity_event_rag import EntityEventRAG, QueryParam, utils
    from entity_event_rag.utils import EmbeddingFunc
    from examples import basic_usage


TEXT = "Watson was wounded at Maiwand. Watson later returned to England."


class CharacterTokenizer:
    def encode(self, text):
        return list(map(ord, text))

    def decode(self, tokens):
        return "".join(map(chr, tokens))


@pytest.fixture
def example(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("OPENAI_API_KEY", "offline-fixture")
    monkeypatch.setattr(utils, "ENCODER", CharacterTokenizer())
    monkeypatch.setattr(basic_usage, "__file__", str(tmp_path / "examples/basic_usage.py"))
    documents = tmp_path / "RAG_documents"
    documents.mkdir()
    (documents / "Lady_Susan.txt").write_text(TEXT, encoding="utf-8")
    state = {"inserts": [], "combined": [], "responses": [], "prompts": []}

    async def embed(texts):
        return np.asarray([[1.0, 0.0, 0.0] for _ in texts], dtype=np.float32)

    async def model(prompt, system_prompt=None, **kwargs):
        if system_prompt is not None:
            assert "WATSON WOUNDED" in system_prompt
            assert TEXT in system_prompt
            state["prompts"].append(system_prompt)
            return "Watson was wounded and then returned to England."
        if "Event_types:" in prompt:
            return ('("entity"<|>WATSON WOUNDED<|>BATTLE<|>Watson was wounded at Maiwand)'
                    '##("entity"<|>RETURN TO ENGLAND<|>JOURNEY<|>Watson returned to England)'
                    '##("relationship"<|>WATSON WOUNDED<|>RETURN TO ENGLAND<|>Return followed injury<|>recovery<|>1)'
                    '<|COMPLETE|>')
        if TEXT in prompt:
            return ('("entity"<|>WATSON<|>PERSON<|>Watson was wounded at Maiwand)'
                    '##("entity"<|>ENGLAND<|>GEO<|>Watson returned to England)'
                    '##("relationship"<|>WATSON<|>ENGLAND<|>Watson returned to England<|>return<|>1)'
                    '<|COMPLETE|>')
        return json.dumps({"entities": ["Watson"], "events": ["Watson wounded"]})

    def make_rag(**kwargs):
        directory = Path(kwargs["working_dir"])
        if directory.name == "combined_kg":
            # The new instance must be opened after the graph and KVs exist.
            for name in ("graph_chunk_entity_relation.graphml",
                         "kv_store_text_chunks.json", "kv_store_full_docs.json"):
                assert (directory / name).exists()
            assert Path(kwargs["entity_storage_dir"]).name == "entity_kg"
            assert Path(kwargs["event_storage_dir"]).name == "event_kg"
        rag = EntityEventRAG(
            **kwargs, embedding_func=EmbeddingFunc(3, 8192, embed),
            llm_model_func=model, entity_extract_max_gleaning=0,
        )
        real_insert, real_query = rag.insert, rag.query

        def insert(content):
            state["inserts"].append(kwargs["extraction_type"])
            return real_insert(content)

        def query(question, param):
            # The unrelated hybrid demo is outside this regression's scope.
            if param.mode == "hybrid":
                return "hybrid fixture"
            result = real_query(question, param)
            state["responses"].append(result)
            return result

        rag.insert, rag.query = insert, query
        if directory.name == "combined_kg":
            state["combined"].append(rag)
        return rag

    monkeypatch.setattr(basic_usage, "EntityEventRAG", make_rag)
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    try:
        yield state
    finally:
        loop.close()
        asyncio.set_event_loop(None)


@pytest.mark.parametrize("cached", [False, True])
def test_basic_example_assembles_both_graphs_before_querying(example, cached):
    cache = Path("example_cache")
    if cached:
        basic_usage.main()
        for name in ("graph_chunk_entity_relation.graphml",
                     "kv_store_text_chunks.json", "kv_store_full_docs.json"):
            (cache / "combined_kg" / name).unlink()
        example["inserts"].clear()
    basic_usage.main()
    assert example["inserts"] == ([] if cached else ["entity", "event"])
    _, nodes, _ = example["responses"][-1]
    events = [node for node in nodes if node.get("node_type") == "event"]
    assert len(events) == 2
    assert {node["event_type"] for node in events} == {"BATTLE", "JOURNEY"}
    assert all(node["entity_type"] == "EVENT" for node in events)
    graph = nx.read_graphml(cache / "combined_kg/graph_chunk_entity_relation.graphml")
    assert len(graph) == 4
    assert any(edge["edge_type"] == "entity_event_relation" for _, _, edge in graph.edges(data=True))
    for name in ("kv_store_text_chunks.json", "kv_store_full_docs.json"):
        assert (cache / "combined_kg" / name).read_bytes() == (cache / "entity_kg" / name).read_bytes()
    assert example["prompts"]


@pytest.mark.parametrize("direct_embedding", [False, True])
def test_existing_retrieval_accepts_normalized_combined_graph(example, direct_embedding):
    basic_usage.main()
    rag = example["combined"][-1]
    _, nodes, _ = rag.query(
        "What happened to Watson after Maiwand?",
        QueryParam(mode="entity_event", use_direct_embedding=direct_embedding, event_hop_depth=0),
    )
    assert sum(n.get("node_type") == "event" for n in nodes) == 2


def test_graph_assembly_failure_does_not_open_combined_query_instance(example, monkeypatch, capsys):
    monkeypatch.setattr(basic_usage, "create_combined_kg", lambda *args: None)
    basic_usage.main()
    assert not example["combined"]
    assert "Failed to construct the combined entity-event graph" in capsys.readouterr().out
