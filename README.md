# Entity-Event RAG

**Bipartite Graph-based Retrieval-Augmented Generation**

Entity-Event RAG is a novel RAG system that builds a bipartite knowledge graph connecting **entities** and **events**, enabling more comprehensive retrieval for complex queries involving temporal and narrative understanding.

## Key Features

- **Bipartite Entity-Event Graph**: Instead of traditional entity-relationship graphs, builds a bipartite structure connecting entities to events
- **Multiple Query Modes**: Supports `naive`, `local`, `global`, `hybrid`, and `entity_event` retrieval modes
- **Event-Aware Retrieval**: Hop through events to discover related entities and context
- **HyDE Support**: Optional hypothetical document embedding for improved retrieval
- **Flexible Storage**: NetworkX, Neo4j, or Oracle for graph storage

## Installation

```bash
# Install from source
cd EntityEventRAG
pip install -e .

# With visualization dependencies
pip install -e ".[visualization]"
```

## Setup

Configure your OpenAI API key:

```bash
cp .env.example .env
# Edit .env and add your OPENAI_API_KEY
```

Or run the example script which indexes `RAG_documents/Lady_Susan.txt` and runs sample queries:

```bash
python examples/basic_usage.py
```

Or run the interactive CLI to index a PDF/TXT (or paste text), then query and visualize:

```bash
python examples/interactive_query.py
```

Both examples use `./example_cache` by default. If cached graph files already exist there, `basic_usage.py` skips indexing and `interactive_query.py` lets you reuse them via “Use existing Knowledge Graphs”.

For `entity_event` queries, both examples assemble a combined graph using
`construct_bipartite_graph` and `create_combined_kg` from the interactive example.
The query instance loads that graph and its source chunks from `combined_kg`,
with `entity_storage_dir=entity_kg` and `event_storage_dir=event_kg`. The graph
builder labels event nodes as `EVENT`, preserving their original subtype in
`event_type`. Reusing cached subgraphs still requires this assembly step before
constructing the query instance. Keep the same embedding model used to index
those caches when querying them.

## Quick Start

```python
from entity_event_rag import EntityEventRAG
from entity_event_rag.base import QueryParam

# Initialize with entity extraction
rag = EntityEventRAG(
    working_dir="./my_knowledge_graph",
    extraction_type="entity"  # or "event"
)

# Index a document
with open("document.txt") as f:
    rag.insert(f.read())

# This instance contains only the entity graph; use hybrid retrieval here.
# For entity_event retrieval, use the combined-graph examples above.
response = rag.query(
    "What are the main events in the story?",
    param=QueryParam(mode="hybrid")
)
print(response)
```

## Repository Structure

```
EntityEventRAG/
├── entity_event_rag/     # Core library
│   ├── core.py           # Main EntityEventRAG class
│   ├── operate.py        # Graph operations
│   ├── prompt.py         # Extraction prompts
│   └── ...
├── config/               # Configuration files
│   └── default.py        # Default settings & profiles
├── scripts/              # Utility scripts
│   └── visualize_graph.py
├── experiments/          # Experiment runners
│   └── run_experiment.py
├── examples/             # Usage examples
│   ├── basic_usage.py    # Sample queries for Lady_Susan.txt
│   └── interactive_query.py
├── RAG_documents/        # Benchmark literary works
│   ├── Lady_Susan.txt
│   ├── Pride_and_Prejudice.txt
│   ├── A_Study_in_Scarlet.txt
│   └── ... (16 classic novels)
```

### RAG_documents

This folder contains 16 classic literary works used for benchmarking and evaluation:

1. Lady Susan
2. Pride and Prejudice
3. A Study in Scarlet
4. The Adventures of Sherlock Holmes
5. The Hound of the Baskervilles
6. The Sign of the Four
7. The Mysterious Affair at Styles
8. The Picture of Dorian Gray
9. Anne of Green Gables
10. The Phantom of the Opera
11. The Secret Garden
12. The Wonderful Wizard of Oz
13. Les Misérables
14. The Sorrows of Young Werther
15. Dangerous Connections
16. The Diary of a Nobody

## Configuration

Use Python-based configuration for type safety:

```python
from config import EntityEventRAGConfig, ConfigProfiles

# Use a pre-configured profile
config = ConfigProfiles.narrative_analysis()

# Or customize settings
config = EntityEventRAGConfig()
config.retrieval.event_hop_depth = 4
config.extraction.event_types = ["battle", "discovery", "meeting"]
```

## Query Modes

| Mode | Description |
|------|-------------|
| `naive` | Simple vector similarity search on chunks |
| `local` | Entity-centric retrieval with local context |
| `global` | Relationship-centric retrieval with global context |
| `hybrid` | Combines local and global approaches |
| `entity_event` | Bipartite graph traversal through entities and events |

## Running Experiments

```bash
# Run batch queries
python experiments/run_experiment.py \
    --working-dir ./cache \
    --query "What happened to Jonathan Harker?" \
    --mode entity_event \
    --event-hop-depth 2

# Visualize knowledge graph
python scripts/visualize_graph.py \
    --working-dir ./cache/entity_kg \
    --mode 2d
```


## Citation

If you use Entity-Event RAG in your research, please cite:

```bibtex
@article{zhang2025respecting,
  title={Respecting Temporal-Causal Consistency: Entity-Event Knowledge Graphs for Retrieval-Augmented Generation},
  author={Zhang, Ze Yu and Li, Zitao and Li, Yaliang and Ding, Bolin and Low, Bryan Kian Hsiang},
  journal={arXiv preprint arXiv:2506.05939},
  year={2025}
}
```
## Acknowledgements

This repository is developed with reference to the [LightRAG](https://github.com/HKUDS/LightRAG) project.

## License

MIT License
