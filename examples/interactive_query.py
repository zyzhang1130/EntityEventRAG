#!/usr/bin/env python3
"""
Interactive script to index documents, query, and visualize:
- entity knowledge graph
- event knowledge graph
- bipartite connections
- combined graph
"""

import hashlib
import json
import math
import os
import shutil
import sys
import traceback
import webbrowser
from pathlib import Path
from typing import Optional

import networkx as nx
from dotenv import load_dotenv

load_dotenv()

# Add parent directory to path for development
sys.path.insert(0, str(Path(__file__).parent.parent))

from entity_event_rag import EntityEventRAG
from entity_event_rag.base import QueryParam

try:
    from pyvis.network import Network  # type: ignore
except Exception:
    Network = None

try:
    import readline  # noqa: F401
except Exception:
    readline = None

try:
    import plotly.graph_objects as go  # type: ignore
except Exception:
    go = None

try:
    from PyPDF2 import PdfReader  # type: ignore
except Exception:
    PdfReader = None


WORKING_DIR = "./example_cache"
ENTITY_DIR = os.path.join(WORKING_DIR, "entity_kg")
EVENT_DIR = os.path.join(WORKING_DIR, "event_kg")
COMBINED_DIR = os.path.join(WORKING_DIR, "combined_kg")


def get_input(prompt: str) -> str:
    try:
        return input(prompt)
    except EOFError:
        return "exit"
    except KeyboardInterrupt:
        print("\nInput cancelled.")
        return "exit"


def _graphml_path(working_dir: str) -> str:
    return os.path.join(working_dir, "graph_chunk_entity_relation.graphml")


def _safe_json_load(path: str):
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except FileNotFoundError:
        return None
    except Exception as e:
        print(f"Error reading JSON {path}: {e}")
        return None


def get_color_for_id(id_string: str) -> str:
    return f"#{hashlib.md5(str(id_string).encode()).hexdigest()[:6]}"


def _split_source_ids(source_id_value) -> list[str]:
    if not source_id_value:
        return []
    if isinstance(source_id_value, str):
        return [s for s in source_id_value.split("<SEP>") if s]
    return [str(source_id_value)]


def _load_chunk_store(working_dir: str) -> dict:
    store = _safe_json_load(os.path.join(working_dir, "kv_store_text_chunks.json"))
    return store if isinstance(store, dict) else {}


def get_chunk_ids(G: nx.Graph) -> list[str]:
    chunk_ids: set[str] = set()
    for _, data in G.nodes(data=True):
        chunk_ids.update(_split_source_ids(data.get("source_id", "")))
    return sorted(chunk_ids)


def get_doc_ids(G: nx.Graph, working_dir: str) -> list[str]:
    chunk_store = _load_chunk_store(working_dir)
    chunk_to_doc = {
        chunk_id: data.get("full_doc_id")
        for chunk_id, data in chunk_store.items()
        if isinstance(data, dict) and data.get("full_doc_id")
    }

    doc_ids: set[str] = set()
    for _, data in G.nodes(data=True):
        for chunk_id in _split_source_ids(data.get("source_id", "")):
            doc_id = chunk_to_doc.get(chunk_id)
            if doc_id:
                doc_ids.add(doc_id)
    return sorted(doc_ids)


def visualize_graph_2d(G: nx.Graph, working_dir: str, coloring_mode: str = "document") -> Optional[str]:
    if Network is None:
        print("pyvis is not installed; 2D visualization is unavailable.")
        return None

    try:
        net = Network(height="750px", width="100%", bgcolor="#ffffff", font_color="black", directed=True)

        net.set_options(
            """
        {
          "physics": {
            "enabled": true,
            "solver": "forceAtlas2Based",
            "forceAtlas2Based": {
              "gravitationalConstant": -50,
              "centralGravity": 0.01,
              "springLength": 100,
              "springConstant": 0.08,
              "avoidOverlap": 1.0
            },
            "minVelocity": 0.01
          }
        }
        """
        )

        chunk_to_doc: dict[str, str] = {}
        color_mapping: dict[str, str] = {}
        node_colors = {"entity": "#FF5733", "event": "#33C1FF"}
        edge_colors = {
            "entity_relation": "#FF8C33",
            "event_relation": "#33A1FF",
            "entity_event_relation": "#9933FF",
        }

        if working_dir == COMBINED_DIR:
            legend_title = "Node and Edge Types"
        elif coloring_mode == "chunk":
            chunk_ids = get_chunk_ids(G)
            color_mapping = {chunk_id: get_color_for_id(chunk_id) for chunk_id in chunk_ids}
            legend_title = "Chunk IDs"
        else:
            doc_ids = get_doc_ids(G, working_dir)
            color_mapping = {doc_id: get_color_for_id(doc_id) for doc_id in doc_ids}
            legend_title = "Document IDs"
            chunk_store = _load_chunk_store(working_dir)
            chunk_to_doc = {
                chunk_id: dp.get("full_doc_id", "unknown")
                for chunk_id, dp in chunk_store.items()
                if isinstance(dp, dict)
            }

        for node in G.nodes():
            if working_dir == COMBINED_DIR:
                node_type = str(G.nodes[node].get("node_type", "unknown")).strip('"')
                color = node_colors.get(node_type, "#808080")
                description = G.nodes[node].get("description", "No description")
                title = f"Node: {node}<br>Type: {node_type}<br>Description: {description}"
            else:
                source_ids = _split_source_ids(G.nodes[node].get("source_id", ""))
                chunk_id = source_ids[0] if source_ids else "unknown"
                description = G.nodes[node].get("description", "No description")
                if coloring_mode == "chunk":
                    color = color_mapping.get(chunk_id, "#808080")
                    title = f"Node: {node}<br>Chunk ID: {chunk_id}<br>Description: {description}"
                else:
                    doc_id = chunk_to_doc.get(chunk_id, "unknown")
                    color = color_mapping.get(doc_id, "#808080")
                    title = f"Node: {node}<br>Doc ID: {doc_id}<br>Description: {description}"

            net.add_node(node, title=title, color=color, label=node)

        for src, tgt, edge_data in G.edges(data=True):
            if working_dir == COMBINED_DIR:
                edge_type = str(edge_data.get("edge_type", "unknown")).strip('"')
                color = edge_colors.get(edge_type, "#808080")
                relationship = str(edge_data.get("keywords", "")).strip('"') if "keywords" in edge_data else edge_type
                weight = edge_data.get("weight", "N/A")
                description = str(edge_data.get("description", "No description")).strip('"')
                title = (
                    f"Edge Type: {edge_type}<br>Relationship: {relationship}"
                    f"<br>Weight: {weight}<br>Description: {description}"
                )
            else:
                relationship = str(edge_data.get("keywords", "unknown")).strip('"')
                weight = edge_data.get("weight", "N/A")
                description = str(edge_data.get("description", "No description")).strip('"')
                kg_type = "Event" if working_dir == EVENT_DIR else "Entity"
                title = f"{kg_type} Relationship: {relationship}<br>Weight: {weight}<br>Description: {description}"
                color = "gray"

            net.add_edge(src, tgt, title=title, color=color)

        output_path = os.path.join(working_dir, "knowledge_graph_2d.html")
        net.write_html(output_path)

        if working_dir == COMBINED_DIR:
            legend_html = """
            <div id='legend'>
                <h3>Node Types:</h3>
                <p><span style='color:#FF5733;'>&#9679;</span> Entities</p>
                <p><span style='color:#33C1FF;'>&#9679;</span> Events</p>
                <h3>Edge Types:</h3>
                <p><span style='color:#FF8C33;'>&#9679;</span> Entity Relations</p>
                <p><span style='color:#33A1FF;'>&#9679;</span> Event Relations</p>
                <p><span style='color:#9933FF;'>&#9679;</span> Entity-Event Relations</p>
            </div>
            """
        else:
            legend_html = f"<div id='legend'><h3>{legend_title}:</h3>"
            for id_value, color in sorted(color_mapping.items()):
                legend_html += f"<p><span style='color:{color};'>&#9679;</span> {id_value}</p>"
            legend_html += "</div>"

        custom_style = """
        <style>
        #legend {
            position: fixed;
            top: 10px;
            left: 10px;
            z-index: 999;
            background: white;
            padding: 10px;
            border: 1px solid #ccc;
            border-radius: 4px;
            max-height: 80%;
            overflow-y: auto;
            max-width: 250px;
            font-size: 12px;
            opacity: 0.9;
        }
        #legend h3 { margin-top: 0; margin-bottom: 5px; }
        #legend p { margin: 5px 0; }
        </style>
        """

        with open(output_path, "r", encoding="utf-8") as f:
            content = f.read()
        content = content.replace("<body>", f"<body>{custom_style}{legend_html}")
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(content)

        return output_path
    except Exception as e:
        print(f"Error in visualize_graph_2d: {e}")
        traceback.print_exc()
        return None


def visualize_graph_3d(G: nx.Graph, working_dir: str, coloring_mode: str = "document") -> Optional[str]:
    if go is None:
        print("Plotly not installed; 3D visualization is unavailable.")
        return None

    try:
        pos = nx.spring_layout(G, dim=3)

        if working_dir == COMBINED_DIR:
            node_colors_map = {"entity": "#FF5733", "event": "#33C1FF"}
            edge_colors_map = {
                "entity_relation": "#FF8C33",
                "event_relation": "#33A1FF",
                "entity_event_relation": "#9933FF",
            }
            legend_title = "Node and Edge Types"
        elif coloring_mode == "chunk":
            chunk_ids = get_chunk_ids(G)
            color_mapping = {chunk_id: get_color_for_id(chunk_id) for chunk_id in chunk_ids}
            legend_title = "Chunk IDs"
        else:
            doc_ids = get_doc_ids(G, working_dir)
            color_mapping = {doc_id: get_color_for_id(doc_id) for doc_id in doc_ids}
            legend_title = "Document IDs"

        node_colors: list[str] = []
        node_texts: list[str] = []
        for node in G.nodes():
            if working_dir == COMBINED_DIR:
                node_type = str(G.nodes[node].get("node_type", "unknown")).strip('"')
                color = node_colors_map.get(node_type, "#808080")
                description = G.nodes[node].get("description", "No description")
                node_texts.append(f"Node: {node}<br>Type: {node_type}<br>Description: {description}")
            else:
                source_ids = _split_source_ids(G.nodes[node].get("source_id", ""))
                chunk_id = source_ids[0] if source_ids else "unknown"
                description = G.nodes[node].get("description", "No description")
                if coloring_mode == "chunk":
                    color = color_mapping.get(chunk_id, "#808080")
                    node_texts.append(f"Node: {node}<br>Chunk ID: {chunk_id}<br>Description: {description}")
                else:
                    chunk_store = _load_chunk_store(working_dir)
                    doc_id = chunk_store.get(chunk_id, {}).get("full_doc_id", "unknown")
                    color = color_mapping.get(doc_id, "#808080")
                    node_texts.append(f"Node: {node}<br>Doc ID: {doc_id}<br>Description: {description}")
            node_colors.append(color)

        node_x = [pos[node][0] for node in G.nodes()]
        node_y = [pos[node][1] for node in G.nodes()]
        node_z = [pos[node][2] for node in G.nodes()]

        node_trace = go.Scatter3d(
            x=node_x,
            y=node_y,
            z=node_z,
            mode="markers+text",
            marker=dict(size=5, color=node_colors, line=dict(width=1, color="white")),
            text=list(G.nodes()),
            textposition="top center",
            hovertext=node_texts,
            hoverinfo="text",
        )

        edge_traces = []
        for src, tgt, edge_data in G.edges(data=True):
            x0, y0, z0 = pos[src]
            x1, y1, z1 = pos[tgt]

            if working_dir == COMBINED_DIR:
                edge_type = str(edge_data.get("edge_type", "unknown")).strip('"')
                color = edge_colors_map.get(edge_type, "#808080")
                relationship = str(edge_data.get("keywords", "")).strip('"') if "keywords" in edge_data else edge_type
                weight = edge_data.get("weight", "N/A")
                description = str(edge_data.get("description", "No description")).strip('"')
                hover_text = (
                    f"Edge Type: {edge_type}<br>Relationship: {relationship}"
                    f"<br>Weight: {weight}<br>Description: {description}"
                )
            else:
                relationship = str(edge_data.get("keywords", "unknown")).strip('"')
                weight = edge_data.get("weight", "N/A")
                description = str(edge_data.get("description", "No description")).strip('"')
                kg_type = "Event" if working_dir == EVENT_DIR else "Entity"
                hover_text = f"{kg_type} Relationship: {relationship}<br>Weight: {weight}<br>Description: {description}"
                color = "gray"

            edge_traces.append(
                go.Scatter3d(
                    x=[x0, x1],
                    y=[y0, y1],
                    z=[z0, z1],
                    mode="lines",
                    line=dict(width=3, color=color),
                    text=[hover_text, hover_text],
                    hoverinfo="text",
                )
            )

            dx, dy, dz = x1 - x0, y1 - y0, z1 - z0
            length = (dx * dx + dy * dy + dz * dz) ** 0.5
            if length <= 0:
                continue
            dx, dy, dz = dx / length, dy / length, dz / length
            arrow_size = length * 0.15
            arrow_angle = 0.5
            base_x = x1 - dx * arrow_size
            base_y = y1 - dy * arrow_size
            base_z = z1 - dz * arrow_size
            perpx, perpy, perpz = -dy, dx, 0
            perp_length = (perpx * perpx + perpy * perpy + perpz * perpz) ** 0.5
            if perp_length <= 0:
                continue
            scale = arrow_size * math.sin(arrow_angle) / perp_length
            perpx, perpy, perpz = perpx * scale, perpy * scale, perpz * scale
            left_x, left_y, left_z = base_x + perpx, base_y + perpy, base_z + perpz
            right_x, right_y, right_z = base_x - perpx, base_y - perpy, base_z - perpz
            edge_traces.append(
                go.Scatter3d(
                    x=[left_x, x1, right_x],
                    y=[left_y, y1, right_y],
                    z=[left_z, z1, right_z],
                    mode="lines",
                    line=dict(width=3, color=color),
                    showlegend=False,
                )
            )

        fig = go.Figure(data=[*edge_traces, node_trace])
        fig.update_layout(
            title="Knowledge Graph (3D View)",
            showlegend=False,
            margin=dict(l=0, r=0, t=40, b=0),
            scene=dict(
                xaxis=dict(showticklabels=False, showgrid=False, zeroline=False),
                yaxis=dict(showticklabels=False, showgrid=False, zeroline=False),
                zaxis=dict(showticklabels=False, showgrid=False, zeroline=False),
                aspectmode="data",
                camera=dict(up=dict(x=0, y=0, z=1), center=dict(x=0, y=0, z=0), eye=dict(x=2, y=2, z=2)),
            ),
        )

        output_path = os.path.join(working_dir, "knowledge_graph_3d.html")
        fig.write_html(output_path)

        if working_dir == COMBINED_DIR:
            legend_html = """
            <div id='legend'>
                <h3>Node Types:</h3>
                <p><span style='color:#FF5733;'>&#9679;</span> Entities</p>
                <p><span style='color:#33C1FF;'>&#9679;</span> Events</p>
                <h3>Edge Types:</h3>
                <p><span style='color:#FF8C33;'>&#9679;</span> Entity Relations</p>
                <p><span style='color:#33A1FF;'>&#9679;</span> Event Relations</p>
                <p><span style='color:#9933FF;'>&#9679;</span> Entity-Event Relations</p>
            </div>
            """
        else:
            legend_html = f"<div id='legend'><h3>{legend_title}:</h3>"
            for id_value, color in sorted(color_mapping.items()):
                legend_html += f"<p><span style='color:{color};'>&#9679;</span> {id_value}</p>"
            legend_html += "</div>"

        custom_style = """
        <style>
        #legend {
            position: fixed;
            top: 10px;
            left: 10px;
            z-index: 999;
            background: white;
            padding: 10px;
            border: 1px solid #ccc;
            border-radius: 4px;
            max-height: 80%;
            overflow-y: auto;
            max-width: 250px;
            font-size: 12px;
            opacity: 0.9;
        }
        #legend h3 { margin-top: 0; margin-bottom: 5px; }
        #legend p { margin: 5px 0; }
        </style>
        """

        with open(output_path, "r", encoding="utf-8") as f:
            html_content = f.read()
        html_content = html_content.replace("<body>", f"<body>{custom_style}{legend_html}")
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(html_content)

        return output_path
    except Exception as e:
        print(f"Error in visualize_graph_3d: {e}")
        traceback.print_exc()
        return None


def visualize_graph(working_dir: str) -> None:
    graph_path = _graphml_path(working_dir)
    if not os.path.exists(graph_path):
        print(f"No graph file found at {graph_path}")
        return

    try:
        print(f"Loading graph from {graph_path}...")
        G = nx.read_graphml(graph_path)
        if G is None or len(G.nodes) == 0:
            print("The graph is empty.")
            return
        print(f"Loaded graph with {len(G.nodes)} nodes and {len(G.edges)} edges")

        while True:
            viz_type = get_input("\nChoose visualization type (2/3 for 2D/3D): ").strip()
            if viz_type in {"2", "3", "exit"}:
                break
            print("Invalid choice. Please enter 2 or 3.")
        if viz_type == "exit":
            return

        if working_dir != COMBINED_DIR:
            while True:
                print("\nChoose coloring mode:")
                print("1. Color by chunk ID")
                print("2. Color by document ID")
                coloring_choice = get_input("Enter choice (1/2): ").strip()
                if coloring_choice in {"1", "2", "exit"}:
                    break
                print("Invalid choice. Please enter 1 or 2.")
            if coloring_choice == "exit":
                return
            coloring_mode = "chunk" if coloring_choice == "1" else "document"
        else:
            coloring_mode = "document"

        print("\nGenerating visualization...")
        if viz_type == "2":
            output_path = visualize_graph_2d(G, working_dir, coloring_mode=coloring_mode)
        else:
            output_path = visualize_graph_3d(G, working_dir, coloring_mode=coloring_mode)
            if output_path is None:
                output_path = visualize_graph_2d(G, working_dir, coloring_mode=coloring_mode)

        if output_path:
            print(f"\nGraph visualization saved to: {output_path}")
            webbrowser.open(f"file://{os.path.abspath(output_path)}")
    except Exception as e:
        print(f"Error visualizing graph: {e}")
        traceback.print_exc()


def construct_bipartite_graph(entity_graph_path: str, event_graph_path: str) -> dict[str, list[str]]:
    try:
        entity_G = nx.read_graphml(entity_graph_path)
        event_G = nx.read_graphml(event_graph_path)

        events_by_chunk: dict[str, list[tuple[str, dict]]] = {}
        for event_node, data in event_G.nodes(data=True):
            for chunk_id in _split_source_ids(data.get("source_id", "")):
                events_by_chunk.setdefault(chunk_id, []).append((event_node, data))

        bipartite_graph: dict[str, list[str]] = {}
        for entity_node in entity_G.nodes():
            chunk_idx = entity_node.find("_chunk-")
            if chunk_idx == -1:
                continue
            entity_name = entity_node[:chunk_idx].strip('"')
            entity_chunk_id = entity_node[chunk_idx + 1 :]

            for event_node, data in events_by_chunk.get(entity_chunk_id, []):
                event_description = str(data.get("description", "") or "")
                event_chunk_idx = event_node.find("_chunk-")
                event_name = event_node[:event_chunk_idx].strip('"') if event_chunk_idx != -1 else event_node.strip('"')
                if entity_name.lower() in event_description.lower() or entity_name.lower() in event_name.lower():
                    bipartite_graph.setdefault(entity_node, []).append(event_node)

        return bipartite_graph
    except Exception as e:
        print(f"Error constructing bipartite graph: {e}")
        traceback.print_exc()
        return {}


def create_combined_kg(
    entity_graph_path: str,
    event_graph_path: str,
    bipartite_graph: dict[str, list[str]],
    output_dir: str,
) -> Optional[nx.Graph]:
    try:
        entity_G = nx.read_graphml(entity_graph_path)
        event_G = nx.read_graphml(event_graph_path)

        combined_G = nx.DiGraph()

        for node, data in entity_G.nodes(data=True):
            d = dict(data)
            d["node_type"] = "entity"
            d["display_color"] = "#FF5733"
            combined_G.add_node(node, **d)
        for u, v, data in entity_G.edges(data=True):
            d = dict(data)
            d["edge_type"] = "entity_relation"
            d["display_color"] = "#FF8C33"
            combined_G.add_edge(u, v, **d)

        for node, data in event_G.nodes(data=True):
            d = dict(data)
            original_type = str(d.get("entity_type", "")).strip('"')
            if original_type and original_type.upper() != "EVENT":
                d["event_type"] = original_type
            d["entity_type"] = "EVENT"
            d["node_type"] = "event"
            d["display_color"] = "#33C1FF"
            combined_G.add_node(node, **d)
        for u, v, data in event_G.edges(data=True):
            d = dict(data)
            d["edge_type"] = "event_relation"
            d["display_color"] = "#33A1FF"
            combined_G.add_edge(u, v, **d)

        for entity_node, event_nodes in bipartite_graph.items():
            entity_data = entity_G.nodes.get(entity_node, {})
            for event_node in event_nodes:
                event_data = event_G.nodes.get(event_node, {})
                combined_G.add_edge(
                    entity_node,
                    event_node,
                    edge_type="entity_event_relation",
                    description=f"Entity '{entity_data.get('description', '')}' is involved in event '{event_data.get('description', '')}'",
                    weight=1.0,
                    keywords="ENTITY_EVENT_RELATION",
                    source_id=f"{entity_data.get('source_id', '')}<SEP>{event_data.get('source_id', '')}",
                    display_color="#9933FF",
                )

        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(output_dir, "graph_chunk_entity_relation.graphml")
        nx.write_graphml(combined_G, output_path)
        return combined_G
    except Exception as e:
        print(f"Error creating combined KG: {e}")
        traceback.print_exc()
        return None


def visualize_bipartite_graph(
    bipartite_graph: dict[str, list[str]],
    entity_graph_path: str,
    event_graph_path: str,
    output_path: str,
) -> None:
    if Network is None:
        print("pyvis is not installed; bipartite visualization is unavailable.")
        return
    try:
        if not bipartite_graph:
            print("No bipartite edges available to visualize.")
            return

        B = nx.Graph()
        entity_nodes = list(bipartite_graph.keys())
        event_nodes: set[str] = set()
        for events in bipartite_graph.values():
            event_nodes.update(events)
        event_nodes_list = sorted(event_nodes)

        B.add_nodes_from(entity_nodes, bipartite=0)
        B.add_nodes_from(event_nodes_list, bipartite=1)
        for entity, events in bipartite_graph.items():
            for event in events:
                B.add_edge(entity, event)

        entity_G = nx.read_graphml(entity_graph_path)
        event_G = nx.read_graphml(event_graph_path)
        node_descriptions = {}
        for node in entity_G.nodes():
            node_descriptions[node] = entity_G.nodes[node].get("description", "No description")
        for node in event_G.nodes():
            node_descriptions[node] = event_G.nodes[node].get("description", "No description")

        node_mapping = {node.upper(): node for node in B.nodes()}

        print("\nOptions:")
        print("1. Show full bipartite graph")
        print("2. Show neighbors of a specific node")
        choice = get_input("Enter choice (1/2): ").strip()

        if choice == "2":
            print("\nEnter a node name to list neighbors (or 'return' to go back).")
            while True:
                input_node = get_input("Node name: ").strip()
                if input_node.lower() in {"return", "exit"}:
                    return
                maybe = node_mapping.get(input_node.upper())
                if not maybe:
                    print("Node not found (case-insensitive). Try again.")
                    continue
                neigh = list(B.neighbors(maybe))
                side = "Entity" if B.nodes[maybe].get("bipartite") == 0 else "Event"
                print(f"\n{side} node: {maybe}")
                print(f"Neighbors ({len(neigh)}):")
                for n in neigh[:50]:
                    print(f"- {n}")
                if len(neigh) > 50:
                    print(f"... and {len(neigh) - 50} more")
                print("\nEnter another node name (or 'return' to go back).")

        pos = nx.bipartite_layout(B, entity_nodes)

        net = Network(height="750px", width="100%", bgcolor="#ffffff", font_color="black")
        physics_options = {
            "enabled": True,
            "solver": "barnesHut",
            "barnesHut": {
                "gravitationalConstant": -2000,
                "centralGravity": 0.3,
                "springLength": 200,
                "springConstant": 0.04,
                "damping": 0.09,
                "avoidOverlap": 0.1,
            },
            "minVelocity": 0.75,
        }
        net.set_options(json.dumps({"physics": physics_options}))

        for node in B.nodes():
            group = B.nodes[node].get("bipartite")
            color = "#FF5733" if group == 0 else "#33C1FF"
            x, y = pos[node]
            description = node_descriptions.get(node, "No description")
            hover_text = f"{node}\nDescription: {description}"
            net.add_node(node, label="", title=hover_text, color=color, x=float(x) * 500, y=float(y) * 500)

        for u, v in B.edges():
            net.add_edge(u, v)

        legend_html = """
        <div id='legend'>
            <h3>Node Types:</h3>
            <p><span style='color:#FF5733;'>&#9679;</span> Entities</p>
            <p><span style='color:#33C1FF;'>&#9679;</span> Events</p>
        </div>
        """
        custom_style = """
        <style>
        #legend {
            position: fixed;
            top: 10px;
            left: 10px;
            z-index: 999;
            background: white;
            padding: 10px;
            border: 1px solid #ccc;
            border-radius: 4px;
            font-size: 12px;
            opacity: 0.9;
        }
        #legend h3 { margin-top: 0; }
        </style>
        """

        os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
        net.write_html(output_path)
        with open(output_path, "r", encoding="utf-8") as f:
            html_content = f.read()
        html_content = html_content.replace("<body>", f"<body>{custom_style}{legend_html}")
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(html_content)

        print(f"\nGraph visualization saved to: {output_path}")
        webbrowser.open(f"file://{os.path.abspath(output_path)}")
    except Exception as e:
        print(f"Error visualizing bipartite graph: {e}")
        traceback.print_exc()


def _infer_node_type(node_id: str, node_data: Optional[dict] = None) -> str:
    if node_data and node_data.get("node_type"):
        return str(node_data.get("node_type")).strip('"').lower()
    return "entity" if "_chunk-" in (node_id or "") else "event"


def _infer_edge_type(src: str, tgt: str, edge_data: Optional[dict] = None) -> str:
    if edge_data and edge_data.get("edge_type"):
        return str(edge_data.get("edge_type")).strip('"')
    src_is_entity = "_chunk-" in (src or "")
    tgt_is_entity = "_chunk-" in (tgt or "")
    if src_is_entity and tgt_is_entity:
        return "entity_relation"
    if (not src_is_entity) and (not tgt_is_entity):
        return "event_relation"
    return "entity_event_relation"


def visualize_query_result(nodes, edges, output_dir: str):
    if Network is None:
        print("pyvis is not installed; query visualization is unavailable.")
        return None
    try:
        query_results_dir = os.path.join(output_dir, "query_results")
        os.makedirs(query_results_dir, exist_ok=True)

        net = Network(height="750px", width="100%", bgcolor="#ffffff", font_color="black", directed=True)

        node_data = {n.get("entity_name"): n for n in (nodes or []) if isinstance(n, dict) and n.get("entity_name")}
        node_ids: set[str] = set(node_data.keys())

        for edge in edges or []:
            if not isinstance(edge, dict):
                continue
            src_tgt = edge.get("src_tgt")
            if not src_tgt or not isinstance(src_tgt, (list, tuple)) or len(src_tgt) != 2:
                continue
            src, tgt = src_tgt[0], src_tgt[1]
            node_ids.add(src)
            node_ids.add(tgt)

        for node_id in sorted(node_ids):
            n = node_data.get(node_id, {})
            node_type = _infer_node_type(node_id, n)
            matched = bool(n.get("matched", False))
            description = n.get("description", "No description")

            if node_type == "entity":
                color = "#FF5733" if matched else "#FFA07A"
            else:
                color = "#33C1FF" if matched else "#87CEEB"

            title = f"Node: {node_id}<br>Type: {node_type}<br>Matched: {matched}<br>Description: {description}"
            net.add_node(node_id, title=title, color=color, label=node_id)

        edge_colors = {"entity_relation": "#FF8C33", "event_relation": "#33A1FF", "entity_event_relation": "#9933FF"}
        for edge in edges or []:
            if not isinstance(edge, dict):
                continue
            src_tgt = edge.get("src_tgt")
            if not src_tgt or not isinstance(src_tgt, (list, tuple)) or len(src_tgt) != 2:
                continue
            src, tgt = src_tgt[0], src_tgt[1]
            edge_type = edge.get("edge_type") or _infer_edge_type(src, tgt, edge)
            description = edge.get("description", "No description")
            keywords = edge.get("keywords", "")
            color = edge_colors.get(edge_type, "#888888")
            title = f"Edge Type: {edge_type}<br>Description: {description}"
            if keywords:
                title += f"<br>Keywords: {keywords}"
            net.add_edge(src, tgt, title=title, color=color)

        legend_html = """
        <div id='legend'>
            <h3>Matched Nodes:</h3>
            <p><span style='color:#FF5733;'>&#9679;</span> Entities</p>
            <p><span style='color:#33C1FF;'>&#9679;</span> Events</p>
            <h3>Related Nodes:</h3>
            <p><span style='color:#FFA07A;'>&#9679;</span> Entities</p>
            <p><span style='color:#87CEEB;'>&#9679;</span> Events</p>
            <h3>Edge Types:</h3>
            <p><span style='color:#FF8C33;'>&#9679;</span> Entity-Entity Relations</p>
            <p><span style='color:#33A1FF;'>&#9679;</span> Event-Event Relations</p>
            <p><span style='color:#9933FF;'>&#9679;</span> Entity-Event Relations</p>
        </div>
        """

        custom_style = """
        <style>
        #legend {
            position: fixed;
            top: 10px;
            left: 10px;
            z-index: 999;
            background: white;
            padding: 10px;
            border: 1px solid #ccc;
            border-radius: 4px;
            max-height: 80%;
            overflow-y: auto;
            max-width: 250px;
            font-size: 12px;
            opacity: 0.9;
        }
        #legend h3 { margin-top: 0; margin-bottom: 5px; }
        #legend p { margin: 5px 0; }
        </style>
        """

        output_path = os.path.join(query_results_dir, "query_result_graph.html")
        net.write_html(output_path)
        with open(output_path, "r", encoding="utf-8") as f:
            html_content = f.read()
        html_content = html_content.replace("<body>", f"<body>{custom_style}{legend_html}")
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(html_content)

        print(f"Created visualization of query results: {output_path}")
        webbrowser.open(f"file://{os.path.abspath(output_path)}")
        return output_path
    except Exception as e:
        print(f"Error creating query visualization: {e}")
        traceback.print_exc()
        return None


def read_pdf(file_path: str) -> Optional[str]:
    if PdfReader is None:
        print("Error: PyPDF2 not installed. Please install it with 'pip install PyPDF2'")
        return None
    try:
        reader = PdfReader(file_path)
        text = ""
        for page in reader.pages:
            text += (page.extract_text() or "") + "\n"
        return text
    except Exception as e:
        print(f"Error reading PDF: {e}")
        return None


def read_txt(file_path: str) -> Optional[str]:
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            return f.read()
    except Exception as e:
        print(f"Error reading TXT file: {e}")
        return None


def _sync_combined_kv_files() -> None:
    os.makedirs(COMBINED_DIR, exist_ok=True)
    for name in ("kv_store_text_chunks.json", "kv_store_full_docs.json"):
        src = os.path.join(ENTITY_DIR, name)
        dst = os.path.join(COMBINED_DIR, name)
        if os.path.exists(src):
            shutil.copy2(src, dst)


def main():
    os.makedirs(ENTITY_DIR, exist_ok=True)
    os.makedirs(EVENT_DIR, exist_ok=True)
    os.makedirs(COMBINED_DIR, exist_ok=True)

    entity_graph_path = _graphml_path(ENTITY_DIR)
    event_graph_path = _graphml_path(EVENT_DIR)

    print("=" * 60)
    print("Entity-Event KG - Interactive Mode")
    print("=" * 60)

    while True:
        print("\nChoose input source:")
        print("1. PDF file")
        print("2. TXT file")
        print("3. Direct text input")
        print("4. Use existing Knowledge Graphs (if available)")
        print("5. Exit")

        source_type = get_input("Enter choice (1-5): ").strip()
        if source_type.lower() == "exit" or source_type == "5":
            return

        process_content = source_type in {"1", "2", "3"}
        use_existing = source_type == "4"

        if use_existing:
            if not (os.path.exists(entity_graph_path) and os.path.exists(event_graph_path)):
                print("\nNo existing Knowledge Graphs found under ./example_cache.")
                continue
            print("\nUsing existing Knowledge Graphs...")
            break

        if not process_content:
            print("Invalid choice.")
            continue

        content: Optional[str] = None
        if source_type == "1":
            pdf_path = get_input("Enter the path to your PDF file: ").strip()
            if os.path.exists(pdf_path) and pdf_path.lower().endswith(".pdf"):
                content = read_pdf(pdf_path)
            else:
                print("Invalid PDF path.")
        elif source_type == "2":
            txt_path = get_input("Enter the path to your TXT file: ").strip()
            if os.path.exists(txt_path):
                content = read_txt(txt_path)
            else:
                print("Invalid TXT path.")
        elif source_type == "3":
            print("Enter your text (Ctrl+D to finish):")
            lines = []
            while True:
                try:
                    lines.append(input())
                except EOFError:
                    break
                except KeyboardInterrupt:
                    print("\nInput cancelled.")
                    return
            content = "\n".join(lines).strip()

        if not content:
            print("No content provided.")
            continue

        if not os.getenv("OPENAI_API_KEY"):
            print("Error: OPENAI_API_KEY not found in environment variables.")
            print("Indexing requires an API key. You can still choose option 4 to visualize existing graphs.")
            return

        print("\nProcessing content for entity extraction...")
        entity_rag = EntityEventRAG(working_dir=ENTITY_DIR, extraction_type="entity")
        entity_rag.insert(content)
        print("Entity extraction completed!")

        print("\nProcessing content for event extraction...")
        event_rag = EntityEventRAG(working_dir=EVENT_DIR, extraction_type="event")
        event_rag.insert(content)
        print("Event extraction completed!")

        break

    if not (os.path.exists(entity_graph_path) and os.path.exists(event_graph_path)):
        print("\nMissing entity/event KG graph files; nothing to do.")
        return

    print("\nConstructing bipartite graph between entity and event KGs...")
    bipartite_graph = construct_bipartite_graph(entity_graph_path, event_graph_path)
    print(f"Bipartite edges: {sum(len(v) for v in bipartite_graph.values())}")

    print("\nCreating combined Knowledge Graph...")
    combined_G = create_combined_kg(entity_graph_path, event_graph_path, bipartite_graph, COMBINED_DIR)
    if combined_G is None:
        print("Failed to create combined Knowledge Graph.")
        return
    _sync_combined_kv_files()

    print("\nAvailable query modes:")
    print("1. naive")
    print("2. local")
    print("3. global")
    print("4. hybrid")
    print("5. entity_event")

    entity_rag: Optional[EntityEventRAG] = None
    combined_rag: Optional[EntityEventRAG] = None

    while True:
        print("\n" + "=" * 50)
        print("Choose an option:")
        print("1. Provide query")
        print("2. Visualize knowledge graphs")
        print("3. Exit")

        option = get_input("Enter choice (1-3): ").strip()
        if option.lower() == "exit" or option == "3":
            break

        if option == "2":
            while True:
                kg_type = get_input("Choose KG to visualize (1 Entity, 2 Event, 3 Bipartite, 4 Combined): ").strip()
                if kg_type.lower() == "exit":
                    kg_type = ""
                    break
                if kg_type in {"1", "2", "3", "4"}:
                    break
                print("Invalid choice. Please enter 1, 2, 3, or 4.")

            if kg_type == "1":
                visualize_graph(ENTITY_DIR)
            elif kg_type == "2":
                visualize_graph(EVENT_DIR)
            elif kg_type == "3":
                visualize_bipartite_graph(
                    bipartite_graph,
                    entity_graph_path,
                    event_graph_path,
                    os.path.join(WORKING_DIR, "bipartite_graph.html"),
                )
            elif kg_type == "4":
                visualize_graph(COMBINED_DIR)
            continue

        if option == "1":
            if not os.getenv("OPENAI_API_KEY"):
                print("Error: OPENAI_API_KEY not found; querying requires an API key.")
                continue

            query = get_input("\nQuestion: ").strip()
            if not query or query.lower() == "exit":
                continue

            mode_map = {"1": "naive", "2": "local", "3": "global", "4": "hybrid", "5": "entity_event"}
            while True:
                print("Modes: 1=naive, 2=local, 3=global, 4=hybrid, 5=entity_event")
                mode_input = get_input("Enter mode number (1-5): ").strip()
                if mode_input.lower() == "exit":
                    mode_input = ""
                    break
                if mode_input in mode_map:
                    break
                print("Invalid mode.")

            if not mode_input:
                continue

            mode = mode_map[mode_input]
            print(f"\nExecuting query in {mode} mode...")

            if mode == "entity_event":
                if combined_rag is None:
                    combined_rag = EntityEventRAG(
                        working_dir=COMBINED_DIR,
                        extraction_type="entity",
                        entity_storage_dir=ENTITY_DIR,
                        event_storage_dir=EVENT_DIR,
                    )
                response = combined_rag.query(query, param=QueryParam(mode="entity_event", event_hop_depth=2))
                if isinstance(response, tuple) and len(response) == 3:
                    answer, nodes, edges = response
                    print(f"\nResult:\n{answer}")
                    ans = get_input("\nVisualize result subgraph? (y/N): ").strip().lower()
                    if ans in {"y", "yes"}:
                        visualize_query_result(nodes, edges, COMBINED_DIR)
                else:
                    print(f"\nResult:\n{response}")
            else:
                if entity_rag is None:
                    entity_rag = EntityEventRAG(working_dir=ENTITY_DIR, extraction_type="entity")
                response = entity_rag.query(query, param=QueryParam(mode=mode, top_k=10))
                print(f"\nResult:\n{response}")
            continue

        print("Invalid choice.")

    print("Goodbye!")


if __name__ == "__main__":
    main()
