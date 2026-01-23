#!/usr/bin/env python3
"""Simple 2D graph visualization using Pyvis.

Usage:
    python graph_visual_with_html.py path/to/graph.graphml
    python graph_visual_with_html.py path/to/graph.graphml -o output.html
"""

import sys
import argparse
import networkx as nx
from pyvis.network import Network
import random


def visualize_graph(graphml_path: str, output_path: str = "knowledge_graph.html"):
    """Load a GraphML file and create an interactive HTML visualization."""
    # Load the GraphML file
    G = nx.read_graphml(graphml_path)
    
    # Create a Pyvis network
    net = Network(height="100vh", notebook=False)
    
    # Convert NetworkX graph to Pyvis network
    net.from_nx(G)
    
    # Add colors to nodes
    for node in net.nodes:
        node["color"] = "#{:06x}".format(random.randint(0, 0xFFFFFF))
    
    # Save the network
    net.show(output_path)
    print(f"Visualization saved to: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Visualize a GraphML knowledge graph")
    parser.add_argument("graphml_path", help="Path to the GraphML file")
    parser.add_argument("-o", "--output", default="knowledge_graph.html",
                        help="Output HTML file path (default: knowledge_graph.html)")
    
    args = parser.parse_args()
    visualize_graph(args.graphml_path, args.output)


if __name__ == "__main__":
    main()
