import random
from typing import List
import re

import plotly.graph_objects as go
import networkx as nx

from agent import G_und

def show_graph(target_ids, hops=2, max_nodes=50, palette=None):
    """
    target_ids: list of node IDs (strings) to highlight.
                (Assume len(target_ids) > 0.)
    hops: int, the number of BFS layers to traverse.
    max_nodes: int, maximum nodes to sample at each layer.
    palette: optional list of hex colors for distinct classes.
             If None, a default palette is used.
    """
    # Provide a default color palette if none is given.
    if palette is None:
        palette = [
            "#1f77b4",
            "#ff7f0e",
            "#2ca02c",
            "#d62728",
            "#9467bd",
            "#8c564b",
            "#e377c2",
            "#7f7f7f",
            "#bcbd22",
            "#17becf",
        ]

    # Helper: Sample BFS layers manually.
    def sample_bfs_layers(G, target_ids, hops, max_nodes):
        nodes_included = set(target_ids)  # Always include target_ids.
        current_layer = set(target_ids)
        for _ in range(hops):
            next_layer = set()
            for node in current_layer:
                for nbr in G.neighbors(node):
                    if nbr not in nodes_included:
                        next_layer.add(nbr)
            if len(next_layer) > max_nodes:
                next_layer = set(random.sample(list(next_layer), max_nodes // hops))
            nodes_included.update(next_layer)
            current_layer = next_layer
        return nodes_included

    # if empty, use this as a default target node
    if len(target_ids) == 0:
        target_ids = ["Source/0001d6445b831d6a538e2d482186b60d"]

    # Get the nodes to include from the BFS layers.
    nodes_included = sample_bfs_layers(G_und, target_ids, hops, max_nodes)
    subG = G_und.subgraph(nodes_included)
    # Compute positions using Kamada–Kawai layout.
    pos = nx.kamada_kawai_layout(subG)

    # ----------------------------------------------------------------------
    # Helper: Extract class from an _id string formatted as "class/key".
    def get_class(full_id):
        idx = full_id.find("/")
        return full_id[:idx] if idx != -1 else full_id

    # Helper: Truncate text to 15 characters.
    def truncate_label(text):
        return text if len(text) <= 15 else text[:15] + "…"

    # ----------------------------------------------------------------------
    # Build edge traces.
    edge_x, edge_y = [], []
    for u, v in subG.edges():
        x0, y0 = pos[u]
        x1, y1 = pos[v]
        edge_x.extend([x0, x1, None])
        edge_y.extend([y0, y1, None])
    edge_trace = go.Scatter(
        x=edge_x,
        y=edge_y,
        line=dict(width=1, color="#888"),
        hoverinfo="none",
        mode="lines",
        name="Edges",
    )

    # Build edge label trace (at midpoints).
    edge_label_x, edge_label_y, edge_label_text, edge_label_hover = [], [], [], []
    for u, v in subG.edges():
        x0, y0 = pos[u]
        x1, y1 = pos[v]
        xm, ym = (x0 + x1) / 2.0, (y0 + y1) / 2.0
        # Extract full _id and then get the class.
        edge_data = G_und[u][v]
        full_edge_id = str(edge_data.get("_id", "unknown/???"))
        cls = get_class(full_edge_id)
        label = truncate_label(
            cls
        )  # For edge labels, show the class (truncated if necessary)
        edge_label_x.append(xm)
        edge_label_y.append(ym)
        edge_label_text.append(label)
        edge_label_hover.append(f"Edge _id: {full_edge_id}")
    edge_label_trace = go.Scatter(
        x=edge_label_x,
        y=edge_label_y,
        mode="text",
        text=edge_label_text,
        textposition="top center",
        hoverinfo="text",
        hovertext=edge_label_hover,
        textfont=dict(color="#444", size=10),
        name="Edge labels",
    )

    # ----------------------------------------------------------------------
    # Map each node class to a unique color.
    class_to_color = {}
    next_color_idx = 0

    def get_color_for_class(cls):
        nonlocal next_color_idx
        if cls not in class_to_color:
            class_to_color[cls] = palette[next_color_idx % len(palette)]
            next_color_idx += 1
        return class_to_color[cls]

    # Build node trace.
    node_x, node_y = [], []
    node_colors = []
    node_sizes = []
    node_labels = []
    node_hover = []

    for node in subG.nodes():
        x, y = pos[node]
        node_x.append(x)
        node_y.append(y)
        # Get full _id from node data.
        node_data = G_und.nodes[node]
        full_id = str(node_data.get("_id", node))
        cls = get_class(full_id)
        color = get_color_for_class(cls)
        # For target nodes, we want them to be prominent.
        size = 20 if node in target_ids else 12
        node_colors.append(color)
        node_sizes.append(size)
        # Display full _id truncated to 15 chars as label.
        label = truncate_label(full_id)
        node_labels.append(label)
        # Hover shows full _id and the name attribute.
        name_attr = node_data.get("name", "")
        hover_txt = f"_id: {full_id}<br>name: {name_attr}"
        node_hover.append(hover_txt)

    node_trace = go.Scatter(
        x=node_x,
        y=node_y,
        mode="markers+text",
        text=node_labels,
        textposition="top center",
        hoverinfo="text",
        hovertext=node_hover,
        marker=dict(
            color=node_colors,
            size=node_sizes,
            line=dict(width=2, color="#333"),
            opacity=0.9,
        ),
        name="Nodes",
    )

    # ----------------------------------------------------------------------
    # Build the final figure.
    fig = go.Figure(
        data=[edge_trace, edge_label_trace, node_trace],
        layout=go.Layout(
            title="GDELT Open Intelligence Subgraph",
            title_x=0.5,
            showlegend=False,
            hovermode="closest",
            paper_bgcolor="white",
            plot_bgcolor="white",
            margin=dict(b=20, l=5, r=5, t=40),
            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        ),
    )

    return fig


# Regex pattern to capture identifiers in the form "class/key"
pattern = r"\b([A-Za-z]+/[\w\/\-]+)\b"


def parse_node_ids(message: str) -> List[str]:
    """Parse node IDs from a string."""
    matches = re.findall(pattern, message)
    return matches


if __name__ == "__main__":
    msg = """The five most recent events that the actor has participated in are:

"Nyarimirina, RCD soldiers atta" - On August 24, 2005 (Event ID: Event/DRC2834), RCD soldiers attacked peaceful peasants in Nyarimirina, burning several huts, resulting in 10 fatalities and several injuries.

"Both Mayi Mayi and RCD-Goma at" - On January 2, 2005 (Event ID: Event/DRC2673), Mayi Mayi and RCD-Goma attacked civilians in Kanyabayonga and other areas in Nord Kivu.

"Exchange of fire after cattle-" - On June 12, 2004 (Event ID: Event/DRC2550), there was an exchange of fire following a cattle-rustling incident.

"On December 30-31, the MLC, RC" - On December 30-31, 2002 (Event ID: Event/DRC2142), peace deals were independently signed by the MLC, RCD-N, and RCD-Goma with other rebel factions.

"RCD-ML says RCD-N attacks thei" - On December 20, 2002 (Event ID: Event/DRC2127), RCD-ML claimed that RCD-N attacked their positions in Ituri.

These events, identified by their unique IDs, signify significant activities involving this actor."""

    target_ids = parse_node_ids("hello world")
    print(target_ids)

    fig = show_graph(target_ids=[])
    fig.show()
