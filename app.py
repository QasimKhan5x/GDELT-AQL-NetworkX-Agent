import streamlit as st
import time
import random
import plotly.graph_objects as go
import networkx as nx

# ------------------------------ Utility Functions ------------------------------

def show_networkx_graph():
    """Generates and displays a random NetworkX graph using Plotly."""
    G = nx.random_geometric_graph(200, 0.125)

    # Extract edge positions
    edge_x, edge_y = [], []
    for edge in G.edges():
        x0, y0 = G.nodes[edge[0]]['pos']
        x1, y1 = G.nodes[edge[1]]['pos']
        edge_x.extend([x0, x1, None])
        edge_y.extend([y0, y1, None])

    edge_trace = go.Scatter(
        x=edge_x, y=edge_y,
        line=dict(width=0.5, color='#888'),
        hoverinfo='none',
        mode='lines'
    )

    # Extract node positions
    node_x, node_y = [], []
    for node in G.nodes():
        x, y = G.nodes[node]['pos']
        node_x.append(x)
        node_y.append(y)

    node_trace = go.Scatter(
        x=node_x, y=node_y,
        mode='markers',
        hoverinfo='text',
        marker=dict(
            showscale=True,
            colorscale='YlGnBu',
            reversescale=True,
            color=[],
            size=10,
            colorbar=dict(
                thickness=15,
                title=dict(text='Node Connections', side='right'),
                xanchor='left',
            ),
            line_width=2
        )
    )

    # Color nodes based on degree
    node_adjacencies = []
    node_text = []
    for node, adjacencies in G.adjacency():
        node_adjacencies.append(len(adjacencies))
        node_text.append('# of connections: ' + str(len(adjacencies)))
    node_trace.marker.color = node_adjacencies
    node_trace.text = node_text

    # Build figure
    fig = go.Figure(
        data=[edge_trace, node_trace],
        layout=go.Layout(
            title='Random Geometric Graph',
            title_x=0.5,
            showlegend=False,
            hovermode='closest',
            margin=dict(b=20, l=5, r=5, t=40),
            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False)
        )
    )

    st.plotly_chart(fig, use_container_width=True)

def display_chat(messages, placeholder):
    """Displays all chat messages within the given placeholder."""
    with placeholder.container():
        for msg in messages:
            st.chat_message(msg["role"]).write(msg["content"])

def run_pipeline(user_input, pipeline_placeholder):
    """
    Runs a dummy pipeline with three processing steps.
    Updates the pipeline_placeholder with logs and a new network graph.
    Returns the final intermediate result.
    """
    # Dummy processing functions
    def step1(text):
        time.sleep(1)
        return f"Preprocessed: {text.lower()}"

    def step2(text):
        time.sleep(1)
        return f"Inferred: {text[::-1]}"

    def step3(text):
        time.sleep(1)
        return f"Formatted: {text.capitalize()}"

    steps = [
        ("Step 1: Preprocessing", step1),
        ("Step 2: Model Inference", step2),
        ("Step 3: Formatting", step3)
    ]

    with pipeline_placeholder.container():
        st.subheader("Pipeline Processing")
        intermediate_result = user_input

        # Execute each pipeline step once, printing status and result
        for step_name, func in steps:
            st.markdown(f"**{step_name} started...**")
            intermediate_result = func(intermediate_result)
            st.markdown(f"**{step_name} result:** {intermediate_result}")

        st.markdown("**Pipeline complete.**")
        # Show a new random NetworkX graph
        show_networkx_graph()

    return intermediate_result

def generate_pipeline_initial_view(pipeline_placeholder):
    """
    Displays the initial view for the pipeline processing area,
    showing only a header and a random network graph.
    """
    with pipeline_placeholder.container():
        st.subheader("Pipeline Processing")
        show_networkx_graph()

# ------------------------------ Page Configuration & State ------------------------------

st.set_page_config(layout="wide")

# Create two equal-width columns (50% each)
left_col, right_col = st.columns([1, 1])

# Initialize session state for chat messages and user interaction flag
if "messages" not in st.session_state:
    st.session_state["messages"] = [{"role": "assistant", "content": "How can I help you?"}]
if "has_user_interacted" not in st.session_state:
    st.session_state["has_user_interacted"] = False

# ------------------------------ Layout ------------------------------

# Placeholders for chat and pipeline processing
with left_col:
    chat_placeholder = st.empty()
with right_col:
    pipeline_placeholder = st.empty()

# Display chat messages in the left column
display_chat(st.session_state["messages"], chat_placeholder)

# Always show the pipeline processing section.
# On first load, this displays a random network graph with no debug text.
if not st.session_state["has_user_interacted"]:
    generate_pipeline_initial_view(pipeline_placeholder)

# ------------------------------ Chat Input & Processing ------------------------------

user_input = st.chat_input("Ask a question:")

if user_input:
    st.session_state["has_user_interacted"] = True

    # Append and display user message
    st.session_state["messages"].append({"role": "user", "content": user_input})
    display_chat(st.session_state["messages"], chat_placeholder)

    # Clear and run pipeline processing with debug logs
    pipeline_placeholder.empty()
    final_result = run_pipeline(user_input, pipeline_placeholder)

    # Generate a random final response and update chat
    random_prefix = random.choice(["Final response:", "Result:", "Here's what I got:"])
    final_response = f"{random_prefix} {final_result}"
    st.session_state["messages"].append({"role": "assistant", "content": final_response})
    display_chat(st.session_state["messages"], chat_placeholder)