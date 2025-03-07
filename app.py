import streamlit as st
import time
import random
import plotly.graph_objects as go
import networkx as nx

# ------------------------------ Utility Functions ------------------------------

def show_networkx_graph():
    """Generates and displays a random NetworkX graph using Plotly."""
    G = nx.random_geometric_graph(200, 0.125)
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
    node_adjacencies = []
    node_text = []
    for node, adjacencies in G.adjacency():
        node_adjacencies.append(len(adjacencies))
        node_text.append('# of connections: ' + str(len(adjacencies)))
    node_trace.marker.color = node_adjacencies
    node_trace.text = node_text
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
    Runs a dummy pipeline with gradual log updates.
    
    For any input except "xyz", all three steps run immediately.
    
    If the input is exactly "xyz" and we are not already awaiting approval,
    the pipeline runs Step 1 and Step 2, then prints an approval prompt and
    sets a flag to await a textual approval response.
    
    When an approval response arrives (detected via st.session_state.awaiting_approval),
    the pipeline processes that response:
      - "Y" (yes): runs Step 3 and returns the final result.
      - "N" (regenerate): simulates a regeneration branch.
      - Any other input: cancels the operation.
    
    Returns the final result (or None if waiting for approval).
    """
    with pipeline_placeholder.container():
        st.subheader("Pipeline Processing")
        
        # Normal processing for inputs other than "xyz"
        if user_input.strip().lower() != "xyz" and not st.session_state.get("awaiting_approval", False):
            st.markdown("**Step 1: Preprocessing started...**")
            time.sleep(1)
            intermediate_result = f"Preprocessed: {user_input.lower()}"
            st.markdown(f"**Step 1: Preprocessing result:** {intermediate_result}")
            time.sleep(1)
            st.markdown("**Step 2: Model Inference started...**")
            time.sleep(1)
            intermediate_result = f"Inferred: {intermediate_result[::-1]}"
            st.markdown(f"**Step 2: Model Inference result:** {intermediate_result}")
            time.sleep(1)
            st.markdown("**Step 3: Formatting started...**")
            time.sleep(1)
            final_result = f"Formatted: {intermediate_result.capitalize()}"
            st.markdown(f"**Step 3: Formatting result:** {final_result}")
            st.markdown("**Pipeline complete.**")
            show_networkx_graph()
            return final_result
        
        # For input "xyz": if we're not already awaiting approval, run Step 1 and Step 2, then prompt for approval.
        if user_input.strip().lower() == "xyz" and not st.session_state.get("awaiting_approval", False):
            st.markdown("**Step 1: Preprocessing started...**")
            time.sleep(1)
            intermediate_result = f"Preprocessed: {user_input.lower()}"
            st.markdown(f"**Step 1: Preprocessing result:** {intermediate_result}")
            time.sleep(1)
            st.markdown("**Step 2: Model Inference started...**")
            time.sleep(1)
            intermediate_result = f"Inferred: {intermediate_result[::-1]}"
            st.markdown(f"**Step 2: Model Inference result:** {intermediate_result}")
            time.sleep(1)
            st.markdown("**Approval required: Type 'Y' to approve. Type 'N' to regenerate. Any other input will cancel the operation.**")
            st.session_state.awaiting_approval = True
            st.session_state.intermediate_result = intermediate_result
            return None
        
        # If we are awaiting approval, process the approval response.
        if st.session_state.get("awaiting_approval", False):
            approval = user_input.strip().upper()
            intermediate_result = st.session_state.get("intermediate_result", "")
            if approval == "Y":
                st.markdown("**User approved additional processing.**")
                time.sleep(1)
                st.markdown("**Step 3: Formatting started...**")
                time.sleep(1)
                final_result = f"Formatted: {intermediate_result.capitalize()}"
                st.markdown(f"**Step 3: Formatting result:** {final_result}")
            elif approval == "N":
                st.markdown("**User requested regeneration.**")
                time.sleep(1)
                # Simulate regeneration by re-running a similar branch (here we simply alter the output)
                final_result = f"Regenerated: {intermediate_result.capitalize()}"
                st.markdown(f"**Regenerated result:** {final_result}")
            else:
                st.markdown("**Operation cancelled.**")
                final_result = "Operation cancelled"
            st.markdown("**Pipeline complete.**")
            show_networkx_graph()
            # Clear approval flags for future interactions.
            del st.session_state.awaiting_approval
            del st.session_state.intermediate_result
            return final_result

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
left_col, right_col = st.columns([1, 1])
if "messages" not in st.session_state:
    st.session_state["messages"] = [{"role": "assistant", "content": "How can I help you?"}]
if "has_user_interacted" not in st.session_state:
    st.session_state["has_user_interacted"] = False
if "message_added" not in st.session_state:
    st.session_state["message_added"] = False

# ------------------------------ Layout ------------------------------

with left_col:
    chat_placeholder = st.empty()
with right_col:
    pipeline_placeholder = st.empty()

display_chat(st.session_state["messages"], chat_placeholder)
if not st.session_state["has_user_interacted"]:
    generate_pipeline_initial_view(pipeline_placeholder)

# ------------------------------ Chat Input & Processing ------------------------------

user_input = st.chat_input("Ask a question:")

if user_input:
    st.session_state["has_user_interacted"] = True
    # For the very first message or an approval response, always add it immediately.
    if not st.session_state.get("message_added", False):
        st.session_state["messages"].append({"role": "user", "content": user_input})
        st.session_state["message_added"] = True
    display_chat(st.session_state["messages"], chat_placeholder)
    st.session_state["current_input"] = user_input

# Run pipeline if we have a current input.
if "current_input" in st.session_state:
    final_result = run_pipeline(st.session_state["current_input"], pipeline_placeholder)
    # Only update chat when pipeline processing is complete.
    if final_result is not None:
        # For the "xyz" branch, final answer depends on the approval response.
        if st.session_state["current_input"].strip().lower() == "xyz":
            final_response = f"Final response: {final_result}"
        else:
            random_prefix = random.choice(["Final response:", "Result:", "Here's what I got:"])
            final_response = f"{random_prefix} {final_result}"
        st.session_state["messages"].append({"role": "assistant", "content": final_response})
        display_chat(st.session_state["messages"], chat_placeholder)
        del st.session_state["current_input"]
        st.session_state["message_added"] = False