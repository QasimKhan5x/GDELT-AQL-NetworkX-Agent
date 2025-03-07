import streamlit as st
import time
import re
import random
import plotly.graph_objects as go
import networkx as nx

from agent import app

from arangographqachain import ArangoGraphQAChain
from agent import llm, arango_graph, G_adb

chain = ArangoGraphQAChain.from_llm(
    llm=llm,
    graph=arango_graph,
    verbose=True,
    allow_dangerous_requests=True,
)
chain.execute_aql_query = False
config = {"configurable": {"thread_id": "1"}}

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

def run_agent(user_input, pipeline_placeholder):
    state = app.get_state(config).values
    history = state["history"] if state else []
    tools = []
    n = -1 # -1 to account for the initial planning step

    with pipeline_placeholder.container():
        st.markdown(f"**Planning how to answer the question...**")
        for s in app.stream({"task": user_input, "history": history}, config=config):
            if "planning" in s:
                plan = s["planning"]["plan"].plan
                for step in plan:
                    tools.append(step.tool)
                for step in plan:
                    print(step)
            elif "tool" in s:
                tool_name = tools[n]
                tool_code = s["tool"]["code_results"][n]
                if tool_name == "Text2AQL_Read":
                    st.markdown(f"**Step {n+ 1}: AQL Query**")
                elif tool_name == "Text2NetworkX":
                    st.markdown(f"**Step {n + 1}: NetworkX Script**")
                elif tool_name == "Text2ArangoSearch":
                    st.markdown(f"**Step {n + 1}: AQL with ArangoSearch Query**")
                st.markdown(f"```{tool_code}```")
            elif "generate" in s:
                answer = s["generate"]["answer"]
                return answer
            else:
                raise ValueError("Unexpected state")
            n += 1
            if n < len(tools):
                st.markdown(f"*Executing Step {n + 1}...*")
            if "generate" in s:
                st.markdown("*Generating the final answer...*")

def generate_pipeline_initial_view(pipeline_placeholder):
    """
    Displays the initial view for the pipeline processing area:
    a header and a random network graph.
    """
    with pipeline_placeholder.container():
        show_networkx_graph()

# ------------------------------ Page Configuration & State ------------------------------

st.set_page_config(layout="wide")
left_col, right_col = st.columns([1, 1])

# Initialize session state if not present.
if "messages" not in st.session_state:
    st.session_state["messages"] = [{"role": "assistant", "content": "How can I help you?"}]
if "has_user_interacted" not in st.session_state:
    st.session_state["has_user_interacted"] = False
# Flags for controlling approval flow.
if "current_input" not in st.session_state:
    st.session_state["current_input"] = ""
if "awaiting_approval" not in st.session_state:
    st.session_state["awaiting_approval"] = False
if "approval_response" not in st.session_state:
    st.session_state["approval_response"] = ""
# Store variables
if "aql" not in st.session_state:
    st.session_state["aql"] = ""

# ------------------------------ Layout ------------------------------

with left_col:
    chat_placeholder = st.empty()
with right_col:
    st.subheader("Pipeline Processing")  
    pipeline_placeholder = st.empty()

display_chat(st.session_state["messages"], chat_placeholder)
if not st.session_state["has_user_interacted"]:
    generate_pipeline_initial_view(pipeline_placeholder)

# ------------------------------ Chat Input & Processing ------------------------------

# Get user input from the chat box.
user_input = st.chat_input("Ask a question:")

if user_input:
    st.session_state["has_user_interacted"] = True
    # Always add the user's message immediately.
    st.session_state["messages"].append({"role": "user", "content": user_input})
    display_chat(st.session_state["messages"], chat_placeholder)
    
    # If we are already awaiting approval, treat this input as the approval response.
    if st.session_state["awaiting_approval"]:
        st.session_state["approval_response"] = user_input
    else:
        # Otherwise, store this as the new pipeline input.
        st.session_state["current_input"] = user_input

# ------------------------------ Run Pipeline & Control Flow ------------------------------

# If there is a current input and we're not awaiting approval, run the normal pipeline.
if st.session_state["current_input"] and not st.session_state["awaiting_approval"]:
    user_msg = st.session_state["current_input"]

    
    if re.search(r'\b(?:modify|update|change|insert|delete|remove|add)\b', user_msg, re.IGNORECASE):
        # For write operations, use the Text2AQL_Write tool.
        print("Performing Text2AQL_Write")
        tries = 0
        while tries < 3:
            try:
                result = chain.invoke(user_msg)
            except Exception as e:
                tries += 1
            else:
                st.session_state["awaiting_approval"] = True
                aql = result["result"]
                st.session_state["aql"] = aql
                with pipeline_placeholder.container():
                    st.markdown(f"```{aql}```")

                approval_prompt = f"""The following AQL query can modify data.
Type 'Y' to **approve** its execution.
Type 'N' to **cancel** the operation.
Any other input will also **cancel** the operation."""
                st.session_state["messages"].append(
                    {"role": "assistant", "content": approval_prompt}
                )
                display_chat(st.session_state["messages"], chat_placeholder)
                st.session_state["current_input"] = ""
                break
        else:
            error_msg = "The query to perform the operation could not be generated."
            st.session_state["messages"].append(
                {"role": "assistant", "content": error_msg}
            )
            display_chat(st.session_state["messages"], chat_placeholder)
    else:
        response = run_agent(st.session_state["current_input"], pipeline_placeholder)
        if response is not None:
            st.session_state["messages"].append({"role": "assistant", "content": response})
        else:
            st.session_state["messages"].append(
                {"role": "assistant", "content": "I'm sorry, I don't have an answer for that."}
            )
        display_chat(st.session_state["messages"], chat_placeholder)
        st.session_state["current_input"] = ""        

# If we are awaiting approval and an approval response has been provided, process it.
if st.session_state["awaiting_approval"] and st.session_state["approval_response"]:
    approval = st.session_state["approval_response"].strip().upper()
    if approval == "Y":
        G_adb.query(st.session_state["aql"])
        final_response = "The database has been modified successfully."
    else:
        final_response = "The operation was cancelled."
    st.session_state["messages"].append({"role": "assistant", "content": final_response})
    display_chat(st.session_state["messages"], chat_placeholder)
    # Reset approval flags.
    st.session_state["awaiting_approval"] = False
    st.session_state["approval_response"] = ""