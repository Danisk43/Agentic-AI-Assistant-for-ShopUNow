
import os
import streamlit as st
from dotenv import load_dotenv

from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_core.documents import Document
from langchain_community.vectorstores import Chroma
from langgraph.graph import StateGraph, END

from pydantic import BaseModel
from typing import Literal

# -------------------------------
# ENV SETUP
# -------------------------------
load_dotenv()
assert os.getenv("OPENAI_API_KEY"), "OPENAI_API_KEY not found"

# -------------------------------
# STREAMLIT CONFIG
# -------------------------------
st.set_page_config(
    page_title="ShopUNow Agentic AI Assistant",
    page_icon="🛍️",
    layout="centered"
)

st.title("🛍️ ShopUNow Agentic AI Assistant")
st.caption("LangGraph + ChromaDB + Conversational Memory")

# -------------------------------
# MODELS
# -------------------------------
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
embeddings = OpenAIEmbeddings()

# -------------------------------
# LOAD VECTOR DB
# -------------------------------
vectordb = Chroma(
    persist_directory="chroma_db",
    embedding_function=embeddings
)

# -------------------------------
# MEMORY (Session-based)
# -------------------------------
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# -------------------------------
# CONSTANTS
# -------------------------------
VALID_DEPARTMENTS = {"HR", "IT_SUPPORT", "BILLING", "SHIPPING"}

# -------------------------------
# STRUCTURED OUTPUT SCHEMAS
# -------------------------------
class SentimentOutput(BaseModel):
    sentiment: Literal["negative", "neutral"]


class DepartmentOutput(BaseModel):
    department: Literal["HR", "IT_SUPPORT", "BILLING", "SHIPPING", "UNKNOWN"]

# -------------------------------
# AGENT NODES
# -------------------------------
def sentiment_node(state):
    sentiment_llm = llm.with_structured_output(SentimentOutput)

    result = sentiment_llm.invoke(
        f"Classify sentiment (negative or neutral): {state['query']}"
    )

    state["sentiment"] = result.sentiment
    return state


def classify_node(state):
    dept_llm = llm.with_structured_output(DepartmentOutput)

    result = dept_llm.invoke(
        f"""
Choose ONE department strictly from:
HR, IT_SUPPORT, BILLING, SHIPPING, UNKNOWN

User query:
{state['query']}
"""
    )

    state["department"] = result.department
    return state


def rag_node(state):
    docs = vectordb.similarity_search(
        state["query"],
        k=3,
        filter={"department": state["department"]}
    )

    context = "\n".join(d.page_content for d in docs)

    # ---- Conversational Memory ----
    memory_context = "\n".join(
        [f"User: {u}\nAssistant: {a}"
         for u, a in st.session_state.chat_history[-3:]]
    )

    prompt = f"""
You are a helpful assistant for ShopUNow.

Conversation history:
{memory_context}

Context:
{context}

User question:
{state['query']}
"""

    state["response"] = llm.invoke(prompt).content
    return state


def human_node(state):
    state["response"] = (
        "This issue requires human assistance. "
        "A support agent will contact you shortly."
    )
    return state

# -------------------------------
# LANGGRAPH
# -------------------------------
graph = StateGraph(dict)

graph.add_node("sentiment", sentiment_node)
graph.add_node("classify", classify_node)
graph.add_node("rag", rag_node)
graph.add_node("human", human_node)

graph.set_entry_point("sentiment")
graph.add_edge("sentiment", "classify")

def route(state):
    if state["sentiment"] == "negative" or state["department"] not in VALID_DEPARTMENTS:
        return "human"
    return "rag"

graph.add_conditional_edges(
    "classify",
    route,
    {"rag": "rag", "human": "human"}
)

graph.add_edge("rag", END)
graph.add_edge("human", END)

agent = graph.compile()

# -------------------------------
# CHAT UI
# -------------------------------
for user, bot in st.session_state.chat_history:
    with st.chat_message("user"):
        st.write(user)
    with st.chat_message("assistant"):
        st.write(bot)

query = st.chat_input("Ask a question...")

if query:
    with st.chat_message("user"):
        st.write(query)

    result = agent.invoke({"query": query})
    response = result["response"]

    with st.chat_message("assistant"):
        st.write(response)

    st.session_state.chat_history.append((query, response))
