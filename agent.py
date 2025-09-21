import re
from typing import TypedDict, List
from langchain_core.tools import tool
from langchain_milvus import Milvus
from langchain_groq import ChatGroq
from langgraph.graph import StateGraph, END
from pydantic import BaseModel
from langgraph.checkpoint.memory import MemorySaver

import config
from data_handler import FastEmbedEmbeddings
from prompts import PLANNER_PROMPT, DRAFT_PROMPT, REVISER_PROMPT


# --- 1. Define Agent State (No changes) ---
class AgentState(TypedDict):
    task: str
    plan: List[str]
    research_summary: str
    draft: str
    revised_draft: str
    execute_research: bool
    reasoning_log: List[str]


# --- 2. Define Tools (No changes) ---
llm = ChatGroq(model="llama-3.1-8b-instant", temperature=0)
embedding_model = FastEmbedEmbeddings(model_name=config.EMBEDDING_MODEL)
vector_store = Milvus(
    embedding_function=embedding_model,
    collection_name=config.COLLECTION_NAME,
    connection_args={"host": config.MILVUS_HOST, "port": config.MILVUS_PORT},
    drop_old=False,
)


@tool
def vector_database_search(query: str) -> str:
    """Searches the local document knowledge base to find relevant information."""
    print(f"--- Performing vector search for query: '{query}' ---")
    try:
        retrieved_docs = vector_store.similarity_search(query, k=3)
        if not retrieved_docs:
            return f"No information found for query: '{query}'"
        context_parts = [
            f"[Source: {doc.metadata.get('source', 'N/A')}, page: {doc.metadata.get('page', 'N/A')}]\n{doc.page_content}"
            for doc in retrieved_docs
        ]
        return "\n\n---\n\n".join(context_parts)
    except Exception as e:
        return f"Search error: {str(e)}"


# --- 3. Define Graph Nodes (CORRECTED) ---
def planner_node(state: AgentState):
    print("--- 📝 PLANNER ---")
    log = ["Generating a new research plan..."]
    prompt = PLANNER_PROMPT.format(task=state["task"])
    response = llm.invoke(prompt)
    plan_text = response.content
    plan_items = [
        item.strip()
        for item in re.split(r"^\d+\.\s*", plan_text, flags=re.MULTILINE)
        if item.strip()
    ]
    print(f"--- Parsed Plan: {plan_items} ---")
    log.append("Plan generated successfully.")
    return {"plan": plan_items, "reasoning_log": log}


def researcher_node(state: AgentState):
    print("--- 📚 RESEARCHER ---")
    log = state["reasoning_log"] + ["Executing research based on the plan..."]
    research_results = []
    research_results.append(f"**Original Query:** {state['task']}")
    for i, plan_item in enumerate(state["plan"], 1):
        log.append(f"  - Researching step {i}/{len(state['plan'])}: {plan_item}")
        try:
            result = vector_database_search.invoke({"query": plan_item})
            research_results.append(f"**Research for '{plan_item}':**\n{result}")
        except Exception as e:
            research_results.append(f"**Research for '{plan_item}':**\nError: {str(e)}")
    research_summary = "\n\n" + "=" * 50 + "\n\n".join(research_results)
    log.append("Research complete. All sources gathered.")
    return {"research_summary": research_summary, "reasoning_log": log}


def draft_writer_node(state: AgentState):
    print("--- ✍️ DRAFT WRITER ---")
    log = state["reasoning_log"] + ["Writing the first draft of the report..."]
    prompt = DRAFT_PROMPT.format(
        task=state["task"], research_summary=state["research_summary"]
    )

    # THE FIX: Extract the .content attribute from the AIMessage object
    response = llm.invoke(prompt)
    draft_content = response.content

    return {"draft": draft_content, "reasoning_log": log}


def reviser_node(state: AgentState):
    print("--- ✨ REVISER ---")
    log = state["reasoning_log"] + ["Revising and polishing the final report..."]
    prompt = REVISER_PROMPT.format(task=state["task"], draft=state["draft"])

    # THE FIX: Extract the .content attribute from the AIMessage object
    response = llm.invoke(prompt)
    revised_draft_content = response.content

    log.append("Report finalized.")
    return {"revised_draft": revised_draft_content, "reasoning_log": log}


# --- 4. Define the Conditional Edge (No changes) ---
def should_continue(state: AgentState):
    return "continue" if state.get("execute_research") else "pause"


# --- 5. Build and Export the Graph (No changes) ---
graph_builder = StateGraph(AgentState)
graph_builder.add_node("planner", planner_node)
graph_builder.add_node("researcher", researcher_node)
graph_builder.add_node("draft_writer", draft_writer_node)
graph_builder.add_node("reviser", reviser_node)
graph_builder.set_entry_point("planner")
graph_builder.add_conditional_edges(
    "planner", should_continue, {"continue": "researcher", "pause": END}
)
graph_builder.add_edge("researcher", "draft_writer")
graph_builder.add_edge("draft_writer", "reviser")
graph_builder.add_edge("reviser", END)

checkpointer = MemorySaver()
research_agent = graph_builder.compile(checkpointer=checkpointer)
