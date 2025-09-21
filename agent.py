from typing import TypedDict, List
from langchain_core.tools import tool
from langchain_milvus import Milvus
from langchain_groq import ChatGroq
from langgraph.graph import StateGraph, END
from pydantic import BaseModel
from langgraph.checkpoint.memory import MemorySaver
import config
from data_handler import FastEmbedEmbeddings
from prompts import PLANNER_PROMPT, RESEARCHER_PROMPT, DRAFT_PROMPT, REVISER_PROMPT
import re
from langgraph.prebuilt import create_react_agent


# --- 1. Define Agent State ---
class AgentState(TypedDict):
    task: str
    plan: List[str]
    research_summary: str
    draft: str
    revised_draft: str
    # This flag tells the graph whether to proceed after planning.
    execute_research: bool


# --- 2. Define Tools ---
llm = ChatGroq(model="llama-3.1-8b-instant", temperature=0)
embedding_model = FastEmbedEmbeddings(model_name=config.EMBEDDING_MODEL)

# Search parameters - using COSINE for semantic similarity
search_params = {
    "metric_type": "COSINE",
    "params": {"nprobe": 10},
}

# Initialize vector store with proper connection
vector_store = Milvus(
    embedding_function=embedding_model,
    collection_name=config.COLLECTION_NAME,
    connection_args={"host": config.MILVUS_HOST, "port": config.MILVUS_PORT},
    # search_params=search_params,
    drop_old=False,  # Changed to False to keep existing data
)


@tool
def vector_database_search(query: str) -> str:
    """
    Searches the local document knowledge base to find relevant information for a given query.
    """
    print(f"--- Performing vector search for query: '{query}' ---")
    try:
        # Use similarity_search method with proper parameters
        retrieved_docs = vector_store.similarity_search(
            query,
            k=5,
        )

        if not retrieved_docs:
            print("No documents retrieved from vector search")
            return "No information found."

        print(f"Retrieved {len(retrieved_docs)} documents")

        # Format the context properly
        context_parts = []
        for doc in retrieved_docs:
            source = doc.metadata.get("source", "N/A")
            page = doc.metadata.get("page", "N/A")
            content = doc.page_content
            formatted_doc = f"[Source: {source}, page: {page}]\n{content}"
            context_parts.append(formatted_doc)

        context = "\n\n---\n\n".join(context_parts)
        print(f"Context length: {len(context)} characters")
        return context

    except Exception as e:
        print(f"Error in vector search: {e}")
        return f"Search error: {str(e)}"


# --- 3. Define Graph Nodes ---
class Plan(BaseModel):
    """The research plan."""

    items: List[str]


def planner_node(state: AgentState):
    print("--- 📝 PLANNER ---")
    prompt = PLANNER_PROMPT.format(task=state["task"])
    response = llm.invoke(prompt)
    plan_text = response.content

    # Better plan parsing
    plan_items = []
    lines = plan_text.strip().split("\n")
    for line in lines:
        line = line.strip()
        # Match numbered items (1., 2., etc.)
        if re.match(r"^\d+\.", line):
            # Remove the number and period, keep the rest
            item = re.sub(r"^\d+\.\s*", "", line).strip()
            if item:
                plan_items.append(item)

    # Fallback parsing if the above doesn't work
    if not plan_items:
        plan_items = [
            item.strip() for item in re.split(r"\d+\.\s*", plan_text) if item.strip()
        ]

    print(f"--- Parsed Plan: {plan_items} ---")
    return {"plan": plan_items}


def researcher_node(state: AgentState):
    """
    Executes research by calling the vector database search tool for each plan item.
    """
    print("--- 📚 RESEARCHER ---")

    research_results = []

    for i, plan_item in enumerate(state["plan"], 1):
        print(f"--- Researching step {i}: {plan_item} ---")
        try:
            # Directly call the vector database search tool
            result = vector_database_search.invoke({"query": plan_item})
            research_results.append(f"**Research for: {plan_item}**\n\n{result}")
        except Exception as e:
            print(f"Error in research step {i}: {e}")
            research_results.append(f"**Research for: {plan_item}**\n\nError: {str(e)}")

    # Combine all research results
    research_summary = "\n\n" + "=" * 50 + "\n\n".join(research_results)

    print(f"--- Research completed with {len(research_results)} results ---")
    print(research_results)
    return {"research_summary": research_summary}


def draft_writer_node(state: AgentState):
    print("--- ✍️ DRAFT WRITER ---")
    prompt = DRAFT_PROMPT.format(
        task=state["task"], research_summary=state["research_summary"]
    )
    draft = llm.invoke(prompt)
    return {"draft": draft.content}


def reviser_node(state: AgentState):
    print("--- ✨ REVISER ---")
    prompt = REVISER_PROMPT.format(task=state["task"], draft=state["draft"])
    revised_draft = llm.invoke(prompt)
    return {"revised_draft": revised_draft.content}


# --- 4. Define the Conditional Edge ---
def should_continue(state: AgentState):
    if state.get("execute_research"):
        return "continue"
    else:
        return "pause"


# --- 5. Build and Export the Graph ---
graph_builder = StateGraph(AgentState)
graph_builder.add_node("planner", planner_node)
graph_builder.add_node("researcher", researcher_node)
graph_builder.add_node("draft_writer", draft_writer_node)
graph_builder.add_node("reviser", reviser_node)

graph_builder.set_entry_point("planner")

# After planning, check the condition
graph_builder.add_conditional_edges(
    "planner",
    should_continue,
    {
        "continue": "researcher",
        "pause": END,
    },
)
graph_builder.add_edge("researcher", "draft_writer")
graph_builder.add_edge("draft_writer", "reviser")
graph_builder.add_edge("reviser", END)

# Add a checkpointer to manage state between calls
checkpointer = MemorySaver()
research_agent = graph_builder.compile(checkpointer=checkpointer)
