from langchain_core.tools import tool
from langchain_milvus import Milvus
from langchain_groq import ChatGroq
from deepagents import create_deep_agent

import config
from data_handler import FastEmbedEmbeddings  # Import the wrapper

# --- 1. Initialize LLM and Retriever ---
# This setup happens once and is used by our tool.
llm = ChatGroq(model="llama-3.1-8b-instant", temperature=0)
embedding_model = FastEmbedEmbeddings(model_name=config.EMBEDDING_MODEL)
vector_store = Milvus(
    embedding_function=embedding_model,
    collection_name=config.COLLECTION_NAME,
    connection_args={"host": config.MILVUS_HOST, "port": config.MILVUS_PORT},
)
retriever = vector_store.as_retriever(search_kwargs={"k": 7})


# --- 2. Define Custom Tools ---
# The agent will use this tool to query our local knowledge base.
@tool
def vector_database_search(query: str) -> str:
    """
    Searches the local vector database of uploaded documents to find relevant information.
    Use this to answer questions based on the provided knowledge base.
    """
    print(f"--- Performing vector search for query: '{query}' ---")
    retrieved_docs = retriever.invoke(query)

    # Format the results into a string for the LLM to easily understand.
    context = "\n\n".join(
        [
            f"Source: {doc.metadata.get('source', 'N/A')}, Page: {doc.metadata.get('page', 'N/A')}\nContent: {doc.page_content}"
            for doc in retrieved_docs
        ]
    )
    if not context:
        return "No relevant information found in the knowledge base for that query."
    return f"Retrieved the following information:\n\n{context}"


# --- 3. Define the Main Agent's Instructions ---
# This is the primary system prompt for the main agent.
main_agent_instructions = """You are an expert researcher and report writer. Your primary goal is to answer a user's query by conducting thorough research using the tools available.

**Your process must be:**
1.  **Plan:** Start by thinking step-by-step. Use the `write_todos` tool to outline your research plan.
2.  **Research:** Execute your plan by using the `vector_database_search` tool to query the knowledge base for each step of your plan.
3.  **Synthesize:** Once you have gathered all the necessary information, consolidate your findings into a scratchpad file using the `write_file` tool. Name this file `research_findings.md`.
4.  **Report:** Call the `report_synthesizer` sub-agent with the `research_findings.md` file as input. This sub-agent will write the final, polished report. Do NOT write the final report yourself.
"""

# --- 4. Define a Sub-Agent for Synthesis ---
# This sub-agent is specialized for one task: writing the final report.
synthesis_sub_agent = {
    "name": "report_synthesizer",
    "description": "Call this sub-agent ONLY ONCE at the very end to write the final, polished report from the research findings file.",
    "prompt": """You are an expert report writer. Your sole job is to take the provided research findings from the file and synthesize them into a coherent, well-structured, and comprehensive report in Markdown format.

- You MUST cite every piece of information using the sources provided in the context (e.g., `[Source: file.pdf, page: X]`).
- Structure the report with headings, bullet points, and clear paragraphs.
- Do not perform any new research. Your response should ONLY be the final report.
""",
}


# --- 5. Create the Deep Agent ---
def get_deep_research_agent():
    """
    Factory function to create the configured Deep Agent.
    """
    agent_executor = create_deep_agent(
        tools=[vector_database_search],  # Our custom tool
        instructions=main_agent_instructions,
        subagents=[synthesis_sub_agent],
        model=llm,
        # We only need the file system and planning tools for this agent.
        builtin_tools=["write_todos", "write_file", "read_file", "ls", "edit_file"],
    )
    return agent_executor
