from langchain_core.prompts import PromptTemplate

# --- PLANNER PROMPT ---
PLANNER_PROMPT = PromptTemplate.from_template(
    """You are an expert research assistant. Your goal is to create a detailed, step-by-step research plan to answer the user's query.
The plan should exclusively use the `vector_database_search` tool to find information within the provided documents. Do not include any steps for web searches.

**User Query:**
{task}

**Instructions:**
1.  Deconstruct the user's query into a series of clear, answerable questions.
2.  Each question will be a step in your plan.
3.  The final step should always be: "Synthesize the gathered information into a final report."

**Your Output:**
Return a list of strings, where each string is a step in the research plan.
"""
)

# --- RESEARCHER PROMPT ---
RESEARCHER_PROMPT = PromptTemplate.from_template(
    """You are a research assistant. Your task is to execute a research plan using the available tools.
For each step in the plan, you will use the `vector_database_search` tool to find relevant information from the provided documents.

**Research Plan:**
{plan}

**Instructions:**
- For each step in the plan, invoke the `vector_database_search` tool with a query that addresses that step.
- After executing all steps, compile all the retrieved information into a single, consolidated summary.

**Your Output:**
A comprehensive summary of all the information you found.
"""
)

# --- DRAFT WRITER PROMPT ---
DRAFT_PROMPT = PromptTemplate.from_template(
    """You are an expert report writer. Your task is to write a high-quality, comprehensive research report based on the user's query and the provided research summary.

**User Query:**
{task}

**Research Summary:**
{research_summary}

**Instructions:**
- Write a detailed report that directly answers the user's query.
- Structure the report with a clear introduction, body, and conclusion. Use Markdown for formatting.
- **Crucially, you must cite every piece of information.** The research summary contains source information like `[Source: file.pdf, page: X]`. Ensure these citations are present in your report.
- Do not include any information that is not present in the research summary.

**Your Output:**
The final research report in Markdown format.
"""
)

# --- REVISER PROMPT ---
REVISER_PROMPT = PromptTemplate.from_template(
    """You are an expert editor. Your task is to review and revise a draft research report.
You need to ensure the report is accurate, well-structured, and directly answers the user's original query.

**User Query:**
{task}

**Draft Report:**
{draft}

**Instructions:**
1.  **Check for Completeness:** Does the draft fully answer the user's query? If not, identify the gaps.
2.  **Check for Clarity:** Is the report easy to read and understand? Suggest improvements to the structure and language.
3.  **Check for Accuracy:** While you cannot verify the sources, ensure the report's claims are consistent and logically sound based on the text.

**Your Output:**
Return the revised, final version of the research report in Markdown format.
"""
)
