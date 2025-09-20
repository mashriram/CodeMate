import gradio as gr
import markdown
from weasyprint import HTML
import os
import time

from agent import get_deep_research_agent
import config
from data_handler import process_and_embed_pdfs

# --- Agent Initialization ---
agent_executor = get_deep_research_agent()


# --- Helper Functions ---
def generate_exports(markdown_report: str):
    """Generates and saves Markdown and PDF files."""
    md_path = "research_report.md"
    with open(md_path, "w", encoding="utf-8") as f:
        f.write(markdown_report)
    html_report = markdown.markdown(markdown_report)
    pdf_path = "research_report.pdf"
    HTML(string=html_report).write_pdf(pdf_path)
    return md_path, pdf_path


def handle_file_upload(files):
    """Handles the upload and processing of new PDF files."""
    if not files:
        return "No files uploaded."
    file_paths = [file.name for file in files]
    try:
        docs_processed, chunks_ingested = process_and_embed_pdfs(file_paths)
        return f"✅ Successfully processed {docs_processed} file(s) and ingested {chunks_ingested} new chunks."
    except Exception as e:
        return f"❌ Error during file processing: {e}"


# --- Main Application Logic (Corrected) ---
def run_research_agent(query: str, chat_history: list):
    """
    Invokes the deep agent to run to completion and returns the final result.
    """
    # Let the user know the agent has started
    yield (
        chat_history,
        "*Agent is planning and researching...*",
        gr.update(visible=False),
        gr.update(visible=False),
    )

    # deepagents expects this specific input format
    inputs = {"messages": [{"role": "user", "content": query}]}

    # Use .invoke() to run the agent until it finishes
    final_state = agent_executor.invoke(inputs)

    # Extract the final report from the last AI message
    final_report = final_state["messages"][-1].content
    chat_history.append({"role": "assistant", "content": final_report})

    # Extract the reasoning from the agent's internal scratchpad file
    reasoning = final_state.get("files", {}).get(
        "research_findings.md", "Agent did not produce a findings file."
    )
    reasoning_text = f"**Agent's Research Notes:**\n\n---\n\n{reasoning}"

    # Generate export files from the final report
    md_path, pdf_path = generate_exports(final_report)

    # Yield the final, complete result to the UI
    yield (
        chat_history,
        reasoning_text,
        gr.update(value=md_path, visible=True),
        gr.update(value=pdf_path, visible=True),
    )


# --- Gradio UI Definition (Corrected) ---
with gr.Blocks(theme=gr.themes.Soft(), title=config.APP_TITLE) as demo:
    gr.Markdown(f"# {config.APP_TITLE}")

    with gr.Row():
        with gr.Column(scale=2):
            chatbot = gr.Chatbot(
                label="Research Assistant", height=600, type="messages"
            )
            query_box = gr.Textbox(
                label="Enter your research query:",
                container=False,
                placeholder="e.g., What are the key differences between LangGraph and CrewAI?",
            )
            submit_btn = gr.Button("Start Research", variant="primary")

        with gr.Column(scale=1):
            gr.Markdown("### Add to Knowledge Base")
            upload_button = gr.UploadButton(
                "Click to Upload PDFs", file_types=[".pdf"], file_count="multiple"
            )
            upload_status = gr.Markdown(value="*Upload status will appear here*")
            gr.Markdown("### Agent Reasoning")
            reasoning_display = gr.Markdown(value="*Agent is idle...*")
            gr.Markdown("### Export Results")
            download_md = gr.File(label="Download Markdown", visible=False)
            download_pdf = gr.File(label="Download PDF", visible=False)

    def on_submit(query, history):
        """
        This function ONLY updates the UI with the user's message.
        The actual agent call is handled in the .then() block.
        This prevents the duplicate message bug.
        """
        history.append({"role": "user", "content": query})
        return "", history  # Clear textbox, return updated history

    # The .then() event ensures that run_research_agent is called *after*
    # the UI has been updated with the user's message.
    submit_btn.click(
        fn=on_submit,
        inputs=[query_box, chatbot],
        outputs=[query_box, chatbot],
    ).then(
        fn=run_research_agent,
        inputs=[query_box, chatbot],
        outputs=[chatbot, reasoning_display, download_md, download_pdf],
    )

    upload_button.upload(
        fn=handle_file_upload, inputs=[upload_button], outputs=[upload_status]
    )

if __name__ == "__main__":
    demo.launch()
