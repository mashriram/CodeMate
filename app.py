import gradio as gr
import markdown
from weasyprint import HTML
import os
import uuid
from langchain_core.messages import HumanMessage

from agent import research_agent
import config
from data_handler import process_and_embed_pdfs


# --- Helper & File Upload Functions (No changes) ---
def generate_exports(markdown_report: str):
    md_path = "research_report.md"
    with open(md_path, "w", encoding="utf-8") as f:
        f.write(markdown_report)
    html_report = markdown.markdown(markdown_report)
    pdf_path = "research_report.pdf"
    HTML(string=html_report).write_pdf(pdf_path)
    return md_path, pdf_path


def handle_file_upload(files):
    if not files:
        return "No files uploaded."
    file_paths = [file.name for file in files]
    try:
        docs_processed, chunks_ingested = process_and_embed_pdfs(file_paths)
        return f"✅ Successfully processed {docs_processed} file(s) and ingested {chunks_ingested} new chunks."
    except Exception as e:
        return f"❌ Error during file processing: {e}"


# --- Agent Interaction Logic (REBUILT FOR STABILITY) ---
def start_new_research(query: str, chat_history: list):
    """PHASE 1: Plan the research."""
    chat_history.append({"role": "user", "content": query})
    thread_id = str(uuid.uuid4())
    config = {"configurable": {"thread_id": thread_id}}

    # Update UI immediately to show the agent is working
    yield (
        chat_history,
        "*Generating research plan...*",
        thread_id,
        None,
        gr.update(interactive=False),
        gr.update(visible=False),
    )

    # Run the planning phase
    result = research_agent.invoke(
        {"task": query, "execute_research": False}, config=config
    )
    plan = result["plan"]
    plan_markdown = "### Research Plan\n" + "\n".join(f"1. {step}" for step in plan)

    # Final update for this phase
    yield (
        chat_history,
        plan_markdown,
        thread_id,
        plan,
        gr.update(interactive=False),
        gr.update(visible=True, interactive=True),
    )


def execute_research(thread_id: str, plan: list, chat_history: list):
    """PHASE 2: Execute the plan and generate the report."""
    config = {"configurable": {"thread_id": thread_id}}

    # Update UI immediately
    yield (
        chat_history,
        "Executing plan... This may take a moment.",
        gr.update(visible=False),
        gr.update(visible=False),
        gr.update(interactive=False),
        gr.update(visible=False, interactive=False),
    )

    # Run the execution phase
    final_state = research_agent.invoke({"execute_research": True}, config=config)

    final_report = final_state["revised_draft"]
    reasoning_log = "\n".join(f"- {step}" for step in final_state["reasoning_log"])
    chat_history.append({"role": "assistant", "content": final_report})
    md_path, pdf_path = generate_exports(final_report)

    # Final update for the entire process
    yield (
        chat_history,
        reasoning_log,
        gr.update(value=md_path, visible=True),
        gr.update(value=pdf_path, visible=True),
        gr.update(interactive=True),  # Re-enable start button
        gr.update(visible=False),
    )


# --- Gradio UI Definition ---
with gr.Blocks(theme=gr.themes.Soft(), title=config.APP_TITLE) as demo:
    plan_state = gr.State()
    thread_id_state = gr.State()

    gr.Markdown(f"# {config.APP_TITLE}")
    with gr.Row():
        with gr.Column(scale=2):
            chatbot = gr.Chatbot(
                label="Research Assistant", height=600, type="messages"
            )
            query_box = gr.Textbox(label="Enter your research query:", container=False)
            with gr.Row():
                plan_button = gr.Button("Start Research", variant="primary")
                execute_button = gr.Button(
                    "Execute Plan", variant="primary", visible=False, interactive=False
                )

        with gr.Column(scale=1):
            gr.Markdown("### Add to Knowledge Base")
            upload_button = gr.UploadButton(
                "Upload PDFs", file_types=[".pdf"], file_count="multiple"
            )
            upload_status = gr.Markdown("*Upload status...*")
            gr.Markdown("### Agent Reasoning Steps")
            reasoning_display = gr.Markdown("*Agent is idle...*")
            gr.Markdown("### Export Results")
            download_md = gr.File(label="Download Markdown", visible=False)
            download_pdf = gr.File(label="Download PDF", visible=False)

    plan_button.click(
        fn=start_new_research,
        inputs=[query_box, chatbot],
        outputs=[
            chatbot,
            reasoning_display,
            thread_id_state,
            plan_state,
            plan_button,
            execute_button,
        ],
    )

    execute_button.click(
        fn=execute_research,
        inputs=[thread_id_state, plan_state, chatbot],
        outputs=[
            chatbot,
            reasoning_display,
            download_md,
            download_pdf,
            plan_button,
            execute_button,
        ],
    )

    upload_button.upload(
        fn=handle_file_upload, inputs=[upload_button], outputs=[upload_status]
    )

if __name__ == "__main__":
    demo.launch()
