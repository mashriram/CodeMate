import gradio as gr
import markdown
from weasyprint import HTML
import os
import uuid

from agent import research_agent
import config
from data_handler import process_and_embed_pdfs


# --- Helper Functions (No changes) ---
def generate_exports(markdown_report: str):
    md_path = "research_report.md"
    with open(md_path, "w", encoding="utf-8") as f:
        f.write(markdown_report)
    html_report = markdown.markdown(markdown_report)
    pdf_path = "research_report.pdf"
    HTML(string=html_report).write_pdf(pdf_path)
    return md_path, pdf_path


# --- File Upload Handler (No changes) ---
def handle_file_upload(files):
    if not files:
        return "No files uploaded. Please select one or more PDF files."

    file_paths = []
    for file in files:
        # This logic correctly handles Gradio's temporary file objects
        content = ""
        with open(file.name, "rb") as f:
            content = f.read()

        filepath = os.path.join(config.DATA_DIRECTORY, os.path.basename(file.name))
        with open(filepath, "wb") as f:
            f.write(content)
        file_paths.append(filepath)

    try:
        docs_processed, chunks_ingested = process_and_embed_pdfs(file_paths)
        return f"✅ Successfully processed {docs_processed} file(s) and ingested {chunks_ingested} new chunks into the knowledge base."
    except Exception as e:
        return f"❌ Error during file processing: {e}"


# --- Multi-Stage Agent Logic (CORRECTED) ---
def run_planning_phase(query: str, chat_history: list):
    chat_history.append({"role": "user", "content": query})
    yield (
        chat_history,
        "*Generating research plan...*",
        None,
        None,
        gr.update(interactive=False),
        gr.update(visible=False),
    )

    try:
        thread_id = str(uuid.uuid4())
        config = {"configurable": {"thread_id": thread_id}}

        # Invoke the agent to run the planning step.
        research_agent.invoke({"task": query, "execute_research": False}, config=config)

        # THE FIX: Explicitly get the final state for the thread.
        final_state = research_agent.get_state(config)
        plan = final_state.values["plan"]

        plan_markdown = "### Research Plan\n" + "\n".join(f"1. {step}" for step in plan)

        yield (
            chat_history,
            plan_markdown,
            thread_id,
            plan,
            gr.update(interactive=False),
            gr.update(visible=True, interactive=True),
        )
    except Exception as e:
        yield (
            chat_history,
            f"Error during planning: {e}",
            None,
            None,
            gr.update(interactive=True),
            gr.update(visible=False),
        )


def run_execution_phase(thread_id: str, plan: list, chat_history: list):
    yield (
        chat_history,
        "Executing plan... (Researching, Drafting, Revising)",
        gr.update(visible=False),
        gr.update(visible=False),
        gr.update(interactive=False),
        gr.update(visible=False),
    )

    try:
        config = {"configurable": {"thread_id": thread_id}}

        # Invoke the agent to resume and run the execution steps.
        research_agent.invoke({"execute_research": True}, config=config)

        # THE FIX: Explicitly get the final state for the thread.
        final_state = research_agent.get_state(config)
        final_report = final_state.values["revised_draft"]

        chat_history.append({"role": "assistant", "content": final_report})
        md_path, pdf_path = generate_exports(final_report)

        yield (
            chat_history,
            "Execution complete. Final report below.",
            gr.update(value=md_path, visible=True),
            gr.update(value=pdf_path, visible=True),
            gr.update(interactive=True),
            gr.update(visible=False),
        )
    except Exception as e:
        error_message = f"Error during execution: {e}"
        chat_history.append({"role": "assistant", "content": error_message})
        yield (
            chat_history,
            error_message,
            gr.update(visible=False),
            gr.update(visible=False),
            gr.update(interactive=True),
            gr.update(visible=False),
        )
        raise e


# --- Gradio UI Definition (No changes) ---
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
                plan_button = gr.Button("Plan Research", variant="primary")
                execute_button = gr.Button(
                    "Execute Plan", variant="primary", visible=False, interactive=False
                )
        with gr.Column(scale=1):
            gr.Markdown("### Add to Knowledge Base")
            upload_button = gr.UploadButton(
                "Upload PDFs", file_types=[".pdf"], file_count="multiple"
            )
            upload_status = gr.Markdown("*Upload status...*")
            gr.Markdown("### Agent Plan & Reasoning")
            reasoning_display = gr.Markdown("*Agent is idle...*")
            gr.Markdown("### Export Results")
            download_md = gr.File(label="Download Markdown", visible=False)
            download_pdf = gr.File(label="Download PDF", visible=False)

    plan_button.click(
        fn=run_planning_phase,
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
        fn=run_execution_phase,
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
