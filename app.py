import os
import gradio as gr
import google.generativeai as genai
from datetime import datetime
import uuid
from pathlib import Path
from utils.auth import check_password, create_token, verify_token
from utils.document_processor import DocumentProcessor
from utils.db_handler import DatabaseHandler
from dotenv import load_dotenv

# Initialize components
load_dotenv()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
if not GEMINI_API_KEY:
    raise ValueError("GEMINI_API_KEY not found in environment variables")

genai.configure(api_key=GEMINI_API_KEY)
doc_processor = DocumentProcessor(GEMINI_API_KEY)
db_handler = DatabaseHandler()

# Configure Gemini model
generation_config = {
    "temperature": 0.7,
    "top_p": 1,
    "top_k": 1,
    "max_output_tokens": 2048,
}

safety_settings = [
    {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
    {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
    {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
    {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
]

model = genai.GenerativeModel(
    model_name="gemini-1.5-flash-8b",
    generation_config=generation_config,
    safety_settings=safety_settings
)

# Clean, minimalistic CSS
custom_css = """
:root {
    --primary-color: #2c5282;
    --background-color: #ffffff;
    --text-color: #2d3748;
    --border-color: #e2e8f0;
}

#app-container {
    max-width: 1000px;
    margin: 2rem auto;
    padding: 0 1rem;
}

.header {
    margin-bottom: 2rem;
    text-align: center;
    color: var(--text-color);
}

.header h1 {
    font-size: 2rem;
    font-weight: 600;
    margin-bottom: 0.5rem;
}

.header p {
    color: #4a5568;
    font-size: 1rem;
}

.chat-interface {
    border: 1px solid var(--border-color);
    border-radius: 8px;
    padding: 1.5rem;
    background: var(--background-color);
}

.message-bot {
    background: #f7fafc !important;
    padding: 1rem !important;
    border-radius: 8px 8px 8px 0 !important;
    border: 1px solid var(--border-color) !important;
    margin: 0.5rem 0 !important;
}

.message-user {
    background: #ebf8ff !important;
    padding: 1rem !important;
    border-radius: 8px 8px 0 8px !important;
    border: 1px solid #bee3f8 !important;
    margin: 0.5rem 0 !important;
}

.admin-panel {
    padding: 1.5rem;
    border-radius: 8px;
    background: #f7fafc;
    border: 1px solid var(--border-color);
}

.input-row {
    display: flex;
    gap: 0.5rem;
    margin-top: 1rem;
}

button.primary-btn {
    background-color: var(--primary-color) !important;
    color: white !important;
    border-radius: 6px !important;
    padding: 0.5rem 1rem !important;
    font-size: 0.875rem !important;
}

.footer {
    margin-top: 2rem;
    padding-top: 1rem;
    border-top: 1px solid var(--border-color);
    text-align: center;
    font-size: 0.875rem;
    color: #718096;
}

.clear-btn {
    margin-top: 0.5rem;
    opacity: 0.8;
    font-size: 0.875rem;
}

.chat-window {
    margin-bottom: 1rem;
    padding: 1rem;
}

.source-citation {
    font-size: 0.875rem;
    color: #718096;
    border-top: 1px solid var(--border-color);
    margin-top: 0.5rem;
    padding-top: 0.5rem;
}
"""

def admin_login(username, password):
    """Handle admin login"""
    if username == "admin" and check_password(password):
        return create_token(), gr.update(visible=True), gr.update(visible=False)
    return None, gr.update(visible=False), gr.update(visible=True)

def process_file(file, token):
    """Process uploaded file and store in database"""
    if not verify_token(token):
        return "Invalid token. Please log in again."
    
    try:
        file_path = file.name
        file_type = Path(file_path).suffix[1:].lower()
        if file_type not in ["pdf", "txt", "json", "md"]:
            return "Unsupported file type. Please upload PDF, TXT, JSON, or MD files."

        text = doc_processor.extract_text(file_path, file_type)
        embeddings = doc_processor.get_embeddings(text)

        document_id = str(uuid.uuid4())
        metadata = {
            "filename": Path(file_path).name,
            "file_type": file_type,
            "upload_date": datetime.now().isoformat()
        }

        if db_handler.add_document(document_id, text, embeddings, metadata):
            return f"✓ {Path(file_path).name} processed successfully"
        return "× Error storing document in database"

    except Exception as e:
        return f"× Error: {str(e)}"

def get_document_list(token):
    """Get list of uploaded documents"""
    if not verify_token(token):
        return []
    
    documents = db_handler.list_documents()
    if not documents:
        return []
    
    # Convert to list format for dataframe
    return [[doc['filename'], doc['file_type'], doc['upload_date'], doc['id']] 
            for doc in documents]

def delete_selected_document(selected_rows, token):
    """Delete selected document from database"""
    if not verify_token(token):
        return "Invalid token. Please log in again.", None
    
    try:
        if not selected_rows or not isinstance(selected_rows, (list, tuple)):
            return "No document selected", None
        
        # Get the document ID from the selected row (last column)
        doc_id = selected_rows[3]  # ID is in the fourth column
        
        if db_handler.delete_document(doc_id):
            # Get updated document list
            updated_list = get_document_list(token)
            return "✓ Document deleted successfully", updated_list
        return "× Error deleting document", None
    except Exception as e:
        return f"× Error: {str(e)}", None

def format_sources(results):
    """Format source citations"""
    sources = []
    for result in results:
        filename = result.get('metadata', {}).get('filename', 'Unknown Source')
        sources.append(f"- {filename}")
    return "\n".join(sources)

def chat(message, history):
    """Handle user chat interactions"""
    try:
        message_lower = message.lower()
        is_greeting = any(greeting in message_lower for greeting in ["hello", "hi", "hey", "greetings"])
        
        if is_greeting and not history:
            return "Hello! I'm here to help you with information about LPU. How can I assist you?", ""

        # Extract previous messages from history for context
        chat_history = []
        if history:
            for msg in history:
                if isinstance(msg, dict):
                    chat_history.append(f"{msg['role']}: {msg['content']}")

        # Process normal queries
        query_embedding = doc_processor.get_embeddings(message)[0]
        results = db_handler.query_similar(query_embedding)
        
        if not results:
            return "I apologize, but I don't have enough information to answer that question. Please contact LPU support for more details.", ""

        context = "\n".join([result['text'] for result in results])
        sources = format_sources(results)
        
        # Include chat history in prompt for context
        chat_context = "\n".join(chat_history[-4:]) if chat_history else ""  # Last 4 messages for context
        prompt = f"""You are the LPU AI Assistant. Provide a clear, concise response based on the following context:

        Previous Messages:
        {chat_context}

        Context:
        {context}

        Question: {message}

        Keep the response professional and focused on LPU-related information. Include specific details and references where possible."""

        response = model.generate_content(prompt).text

        # Add sources to response with formatting
        response_with_sources = f"{response}\n\n<div class='source-citation'>Sources:\n{sources}</div>"
        
        return response_with_sources, ""  # Return empty string to clear input

    except Exception as e:
        return f"I apologize, but I encountered an error. Please try again or contact support.", ""

def user_message(message, history):
    """Handle user message submission"""
    response, _ = chat(message, history)
    history = history or []
    history.append({"role": "user", "content": message})
    history.append({"role": "assistant", "content": response})
    return "", history

# Create Gradio interface
with gr.Blocks(css=custom_css, theme=gr.themes.Base()) as app:
    with gr.Column(elem_id="app-container"):
        gr.HTML("""
            <div class="header">
                <h1>LPU AI Assistant Prototype</h1>
                <p>Ask me anything about Lovely Professional University</p>
            </div>
        """)

        token_state = gr.State("")

        with gr.Tab("Chat"):
            with gr.Column(elem_classes="chat-interface"):
                chatbot = gr.Chatbot(
                    value=None,
                    label=None,
                    elem_classes=["message-bot", "message-user"],
                    height=450,
                    avatar_images=("https://th.bing.com/th/id/OIP.WIECMJRJhIIAmbZGxVJddwHaGv?rs=1&pid=ImgDetMain", "https://th.bing.com/th/id/OIP.kpO_asrAGtH-pUBQyHiv5AHaE8?rs=1&pid=ImgDetMain"),
                    show_copy_button=True,
                    type="messages",
                )
                with gr.Row(elem_classes="input-row"):
                    txt = gr.Textbox(
                        placeholder="Type your question here...",
                        scale=8,
                        show_label=False,
                        container=False
                    )
                    submit_btn = gr.Button("Ask", elem_classes="primary-btn", scale=1)
                clear_btn = gr.Button("Clear Chat", size="sm", elem_classes="clear-btn")

        with gr.Tab("Admin"):
            with gr.Column(elem_classes="admin-panel") as login_column:
                username = gr.Textbox(label="Username", placeholder="Enter username")
                password = gr.Textbox(label="Password", type="password", placeholder="Enter password")
                login_button = gr.Button("Login", elem_classes="primary-btn")
                login_error = gr.Markdown(visible=False, value="Invalid credentials")

            with gr.Column(visible=False, elem_classes="admin-panel") as admin_panel:
                upload_file = gr.File(label="Upload Document")
                upload_button = gr.Button("Process", elem_classes="primary-btn")
                upload_status = gr.Markdown()

                document_list = gr.Dataframe(
                    headers=["Name", "Type", "Date", "ID"],
                    label="Documents",
                    wrap=True
                )
                with gr.Row():
                    refresh_button = gr.Button("Refresh", elem_classes="primary-btn")
                    delete_button = gr.Button("Delete", elem_classes="primary-btn")

        gr.HTML("""
            <div class="footer">
                <p>AI Assistant by Raj | © 2025 LPU</p>
            </div>
        """)

    # Event handlers
    txt.submit(user_message, [txt, chatbot], [txt, chatbot])
    submit_btn.click(user_message, [txt, chatbot], [txt, chatbot])
    clear_btn.click(lambda: (None, None), None, [chatbot, txt], queue=False)

    login_button.click(
        admin_login,
        inputs=[username, password],
        outputs=[token_state, admin_panel, login_error]
    )

    upload_button.click(
        process_file,
        inputs=[upload_file, token_state],
        outputs=[upload_status]
    )

    refresh_button.click(
        get_document_list,
        inputs=[token_state],
        outputs=[document_list]
    )

    delete_button.click(
        delete_selected_document,
        inputs=[document_list, token_state],
        outputs=[upload_status, document_list]
    )

if __name__ == "__main__":
    app.launch()
