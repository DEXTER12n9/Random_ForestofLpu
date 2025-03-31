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

# Load environment variables
load_dotenv()

# Initialize components
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
if not GEMINI_API_KEY:
    raise ValueError("GEMINI_API_KEY not found in environment variables")

genai.configure(api_key=GEMINI_API_KEY)
doc_processor = DocumentProcessor(GEMINI_API_KEY)
db_handler = DatabaseHandler()

# Configure Gemini model
generation_config = {
    "temperature": 0.7,  # Slightly higher for more natural conversation
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

# Custom CSS for better UI
custom_css = """
#app-container {
    max-width: 1200px;
    margin: auto;
}

.main-header {
    background-color: #1e3d59;
    color: white;
    padding: 20px;
    border-radius: 10px;
    margin-bottom: 20px;
    text-align: center;
}

.chat-container {
    border-radius: 15px;
    box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
}

.admin-panel {
    background-color: #f5f5f5;
    padding: 20px;
    border-radius: 10px;
}

.footer-text {
    text-align: center;
    font-size: 0.9em;
    color: #666;
    margin-top: 20px;
}

/* Custom button styling */
button.primary-btn {
    background-color: #1e3d59 !important;
    color: white !important;
}

/* Chat message styling */
.message-bot {
    background-color: #e8f4f8 !important;
    border-radius: 15px !important;
    padding: 15px !important;
}

.message-user {
    background-color: #f0f0f0 !important;
    border-radius: 15px !important;
    padding: 15px !important;
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

        # Extract text and generate embeddings
        text = doc_processor.extract_text(file_path, file_type)
        embeddings = doc_processor.get_embeddings(text)

        # Store in database
        document_id = str(uuid.uuid4())
        metadata = {
            "filename": Path(file_path).name,
            "file_type": file_type,
            "upload_date": datetime.now().isoformat()
        }

        if db_handler.add_document(document_id, text, embeddings, metadata):
            return f"File processed and stored successfully: {Path(file_path).name}"
        return "Error storing document in database."

    except Exception as e:
        return f"Error processing file: {str(e)}"

def delete_document(doc_id, token):
    """Delete document from database"""
    if not verify_token(token):
        return "Invalid token. Please log in again.", None
    
    if db_handler.delete_document(doc_id):
        return "Document deleted successfully.", get_document_list(token)
    return "Error deleting document.", None

def get_document_list(token):
    """Get list of uploaded documents"""
    if not verify_token(token):
        return None
    
    documents = db_handler.list_documents()
    if not documents:
        return []
    
    return [[doc['filename'], doc['file_type'], doc['upload_date'], doc['id']] 
            for doc in documents]

def chat(message, history):
    """Handle user chat interactions with LPU-specific customization"""
    try:
        # Handle greetings and common queries
        message_lower = message.lower()
        if any(greeting in message_lower for greeting in ["hello", "hi", "hey", "greetings"]):
            return "Hello! I'm the LPU Assistant, here to help you with information about Lovely Professional University. How can I assist you today?"

        # Generate embedding for user query
        query_embedding = doc_processor.get_embeddings(message)[0]
        
        # Get relevant documents
        results = db_handler.query_similar(query_embedding)
        if not results:
            return """I apologize, but I don't have enough information to answer your question at the moment. 
                    Please note that I'm an AI assistant specifically trained to help with LPU-related queries. 
                    You can contact the university directly for more detailed information.
                    
                    This chatbot is maintained by Raj (Developer) and is regularly updated with new information."""

        # Combine relevant documents for context
        context = "\n".join([result['text'] for result in results])
        
        # Prepare prompt for Gemini with LPU-specific instructions
        prompt = f"""You are the AI Assistant for Lovely Professional University (LPU). 
        Respond in a professional yet friendly manner, maintaining the tone of a university assistant.
        Base your response on the following context, and if you cannot find the specific information,
        suggest contacting the relevant department at LPU.

        Context:
        {context}

        Question: {message}

        Remember to:
        - Be polite and professional
        - Use "we" when referring to LPU
        - Acknowledge if you're not sure about something
        - Suggest relevant LPU resources when appropriate
        """

        # Generate response
        response = model.generate_content(prompt)
        
        # Add footer to response
        response_text = response.text + "\n\n_Note: This AI assistant is maintained by Raj (Developer) and is regularly updated with new information to serve you better._"
        return response_text

    except Exception as e:
        return f"""I apologize, but I encountered an error processing your request. Please try again or contact LPU support if the issue persists.
                Error details: {str(e)}
                
                Note: This AI assistant is maintained by Raj (Developer) and is regularly updated with new information."""

# Create Gradio interface
with gr.Blocks(css=custom_css) as app:
    with gr.Column(elem_id="app-container"):
        gr.HTML("""
            <div class="main-header">
                <h1>🎓 LPU AI Assistant</h1>
                <p>Your knowledgeable guide to Lovely Professional University</p>
            </div>
        """)

        # State variables
        token_state = gr.State("")

        # Admin login interface
        with gr.Tab("Admin"):
            with gr.Column(elem_classes="admin-panel") as login_column:
                username = gr.Textbox(label="Username", placeholder="Enter admin username")
                password = gr.Textbox(label="Password", type="password", placeholder="Enter password")
                login_button = gr.Button("Login", elem_classes="primary-btn")
                login_error = gr.Markdown(visible=False, value="Invalid credentials")

            with gr.Column(visible=False, elem_classes="admin-panel") as admin_panel:
                gr.Markdown("### Document Management")
                upload_file = gr.File(label="Upload Document")
                upload_button = gr.Button("Process and Store", elem_classes="primary-btn")
                upload_status = gr.Markdown()

                gr.Markdown("### Document Library")
                document_list = gr.Dataframe(
                    headers=["Filename", "Type", "Upload Date", "ID"],
                    label="Uploaded Documents"
                )
                with gr.Row():
                    refresh_button = gr.Button("Refresh List", elem_classes="primary-btn")
                    delete_button = gr.Button("Delete Selected", elem_classes="primary-btn")

        # User chat interface
        with gr.Tab("Chat"):
            with gr.Column(elem_classes="chat-container"):
                chatbot = gr.ChatInterface(
                    chat,
                    chatbot=gr.Chatbot(
                        show_label=False,
                        elem_classes=["message-bot", "message-user"],
                        height=500
                    ),
                    textbox=gr.Textbox(
                        placeholder="Ask me anything about LPU...",
                        container=False,
                        scale=7
                    ),
                    title="Chat with LPU Assistant",
                    description="I'm here to help you with information about Lovely Professional University.",
                    theme="soft"
                )

        gr.HTML("""
            <div class="footer-text">
                <p>© 2024 Lovely Professional University | AI Assistant developed and maintained by Raj</p>
                <p>Regularly updated with new information to serve you better</p>
            </div>
        """)

    # Event handlers
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
        delete_document,
        inputs=[document_list, token_state],
        outputs=[upload_status, document_list]
    )

if __name__ == "__main__":
    app.launch()
