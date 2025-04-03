import os
import gradio as gr
import google.generativeai as genai
from datetime import datetime
import random
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

# Clean, minimalistic CSS with friendly colors
custom_css = """
:root {
    --primary-color: #4f46e5;
    --background-color: #ffffff;
    --text-color: #2d3748;
    --border-color: #e2e8f0;
    --accent-color: #8b5cf6;
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
    animation: fadeIn 0.5s ease-in;
}

.header h1 {
    font-size: 2.5rem;
    font-weight: 600;
    margin-bottom: 0.5rem;
    background: linear-gradient(120deg, var(--primary-color), var(--accent-color));
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}

.header p {
    color: #4a5568;
    font-size: 1.1rem;
}

.chat-interface {
    border: 1px solid var(--border-color);
    border-radius: 12px;
    padding: 1.5rem;
    background: var(--background-color);
    box-shadow: 0 4px 6px rgba(0, 0, 0, 0.05);
}

.message-bot {
    background: #f8f7ff !important;
    padding: 1rem !important;
    border-radius: 12px 12px 12px 0 !important;
    border: 1px solid #e4e2ff !important;
    margin: 0.5rem 0 !important;
    transition: all 0.2s ease;
}

.message-user {
    background: #ebf8ff !important;
    padding: 1rem !important;
    border-radius: 12px 12px 0 12px !important;
    border: 1px solid #bee3f8 !important;
    margin: 0.5rem 0 !important;
    transition: all 0.2s ease;
}

.admin-panel {
    padding: 1.5rem;
    border-radius: 12px;
    background: #f8f7ff;
    border: 1px solid var(--border-color);
}

.input-row {
    display: flex;
    gap: 0.5rem;
    margin-top: 1rem;
}

button.primary-btn {
    background: linear-gradient(120deg, var(--primary-color), var(--accent-color)) !important;
    color: white !important;
    border-radius: 8px !important;
    padding: 0.5rem 1.2rem !important;
    font-size: 0.95rem !important;
    transition: all 0.2s ease !important;
    border: none !important;
}

button.primary-btn:hover {
    transform: translateY(-1px) !important;
    box-shadow: 0 4px 12px rgba(79, 70, 229, 0.2) !important;
}

.footer {
    margin-top: 2rem;
    padding-top: 1rem;
    border-top: 1px solid var(--border-color);
    text-align: center;
    font-size: 0.95rem;
    color: #718096;
}

.clear-btn {
    margin-top: 0.5rem;
    opacity: 0.9;
    font-size: 0.95rem;
    transition: all 0.2s ease;
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

.document-row {
    display: flex;
    align-items: center;
    padding: 0.5rem;
    border-bottom: 1px solid var(--border-color);
}

.document-info {
    flex-grow: 1;
}

.document-actions {
    display: flex;
    gap: 0.5rem;
}

@keyframes fadeIn {
    from {
        opacity: 0;
        transform: translateY(-10px);
    }
    to {
        opacity: 1;
        transform: translateY(0);
    }
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
        return "Invalid token. Please log in again.", None
    
    try:
        if not file:
            return "No file selected.", None
            
        file_path = file.name
        file_type = Path(file_path).suffix[1:].lower()
        if file_type not in ["pdf", "txt", "json", "md"]:
            return "Let's try with a PDF, TXT, JSON, or MD file to enhance our knowledge base! 📚", None

        text = doc_processor.extract_text(file_path, file_type)
        embeddings = doc_processor.get_embeddings(text)

        document_id = str(uuid.uuid4())
        metadata = {
            "filename": Path(file_path).name,
            "file_type": file_type,
            "upload_date": datetime.now().isoformat()
        }

        if db_handler.add_document(document_id, text, embeddings, metadata):
            updated_list = [[doc['filename'], doc['file_type'], 
                           doc['upload_date'][:16].replace('T', ' '), 
                           "🗑️ Delete"] 
                          for doc in db_handler.list_documents()]
            return f"✨ Successfully added {Path(file_path).name} to our knowledge base! Thank you for helping me learn more about LPU!", updated_list
        return "I encountered a small challenge while storing the document. Let's try again! 🌟", None

    except Exception as e:
        return f"A learning opportunity arose! Let's try that again. Error details: {str(e)} 🔄", None

def delete_document(evt: gr.SelectData, token, documents):
    """Delete a document when its delete button is clicked"""
    if not verify_token(token):
        return "Invalid token. Please log in again.", None
    
    try:
        if evt.column != 3:
            return None, documents
            
        doc_to_delete = documents[evt.index]
        filename = doc_to_delete[0]
        
        all_docs = db_handler.list_documents()
        doc_id = next((doc['id'] for doc in all_docs if doc['filename'] == filename), None)
        
        if not doc_id:
            return "I couldn't find that document in my records. Let's try something else! ✨", documents
            
        if db_handler.delete_document(doc_id):
            updated_list = [[doc['filename'], doc['file_type'], 
                           doc['upload_date'][:16].replace('T', ' '), 
                           "🗑️ Delete"] 
                          for doc in db_handler.list_documents()]
            return f"✨ {filename} has been successfully removed from our collection!", updated_list
        return "I encountered a small challenge while removing the document. Let's try again! 🌟", documents
    except Exception as e:
        return f"A learning opportunity arose! Let's try that again. Error details: {str(e)} 🔄", documents

def format_sources(results):
    """Format source citations"""
    sources = []
    for result in results:
        filename = result.get('metadata', {}).get('filename', 'Our Knowledge Base')
        sources.append(f"- {filename}")
    return "\n".join(sources)

def chat(message, history):
    """Handle user chat interactions"""
    try:
        message_lower = message.lower()
        is_greeting = any(greeting in message_lower for greeting in ["hello", "hi", "hey", "greetings", "namaste"])
        
        if is_greeting and not history:
            greetings = [
                "Namaste! I'm your friendly LPU companion! I'd love to tell you all about our amazing university. What would you like to know? 😊",
                "Hi there! Welcome to LPU's virtual family! I'm here to share exciting things about our wonderful campus. What interests you? 🎓",
                "Hey! I'm thrilled to connect with you! LPU is an incredible place of learning and growth. How can I help you explore it? ✨"
            ]
            return random.choice(greetings), ""

        chat_history = []
        if history:
            for msg in history:
                if isinstance(msg, dict):
                    chat_history.append(f"{msg['role']}: {msg['content']}")

        query_embedding = doc_processor.get_embeddings(message)[0]
        results = db_handler.query_similar(query_embedding)
        
        if not results:
            alternatives = [
                "I'm still learning about that aspect of LPU! Our university is so vast and wonderful that there's always something new to discover. Could you ask me about another exciting aspect of LPU? 🌟",
                "While I continue enhancing my knowledge about our fantastic university, maybe I can tell you about other amazing things at LPU? What else interests you? 🎯",
                "I'm currently expanding my understanding of that topic! LPU has so many remarkable features - would you like to explore something else about our prestigious institution? 🌈"
            ]
            return random.choice(alternatives), ""

        context = "\n".join([result['text'] for result in results])
        sources = format_sources(results)
        
        chat_context = "\n".join(chat_history[-4:]) if chat_history else ""
        prompt = f"""You are the friendly and enthusiastic LPU AI Assistant. You love LPU and are incredibly proud of the university's achievements. Remember to:

        - Be warm and engaging while maintaining professionalism
        - Share information Professionally and use data to support your responses , facts and highlight important things.
        - Always use Data driven responses.
        - Highlight LPU's strengths and achievements with pride
        - If discussing challenges, frame them as opportunities for growth
        - Share relatable examples and success stories when relevant
        - Always be encouraging and supportive
        - Include specific details that showcase LPU's excellence

        Previous Messages:
        {chat_context}

        Context:
        {context}

        Question: {message}

        Respond in a way that makes the user feel Satisfied of its questions. Balance friendliness with informative content with data and facts with professionalism."""

        response = model.generate_content(prompt).text

        engagement_phrases = [
            "\n\nIs there anything specific about this that you'd like to explore further? 🤔",
            "\n\nI'm excited to share more about LPU's excellence! What aspect interests you most? ✨",
            "\n\nThis is just one of the many amazing things about LPU! Would you like to know more? 🌟"
        ]
        
        response_with_sources = f"{response}{random.choice(engagement_phrases)}\n\n<div class='source-citation'>Sources:\n{sources}</div>"
        
        return response_with_sources, ""

    except Exception as e:
        return "I'm Still learning my devloper Raj is still building me. At LPU, we believe in continuous improvement. Please try again, and I'll be happy to assist you! 🌟", ""

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
                <h1>Welcome to Your LPU Friend! 🎓</h1>
                <p>Let's explore the wonderful world of Lovely Professional University together! ✨</p>
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
                        placeholder="Share your thoughts or questions about LPU! 💭",
                        scale=8,
                        show_label=False,
                        container=False
                    )
                    submit_btn = gr.Button("Let's Chat! 💫", elem_classes="primary-btn", scale=1)
                clear_btn = gr.Button("Start Fresh ✨", size="sm", elem_classes="clear-btn")

        with gr.Tab("Admin"):
            with gr.Column(elem_classes="admin-panel") as login_column:
                username = gr.Textbox(label="Username", placeholder="Enter your username ✨")
                password = gr.Textbox(label="Password", type="password", placeholder="Enter your password 🔒")
                login_button = gr.Button("Log In ✨", elem_classes="primary-btn")
                login_error = gr.Markdown(visible=False, value="Let's try those credentials again! 🔄")

            with gr.Column(visible=False, elem_classes="admin-panel") as admin_panel:
                with gr.Column() as upload_section:
                    gr.Markdown("### Share Your Knowledge 📚")
                    upload_file = gr.File(label="Choose a Document to Share ✨")
                    with gr.Row():
                        with gr.Column(scale=4):
                            upload_status = gr.Markdown()
                        with gr.Column(scale=1):
                            with gr.Row():
                                upload_button = gr.Button("Process ✨", elem_classes="primary-btn")

                with gr.Column() as document_section:
                    with gr.Row():
                        gr.Markdown("### Our Knowledge Collection 📚")
                    with gr.Row():
                        refresh_btn = gr.Button("🔄 Refresh List", elem_classes="primary-btn", scale=0)
                    document_list = gr.Dataframe(
                            headers=["Name", "Type", "Date", "Actions"],
                            label="",
                            value=[],
                            interactive=False,
                            wrap=True,
                            row_count=(5, "fixed")
                        )

        gr.HTML("""
            <div class="footer">
                <p>Your Friendly AI Guide by Raj | © 2025 Lovely Professional University - Think Big 🌟</p>
            </div>
        """)

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
        outputs=[upload_status, document_list],
        show_progress="full"
    )

    document_list.select(
        delete_document,
        inputs=[token_state, document_list],
        outputs=[upload_status, document_list]
    )

    def refresh_documents(token):
        """Refresh the list of documents"""
        if not verify_token(token):
            return None
        return [[doc['filename'], doc['file_type'], 
                doc['upload_date'][:16].replace('T', ' '), 
                "🗑️ Delete"] 
               for doc in db_handler.list_documents()]

    refresh_btn.click(
        refresh_documents,
        inputs=[token_state],
        outputs=[document_list]
    )

if __name__ == "__main__":
    app.launch(server_name="0.0.0.0", server_port=7860, share=True)
