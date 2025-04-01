---
title: LPU AIBOT PROTOTYPE
emoji: 😻
colorFrom: gray
colorTo: red
sdk: gradio
sdk_version: 5.23.2
app_file: app.py
pinned: false
license: apache-2.0
short_description: Prototype version of lpu data bot
---


# RAG Chat Application with Gemini AI

A Retrieval-Augmented Generation (RAG) chat application built with Gradio, Google's Gemini AI, and FAISS. The application features a secure admin interface for document management and a user chat interface that provides answers based on the uploaded knowledge base.

## Features

- **Secure Admin Interface**
  - Password-protected access
  - Upload documents (PDF, TXT, JSON, Markdown)
  - View and manage uploaded documents
  - Delete documents from the knowledge base

- **User Chat Interface**
  - Chat with AI using the knowledge from uploaded documents
  - Contextual responses based on document content
  - Semantic search using embeddings

- **Technical Features**
  - Document text extraction and processing
  - Gemini AI for embeddings and chat responses
  - Efficient vector search with FAISS
  - Persistent storage of embeddings and metadata
  - JWT-based admin authentication

## Setup

1. Clone the repository
```bash
git clone <repository-url>
cd <repository-name>
```

2. Create and activate a virtual environment (recommended)
```bash
python -m venv venv
# On Windows
.\venv\Scripts\activate
# On macOS/Linux
source venv/bin/activate
```

3. Install dependencies
```bash
pip install -r requirements.txt
```

4. Set up environment variables
```bash
# Copy the example .env file
cp .env.example .env
# Edit .env and add your Gemini API key
```

5. Run the application
```bash
python app.py
```

## Usage

### Admin Interface
1. Access the admin interface through the "Admin" tab
2. Login with credentials:
   - Username: admin
   - Password: admin123 (change this in production)
3. Upload documents:
   - Click "Upload Document"
   - Select a file (PDF, TXT, JSON, or MD)
   - Click "Process and Store"
4. Manage documents:
   - View uploaded documents in the table
   - Delete documents using the delete button
   - Refresh the list to see updates

### User Chat Interface
1. Switch to the "Chat" tab
2. Enter your question in the chat input
3. The AI will respond based on the knowledge from uploaded documents

## Security Notes

For production deployment:
1. Change the default admin password in `utils/auth.py`
2. Use a secure secret key for JWT token generation
3. Enable HTTPS
4. Consider implementing additional authentication methods

## File Structure
```
├── app.py              # Main application file
├── requirements.txt    # Python dependencies
├── .env               # Environment variables
├── vector_db/         # FAISS index and metadata storage
│   ├── faiss.index   # Vector embeddings
│   └── metadata.pkl  # Document metadata
└── utils/
    ├── auth.py        # Authentication utilities
    ├── db_handler.py  # Database operations using FAISS
    └── document_processor.py  # Document processing utilities
```

## Dependencies

- gradio - Web interface
- google-generativeai - Gemini AI integration
- faiss-cpu - Vector similarity search
- python-dotenv - Environment variables
- pypdf2 - PDF processing
- markdown2 - Markdown processing
- PyJWT - Authentication
- python-multipart - File upload handling
- numpy - Numerical operations

## Data Persistence

- Document embeddings are stored in a FAISS index
- Document metadata and text are stored in a pickle file
- Both the index and metadata are automatically persisted to disk
- Data can be preserved across application restarts and deployments

## Vector Search

The application uses FAISS (Facebook AI Similarity Search) for efficient similarity search:
- Fast and memory-efficient vector similarity search.
- Optimized for production use
- Supports large-scale document collections
- Easy to deploy and maintain

## Error Handling

- The application includes comprehensive error handling for:
  - File processing errors
  - Database operations
  - Authentication issues
  - API communication errors
  - Vector search operations

## Contributing

Feel free to submit issues and enhancement requests!


