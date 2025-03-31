import json
import markdown2
from PyPDF2 import PdfReader
import google.generativeai as genai
from pathlib import Path

class DocumentProcessor:
    def __init__(self, api_key):
        genai.configure(api_key=api_key)
        self.model = genai.GenerativeModel('gemini-1.5-flash-8b')
        self.embedding_model = 'models/embedding-001'  # This is still used for embeddings as FLASH-8B doesn't generate embeddings

    def extract_text(self, file_path: str, file_type: str) -> str:
        """Extract text content from different file types"""
        path = Path(file_path)
        
        if file_type == "pdf":
            return self._extract_from_pdf(path)
        elif file_type == "txt":
            return self._extract_from_text(path)
        elif file_type == "json":
            return self._extract_from_json(path)
        elif file_type == "md":
            return self._extract_from_markdown(path)
        else:
            raise ValueError(f"Unsupported file type: {file_type}")

    def _extract_from_pdf(self, path: Path) -> str:
        """Extract text from PDF files"""
        reader = PdfReader(path)
        text = ""
        for page in reader.pages:
            text += page.extract_text() + "\n"
        return text

    def _extract_from_text(self, path: Path) -> str:
        """Extract text from plain text files"""
        return path.read_text(encoding='utf-8')

    def _extract_from_json(self, path: Path) -> str:
        """Extract text from JSON files"""
        data = json.loads(path.read_text(encoding='utf-8'))
        return json.dumps(data, indent=2)

    def _extract_from_markdown(self, path: Path) -> str:
        """Extract text from Markdown files"""
        md_text = path.read_text(encoding='utf-8')
        html = markdown2.markdown(md_text)
        return html

    def get_embeddings(self, text: str) -> list:
        """Generate embeddings for the text using Gemini API"""
        # Split text into chunks if it's too long
        chunks = self._chunk_text(text, max_length=1000)
        embeddings = []
        
        for chunk in chunks:
            try:
                embedding = genai.embed_content(
                    model=self.embedding_model,
                    content=chunk,
                    task_type="retrieval_document"
                )
                embeddings.append(embedding['embedding'])
            except Exception as e:
                print(f"Error generating embedding: {e}")
                continue
        
        return embeddings

    def _chunk_text(self, text: str, max_length: int) -> list:
        """Split text into chunks of maximum length"""
        words = text.split()
        chunks = []
        current_chunk = []
        current_length = 0

        for word in words:
            if current_length + len(word) + 1 > max_length:
                chunks.append(" ".join(current_chunk))
                current_chunk = [word]
                current_length = len(word)
            else:
                current_chunk.append(word)
                current_length += len(word) + 1

        if current_chunk:
            chunks.append(" ".join(current_chunk))

        return chunks
