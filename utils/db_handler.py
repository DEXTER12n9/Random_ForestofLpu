import faiss
import numpy as np
import pickle
import os
from pathlib import Path
from datetime import datetime

class DatabaseHandler:
    def __init__(self, persist_directory: str = "vector_db"):
        """Initialize the FAISS index and storage"""
        self.persist_directory = Path(persist_directory)
        self.persist_directory.mkdir(parents=True, exist_ok=True)
        
        self.index_path = self.persist_directory / "faiss.index"
        self.metadata_path = self.persist_directory / "metadata.pkl"
        
        # Initialize or load the FAISS index
        self.dimension = 768  # Gemini's embedding dimension
        if self.index_path.exists():
            self.index = faiss.read_index(str(self.index_path))
            with open(self.metadata_path, 'rb') as f:
                self.metadata = pickle.load(f)
        else:
            self.index = faiss.IndexFlatL2(self.dimension)
            self.metadata = {
                'documents': {},  # Document text and metadata
                'id_map': []     # Map FAISS ids to document ids
            }

    def add_document(self, document_id: str, text: str, embeddings: list, metadata: dict):
        """Add a document and its embeddings to the database"""
        try:
            # Convert embeddings to numpy array
            embeddings_array = np.array(embeddings).astype('float32')
            
            # Add embeddings to FAISS index
            self.index.add(embeddings_array)
            
            # Store document metadata
            start_idx = len(self.metadata['id_map'])
            for i in range(len(embeddings)):
                self.metadata['id_map'].append(document_id)
            
            # Store document information
            self.metadata['documents'][document_id] = {
                'text': text,
                'metadata': metadata,
                'chunk_indices': list(range(start_idx, start_idx + len(embeddings)))
            }
            
            # Persist changes
            self.persist()
            return True
            
        except Exception as e:
            print(f"Error adding document to database: {e}")
            return False

    def query_similar(self, query_embedding: list, n_results: int = 5) -> list:
        """Query the database for similar documents"""
        try:
            # Convert query embedding to numpy array
            query_array = np.array([query_embedding]).astype('float32')
            
            # Search the FAISS index
            D, I = self.index.search(query_array, n_results)
            
            results = []
            seen_docs = set()
            
            # Get unique documents from the results
            for idx in I[0]:
                if idx >= len(self.metadata['id_map']):
                    continue
                    
                doc_id = self.metadata['id_map'][idx]
                if doc_id in seen_docs:
                    continue
                
                seen_docs.add(doc_id)
                doc_info = self.metadata['documents'][doc_id]
                results.append({
                    'id': doc_id,
                    'text': doc_info['text'],
                    'metadata': doc_info['metadata']
                })
            
            return results
            
        except Exception as e:
            print(f"Error querying database: {e}")
            return []

    def delete_document(self, document_id: str) -> bool:
        """Delete a document from the database"""
        try:
            if document_id not in self.metadata['documents']:
                return False
            
            # Get indices to remove
            indices_to_remove = set(self.metadata['documents'][document_id]['chunk_indices'])
            
            # Create a new index
            new_index = faiss.IndexFlatL2(self.dimension)
            
            # Get all vectors
            vectors = self.index.reconstruct_n(0, self.index.ntotal)
            
            # Create new metadata
            new_metadata = {
                'documents': {},
                'id_map': []
            }
            
            # Add vectors and update metadata, skipping the deleted document
            current_idx = 0
            for old_idx in range(len(self.metadata['id_map'])):
                if old_idx not in indices_to_remove:
                    doc_id = self.metadata['id_map'][old_idx]
                    new_index.add(vectors[old_idx:old_idx+1])
                    new_metadata['id_map'].append(doc_id)
                    
                    # Update document indices if needed
                    if doc_id != document_id:
                        if doc_id not in new_metadata['documents']:
                            new_metadata['documents'][doc_id] = {
                                'text': self.metadata['documents'][doc_id]['text'],
                                'metadata': self.metadata['documents'][doc_id]['metadata'],
                                'chunk_indices': []
                            }
                        new_metadata['documents'][doc_id]['chunk_indices'].append(current_idx)
                        current_idx += 1
            
            # Update the instance variables
            self.index = new_index
            self.metadata = new_metadata
            
            # Persist changes
            self.persist()
            return True
            
        except Exception as e:
            print(f"Error deleting document: {e}")
            return False

    def list_documents(self) -> list:
        """List all unique documents in the database"""
        try:
            documents = []
            for doc_id, doc_info in self.metadata['documents'].items():
                documents.append({
                    'id': doc_id,
                    'filename': doc_info['metadata'].get('filename', ''),
                    'file_type': doc_info['metadata'].get('file_type', ''),
                    'upload_date': doc_info['metadata'].get('upload_date', '')
                })
            return documents
        except Exception as e:
            print(f"Error listing documents: {e}")
            return []

    def persist(self):
        """Persist the index and metadata to disk"""
        try:
            # Save FAISS index
            faiss.write_index(self.index, str(self.index_path))
            
            # Save metadata
            with open(self.metadata_path, 'wb') as f:
                pickle.dump(self.metadata, f)
            
            return True
        except Exception as e:
            print(f"Error persisting database: {e}")
            return False
