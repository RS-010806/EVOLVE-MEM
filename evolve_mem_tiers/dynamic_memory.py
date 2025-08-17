"""
EVOLVE-MEM: Dynamic Memory Network (Tier 1)

Handles raw experience ingestion, embedding, and note creation.
- Uses gte-small for embeddings (open-source, fast, accurate)
- Stores notes in ChromaDB for efficient similarity search
- Simplified and focused on core functionality

"""
from sentence_transformers import SentenceTransformer
import chromadb
from chromadb.config import Settings
from datetime import datetime
import uuid
import json
from typing import Dict, List, Optional
import logging

class DynamicMemoryNetwork:
    """
    Tier 1: Ingests raw experiences and creates structured memory notes.
    Simplified version focused on core functionality.
    """
    def __init__(self, embedding_model: str = 'thenlper/gte-small'):
        """Initialize the dynamic memory network."""
        try:
            self.model = SentenceTransformer(embedding_model)
            logging.info(f"Loaded embedding model: {embedding_model}")
        except Exception as e:
            logging.error(f"Failed to load embedding model: {e}", exc_info=True)
            raise
        
        try:
            self.chroma_client = chromadb.Client(Settings(anonymized_telemetry=False))
            # Use get_or_create to avoid UniqueConstraintError when rerunning
            if hasattr(self.chroma_client, 'get_or_create_collection'):
                self.collection = self.chroma_client.get_or_create_collection('evolve_mem_dynamic')
            else:
                # Fallback for older clients: try create, else get
                try:
                    self.collection = self.chroma_client.create_collection('evolve_mem_dynamic')
                except Exception:
                    self.collection = self.chroma_client.get_collection('evolve_mem_dynamic')
            logging.info("ChromaDB collection created successfully")
        except Exception as e:
            logging.error(f"Failed to initialize ChromaDB: {e}", exc_info=True)
            raise
        
        self.notes = {}  # id -> note dict
        self.note_count = 0

    def add_note(self, text: str, metadata: Optional[Dict] = None) -> Dict:
        """Add a new experience as a memory note."""
        if not text or not text.strip():
            logging.warning("Attempted to add empty text, skipping")
            # Return a minimal note to prevent pipeline failure
            return {
                'id': f"empty_{int(datetime.now().timestamp())}",
                'content': '',
                'timestamp': datetime.now().isoformat(),
                'embedding': None,
                'tags': [],
                'context': '',
                'keywords': [],
                'links': [],
                'metadata': {'error': 'Empty text provided'}
            }
        
        try:
            # Generate embedding
            embedding = self.model.encode(text).tolist()
            
            # Create note ID
            note_id = str(uuid.uuid4())
            
            # Prepare metadata
            note_metadata = {
                'timestamp': datetime.now().isoformat(),
                'note_count': self.note_count,
                'content_length': len(text)
            }
            if metadata:
                note_metadata.update(metadata)
            
            # Create note structure
            note = {
                'id': note_id,
                'content': text,
                'timestamp': note_metadata['timestamp'],
                'embedding': embedding,
                'tags': [],
                'context': '',
                'keywords': [],
                'links': [],
                'metadata': note_metadata
            }
            
            # Store in memory
            self.notes[note_id] = note
            
            # Store in ChromaDB
            try:
                self.collection.add(
                    documents=[text],
                    embeddings=[embedding],
                    ids=[note_id],
                    metadatas=[note_metadata]
                )
            except Exception as e:
                logging.error(f"Failed to add note to ChromaDB: {e}")
                # Note is still stored in memory as fallback
            
            self.note_count += 1
            logging.info(f"Added note {note_id} (total: {self.note_count})")
            
            return note
            
        except Exception as e:
            logging.error(f"Failed to add note: {e}")
            # Return a fallback note to prevent pipeline failure
            return {
                'id': f"fallback_{int(datetime.now().timestamp())}",
                'content': text[:100] + "..." if len(text) > 100 else text,
                'timestamp': datetime.now().isoformat(),
                'embedding': None,
                'tags': [],
                'context': '',
                'keywords': [],
                'links': [],
                'metadata': {'error': str(e)}
            }

    def get_note(self, note_id: str) -> Optional[Dict]:
        """Retrieve a note by ID."""
        return self.notes.get(note_id)

    def search(self, query: str, top_k: int = 5) -> Dict:
        """Search for similar notes using embeddings."""
        if not query or not query.strip():
            logging.warning("Empty query provided for search")
            return {'ids': [], 'documents': [], 'metadatas': [], 'distances': []}
        
        try:
            # Generate query embedding
            query_embedding = self.model.encode(query).tolist()
            
            # Search in ChromaDB
            try:
                results = self.collection.query(
                    query_embeddings=[query_embedding], 
                    n_results=top_k
                )
                return results
            except Exception as e:
                logging.error(f"ChromaDB search failed: {e}")
                # Fallback to memory search
                return self._fallback_search(query, top_k)
            
        except Exception as e:
            logging.error(f"Search failed: {e}")
            return {'ids': [], 'documents': [], 'metadatas': [], 'distances': []}
    
    def _fallback_search(self, query: str, top_k: int) -> Dict:
        """Fallback search using memory dict when ChromaDB fails."""
        try:
            # Simple text-based search as fallback
            query_lower = query.lower()
            matches = []
            
            for note_id, note in self.notes.items():
                if query_lower in note['content'].lower():
                    matches.append({
                        'id': note_id,
                        'content': note['content'],
                        'metadata': note.get('metadata', {}),
                        'score': 1.0  # Simple binary match
                    })
            
            # Sort by score and take top_k
            matches.sort(key=lambda x: x['score'], reverse=True)
            matches = matches[:top_k]
            
            return {
                'ids': [[m['id'] for m in matches]],
                'documents': [[m['content'] for m in matches]],
                'metadatas': [[m['metadata'] for m in matches]],
                'distances': [[1.0 - m['score'] for m in matches]]
            }
        except Exception as e:
            logging.error(f"Fallback search failed: {e}")
            return {'ids': [], 'documents': [], 'metadatas': [], 'distances': []}

    def get_stats(self) -> Dict:
        """Get statistics about the memory network."""
        return {
            'total_notes': len(self.notes),
            'note_count': self.note_count,
            'embedding_dimension': len(self.model.encode("test")) if self.notes else 0
        }

    def clear(self):
        """Clear all notes (for testing/reset)."""
        self.notes.clear()
        self.note_count = 0
        try:
            self.chroma_client.delete_collection('evolve_mem_dynamic')
            self.collection = self.chroma_client.create_collection('evolve_mem_dynamic')
            logging.info("Memory network cleared")
        except Exception as e:
            logging.error(f"Failed to clear ChromaDB: {e}") 