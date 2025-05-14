# import os
# import numpy as np
# import faiss
# import pickle
# from sentence_transformers import SentenceTransformer

# class VectorDB:
#     def __init__(self, db_path="vector_db", embedding_model="all-MiniLM-L6-v2"):
#         """
#         Initialize the vector database.
        
#         Args:
#             db_path: Path to store the vector database files
#             embedding_model: Name of the sentence-transformers model to use
#         """
#         self.db_path = db_path
#         self.index_path = os.path.join(db_path, "faiss_index.bin")
#         self.documents_path = os.path.join(db_path, "documents.pkl")
#         self.model = SentenceTransformer(embedding_model)
#         self.dimension = self.model.get_sentence_embedding_dimension()
        
#         # Create or load the database
#         self.index, self.documents = self._create_or_load_db()
    
#     def _create_or_load_db(self):
#         """Create a new vector database if it doesn't exist, or load existing one."""
#         if not os.path.exists(self.db_path):
#             os.makedirs(self.db_path)
            
#         if os.path.exists(self.index_path) and os.path.exists(self.documents_path):
#             # Load existing database
#             index = faiss.read_index(self.index_path)
#             with open(self.documents_path, 'rb') as f:
#                 documents = pickle.load(f)
#             print(f"Loaded existing database with {len(documents)} documents")
#         else:
#             # Create new database with cosine similarity
#             index = faiss.IndexFlatIP(self.dimension)  # Inner product for cosine similarity
#             documents = []
#             print("Created new vector database")
            
#         return index, documents


    
#     def add_document(self, text):
#         """
#         Embed a text document and add it to the database.
        
#         Args:
#             text: The text document to embed and add
            
#         Returns:
#             int: The document ID
#         """
#         # Generate embedding
#         embedding = self.model.encode([text])[0]
#         embedding = np.array([embedding]).astype('float32')
#         faiss.normalize_L2(embedding)
#         # Add to index
#         self.index.add(embedding)
        
#         # Store the original document
#         doc_id = len(self.documents)
#         self.documents.append(text)
        
#         # Save the updated database
#         self._save_db()
        
#         return doc_id
    
#     def add_documents(self, texts):
#         """
#         Embed multiple text documents and add them to the database.
        
#         Args:
#             texts: List of text documents to embed and add
            
#         Returns:
#             list: List of document IDs
#         """
#         # Generate embeddings
#         embeddings = self.model.encode(texts)
#         embeddings = np.array(embeddings).astype('float32')
        
#         # Add to index
#         self.index.add(embeddings)
        
#         # Store the original documents
#         start_id = len(self.documents)
#         self.documents.extend(texts)
        
#         # Save the updated database
#         self._save_db()
        
#         return list(range(start_id, len(self.documents)))
    
#     def search(self, query, k=5):
#         """
#         Search the database for documents similar to the query.
        
#         Args:
#             query: The query text
#             k: Number of results to return
            
#         Returns:
#             list: List of (document, similarity score) tuples
#         """
#         # Normalize the query embedding for cosine similarity
#         query_embedding = self.model.encode([query])[0]
#         query_embedding = np.array([query_embedding]).astype('float32')
#         faiss.normalize_L2(query_embedding)  # Normalize for cosine similarity
        
#         # Search (with inner product, higher is better)
#         distances, indices = self.index.search(query_embedding, k)
        
#         # Return results
#         results = []
#         for i, idx in enumerate(indices[0]):
#             if idx != -1:
#                 # For inner product with normalized vectors, the score is already in [-1,1]
#                 # Convert to [0,1] range
#                 similarity = (distances[0][i] + 1) / 2
#                 results.append((self.documents[idx], similarity))
                    
#         return results
    
#     def _save_db(self):
#         """Save the database to disk."""
#         faiss.write_index(self.index, self.index_path)
#         with open(self.documents_path, 'wb') as f:
#             pickle.dump(self.documents, f)

import os
import numpy as np
import faiss
import pickle
from sentence_transformers import SentenceTransformer

class VectorDB:
    def __init__(self, db_path="vector_db", embedding_model="all-MiniLM-L6-v2"):
        """
        Initialize the vector database.
        
        Args:
            db_path: Path to store the vector database files
            embedding_model: Name of the sentence-transformers model to use
        """
        self.db_path = db_path
        self.index_path = os.path.join(db_path, "faiss_index.bin")
        self.documents_path = os.path.join(db_path, "documents.pkl")
        self.model = SentenceTransformer(embedding_model)
        self.dimension = self.model.get_sentence_embedding_dimension()
        
        # Create or load the database
        self.index, self.documents = self._create_or_load_db()
    
    def _create_or_load_db(self):
        """Create a new vector database if it doesn't exist, or load existing one."""
        if not os.path.exists(self.db_path):
            os.makedirs(self.db_path)
            
        if os.path.exists(self.index_path) and os.path.exists(self.documents_path):
            # Load existing database
            index = faiss.read_index(self.index_path)
            with open(self.documents_path, 'rb') as f:
                documents = pickle.load(f)
            print(f"Loaded existing database with {len(documents)} documents")
        else:
            # Create new database with cosine similarity
            # First normalize vectors, then use inner product for cosine similarity
            index = faiss.IndexFlatIP(self.dimension)  # Inner product for cosine similarity
            documents = []
            print("Created new vector database")
            
        return index, documents
    
    def add_document(self, text):
        """
        Embed a text document and add it to the database.
        
        Args:
            text: The text document to embed and add
            
        Returns:
            int: The document ID
        """
        # Generate embedding
        embedding = self.model.encode([text])[0]
        embedding = np.array([embedding]).astype('float32')
        
        # Normalize for cosine similarity
        faiss.normalize_L2(embedding)
        
        # Add to index
        self.index.add(embedding)
        
        # Store the original document
        doc_id = len(self.documents)
        self.documents.append(text)
        
        # Save the updated database
        self._save_db()
        
        return doc_id
    
    def add_documents(self, texts):
        """
        Embed multiple text documents and add them to the database.
        
        Args:
            texts: List of text documents to embed and add
            
        Returns:
            list: List of document IDs
        """
        if not texts:
            return []
            
        # Generate embeddings
        embeddings = self.model.encode(texts)
        embeddings = np.array(embeddings).astype('float32')
        
        # Normalize for cosine similarity
        faiss.normalize_L2(embeddings)
        
        # Add to index
        self.index.add(embeddings)
        
        # Store the original documents
        start_id = len(self.documents)
        self.documents.extend(texts)
        
        # Save the updated database
        self._save_db()
        
        return list(range(start_id, len(self.documents)))
    
    def search(self, query, k=5):
        """
        Search the database for documents similar to the query.
        
        Args:
            query: The query text
            k: Number of results to return
            
        Returns:
            list: List of (document, similarity score) tuples
        """
        # Handle empty database case
        if len(self.documents) == 0:
            return []
            
        # Limit k to the number of documents
        k = min(k, len(self.documents))
        
        # Generate query embedding
        query_embedding = self.model.encode([query])[0]
        query_embedding = np.array([query_embedding]).astype('float32')
        
        # Normalize for cosine similarity
        faiss.normalize_L2(query_embedding)
        
        # Search (with inner product, higher is better for similarity)
        distances, indices = self.index.search(query_embedding, k)
        
        # Return results
        results = []
        for i, idx in enumerate(indices[0]):
            if idx != -1:  # FAISS returns -1 for empty slots
                # For normalized vectors with inner product, the score is in [-1,1]
                # Convert to [0,1] range
                similarity = (distances[0][i] + 1) / 2
                results.append((self.documents[idx], similarity))
                
        return results
    
    def _save_db(self):
        """Save the database to disk."""
        faiss.write_index(self.index, self.index_path)
        with open(self.documents_path, 'wb') as f:
            pickle.dump(self.documents, f)
            
    def clear_db(self):
        """Clear all documents from the database."""
        # Create new empty index
        self.index = faiss.IndexFlatIP(self.dimension)
        self.documents = []
        
        # Save the empty database
        self._save_db()
        print("Database cleared")
        
    def get_document_count(self):
        """Get the number of documents in the database."""
        return len(self.documents)