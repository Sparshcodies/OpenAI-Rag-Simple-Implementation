import os
from typing import List, Dict, Optional, Tuple
from utils.logging_utils import log_error
try:
    import chromadb
    from sentence_transformers import SentenceTransformer
except Exception:
    raise ImportError("Install required modules first: pip install -r requirements.txt")


class VectorStore:
    def __init__(self, db_path: str = "./chroma_db", model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
        try:
            os.makedirs(db_path, exist_ok=True)
            self.client = chromadb.PersistentClient(path=db_path)
            self.col = self.client.get_or_create_collection("chunks", metadata={"hnsw:space": "cosine"})
            self.model = SentenceTransformer(model_name)
        except Exception as e:
            log_error(f"VectorStore init failed: {str(e)}")
            print("Error initializing VectorStore. Check logs for details.")
            
    def embed(self, texts: List[str]) -> List[List[float]]:
        if not texts:
            log_error("Embed called with empty text list.")
            return []
        return self.model.encode(texts, normalize_embeddings=True).tolist()

    def upsert(self, ids: List[str], texts: List[str], metadatas: Optional[List[Dict]] = None) -> None:
        if not ids or not texts:
            log_error("Upsert skipped due to missing ids/texts.")
            return
        try:
            vecs = self.embed(texts)
            self.col.delete(ids=ids)
            self.col.add(ids=ids, embeddings=vecs, documents=texts, metadatas=metadatas or None)
        except Exception as e:
            log_error(f"Upsert failed: {str(e)}")

    def delete(self, ids: List[str]) -> None:
        try:
            self.col.delete(ids=ids)
        except Exception as e:
            log_error(f"Delete failed: {str(e)}")

    def search(self, text: str, k: int = 5) -> List[Tuple[str, float, str, Dict]]:
        try:
            qvec = self.embed([text])[0]
            r = self.col.query(query_embeddings=[qvec], n_results=k)

            ids = r.get("ids", [[]])[0]
            docs = r.get("documents", [[]])[0]
            metas = r.get("metadatas", [[]])[0]
            dists = r.get("distances", [[]])[0]
            

            results = [(i, max(0, min(1, float(1 - d))), doc, meta) for i, d, doc, meta in zip(ids, dists, docs, metas)]
            return results
        except Exception as e:
            log_error(f"Search failed: {str(e)}")
            
    def delete_by_filename(self, file_name: str):
        try:
            all_docs = self.col.get()
            ids_to_delete = []

            for ids, metas in zip(all_docs["ids"], all_docs["metadatas"]):
                if metas and metas.get("file") == file_name:
                    ids_to_delete.append(ids)

            if ids_to_delete:
                self.col.delete(ids=ids_to_delete)
        except Exception as e:
            log_error(f"Delete by filename failed: {str(e)}")
            
store = VectorStore()
