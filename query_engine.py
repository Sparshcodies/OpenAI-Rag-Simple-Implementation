import os
from typing import Dict, Any
from dotenv import load_dotenv
from utils.vector_utils import VectorStore
from utils.logging_utils import log_error, log_query
from openai import OpenAI

load_dotenv()

class QueryEngine:
    def __init__(self, db_path: str = "./chroma_db", model_name: str = "gpt-4o-mini"):
        try:
            self.store = VectorStore(db_path=db_path)
            key = os.environ.get("OPENAI_API_KEY")
            self.client = OpenAI(api_key=key) if key else None
            self.model = model_name
        except Exception as e:
            log_error(f"Initialization failed: {str(e)}")
            print("Error initializing QueryEngine. Check logs for details.")

    def ask(self, query: str, top_k: int = 4, threshold: float = 0.3) -> Dict[str, Any]:
        try:
            hits = self.store.search(query, k=top_k)
            if not hits or hits[0][1] < threshold:
                log_error("No relevant context found for the query.")
                log_query(query, answer)
                return {"answer": "I don't have enough information in the uploaded documents.", "sources": []}

            context = "\n\n".join([h[2] for h in hits])

            if not self.client:
                answer = hits[0][2]
                log_error("API key missing. Returning top context instead.")
                log_query(query, answer)
                return {"answer": answer, "sources": [{"id": h[0], "similarity": h[1]} for h in hits]}

            prompt = f"""
            You answer strictly from the context below. If the answer is not fully present, reply only:
            "I don't have enough information in the uploaded documents."

            Context:
            {context}

            Question: {query}

            Answer:
            """.strip()

            res = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0
            )

            answer = res.choices[0].message.content.strip()
            log_query(query, answer)
            return {"answer": answer, "sources": [{"id": h[0], "similarity": h[1]} for h in hits]}
        except Exception as e:
            log_error(f"Error in ask(): {str(e)}")
            return {"answer": "An internal error occurred while processing the query.", "sources": []}